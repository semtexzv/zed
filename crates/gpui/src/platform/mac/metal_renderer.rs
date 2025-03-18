use super::metal_atlas::MetalAtlas;
use crate::{
    point, size, AtlasTextureId, AtlasTextureKind, AtlasTile, Background, Bounds, ContentMask,
    DevicePixels, MonochromeSprite, PaintSurface, Path, PathId, PathVertex, PolychromeSprite, 
    PrimitiveBatch, Quad, ScaledPixels, Scene, Shadow, Size, Surface, Underline, scene::DamageRegion,
};
use anyhow::{anyhow, Result};
use block::ConcreteBlock;
use cocoa::{
    base::{NO, YES},
    foundation::{NSSize, NSUInteger},
    quartzcore::AutoresizingMask,
};
use collections::HashMap;
use core_foundation::base::TCFType;
use foreign_types::ForeignType;
use media::core_video::CVMetalTextureCache;
use metal::{CAMetalLayer, CommandQueue, MTLPixelFormat, MTLResourceOptions, NSRange};
use objc::{self, msg_send, sel, sel_impl};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{cell::Cell, ffi::c_void, mem, ptr, sync::Arc};

// Exported to metal
pub(crate) type PointF = crate::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: crate::Size<f32>,
    _transparent: bool,
) -> Renderer {
    MetalRenderer::new(context)
}

pub(crate) struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(&mut self, device: &metal::Device) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            device.new_buffer(
                self.buffer_size as u64,
                MTLResourceOptions::StorageModeManaged,
            )
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub(crate) struct MetalRenderer {
    device: metal::Device,
    layer: metal::MetalLayer,
    presents_with_transaction: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    stencil_pipeline_state: metal::RenderPipelineState, // Pipeline for stencil buffer operations
    unit_vertices: metal::Buffer,
    quad_vertices: metal::Buffer, // Buffer for stencil quad vertices
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: CVMetalTextureCache,
    stencil_texture: Option<metal::Texture>, // Cache for stencil texture
}

// Structure for representing a damage region quad for stencil marking
#[repr(C)]
struct StencilQuad {
    origin: PointF,
    size: Size<f32>,
}

impl MetalRenderer {
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        let mut devices = metal::Device::all();
        devices.sort_by_key(|device| (device.is_removable(), device.is_low_power()));
        let Some(device) = devices.pop() else {
            log::error!("unable to access a compatible graphics device");
            std::process::exit(1);
        };

        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_opaque(false);
        layer.set_maximum_drawable_count(3);
        unsafe {
            let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: NO];
            let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: YES];
            let _: () = msg_send![
                &*layer,
                setAutoresizingMask: AutoresizingMask::WIDTH_SIZABLE
                    | AutoresizingMask::HEIGHT_SIZABLE
            ];
        }
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::R16Float,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );

        let command_queue = device.new_command_queue();
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), PATH_SAMPLE_COUNT));
        let core_video_texture_cache =
            unsafe { CVMetalTextureCache::new(device.as_ptr()).unwrap() };
            
        // Create the stencil pipeline state
        let stencil_pipeline_state = build_stencil_pipeline_state(
            &device,
            &library,
            "stencil_marker",
            "stencil_marker_vertex",
            "stencil_marker_fragment"
        );
        
        // Create quad vertices for stencil operations (unit square)
        let quad_vertices = [
            point(0.0, 0.0),
            point(1.0, 0.0),
            point(0.0, 1.0),
            point(1.0, 0.0),
            point(1.0, 1.0),
            point(0.0, 1.0),
        ];
        let quad_vertices = device.new_buffer_with_data(
            quad_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&quad_vertices) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        Self {
            device,
            layer,
            presents_with_transaction: false,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            stencil_pipeline_state,
            unit_vertices,
            quad_vertices,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            stencil_texture: None,
        }
    }

    pub fn layer(&self) -> &metal::MetalLayerRef {
        &self.layer
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer.as_ptr()
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        self.layer
            .set_presents_with_transaction(presents_with_transaction);
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let size = NSSize {
            width: size.width.0 as f64,
            height: size.height.0 as f64,
        };
        unsafe {
            let _: () = msg_send![
                self.layer(),
                setDrawableSize: size
            ];
        }
    }

    pub fn update_transparency(&self, _transparent: bool) {
        // todo(mac)?
    }

    pub fn destroy(&self) {
        // nothing to do
    }
    
    /// Draw the scene using stencil to only redraw damaged regions
    pub fn draw_with_damage(&mut self, scene: &Scene, damage_region: &DamageRegion) -> Result<()> {
        if damage_region.is_empty() {
            return Ok(()); // Nothing to draw
        }
        
        let layer = self.layer.clone();
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return Ok(());
        };

        // Create stencil texture if needed
        let stencil_texture = self.get_stencil_texture(viewport_size);
        
        // Step 1: Set up render pass with stencil attachment
        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
            
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(metal::MTLLoadAction::Load); // Keep existing content
        color_attachment.set_store_action(metal::MTLStoreAction::Store);
        
        // Set up stencil attachment
        let stencil_attachment = render_pass_descriptor
            .stencil_attachment()
            .unwrap();
            
        stencil_attachment.set_texture(Some(&stencil_texture));
        stencil_attachment.set_load_action(metal::MTLLoadAction::Clear);
        stencil_attachment.set_store_action(metal::MTLStoreAction::DontCare);
        stencil_attachment.set_clear_stencil(0); // Clear stencil to 0
        
        // Create a local reference to the command queue to prevent borrow issues
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        
        // Step 2: Fill stencil buffer with 1s in damaged regions
        let stencil_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        stencil_encoder.set_render_pipeline_state(&self.stencil_pipeline_state);
        
        // Set stencil state to always write 1 to stencil buffer
        let stencil_descriptor = metal::StencilDescriptor::new();
        stencil_descriptor.set_stencil_compare_function(metal::MTLCompareFunction::Always);
        stencil_descriptor.set_stencil_failure_operation(metal::MTLStencilOperation::Keep);
        stencil_descriptor.set_depth_failure_operation(metal::MTLStencilOperation::Keep);
        stencil_descriptor.set_depth_stencil_pass_operation(metal::MTLStencilOperation::Replace);
        stencil_descriptor.set_read_mask(0xFF);
        stencil_descriptor.set_write_mask(0xFF);
        
        let depth_stencil_descriptor = metal::DepthStencilDescriptor::new();
        depth_stencil_descriptor.set_front_face_stencil(Some(&stencil_descriptor));
        depth_stencil_descriptor.set_back_face_stencil(Some(&stencil_descriptor));
        
        let depth_stencil_state = self.device.new_depth_stencil_state(&depth_stencil_descriptor);
        stencil_encoder.set_depth_stencil_state(&depth_stencil_state);
        stencil_encoder.set_stencil_reference_value(1); // Write 1s to stencil
        
        // Set viewport to match drawable size
        stencil_encoder.set_viewport(metal::MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: i32::from(viewport_size.width) as f64,
            height: i32::from(viewport_size.height) as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        
        // Draw quads for each damage region
        stencil_encoder.set_vertex_buffer(0, Some(&self.quad_vertices), 0);
        stencil_encoder.set_vertex_bytes(
            2, 
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _
        );
        
        // Create buffer for damage region quads
        let quads: Vec<StencilQuad> = damage_region.bounds.iter().map(|bounds| {
            StencilQuad {
                origin: bounds.origin.map(|p| p.0),
                size: bounds.size.map(|s| s.0),
            }
        }).collect();
        
        if !quads.is_empty() {
            let quad_buffer = self.device.new_buffer_with_data(
                quads.as_ptr() as *const c_void,
                (mem::size_of::<StencilQuad>() * quads.len()) as u64,
                MTLResourceOptions::StorageModeManaged
            );
            
            stencil_encoder.set_vertex_buffer(1, Some(&quad_buffer), 0);
            stencil_encoder.draw_primitives_instanced(
                metal::MTLPrimitiveType::Triangle,
                0,
                6, // 6 vertices for quad (2 triangles)
                quads.len() as u64
            );
        }
        
        stencil_encoder.end_encoding();
        
        // Step 3: Render scene but only where stencil == 1
        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
            
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(metal::MTLLoadAction::Load);
        color_attachment.set_store_action(metal::MTLStoreAction::Store);
        
        // Reuse stencil texture
        let stencil_attachment = render_pass_descriptor
            .stencil_attachment()
            .unwrap();
            
        stencil_attachment.set_texture(Some(&stencil_texture));
        stencil_attachment.set_load_action(metal::MTLLoadAction::Load); // Keep stencil content
        stencil_attachment.set_store_action(metal::MTLStoreAction::DontCare);
        
        loop {
            let mut instance_buffer = self.instance_buffer_pool.lock().acquire(&self.device);

            let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
            
            // Set stencil test to only render where stencil == 1
            let stencil_descriptor = metal::StencilDescriptor::new();
            stencil_descriptor.set_stencil_compare_function(metal::MTLCompareFunction::Equal);
            stencil_descriptor.set_stencil_failure_operation(metal::MTLStencilOperation::Keep);
            stencil_descriptor.set_depth_failure_operation(metal::MTLStencilOperation::Keep);
            stencil_descriptor.set_depth_stencil_pass_operation(metal::MTLStencilOperation::Keep);
            stencil_descriptor.set_read_mask(0xFF);
            stencil_descriptor.set_write_mask(0x00); // Don't modify stencil during rendering
            
            let depth_stencil_descriptor = metal::DepthStencilDescriptor::new();
            depth_stencil_descriptor.set_front_face_stencil(Some(&stencil_descriptor));
            depth_stencil_descriptor.set_back_face_stencil(Some(&stencil_descriptor));
            
            let depth_stencil_state = self.device.new_depth_stencil_state(&depth_stencil_descriptor);
            command_encoder.set_depth_stencil_state(&depth_stencil_state);
            command_encoder.set_stencil_reference_value(1); // Test against 1
            
            command_encoder.set_viewport(metal::MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: i32::from(viewport_size.width) as f64,
                height: i32::from(viewport_size.height) as f64,
                znear: 0.0,
                zfar: 1.0,
            });
            
            // Draw scene primitives - to avoid a borrow issue, we need to use low-level primitive drawing instead
            let mut instance_offset = 0;
            
            // Create a local clone of the command queue to avoid borrow issues
            let command_queue = self.command_queue.clone();
            
            // First rasterize paths
            let path_tiles = match self.rasterize_paths(
                scene.paths(),
                &mut instance_buffer,
                &mut instance_offset,
                command_queue.new_command_buffer(),
            ) {
                Some(tiles) => tiles,
                None => {
                    command_encoder.end_encoding();
                    return Err(anyhow!("failed to rasterize {} paths", scene.paths().len()));
                }
            };
            
            // Draw each batch type
            let mut success = true;
            for batch in scene.batches() {
                let ok = match batch {
                    PrimitiveBatch::Shadows(shadows) => self.draw_shadows(
                        shadows,
                        &mut instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_encoder,
                    ),
                    PrimitiveBatch::Quads(quads) => self.draw_quads(
                        quads,
                        &mut instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_encoder,
                    ),
                    PrimitiveBatch::Paths(paths) => self.draw_paths(
                        paths,
                        &path_tiles,
                        &mut instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_encoder,
                    ),
                    PrimitiveBatch::Underlines(underlines) => self.draw_underlines(
                        underlines,
                        &mut instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_encoder,
                    ),
                    PrimitiveBatch::MonochromeSprites {
                        texture_id,
                        sprites,
                    } => self.draw_monochrome_sprites(
                        texture_id,
                        sprites,
                        &mut instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_encoder,
                    ),
                    PrimitiveBatch::PolychromeSprites {
                        texture_id,
                        sprites,
                    } => self.draw_polychrome_sprites(
                        texture_id,
                        sprites,
                        &mut instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_encoder,
                    ),
                    PrimitiveBatch::Surfaces(surfaces) => self.draw_surfaces(
                        surfaces,
                        &mut instance_buffer,
                        &mut instance_offset,
                        viewport_size,
                        command_encoder,
                    ),
                };
                if !ok {
                    success = false;
                    break;
                }
            }
            
            instance_buffer.metal_buffer.did_modify_range(NSRange {
                location: 0,
                length: instance_offset as NSUInteger,
            });
            
            let result = if success {
                Ok(())
            } else {
                Err(anyhow!("scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                ))
            };
            
            command_encoder.end_encoding();
            
            match result {
                Ok(_) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    if self.presents_with_transaction {
                        command_buffer.commit();
                        command_buffer.wait_until_scheduled();
                        drawable.present();
                    } else {
                        command_buffer.present_drawable(drawable);
                        command_buffer.commit();
                    }
                    return Ok(());
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        log::error!("instance buffer size grew too large: {}", buffer_size);
                        break;
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
        Ok(())
    }

    /// Helper method for drawing scene primitives with an existing command encoder
    fn draw_scene_primitives(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        // First rasterize paths
        let path_tiles = match self.rasterize_paths(
            scene.paths(),
            instance_buffer,
            instance_offset,
            self.command_queue.new_command_buffer(),
        ) {
            Some(tiles) => tiles,
            None => return Err(anyhow!("failed to rasterize {} paths", scene.paths().len())),
        };
        
        // Draw scene batches
        for batch in scene.batches() {
            let ok = match batch {
                PrimitiveBatch::Shadows(shadows) => self.draw_shadows(
                    shadows,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Quads(quads) => self.draw_quads(
                    quads,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Paths(paths) => self.draw_paths(
                    paths,
                    &path_tiles,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Underlines(underlines) => self.draw_underlines(
                    underlines,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites,
                } => self.draw_monochrome_sprites(
                    texture_id,
                    sprites,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites,
                } => self.draw_polychrome_sprites(
                    texture_id,
                    sprites,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Surfaces(surfaces) => self.draw_surfaces(
                    surfaces,
                    instance_buffer,
                    instance_offset,
                    viewport_size,
                    command_encoder,
                ),
            };

            if !ok {
                return Err(anyhow!("scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                ));
            }
        }
        
        instance_buffer.metal_buffer.did_modify_range(NSRange {
            location: 0,
            length: *instance_offset as NSUInteger,
        });
        
        Ok(())
    }

    pub fn draw(&mut self, scene: &Scene) -> Result<()> {
        let layer = self.layer.clone();
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return Ok(());
        };

        loop {
            let mut instance_buffer = self.instance_buffer_pool.lock().acquire(&self.device);

            let command_buffer =
                self.draw_primitives(scene, &mut instance_buffer, drawable, viewport_size);

            match command_buffer {
                Ok(command_buffer) => {
                    let instance_buffer_pool = self.instance_buffer_pool.clone();
                    let instance_buffer = Cell::new(Some(instance_buffer));
                    let block = ConcreteBlock::new(move |_| {
                        if let Some(instance_buffer) = instance_buffer.take() {
                            instance_buffer_pool.lock().release(instance_buffer);
                        }
                    });
                    let block = block.copy();
                    command_buffer.add_completed_handler(&block);

                    if self.presents_with_transaction {
                        command_buffer.commit();
                        command_buffer.wait_until_scheduled();
                        drawable.present();
                    } else {
                        command_buffer.present_drawable(drawable);
                        command_buffer.commit();
                    }
                    return Ok(());
                }
                Err(err) => {
                    log::error!(
                        "failed to render: {}. retrying with larger instance buffer size",
                        err
                    );
                    let mut instance_buffer_pool = self.instance_buffer_pool.lock();
                    let buffer_size = instance_buffer_pool.buffer_size;
                    if buffer_size >= 256 * 1024 * 1024 {
                        log::error!("instance buffer size grew too large: {}", buffer_size);
                        break;
                    }
                    instance_buffer_pool.reset(buffer_size * 2);
                    log::info!(
                        "increased instance buffer size to {}",
                        instance_buffer_pool.buffer_size
                    );
                }
            }
        }
        Ok(())
    }

    fn draw_primitives(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        drawable: &metal::MetalDrawableRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        let mut instance_offset = 0;

        let Some(path_tiles) = self.rasterize_paths(
            scene.paths(),
            instance_buffer,
            &mut instance_offset,
            command_buffer,
        ) else {
            return Err(anyhow!("failed to rasterize {} paths", scene.paths().len()));
        };

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();

        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_store_action(metal::MTLStoreAction::Store);
        let alpha = if self.layer.is_opaque() { 1. } else { 0. };
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., alpha));
        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);

        command_encoder.set_viewport(metal::MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: i32::from(viewport_size.width) as f64,
            height: i32::from(viewport_size.height) as f64,
            znear: 0.0,
            zfar: 1.0,
        });

        for batch in scene.batches() {
            let ok = match batch {
                PrimitiveBatch::Shadows(shadows) => self.draw_shadows(
                    shadows,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Quads(quads) => self.draw_quads(
                    quads,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Paths(paths) => self.draw_paths(
                    paths,
                    &path_tiles,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Underlines(underlines) => self.draw_underlines(
                    underlines,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    sprites,
                } => self.draw_monochrome_sprites(
                    texture_id,
                    sprites,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    sprites,
                } => self.draw_polychrome_sprites(
                    texture_id,
                    sprites,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::Surfaces(surfaces) => self.draw_surfaces(
                    surfaces,
                    instance_buffer,
                    &mut instance_offset,
                    viewport_size,
                    command_encoder,
                ),
            };

            if !ok {
                command_encoder.end_encoding();
                return Err(anyhow!("scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                ));
            }
        }

        command_encoder.end_encoding();

        instance_buffer.metal_buffer.did_modify_range(NSRange {
            location: 0,
            length: instance_offset as NSUInteger,
        });
        Ok(command_buffer.to_owned())
    }

    fn rasterize_paths(
        &self,
        paths: &[Path<ScaledPixels>],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        command_buffer: &metal::CommandBufferRef,
    ) -> Option<HashMap<PathId, AtlasTile>> {
        self.sprite_atlas.clear_textures(AtlasTextureKind::Path);

        let mut tiles = HashMap::default();
        let mut vertices_by_texture_id = HashMap::default();
        for path in paths {
            let clipped_bounds = path.bounds.intersect(&path.content_mask.bounds);

            let tile = self
                .sprite_atlas
                .allocate(clipped_bounds.size.map(Into::into), AtlasTextureKind::Path)?;
            vertices_by_texture_id
                .entry(tile.texture_id)
                .or_insert(Vec::new())
                .extend(path.vertices.iter().map(|vertex| PathVertex {
                    xy_position: vertex.xy_position - clipped_bounds.origin
                        + tile.bounds.origin.map(Into::into),
                    st_position: vertex.st_position,
                    content_mask: ContentMask {
                        bounds: tile.bounds.map(Into::into),
                    },
                }));
            tiles.insert(path.id, tile);
        }

        for (texture_id, vertices) in vertices_by_texture_id {
            align_offset(instance_offset);
            let vertices_bytes_len = mem::size_of_val(vertices.as_slice());
            let next_offset = *instance_offset + vertices_bytes_len;
            if next_offset > instance_buffer.size {
                return None;
            }

            let render_pass_descriptor = metal::RenderPassDescriptor::new();
            let color_attachment = render_pass_descriptor
                .color_attachments()
                .object_at(0)
                .unwrap();

            let texture = self.sprite_atlas.metal_texture(texture_id);
            let msaa_texture = self.sprite_atlas.msaa_texture(texture_id);

            if let Some(msaa_texture) = msaa_texture {
                color_attachment.set_texture(Some(&msaa_texture));
                color_attachment.set_resolve_texture(Some(&texture));
                color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
            } else {
                color_attachment.set_texture(Some(&texture));
                color_attachment.set_load_action(metal::MTLLoadAction::Clear);
                color_attachment.set_store_action(metal::MTLStoreAction::Store);
            }
            color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 1.));

            let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
            command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);
            command_encoder.set_vertex_buffer(
                PathRasterizationInputIndex::Vertices as u64,
                Some(&instance_buffer.metal_buffer),
                *instance_offset as u64,
            );
            let texture_size = Size {
                width: DevicePixels::from(texture.width()),
                height: DevicePixels::from(texture.height()),
            };
            command_encoder.set_vertex_bytes(
                PathRasterizationInputIndex::AtlasTextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );

            let buffer_contents = unsafe {
                (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset)
            };
            unsafe {
                ptr::copy_nonoverlapping(
                    vertices.as_ptr() as *const u8,
                    buffer_contents,
                    vertices_bytes_len,
                );
            }

            command_encoder.draw_primitives(
                metal::MTLPrimitiveType::Triangle,
                0,
                vertices.len() as u64,
            );
            command_encoder.end_encoding();
            *instance_offset = next_offset;
        }

        Some(tiles)
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if shadows.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let shadow_bytes_len = mem::size_of_val(shadows);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + shadow_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                shadows.as_ptr() as *const u8,
                buffer_contents,
                shadow_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if quads.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let quad_bytes_len = mem::size_of_val(quads);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + quad_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(quads.as_ptr() as *const u8, buffer_contents, quad_bytes_len);
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_paths(
        &self,
        paths: &[Path<ScaledPixels>],
        tiles_by_path_id: &HashMap<PathId, AtlasTile>,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if paths.is_empty() {
            return true;
        }

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let mut prev_texture_id = None;
        let mut sprites = SmallVec::<[_; 1]>::new();
        let mut paths_and_tiles = paths
            .iter()
            .map(|path| (path, tiles_by_path_id.get(&path.id).unwrap()))
            .peekable();

        loop {
            if let Some((path, tile)) = paths_and_tiles.peek() {
                if prev_texture_id.map_or(true, |texture_id| texture_id == tile.texture_id) {
                    prev_texture_id = Some(tile.texture_id);
                    let origin = path.bounds.intersect(&path.content_mask.bounds).origin;
                    sprites.push(PathSprite {
                        bounds: Bounds {
                            origin: origin.map(|p| p.floor()),
                            size: tile.bounds.size.map(Into::into),
                        },
                        color: path.color,
                        tile: (*tile).clone(),
                    });
                    paths_and_tiles.next();
                    continue;
                }
            }

            if sprites.is_empty() {
                break;
            } else {
                align_offset(instance_offset);
                let texture_id = prev_texture_id.take().unwrap();
                let texture: metal::Texture = self.sprite_atlas.metal_texture(texture_id);
                let texture_size = size(
                    DevicePixels(texture.width() as i32),
                    DevicePixels(texture.height() as i32),
                );

                command_encoder.set_vertex_buffer(
                    SpriteInputIndex::Sprites as u64,
                    Some(&instance_buffer.metal_buffer),
                    *instance_offset as u64,
                );
                command_encoder.set_vertex_bytes(
                    SpriteInputIndex::AtlasTextureSize as u64,
                    mem::size_of_val(&texture_size) as u64,
                    &texture_size as *const Size<DevicePixels> as *const _,
                );
                command_encoder.set_fragment_buffer(
                    SpriteInputIndex::Sprites as u64,
                    Some(&instance_buffer.metal_buffer),
                    *instance_offset as u64,
                );
                command_encoder
                    .set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

                let sprite_bytes_len = mem::size_of_val(sprites.as_slice());
                let next_offset = *instance_offset + sprite_bytes_len;
                if next_offset > instance_buffer.size {
                    return false;
                }

                let buffer_contents = unsafe {
                    (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset)
                };

                unsafe {
                    ptr::copy_nonoverlapping(
                        sprites.as_ptr() as *const u8,
                        buffer_contents,
                        sprite_bytes_len,
                    );
                }

                command_encoder.draw_primitives_instanced(
                    metal::MTLPrimitiveType::Triangle,
                    0,
                    6,
                    sprites.len() as u64,
                );
                *instance_offset = next_offset;
                sprites.clear();
            }
        }
        true
    }

    fn draw_underlines(
        &self,
        underlines: &[Underline],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if underlines.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );

        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let underline_bytes_len = mem::size_of_val(underlines);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + underline_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                underlines.as_ptr() as *const u8,
                buffer_contents,
                underline_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[MonochromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let sprite_bytes_len = mem::size_of_val(sprites);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[PolychromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_buffer.metal_buffer),
            *instance_offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        let sprite_bytes_len = mem::size_of_val(sprites);
        let buffer_contents =
            unsafe { (instance_buffer.metal_buffer.contents() as *mut u8).add(*instance_offset) };

        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        *instance_offset = next_offset;
        true
    }

    /// Helper to create or get stencil texture
    fn get_stencil_texture(&mut self, size: Size<DevicePixels>) -> metal::Texture {
        // Check if we have a cached texture of the right size
        if let Some(texture) = self.stencil_texture.as_ref() {
            if texture.width() == size.width.0 as u64 && 
               texture.height() == size.height.0 as u64 {
                return texture.clone();
            }
        }
        
        // Create new stencil texture
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_texture_type(metal::MTLTextureType::D2);
        descriptor.set_width(size.width.0 as u64);
        descriptor.set_height(size.height.0 as u64);
        descriptor.set_pixel_format(metal::MTLPixelFormat::Stencil8); // 8-bit stencil
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
        
        let texture = self.device.new_texture(&descriptor);
        self.stencil_texture = Some(texture.clone());
        texture
    }
    
    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> bool {
        command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for surface in surfaces {
            let texture_size = size(
                DevicePixels::from(surface.image_buffer.width() as i32),
                DevicePixels::from(surface.image_buffer.height() as i32),
            );

            assert_eq!(
                surface.image_buffer.pixel_format_type(),
                media::core_video::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let y_texture = unsafe {
                self.core_video_texture_cache
                    .create_texture_from_image(
                        surface.image_buffer.as_concrete_TypeRef(),
                        ptr::null(),
                        MTLPixelFormat::R8Unorm,
                        surface.image_buffer.plane_width(0),
                        surface.image_buffer.plane_height(0),
                        0,
                    )
                    .unwrap()
            };
            let cb_cr_texture = unsafe {
                self.core_video_texture_cache
                    .create_texture_from_image(
                        surface.image_buffer.as_concrete_TypeRef(),
                        ptr::null(),
                        MTLPixelFormat::RG8Unorm,
                        surface.image_buffer.plane_width(1),
                        surface.image_buffer.plane_height(1),
                        1,
                    )
                    .unwrap()
            };

            align_offset(instance_offset);
            let next_offset = *instance_offset + mem::size_of::<Surface>();
            if next_offset > instance_buffer.size {
                return false;
            }

            command_encoder.set_vertex_buffer(
                SurfaceInputIndex::Surfaces as u64,
                Some(&instance_buffer.metal_buffer),
                *instance_offset as u64,
            );
            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            command_encoder.set_fragment_texture(
                SurfaceInputIndex::YTexture as u64,
                Some(y_texture.as_texture_ref()),
            );
            command_encoder.set_fragment_texture(
                SurfaceInputIndex::CbCrTexture as u64,
                Some(cb_cr_texture.as_texture_ref()),
            );

            unsafe {
                let buffer_contents = (instance_buffer.metal_buffer.contents() as *mut u8)
                    .add(*instance_offset)
                    as *mut SurfaceBounds;
                ptr::write(
                    buffer_contents,
                    SurfaceBounds {
                        bounds: surface.bounds,
                        content_mask: surface.content_mask.clone(),
                    },
                );
            }

            command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
            *instance_offset = next_offset;
        }
        true
    }
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

/// Build a pipeline state for stencil operations
fn build_stencil_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating stencil vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating stencil fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
    color_attachment.set_write_mask(metal::MTLColorWriteMask::None); // Don't write to color buffer
    
    // Enable stencil attachment
    descriptor.set_stencil_attachment_pixel_format(metal::MTLPixelFormat::Stencil8);
    
    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create stencil pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(true);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

// Align to multiples of 256 make Metal happy.
fn align_offset(offset: &mut usize) {
    *offset = ((*offset + 255) / 256) * 256;
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    AtlasTextureSize = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
    pub color: Background,
    pub tile: AtlasTile,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}
