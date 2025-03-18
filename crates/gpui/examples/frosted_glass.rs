use gpui::{
    div, hsla, prelude::*, px, rgb, rgba, size, App, Application, Bounds, Context,
    FrostedGlassBlending, FrostedGlassMaterial, SharedString, TitlebarOptions,
    WindowBackgroundAppearance, WindowBounds, WindowOptions,
};

struct FrostedGlassExample {
    content: SharedString,
}

impl Render for FrostedGlassExample {
    fn render(&mut self, _window: &mut gpui::Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()

            .rounded_md()
            .flex()
            .flex_col()
            .gap_4()
            .p_4()
            .mt_128()
            .justify_center()
            .items_center()
            .text_xl()
            .text_color(rgb(0xffffff))
            .child(
                div()
                    .p_4()
                    .rounded_md()
                    .bg(rgba(0x00000080))
                    .text_2xl()
                    .font_weight(gpui::FontWeight::BOLD)
                    .child(format!("{}", &self.content))
            )
            .child(
                div()
                    .flex_row()
                    .gap_2()
                    .child(
                        div()
                            .size_16()
                            .rounded_full()
                            .bg(gpui::red())
                    )
                    .child(
                        div()
                            .size_16()
                            .rounded_full()
                            .bg(gpui::green())
                    )
                    .child(
                        div()
                            .size_16()
                            .rounded_full()
                            .bg(gpui::blue())
                    )
            )
            .child(
                div()
                    .p_4()
                    .rounded_md()
                    .bg(rgba(0xffffff33))
                    .text_color(rgb(0x000000))
                    .text_sm()
                    .child("The frosted glass effect is implemented using NSVisualEffectView on macOS.")
                    .child(
                        div()
                            .pt_2()
                            .child("Try moving this window over different background content!")
                    )
            )
            .bg(rgba(0xffffff33))
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(500.), px(500.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_background: WindowBackgroundAppearance::Blurred,
                titlebar: Some(TitlebarOptions {
                    appears_transparent: true,
                    ..Default::default()
                }),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| FrostedGlassExample {
                    content: "✨ Frosted Glass Effect ✨".into(),
                })
            },
        )
        .unwrap();
    });
}
