#![allow(clippy::missing_safety_doc)]
pub mod app;
pub mod graphics;

use anyhow::Result;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use app::App;

fn main() -> Result<()> {
    pretty_env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkan")
        .build(&event_loop)?;
    let mut app = unsafe { App::create(&window) }?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => unsafe { app.render(&window) }.unwrap(),
            Event::WindowEvent {
                event: window_event,
                ..
            } => match window_event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(size) => {
                    //resize
                }
                _ => {}
            },
            _ => {}
        }
    });
}
