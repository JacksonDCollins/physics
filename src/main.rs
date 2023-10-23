#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
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

    let mut destroying = false;
    let mut minimized = false;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared if !destroying && !minimized => {
                unsafe { app.render(&window) }.unwrap()
            }
            Event::WindowEvent {
                event: window_event,
                ..
            } => match window_event {
                WindowEvent::CloseRequested => {
                    destroying = true;
                    *control_flow = ControlFlow::Exit;
                    unsafe {
                        app.device_wait_idle().unwrap();
                        app.destroy();
                    }
                }
                WindowEvent::Resized(size) => {
                    if size.width == 0 || size.height == 0 {
                        minimized = true;
                    } else {
                        minimized = false;
                        app.resized = true;
                    }
                }
                _ => {}
            },
            _ => {}
        }
    });
}

//https://www.google.com/search?q=vulkan+uniformbuffer+best+practice&client=firefox-b-d&sca_esv=575810318&sxsrf=AM9HkKlfXWui_2hH5EYS0Ru6nfYTiekTzw%3A1698073422428&ei=Tos2ZYroGZ_f2roPlaSnoA0&ved=0ahUKEwiKx5W5uIyCAxWfr1YBHRXSCdQQ4dUDCA8&uact=5&oq=vulkan+uniformbuffer+best+practice&gs_lp=Egxnd3Mtd2l6LXNlcnAiInZ1bGthbiB1bmlmb3JtYnVmZmVyIGJlc3QgcHJhY3RpY2UyBxAhGKABGAoyBxAhGKABGApI3ixQAFjEK3AAeACQAQCYAe8BoAGVMaoBBzAuMjEuMTG4AQPIAQD4AQHCAgQQIxgnwgIIEAAYigUYkQLCAgcQABiKBRhDwgIREC4YgAQYsQMYgwEYxwEY0QPCAgsQABiABBixAxiDAcICDhAuGIoFGLEDGMcBGNEDwgIKEAAYigUYsQMYQ8ICCxAuGIAEGMcBGK8BwgIKEAAYgAQYFBiHAsICBRAAGIAEwgIHECMYsAIYJ8ICBxAAGA0YgATCAgYQABgeGA3CAggQABgIGB4YDcICBhAAGBYYHsICCBAAGIoFGIYDwgIGECEYFRgKwgIEECEYCsICChAhGBYYHhgdGAriAwQYACBBiAYB&sclient=gws-wiz-serp

//https://kylemayes.github.io/vulkanalia/texture/images.html
