#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub mod app;
pub mod controller;
pub mod graphics;
pub mod input;

use anyhow::Result;

use winit::{
    event::{DeviceEvent, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, WindowBuilder},
};

use app::App;

fn main() -> Result<()> {
    pretty_env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkan")
        .build(&event_loop)?;

    window
        .set_cursor_grab(CursorGrabMode::Confined)
        .or_else(|_e| window.set_cursor_grab(CursorGrabMode::Locked))?;

    window.set_cursor_visible(false);

    let mut app = unsafe { App::create(&window) }?;

    let mut destroying = false;
    let mut minimized = false;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::MainEventsCleared if !destroying && !minimized => {
                app.tick();
                unsafe { app.render(&window) }.unwrap();
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => app.update_mouse_motion(delta),
            Event::WindowEvent {
                event: window_event,
                ..
            } => {
                app.window_input(&window_event);
                match window_event {
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => {
                        destroying = true;
                        *control_flow = ControlFlow::Exit;
                        unsafe {
                            app.device_wait_idle().unwrap();
                            app.destroy();
                        }
                    }
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
                }
            }
            _ => {}
        }
    });
}

// https://kylemayes.github.io/vulkanalia/dynamic/push_constants.html
