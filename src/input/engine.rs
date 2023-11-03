use std::collections::HashMap;

use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

pub struct InputEngine {
    pub keydata: HashMap<VirtualKeyCode, ElementState>,
    pub recent_delta: (f64, f64),
}

impl InputEngine {
    pub fn create() -> Self {
        Self {
            keydata: HashMap::new(),
            recent_delta: (0.0, 0.0),
        }
    }

    pub fn update(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                self.keydata
                    .entry(*keycode)
                    .and_modify(|e| *e = *state)
                    .or_insert(*state);
            }
            _ => {}
        }
    }

    pub fn update_mouse_motion(&mut self, delta: (f64, f64)) {
        self.recent_delta = delta;
    }

    pub fn clear_old_input(&mut self) {
        self.keydata.retain(|_, v| *v == ElementState::Pressed);
        self.recent_delta = (0.0, 0.0);
    }
}
