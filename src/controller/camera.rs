use std::collections::HashMap;

use crate::graphics::types::{self as g_types, Vec3};

use cgmath::{Angle, InnerSpace};
use winit::event::{ElementState, VirtualKeyCode};

pub struct Camera {
    pub eye: g_types::Point3,
    pub facing: g_types::Vec3,
    pub up: g_types::Vec3,
    pub right: g_types::Vec3,
    pub pitch: cgmath::Rad<f64>,
    pub yaw: cgmath::Rad<f64>,
    pub roll: cgmath::Rad<f64>,
    pub movement_speed: f32,
    pub mouse_sensitivity: f64,
}

impl Camera {
    pub fn create() -> Self {
        Self {
            eye: g_types::point3(0.0, 0.0, 10.0),
            facing: Vec3::unit_x(),
            up: Vec3::unit_y(),
            right: Vec3::unit_z(),
            pitch: cgmath::Rad(0.0),
            yaw: cgmath::Rad(0.0),
            roll: cgmath::Rad(0.0),
            movement_speed: 0.01,
            mouse_sensitivity: 0.005,
        }
    }

    pub fn update(&mut self, keydata: &HashMap<VirtualKeyCode, ElementState>) {
        keydata.iter().for_each(|(keycode, state)| {
            if *state == ElementState::Pressed {
                match *keycode {
                    VirtualKeyCode::W => {
                        self.eye +=
                            self.right.cross(Vec3::unit_z()).normalize() * self.movement_speed;
                    }
                    VirtualKeyCode::S => {
                        self.eye -=
                            self.right.cross(Vec3::unit_z()).normalize() * self.movement_speed;
                    }
                    VirtualKeyCode::A => {
                        self.eye += self.right * self.movement_speed;
                    }
                    VirtualKeyCode::D => {
                        self.eye -= self.right * self.movement_speed;
                    }
                    VirtualKeyCode::Space => {
                        self.eye += g_types::vec3(0.0, 0.0, 1.0) * self.movement_speed;
                    }
                    VirtualKeyCode::LShift => {
                        self.eye -= g_types::vec3(0.0, 0.0, 1.0) * self.movement_speed;
                    }
                    _ => {}
                }
            }
        })
    }

    pub fn update_mouse_motion(&mut self, delta: (f64, f64)) {
        self.yaw -= cgmath::Rad(delta.0 * self.mouse_sensitivity);
        self.pitch -= cgmath::Rad(delta.1 * self.mouse_sensitivity);

        self.pitch.0 = self
            .pitch
            .0
            .clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);

        self.facing = g_types::vec3(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        )
        .normalize()
        .map(|e| e as f32);

        self.right = g_types::vec3(0.0, 0.0, 1.0).cross(self.facing).normalize();
        self.up = self.facing.cross(self.right).normalize();
    }

    pub fn get_view_matrix(&self) -> g_types::Mat4 {
        g_types::Mat4::look_at_rh(self.eye, self.eye + self.facing, self.up)
    }
}
