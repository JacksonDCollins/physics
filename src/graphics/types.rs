
use cgmath::SquareMatrix;
pub use cgmath::{point3, vec2, vec3, Deg};
use serde::{Serialize};
use std::{mem::size_of};
use vulkanalia::prelude::v1_2::*;

pub type Vec2 = cgmath::Vector2<f32>;
pub type Vec3 = cgmath::Vector3<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;

pub static VERTICES: [Vertex; 4] = [
    Vertex::new(vec2(-0.5, -0.5), vec3(1.0, 0.0, 0.0)),
    Vertex::new(vec2(0.5, -0.5), vec3(0.0, 1.0, 0.0)),
    Vertex::new(vec2(0.5, 0.5), vec3(0.0, 0.0, 1.0)),
    Vertex::new(vec2(-0.5, 0.5), vec3(1.0, 1.0, 1.0)),
];

pub const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Serialize)]
pub struct Vertex {
    pos: Vec2,
    color: Vec3,
}

impl Vertex {
    const fn new(pos: Vec2, color: Vec3) -> Self {
        Self { pos, color }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec2>() as u32)
            .build();
        [pos, color]
    }
}

#[derive(Clone, Debug, Default, Copy)]
pub struct CommandPoolSet {
    // data: HashMap<u32, vk::CommandPool>,
    pub present: vk::CommandPool,
    pub graphics: vk::CommandPool,
    pub transfer: vk::CommandPool,
    pub compute: vk::CommandPool,
}

impl CommandPoolSet {
    pub fn create(
        present: vk::CommandPool,
        graphics: vk::CommandPool,
        transfer: vk::CommandPool,
        compute: vk::CommandPool,
    ) -> Self {
        Self {
            present,
            graphics,
            transfer,
            compute,
        }
    }

    pub fn destroy(&self, logical_device: &Device) {
        unsafe {
            logical_device.destroy_command_pool(self.present, None);
            logical_device.destroy_command_pool(self.graphics, None);
            logical_device.destroy_command_pool(self.transfer, None);
            logical_device.destroy_command_pool(self.compute, None);
        }
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

impl Default for UniformBufferObject {
    fn default() -> Self {
        Self {
            model: Mat4::identity(),
            view: Mat4::identity(),
            proj: Mat4::identity(),
        }
    }
}
