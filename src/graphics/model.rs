use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::mem::size_of;

use anyhow::anyhow;
use anyhow::Result;
use std::ptr::copy_nonoverlapping as memcpy;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_2::*;
use winit::window::{Window, WindowBuilder};

use super::texture;
use super::Vertex;

#[derive(Clone, Debug)]
pub struct Wall {
    pub model: Model,
    pub texture: texture::Texture,
    pub width: f32,
}

impl Wall {
    pub unsafe fn new(
        width: f32,
        texture: texture::Texture,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        Ok(Self {
            model: Model::from_width(
                width,
                instance,
                device,
                physical_device,
                command_pool,
                queue,
            )?,
            texture,

            width,
        })
    }

    pub fn destroy(&self, logical_device: &Device) {
        unsafe {
            self.model.destroy(logical_device);
            self.texture.destroy(logical_device);
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub indices: Vec<u32>,
    vertices: Vec<super::Vertex>,
    pub vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
}

impl Model {
    pub unsafe fn from_width(
        width: f32,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        let indices = vec![0, 1, 2, 2, 3, 0];
        let vertices = vec![
            Vertex {
                pos: crate::graphics::vec3(-width, -width, 0.0),
                color: crate::graphics::vec3(1.0, 1.0, 1.0),
                tex_coord: crate::graphics::vec2(0.0, 0.0),
            },
            Vertex {
                pos: crate::graphics::vec3(width, -width, 0.0),
                color: crate::graphics::vec3(1.0, 1.0, 1.0),
                tex_coord: crate::graphics::vec2(1.0, 0.0),
            },
            Vertex {
                pos: crate::graphics::vec3(width, width, 0.0),
                color: crate::graphics::vec3(1.0, 1.0, 1.0),
                tex_coord: crate::graphics::vec2(1.0, 1.0),
            },
            Vertex {
                pos: crate::graphics::vec3(-width, width, 0.0),
                color: crate::graphics::vec3(1.0, 1.0, 1.0),
                tex_coord: crate::graphics::vec2(0.0, 1.0),
            },
        ];

        let (vertex_buffer, vertex_buffer_memory) = create_vertex_buffer(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            &vertices,
        )?;

        let (index_buffer, index_buffer_memory) = create_index_buffer(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            &indices,
        )?;

        Ok(Self {
            vertices,
            indices,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
        })
    }
    pub unsafe fn new(
        path: &str,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);

        let (models, _) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions {
                triangulate: true,
                ..Default::default()
            },
            |_| Ok(Default::default()),
        )?;

        let mut unique_vertices = HashMap::new();
        let mut indices = vec![];
        let mut vertices = vec![];

        for model in &models {
            for index in &model.mesh.indices {
                let pos_offset = (3 * index) as usize;
                let tex_coord_offset = (2 * index) as usize;

                let vertex = crate::graphics::Vertex {
                    pos: crate::graphics::vec3(
                        model.mesh.positions[pos_offset],
                        model.mesh.positions[pos_offset + 1],
                        model.mesh.positions[pos_offset + 2],
                    ),
                    color: crate::graphics::vec3(1.0, 1.0, 1.0),
                    tex_coord: crate::graphics::vec2(
                        model.mesh.texcoords[tex_coord_offset],
                        1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                    ),
                };

                if let Some(index) = unique_vertices.get(&vertex) {
                    indices.push(*index as u32);
                } else {
                    let index = vertices.len();
                    unique_vertices.insert(vertex, index);
                    vertices.push(vertex);
                    indices.push(index as u32);
                }
            }
        }

        let (vertex_buffer, vertex_buffer_memory) = create_vertex_buffer(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            &vertices,
        )?;

        let (index_buffer, index_buffer_memory) = create_index_buffer(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            &indices,
        )?;

        Ok(Self {
            vertices,
            indices,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
        })
    }

    pub unsafe fn destroy(&self, logical_device: &Device) {
        logical_device.destroy_buffer(self.vertex_buffer, None);
        logical_device.free_memory(self.vertex_buffer_memory, None);
        logical_device.destroy_buffer(self.index_buffer, None);
        logical_device.free_memory(self.index_buffer_memory, None);
    }
}

unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    vertices: &Vec<Vertex>,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let size = (size_of::<super::Vertex>() * vertices.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = crate::graphics::utils::create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(vertices.as_ptr(), memory.cast(), vertices.len());

    device.unmap_memory(staging_buffer_memory);

    let (vertex_buffer, vertex_buffer_memory) = crate::graphics::utils::create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    crate::graphics::utils::copy_buffer(
        device,
        command_pool,
        queue,
        staging_buffer,
        vertex_buffer,
        size,
    )?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok((vertex_buffer, vertex_buffer_memory))
}

unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    indices: &Vec<u32>,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let size = (size_of::<u32>() * indices.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = crate::graphics::utils::create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(indices.as_ptr(), memory.cast(), indices.len());

    device.unmap_memory(staging_buffer_memory);

    let (index_buffer, index_buffer_memory) = crate::graphics::utils::create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    crate::graphics::utils::copy_buffer(
        device,
        command_pool,
        queue,
        staging_buffer,
        index_buffer,
        size,
    )?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok((index_buffer, index_buffer_memory))
}
