use std::collections::HashMap;
use std::collections::HashSet;
use std::ffi::CStr;
use std::iter::once;
use std::mem::size_of;
use std::os::raw::c_void;

use crate::graphics::objects as g_objects;
use crate::graphics::types as g_types;
use crate::graphics::utils as g_utils;
use anyhow::{anyhow, Error, Result};
use thiserror::Error;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension};
use vulkanalia::{Entry, Version};
use winit::window::Window;

use super::types::Vertex;

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    pub unsafe fn create(
        window: &Window,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        queue_family_indices: &g_utils::QueueFamilyIndices,
        swapchain_support: &g_utils::SwapchainSupport,
    ) -> Result<Self> {
        let (swapchain, images, extent, format) = g_utils::create_swapchain(
            window,
            instance,
            logical_device,
            physical_device,
            surface,
            queue_family_indices,
            swapchain_support,
        )?;

        let image_views = g_utils::create_swapchain_image_views(logical_device, &images, format)?;

        Ok(Self {
            swapchain,
            images,
            extent,
            format,
            image_views,
        })
    }

    pub unsafe fn destroy(&self, logical_device: &Device) {
        self.image_views
            .iter()
            .for_each(|image_view| logical_device.destroy_image_view(*image_view, None));
        logical_device.destroy_swapchain_khr(self.swapchain, None);
    }
}

pub struct Pipeline {
    pipeline_layout: vk::PipelineLayout,
    pub render_pass: vk::RenderPass,
    pub pipeline: vk::Pipeline,
}

impl Pipeline {
    pub unsafe fn create(logical_device: &Device, swapchain: &Swapchain) -> Result<Self> {
        let (pipeline_layout, render_pass, pipeline) =
            g_utils::create_pipeline_and_renderpass(logical_device, swapchain)?;

        Ok(Self {
            pipeline_layout,
            render_pass,
            pipeline,
        })
    }

    pub unsafe fn destroy(&self, logical_device: &Device) {
        logical_device.destroy_pipeline(self.pipeline, None);
        logical_device.destroy_render_pass(self.render_pass, None);
        logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
    }
}

pub struct Presenter {
    framebuffers: Vec<vk::Framebuffer>,
    command_pool_set: g_types::CommandPoolSet,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
}

impl Presenter {
    pub unsafe fn create(
        logical_device: &Device,
        swapchain: &Swapchain,
        pipeline: &Pipeline,
        queue_family_indices: &g_utils::QueueFamilyIndices,
        buffer_memory_allocator: &mut BufferMemoryAllocator,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
    ) -> Result<Self> {
        let framebuffers =
            g_utils::create_framebuffers(logical_device, pipeline.render_pass, swapchain)?;

        let command_pool_set = g_utils::create_command_pools(logical_device, queue_family_indices)?;

        buffer_memory_allocator.allocate_memory(
            instance,
            logical_device,
            physical_device,
            queue_set,
            command_pool_set,
        )?;

        let command_buffers = g_utils::create_command_buffers(
            logical_device,
            command_pool_set,
            &framebuffers,
            swapchain,
            pipeline,
            buffer_memory_allocator,
        )?;

        let (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        ) = g_utils::create_sync_objects(logical_device, swapchain.images.len())?;

        Ok(Self {
            framebuffers,
            command_pool_set,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        })
    }

    pub unsafe fn destroy(&self, logical_device: &Device) {
        self.in_flight_fences
            .iter()
            .for_each(|fence| logical_device.destroy_fence(*fence, None));
        self.image_available_semaphores
            .iter()
            .for_each(|semaphore| logical_device.destroy_semaphore(*semaphore, None));
        self.render_finished_semaphores
            .iter()
            .for_each(|semaphore| logical_device.destroy_semaphore(*semaphore, None));
        // logical_device.destroy_command_pool(self.command_pool, None);
        self.command_pool_set.destroy(logical_device);
        self.framebuffers
            .iter()
            .for_each(|framebuffer| logical_device.destroy_framebuffer(*framebuffer, None));
    }
}

pub struct VertexBuffer {
    pub buffer: vk::Buffer,
    pub vertices: Vec<g_types::Vertex>,
    pub size: u64,
    pub offset: u64,
}

impl VertexBuffer {
    pub unsafe fn create(logical_device: &Device, vertices: &[g_types::Vertex]) -> Result<Self> {
        let size = std::mem::size_of_val(vertices) as u64;
        let vertex_buffer = g_utils::create_buffer(
            logical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        Ok(Self {
            buffer: vertex_buffer,

            vertices: vertices.to_vec(),
            size,
            offset: 0,
        })
    }

    pub unsafe fn destroy(&self, logical_device: &Device) {
        logical_device.destroy_buffer(self.buffer, None);
    }
}

pub struct IndexBuffer {
    pub buffer: vk::Buffer,
    pub indices: Vec<u16>,
    pub size: u64,
    pub offset: u64,
}

impl IndexBuffer {
    pub unsafe fn create(logical_device: &Device, indices: &[u16]) -> Result<Self> {
        let size = std::mem::size_of_val(indices) as u64;
        let index_buffer = g_utils::create_buffer(
            logical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        Ok(Self {
            buffer: index_buffer,
            indices: indices.to_vec(),
            size,
            offset: 0,
        })
    }

    pub unsafe fn destroy(&self, logical_device: &Device) {
        logical_device.destroy_buffer(self.buffer, None);
    }

    pub unsafe fn null() -> Self {
        Self {
            buffer: vk::Buffer::null(),
            indices: Vec::new(),
            size: 0,
            offset: 0,
        }
    }
}

pub struct BufferMemoryAllocator {
    pub vertex_buffers_to_allocate: Vec<VertexBuffer>,
    pub memory: vk::DeviceMemory,
    pub buffer: vk::Buffer,
    pub index_buffer_to_allocate: IndexBuffer,
    // pub index_memory: vk::DeviceMemory,
    // pub index_buffer: vk::Buffer,
}

impl BufferMemoryAllocator {
    pub unsafe fn create() -> Result<Self> {
        Ok(Self {
            vertex_buffers_to_allocate: Vec::new(),
            memory: vk::DeviceMemory::null(),
            buffer: vk::Buffer::null(),
            index_buffer_to_allocate: IndexBuffer::null(),
            // index_memory: vk::DeviceMemory::null(),
            // index_buffer: vk::Buffer::null(),
        })
    }

    pub unsafe fn get_vertex_buffers(&self) -> Vec<vk::Buffer> {
        self.vertex_buffers_to_allocate
            .iter()
            .map(|buffer| buffer.buffer)
            .collect::<Vec<_>>()
    }

    pub unsafe fn get_vertex_buffers_offsets(&self) -> Vec<u64> {
        self.vertex_buffers_to_allocate
            .iter()
            .map(|buffer| 0) //buffer.offset)
            .collect::<Vec<_>>()
    }

    pub unsafe fn add_vertex_buffer(&mut self, buffer: VertexBuffer) {
        self.vertex_buffers_to_allocate.push(buffer);
    }

    pub unsafe fn set_index_buffer(&mut self, buffer: IndexBuffer) {
        self.index_buffer_to_allocate = buffer;
    }

    pub unsafe fn allocate_memory(
        &mut self,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
        command_pool_set: g_types::CommandPoolSet,
    ) -> Result<()> {
        let vertex_size = self
            .vertex_buffers_to_allocate
            .iter()
            .fold(0, |acc, buffer| acc + buffer.size);

        let index_size = self.index_buffer_to_allocate.size;

        let total_size = vertex_size + index_size;

        let (staging_buffer, staging_buffer_memory, _) = g_utils::create_buffer_and_memory(
            instance,
            logical_device,
            physical_device,
            total_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let memory = logical_device.map_memory(
            staging_buffer_memory,
            0,
            total_size,
            vk::MemoryMapFlags::empty(),
        )?;

        let mut offset = 0;
        self.vertex_buffers_to_allocate
            .iter_mut()
            .for_each(|buffer| {
                let dst = memory.add(offset as usize).cast();
                std::ptr::copy_nonoverlapping(buffer.vertices.as_ptr(), dst, buffer.vertices.len());
                buffer.offset = offset;
                offset += buffer.size;
            });

        let dst = memory.add(offset as usize).cast();
        std::ptr::copy_nonoverlapping(
            self.index_buffer_to_allocate.indices.as_ptr(),
            dst,
            self.index_buffer_to_allocate.indices.len(),
        );
        self.index_buffer_to_allocate.offset = offset;
        // offset += self.index_buffer_to_allocate.size;

        logical_device.unmap_memory(staging_buffer_memory);

        (self.buffer, self.memory, _) = g_utils::create_buffer_and_memory(
            instance,
            logical_device,
            physical_device,
            total_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        g_utils::copy_buffer(
            logical_device,
            queue_set.transfer,
            command_pool_set.transfer,
            staging_buffer,
            self.buffer,
            total_size,
            0,
        )?;

        for buffer in self.vertex_buffers_to_allocate.iter() {
            logical_device.bind_buffer_memory(buffer.buffer, self.memory, buffer.offset)?;
        }

        logical_device.bind_buffer_memory(
            self.index_buffer_to_allocate.buffer,
            self.memory,
            self.index_buffer_to_allocate.offset,
        )?;

        logical_device.destroy_buffer(staging_buffer, None);
        logical_device.free_memory(staging_buffer_memory, None);

        Ok(())
    }

    pub unsafe fn destroy(&self, logical_device: &Device) {
        self.vertex_buffers_to_allocate
            .iter()
            .for_each(|buffer| buffer.destroy(logical_device));

        logical_device.destroy_buffer(self.buffer, None);
        logical_device.free_memory(self.memory, None);

        self.index_buffer_to_allocate.destroy(logical_device);

        // logical_device.destroy_buffer(self.index_buffer, None);
        // logical_device.free_memory(self.index_memory, None);
    }
}
