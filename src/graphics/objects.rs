use std::os::raw::c_void;

use crate::graphics::types as g_types;
use crate::graphics::utils as g_utils;
use anyhow::Result;

use rand::distributions::uniform;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::KhrSwapchainExtension;

use winit::window::Window;

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    pub unsafe fn create(
        window: &Window,
        logical_device: &Device,
        surface: vk::SurfaceKHR,
        queue_family_indices: &g_utils::QueueFamilyIndices,
        swapchain_support: &g_utils::SwapchainSupport,
    ) -> Result<Self> {
        let (swapchain, images, extent, format) = g_utils::create_swapchain(
            window,
            logical_device,
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
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pub pipeline_layout: vk::PipelineLayout,
    pub render_pass: vk::RenderPass,
    pub pipeline: vk::Pipeline,
}

impl Pipeline {
    pub unsafe fn create(logical_device: &Device, swapchain: &Swapchain) -> Result<Self> {
        let descriptor_set_layout = g_utils::create_descriptor_set_layout(logical_device)?;

        let descriptor_pool =
            g_utils::create_descriptor_pool(logical_device, swapchain.images.len() as u32)?;

        let (pipeline_layout, render_pass, pipeline) = g_utils::create_pipeline_and_renderpass(
            logical_device,
            swapchain,
            descriptor_set_layout,
        )?;

        Ok(Self {
            descriptor_set_layout,
            descriptor_pool,
            pipeline_layout,
            render_pass,
            pipeline,
        })
    }

    pub unsafe fn destroy(&self, logical_device: &Device) {
        logical_device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        logical_device.destroy_descriptor_pool(self.descriptor_pool, None);
        logical_device.destroy_pipeline(self.pipeline, None);
        logical_device.destroy_render_pass(self.render_pass, None);
        logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
    }
}

pub struct Presenter {
    descriptor_sets: Vec<vk::DescriptorSet>,
    framebuffers: Vec<vk::Framebuffer>,
    pub command_pool_set: g_types::CommandPoolSet,
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
        let descriptor_sets = g_utils::create_descriptor_sets(
            logical_device,
            pipeline.descriptor_set_layout,
            pipeline.descriptor_pool,
            swapchain.images.len(),
        )?;

        let framebuffers =
            g_utils::create_framebuffers(logical_device, pipeline.render_pass, swapchain)?;

        let command_pool_set = g_utils::create_command_pools(logical_device, queue_family_indices)?;

        buffer_memory_allocator.create_buffers(logical_device)?;

        buffer_memory_allocator.allocate_memory(
            instance,
            logical_device,
            physical_device,
            queue_set,
            command_pool_set,
        )?;

        g_utils::update_descriptor_sets(
            logical_device,
            swapchain.images.len(),
            &buffer_memory_allocator.uniform_buffers_to_allocate,
            &descriptor_sets,
        );

        let command_buffers = g_utils::create_command_buffers(
            logical_device,
            command_pool_set,
            &framebuffers,
            swapchain,
            pipeline,
            buffer_memory_allocator,
            &descriptor_sets,
        )?;

        let (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        ) = g_utils::create_sync_objects(logical_device, swapchain.images.len())?;

        Ok(Self {
            descriptor_sets,
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

#[derive(Debug)]
pub struct VertexBuffer {
    buffer: vk::Buffer,
    vertices: Vec<g_types::Vertex>,
    size: Option<u64>,
    offset: Option<u64>,
    changed: bool,
}

impl VertexBuffer {
    pub unsafe fn create(_logical_device: &Device, vertices: &[g_types::Vertex]) -> Result<Self> {
        let size = std::mem::size_of_val(vertices) as u64;

        Ok(Self {
            buffer: vk::Buffer::null(), //vertex_buffer,
            vertices: vertices.to_vec(),
            size: Some(size),
            offset: None,
            changed: false,
        })
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
        logical_device.destroy_buffer(self.buffer, None);
        self.changed = true;
    }

    pub unsafe fn create_buffer(&mut self, logical_device: &Device) -> Result<()> {
        if self.buffer.is_null() && self.size.is_some() {
            self.buffer = g_utils::create_buffer(
                logical_device,
                self.size.unwrap(),
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct UniformBuffer {
    buffer: vk::Buffer,
    ubo: g_types::UniformBufferObject,
    size: Option<u64>,
    offset: Option<u64>,
    changed: bool,
}

impl UniformBuffer {
    pub unsafe fn create(
        _logical_device: &Device,
        ubo: g_types::UniformBufferObject,
    ) -> Result<Self> {
        let size = std::mem::size_of_val(&ubo) as u64;

        Ok(Self {
            buffer: vk::Buffer::null(), //uniform_buffer,
            ubo,
            size: Some(size),
            offset: None,
            changed: false,
        })
    }

    pub unsafe fn get_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub unsafe fn get_size(&self) -> u64 {
        self.size.unwrap()
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
        logical_device.destroy_buffer(self.buffer, None);
        self.changed = true;
    }

    pub unsafe fn update(&mut self, ubo: g_types::UniformBufferObject) {
        self.size = Some(std::mem::size_of_val(&ubo) as u64);
        self.ubo = ubo;
    }

    pub unsafe fn create_buffer(&mut self, logical_device: &Device) -> Result<()> {
        if self.buffer.is_null() && self.size.is_some() {
            self.buffer = g_utils::create_buffer(
                logical_device,
                self.size.unwrap(),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct IndexBuffer {
    buffer: vk::Buffer,
    indices: Vec<u16>,
    size: Option<u64>,
    offset: Option<u64>,
    changed: bool,
}

impl IndexBuffer {
    pub unsafe fn create(_logical_device: &Device, indices: &[u16]) -> Result<Self> {
        let size = std::mem::size_of_val(indices) as u64;

        Ok(Self {
            buffer: vk::Buffer::null(), // index_buffer,
            indices: indices.to_vec(),
            size: Some(size),
            offset: None,
            changed: false,
        })
    }

    pub fn get_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn get_indice_count(&self) -> u32 {
        self.indices.len() as u32
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
        logical_device.destroy_buffer(self.buffer, None);
        self.changed = true;
    }

    pub unsafe fn null() -> Self {
        Self {
            buffer: vk::Buffer::null(),
            indices: Vec::new(),
            size: None,
            offset: None,
            changed: false,
        }
    }

    pub unsafe fn create_buffer(&mut self, logical_device: &Device) -> Result<()> {
        if self.buffer.is_null() && self.size.is_some() {
            self.buffer = g_utils::create_buffer(
                logical_device,
                self.size.unwrap(),
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            )?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct BufferMemoryAllocator {
    pub vertex_index_memory: vk::DeviceMemory,
    pub vertex_index_buffer: vk::Buffer,
    pub uniform_memory: vk::DeviceMemory,
    pub uniform_buffer: vk::Buffer,
    pub uniform_memory_ptr: *mut c_void,
    pub staging_memory: vk::DeviceMemory,
    pub staging_buffer: vk::Buffer,
    pub stage_memory_ptr: *mut c_void,
    pub vertex_buffers_to_allocate: Vec<VertexBuffer>,
    pub index_buffer_to_allocate: IndexBuffer,
    pub uniform_buffers_to_allocate: Vec<UniformBuffer>,
    pub changed: bool,
}

impl BufferMemoryAllocator {
    pub unsafe fn create() -> Result<Self> {
        log::info!("Creating buffer memory allocator");

        Ok(Self {
            vertex_index_memory: vk::DeviceMemory::null(),
            vertex_index_buffer: vk::Buffer::null(),
            uniform_memory: vk::DeviceMemory::null(),
            uniform_buffer: vk::Buffer::null(),
            uniform_memory_ptr: std::ptr::null_mut(),
            staging_buffer: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            stage_memory_ptr: std::ptr::null_mut(),
            vertex_buffers_to_allocate: Vec::new(),
            index_buffer_to_allocate: IndexBuffer::null(),
            uniform_buffers_to_allocate: Vec::new(),
            changed: false,
        })
    }

    pub unsafe fn get_vertex_buffers(&self) -> Vec<vk::Buffer> {
        log::info!("Getting vertex buffers");
        self.vertex_buffers_to_allocate
            .iter()
            .map(|buffer| buffer.buffer)
            .collect::<Vec<_>>()
    }

    pub unsafe fn get_vertex_buffers_offsets(&self) -> Vec<u64> {
        log::info!("Getting vertex buffer offsets");
        self.vertex_buffers_to_allocate
            .iter()
            .map(|_buffer| 0) //buffer.offset)
            .collect::<Vec<_>>()
    }

    pub unsafe fn add_vertex_buffer(&mut self, buffer: VertexBuffer) {
        log::info!("Adding vertex buffer");
        self.vertex_buffers_to_allocate.push(buffer);
        self.changed = true;
    }

    pub unsafe fn set_index_buffer(&mut self, buffer: IndexBuffer) {
        log::info!("Setting index buffer");
        self.index_buffer_to_allocate = buffer;
        self.changed = true;
    }

    pub unsafe fn add_uniform_buffer(&mut self, buffer: UniformBuffer) {
        log::info!("Adding uniform buffer");
        self.uniform_buffers_to_allocate.push(buffer);
        self.changed = true;
    }

    unsafe fn check_for_changes(&mut self) {
        for buffer in self.vertex_buffers_to_allocate.iter() {
            if buffer.changed {
                self.changed = true;
                return;
            }
        }
        if self.index_buffer_to_allocate.changed {
            self.changed = true;
            return;
        }
        for buffer in self.uniform_buffers_to_allocate.iter() {
            if buffer.changed {
                self.changed = true;
                return;
            }
        }
    }

    unsafe fn reset_changes(&mut self) {
        for buffer in self.vertex_buffers_to_allocate.iter_mut() {
            buffer.changed = false;
        }
        self.index_buffer_to_allocate.changed = false;
        for buffer in self.uniform_buffers_to_allocate.iter_mut() {
            buffer.changed = false;
        }
    }

    unsafe fn create_and_map_staging_buffer_and_memory(
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        size: u64,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, *mut c_void)> {
        let (staging_buffer, staging_buffer_memory, _) = g_utils::create_buffer_and_memory(
            instance,
            logical_device,
            physical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let memory = logical_device.map_memory(
            staging_buffer_memory,
            0,
            size,
            vk::MemoryMapFlags::empty(),
        )?;

        Ok((staging_buffer, staging_buffer_memory, memory))
    }

    pub unsafe fn update_uniform_buffer(
        &mut self,
        ubo: g_types::UniformBufferObject,
        buffer_index: usize,
    ) -> Result<()> {
        let buffer = &mut self.uniform_buffers_to_allocate[buffer_index];
        buffer.update(ubo);

        g_utils::memcpy(
            &[buffer.ubo],
            self.uniform_memory_ptr
                .add(buffer.offset.unwrap() as usize)
                .cast(),
            1,
        );

        Ok(())
    }

    pub unsafe fn allocate_memory(
        &mut self,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
        command_pool_set: g_types::CommandPoolSet,
    ) -> Result<()> {
        self.check_for_changes();
        if !self.changed {
            return Ok(());
        }
        // self.destroy_buffers(logical_device);
        // self.create_buffers(logical_device)?;
        self.reset_changes();
        self.changed = false;

        log::info!("Allocating memory for buffers");

        let vertex_size = self
            .vertex_buffers_to_allocate
            .iter()
            .filter(|buffer| buffer.size.is_some())
            .fold(0, |acc, buffer| acc + buffer.size.unwrap());

        let index_size = self.index_buffer_to_allocate.size.unwrap_or(0);

        let vertex_index_size = vertex_size + index_size;

        if self.stage_memory_ptr.is_null() {
            let (staging_buffer, staging_buffer_memory, memory_ptr) =
                Self::create_and_map_staging_buffer_and_memory(
                    instance,
                    logical_device,
                    physical_device,
                    vertex_index_size * 2,
                )?;
            self.staging_buffer = staging_buffer;
            self.staging_memory = staging_buffer_memory;
            self.stage_memory_ptr = memory_ptr;
        }

        if self.uniform_memory_ptr.is_null() {
            let size = (std::mem::size_of::<g_types::UniformBufferObject>()
                * self.uniform_buffers_to_allocate.len()) as u64;
            let (uniform_buffer, uniform_buffer_memory, _) = g_utils::create_buffer_and_memory(
                instance,
                logical_device,
                physical_device,
                size * 4,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?;

            let memory_ptr = logical_device.map_memory(
                uniform_buffer_memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            self.uniform_buffer = uniform_buffer;
            self.uniform_memory = uniform_buffer_memory;
            self.uniform_memory_ptr = memory_ptr;
        }

        let mut offset = 0;
        self.vertex_buffers_to_allocate
            .iter_mut()
            .filter(|buffer| buffer.size.is_some())
            .for_each(|buffer| {
                let alignment = logical_device
                    .get_buffer_memory_requirements(buffer.buffer)
                    .alignment;
                offset = g_utils::align_up(offset, alignment);
                let dst = self.stage_memory_ptr.add(offset as usize).cast();
                g_utils::memcpy(buffer.vertices.as_ptr(), dst, buffer.vertices.len());
                buffer.offset = Some(offset);
                offset += buffer.size.unwrap();
            });

        if self.index_buffer_to_allocate.size.is_some() {
            let alignment = logical_device
                .get_buffer_memory_requirements(self.index_buffer_to_allocate.buffer)
                .alignment;
            offset = g_utils::align_up(offset, alignment);
            let dst = self.stage_memory_ptr.add(offset as usize).cast();
            g_utils::memcpy(
                self.index_buffer_to_allocate.indices.as_ptr(),
                dst,
                self.index_buffer_to_allocate.indices.len(),
            );
            self.index_buffer_to_allocate.offset = Some(offset);
        }

        (self.vertex_index_buffer, self.vertex_index_memory, _) =
            g_utils::create_buffer_and_memory(
                instance,
                logical_device,
                physical_device,
                vertex_index_size * 4,
                vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

        g_utils::copy_buffer(
            logical_device,
            queue_set.transfer,
            command_pool_set.transfer,
            self.staging_buffer,
            self.vertex_index_buffer,
            vertex_index_size,
            0,
            0,
        )?;

        for buffer in self
            .vertex_buffers_to_allocate
            .iter()
            .filter(|buffer| buffer.offset.is_some())
        {
            logical_device.bind_buffer_memory(
                buffer.buffer,
                self.vertex_index_memory,
                buffer.offset.unwrap(),
            )?;
        }

        if self.index_buffer_to_allocate.offset.is_some() {
            logical_device.bind_buffer_memory(
                self.index_buffer_to_allocate.buffer,
                self.vertex_index_memory,
                self.index_buffer_to_allocate.offset.unwrap(),
            )?;
        }

        let mut offset = 0;
        for buffer in self
            .uniform_buffers_to_allocate
            .iter_mut()
            .filter(|buffer| buffer.size.is_some())
        {
            let alignment = logical_device
                .get_buffer_memory_requirements(buffer.buffer)
                .alignment;
            offset = g_utils::align_up(offset, alignment);

            logical_device.bind_buffer_memory(buffer.buffer, self.uniform_memory, offset)?;
            buffer.offset = Some(offset);
            offset += buffer.size.unwrap();
        }

        Ok(())
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
        log::info!("Destroying buffer memory allocator");
        self.destroy_buffers(logical_device);

        logical_device.destroy_buffer(self.vertex_index_buffer, None);
        logical_device.free_memory(self.vertex_index_memory, None);

        logical_device.unmap_memory(self.staging_memory);
        self.stage_memory_ptr = std::ptr::null_mut();
        logical_device.destroy_buffer(self.staging_buffer, None);
        logical_device.free_memory(self.staging_memory, None);

        logical_device.unmap_memory(self.uniform_memory);
        self.uniform_memory_ptr = std::ptr::null_mut();
        logical_device.destroy_buffer(self.uniform_buffer, None);
        logical_device.free_memory(self.uniform_memory, None);
    }

    pub unsafe fn destroy_buffers(&mut self, logical_device: &Device) {
        log::info!("Destroying buffers");
        self.vertex_buffers_to_allocate
            .iter_mut()
            .filter(|buffer| !buffer.buffer.is_null())
            .for_each(|buffer| buffer.destroy(logical_device));
        if !self.index_buffer_to_allocate.buffer.is_null() {
            self.index_buffer_to_allocate.destroy(logical_device);
        }
        self.uniform_buffers_to_allocate
            .iter_mut()
            .filter(|buffer| !buffer.buffer.is_null())
            .for_each(|buffer| buffer.destroy(logical_device));
    }

    pub unsafe fn create_buffers(&mut self, logical_device: &Device) -> Result<()> {
        log::info!("Creating buffers");
        for buffer in self.vertex_buffers_to_allocate.iter_mut() {
            buffer.create_buffer(logical_device)?;
        }
        self.index_buffer_to_allocate
            .create_buffer(logical_device)?;
        for buffer in self.uniform_buffers_to_allocate.iter_mut() {
            buffer.create_buffer(logical_device)?;
        }
        Ok(())
    }
}

pub struct TextureEngine {
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
}

impl TextureEngine {
    pub unsafe fn create(
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let (texture_image, texture_image_memory) =
            g_utils::create_texture_image(instance, logical_device, physical_device)?;
        Ok(Self {
            texture_image,
            texture_image_memory,
        })
    }

    pub unsafe fn destroy(&self, logical_device: &Device) {}
}
