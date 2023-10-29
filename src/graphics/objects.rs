use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::os::raw::c_void;

use crate::graphics::types as g_types;
use crate::graphics::utils as g_utils;
use anyhow::Result;

use rand::distributions::uniform;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::KhrSwapchainExtension;

use winit::window::Window;

use super::types::Vertex;

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
    pub unsafe fn create(
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        swapchain: &Swapchain,
        msaa_samples: vk::SampleCountFlags,
        texture_engine: &TextureMemoryAllocator,
    ) -> Result<Self> {
        let descriptor_set_layout = g_utils::create_descriptor_set_layout(
            logical_device,
            texture_engine.textures.len() as u32,
            texture_engine.samplers.len() as u32,
        )?;

        let descriptor_pool = g_utils::create_descriptor_pool(
            logical_device,
            swapchain.images.len() as u32,
            texture_engine.textures.len() as u32,
            texture_engine.samplers.len() as u32,
        )?;

        let (pipeline_layout, render_pass, pipeline) = g_utils::create_pipeline_and_renderpass(
            instance,
            logical_device,
            physical_device,
            swapchain,
            descriptor_set_layout,
            msaa_samples,
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
    pub command_pool_set: g_types::CommandPoolSet,
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    color_image: vk::Image,
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,
    framebuffers: Vec<vk::Framebuffer>,
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
        texture_engine: &mut TextureMemoryAllocator,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<Self> {
        let descriptor_sets = g_utils::create_descriptor_sets(
            logical_device,
            pipeline.descriptor_set_layout,
            pipeline.descriptor_pool,
            swapchain.images.len(),
        )?;

        let command_pool_set = g_utils::create_command_pools(logical_device, queue_family_indices)?;

        let (depth_image, depth_image_memory, depth_image_view) = g_utils::create_depth_objects(
            instance,
            logical_device,
            physical_device,
            swapchain.extent,
            &command_pool_set,
            queue_set,
            msaa_samples,
        )?;

        let (color_image, color_image_memory, color_image_view) = g_utils::create_color_objects(
            instance,
            logical_device,
            physical_device,
            swapchain,
            msaa_samples,
        )?;

        let framebuffers = g_utils::create_framebuffers(
            logical_device,
            pipeline.render_pass,
            swapchain,
            depth_image_view,
            color_image_view,
        )?;

        buffer_memory_allocator.create_buffers(logical_device)?;

        buffer_memory_allocator.allocate_memory(
            instance,
            logical_device,
            physical_device,
            queue_set,
            command_pool_set,
        )?;

        texture_engine.create_textures(instance, logical_device, physical_device)?;

        texture_engine.allocate_memory(
            instance,
            logical_device,
            physical_device,
            queue_set,
            &command_pool_set,
        )?;

        g_utils::update_descriptor_sets(
            logical_device,
            swapchain.images.len(),
            &buffer_memory_allocator.uniform_buffers_to_allocate,
            &descriptor_sets,
            texture_engine, //.texture_image_view,
                            // texture_engine, //.texture_sampler,
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
            depth_image,
            depth_image_memory,
            depth_image_view,
            color_image,
            color_image_memory,
            color_image_view,
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
        self.command_pool_set.destroy(logical_device);
        self.framebuffers
            .iter()
            .for_each(|framebuffer| logical_device.destroy_framebuffer(*framebuffer, None));
        logical_device.destroy_image_view(self.depth_image_view, None);
        logical_device.destroy_image(self.depth_image, None);
        logical_device.free_memory(self.depth_image_memory, None);
        logical_device.destroy_image_view(self.color_image_view, None);
        logical_device.destroy_image(self.color_image, None);
        logical_device.free_memory(self.color_image_memory, None);
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
    pub unsafe fn create(vertices: &[g_types::Vertex]) -> Result<Self> {
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
    indices: Vec<u32>,
    size: Option<u64>,
    offset: Option<u64>,
    changed: bool,
}

impl IndexBuffer {
    pub unsafe fn create(indices: &[u32]) -> Result<Self> {
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
                size * 2,
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

pub struct TextureMemoryAllocator {
    pub textures: Vec<Texture>,
    pub staging_buffer: vk::Buffer,
    pub staging_memory: vk::DeviceMemory,
    pub staging_memory_ptr: *mut c_void,
    pub texture_memorys: HashMap<u32, vk::DeviceMemory>,
    pub samplers: HashMap<u32, vk::Sampler>,
    changed: bool,
}

impl TextureMemoryAllocator {
    pub unsafe fn create() -> Result<Self> {
        Ok(Self {
            textures: Vec::new(),
            staging_buffer: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            staging_memory_ptr: std::ptr::null_mut(),
            texture_memorys: HashMap::new(), //vk::DeviceMemory::null(),
            samplers: HashMap::new(),        //vk::Sampler::null(),
            changed: false,
        })
    }

    pub unsafe fn add_texture(&mut self, texture: Texture) {
        self.textures.push(texture);
        self.changed = true;
    }

    unsafe fn create_and_map_staging_buffer_and_memory(
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        size: u64,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, *mut c_void)> {
        let (staging_buffer, staging_memory, _) = g_utils::create_buffer_and_memory(
            instance,
            logical_device,
            physical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let memory =
            logical_device.map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())?;

        Ok((staging_buffer, staging_memory, memory))
    }

    pub unsafe fn create_textures(
        &mut self,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<()> {
        log::info!("Creating textures");
        for texture in self.textures.iter_mut() {
            texture.create_image_objects(instance, logical_device, physical_device)?;
        }

        Ok(())
    }

    fn check_for_changes(&mut self) {
        for texture in self.textures.iter() {
            if texture.changed {
                self.changed = true;
                return;
            }
        }
    }

    fn reset_changes(&mut self) {
        for texture in self.textures.iter_mut() {
            texture.changed = false;
        }
    }

    pub unsafe fn prepare_samplers(&mut self, logical_device: &Device) -> Result<()> {
        let all_mip_levels = self
            .textures
            .iter()
            .map(|texture| texture.mip_levels)
            .collect::<HashSet<_>>();

        for mip_levels in all_mip_levels {
            let sampler = g_utils::create_texture_sampler(logical_device, mip_levels)?;
            self.samplers.insert(mip_levels, sampler);
        }

        Ok(())
    }

    pub unsafe fn allocate_memory(
        &mut self,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
        command_pool_set: &g_types::CommandPoolSet,
    ) -> Result<()> {
        self.check_for_changes();
        if !self.changed {
            return Ok(());
        }
        self.reset_changes();
        self.changed = false;

        let memory_type_indexes = self
            .textures
            .iter()
            .map(|texture| texture.memory_type_index)
            .collect::<HashSet<_>>();

        let total_size = self
            .textures
            .iter()
            .fold(0, |acc, texture| acc + texture.reqs.unwrap().size);

        if self.staging_memory_ptr.is_null() {
            let (staging_buffer, staging_buffer_memory, memory_ptr) =
                Self::create_and_map_staging_buffer_and_memory(
                    instance,
                    logical_device,
                    physical_device,
                    total_size,
                )?;
            self.staging_buffer = staging_buffer;
            self.staging_memory = staging_buffer_memory;
            self.staging_memory_ptr = memory_ptr;
        }

        for memory_type_index in memory_type_indexes {
            let required_size = self
                .textures
                .iter()
                .filter(|texture| texture.memory_type_index == memory_type_index)
                .fold(0, |acc, texture| acc + texture.reqs.unwrap().size);

            let mut offset = 0;
            self.textures
                .iter_mut()
                .filter(|texture| texture.memory_type_index == memory_type_index)
                .for_each(|texture| {
                    log::info!("Copying texture for {:?}", texture.image);
                    offset = g_utils::align_up(offset, texture.reqs.unwrap().alignment);
                    let dst = self.staging_memory_ptr.add(offset as usize).cast();
                    g_utils::memcpy(texture.pixels.as_ptr(), dst, texture.pixels.len());
                    texture.offset = Some(offset);
                    offset += texture.reqs.unwrap().size;
                });

            let texture_memory = g_utils::create_memory_with_mem_type_index(
                logical_device,
                required_size,
                memory_type_index,
            )?;

            self.texture_memorys
                .insert(memory_type_index, texture_memory);

            for texture in self.textures.iter_mut().filter(|texture| {
                texture.memory_type_index == memory_type_index && texture.offset.is_some()
            }) {
                log::info!("Binding memory for {:?}", texture.image);

                logical_device.bind_image_memory(
                    texture.image,
                    *self.texture_memorys.get(&memory_type_index).unwrap(),
                    texture.offset.unwrap(),
                )?;

                g_utils::transition_image_layout(
                    logical_device,
                    command_pool_set.graphics,
                    queue_set.graphics,
                    texture.image,
                    texture.mip_levels,
                    vk::Format::R8G8B8A8_SRGB,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::QUEUE_FAMILY_IGNORED,
                    vk::QUEUE_FAMILY_IGNORED,
                )?;

                g_utils::copy_buffer_to_image(
                    logical_device,
                    command_pool_set.graphics,
                    queue_set.graphics,
                    self.staging_buffer,
                    texture.image,
                    texture.width,
                    texture.height,
                    texture.offset.unwrap(),
                )?;

                g_utils::generate_mipmaps(
                    instance,
                    logical_device,
                    physical_device,
                    command_pool_set.graphics,
                    queue_set.graphics,
                    texture.image,
                    vk::Format::R8G8B8A8_SRGB,
                    texture.width,
                    texture.height,
                    texture.mip_levels,
                )?;

                texture.image_view = g_utils::create_texture_image_view(
                    logical_device,
                    texture.image,
                    texture.mip_levels,
                )?;
            }
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
        self.textures.iter().for_each(|texture| {
            log::info!("Destroying texture for {:?}", texture.image);
            texture.destroy(logical_device);
        });

        for (_, sampler) in self.samplers.iter() {
            logical_device.destroy_sampler(*sampler, None);
        }

        for (_, texture_memory) in self.texture_memorys.iter() {
            logical_device.free_memory(*texture_memory, None);
        }

        logical_device.unmap_memory(self.staging_memory);
        logical_device.destroy_buffer(self.staging_buffer, None);
        self.staging_memory_ptr = std::ptr::null_mut();
        logical_device.free_memory(self.staging_memory, None);
    }
}

pub struct Texture {
    pub image: vk::Image,
    pub pixels: Vec<u8>,
    // pub image_memory: vk::DeviceMemory,
    pub image_view: vk::ImageView,
    pub pixels_size: u64,
    pub reqs: Option<vk::MemoryRequirements>,
    pub offset: Option<u64>,
    pub mip_levels: u32,
    pub width: u32,
    pub height: u32,
    pub memory_type_index: u32,
    changed: bool,
}

impl Texture {
    pub unsafe fn create(path: &str) -> Result<Self> {
        let image = File::open(path)?;
        let decoder = png::Decoder::new(image);
        let mut reader = decoder.read_info()?;

        let mut pixels = vec![0; reader.info().raw_bytes()];
        reader.next_frame(&mut pixels)?;

        let pixels_size = reader.info().raw_bytes() as u64;
        let (width, height) = reader.info().size();

        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        Ok(Self {
            image: vk::Image::null(),
            pixels,
            image_view: vk::ImageView::null(),
            pixels_size,
            reqs: None,
            offset: None,
            mip_levels,
            width,
            height,
            memory_type_index: 0,
            changed: false,
        })
    }

    pub unsafe fn create_image_objects(
        &mut self,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<()> {
        if self.image.is_null() {
            let info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::_2D)
                .extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                })
                .mip_levels(self.mip_levels)
                .array_layers(1)
                .format(vk::Format::R8G8B8A8_SRGB)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .samples(vk::SampleCountFlags::_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            self.image = logical_device.create_image(&info, None)?;
            let reqs = logical_device.get_image_memory_requirements(self.image);
            self.reqs = Some(reqs);

            self.memory_type_index = g_utils::get_memory_type_index(
                instance,
                physical_device,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                reqs,
            )?;
        }

        Ok(())
    }

    pub unsafe fn destroy(&self, logical_device: &Device) {
        logical_device.destroy_image_view(self.image_view, None);
        logical_device.destroy_image(self.image, None);
        // logical_device.free_memory(self.image_memory, None);
    }
}

pub struct Model {
    pub vertex_buffer: VertexBuffer,
    pub index_buffer: IndexBuffer,
}

impl Model {
    pub unsafe fn create() -> Result<Self> {
        let (vertices, indices) = g_utils::load_model()?;

        let vertex_buffer = VertexBuffer::create(&vertices)?;

        let index_buffer = IndexBuffer::create(&indices)?;

        Ok(Self {
            vertex_buffer,
            index_buffer,
        })
    }
}
