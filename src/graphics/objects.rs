use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::hash::Hash;
use std::mem::size_of;
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
    // descriptor_set_layout: vk::DescriptorSetLayout,
    // descriptor_pool: vk::DescriptorPool,
    // pub pipeline_layout: vk::PipelineLayout,
    pub render_pass: vk::RenderPass,
    // pub pipeline: vk::Pipeline,
}

impl Pipeline {
    pub unsafe fn create(
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        swapchain: &Swapchain,
        msaa_samples: vk::SampleCountFlags,
        // model_manager: &mut ModelManager,
    ) -> Result<Self> {
        // let descriptor_set_layout = g_utils::create_descriptor_set_layout(logical_device, 1, 1)?;

        // let descriptor_pool =
        //     g_utils::create_descriptor_pool(logical_device, swapchain.images.len() as u32, 1, 1)?;

        // let (pipeline_layout, render_pass, pipeline) = g_utils::create_pipeline_and_renderpass(
        //     instance,
        //     logical_device,
        //     physical_device,
        //     swapchain,
        //     descriptor_set_layout,
        //     msaa_samples,
        // )?;

        let render_pass = g_utils::create_render_pass(
            instance,
            logical_device,
            physical_device,
            swapchain,
            msaa_samples,
        )?;

        Ok(Self {
            // descriptor_set_layout,
            // descriptor_pool,
            // pipeline_layout,
            render_pass,
            // pipeline,
        })
    }

    pub unsafe fn destroy(&self, logical_device: &Device) {
        // logical_device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        // logical_device.destroy_descriptor_pool(self.descriptor_pool, None);
        // logical_device.destroy_pipeline(self.pipeline, None);
        logical_device.destroy_render_pass(self.render_pass, None);
        // logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
    }
}

pub struct Presenter {
    // pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub command_pool_sets: Vec<g_types::CommandPoolSet>,
    pub master_command_pool_set: g_types::CommandPoolSet,
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    color_image: vk::Image,
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
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
        model_manager: &mut ModelManager,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<Self> {
        // let descriptor_sets = g_utils::create_descriptor_sets(
        //     logical_device,
        //     pipeline.descriptor_set_layout,
        //     pipeline.descriptor_pool,
        //     swapchain.images.len(),
        // )?;

        let master_command_pool_set =
            g_utils::create_command_pool_set(logical_device, queue_family_indices)?;

        let command_pool_sets = g_utils::create_command_pool_sets(
            logical_device,
            swapchain.images.len() as u32,
            queue_family_indices,
        )?;

        let (depth_image, depth_image_memory, depth_image_view) = g_utils::create_depth_objects(
            instance,
            logical_device,
            physical_device,
            swapchain.extent,
            &master_command_pool_set,
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

        model_manager.create_buffers(logical_device)?;

        model_manager.allocate_memory_for_buffers(
            instance,
            logical_device,
            physical_device,
            queue_set,
            master_command_pool_set,
        )?;

        model_manager
            // .texture_engine
            .create_textures(instance, logical_device, physical_device)?;

        model_manager
            //.texture_engine
            .allocate_texture_memory(
                instance,
                logical_device,
                physical_device,
                queue_set,
                master_command_pool_set,
            )?;

        model_manager.update_descriptor_sets(logical_device, swapchain.images.len())?;

        let command_buffers = g_utils::create_command_buffers(
            logical_device,
            &command_pool_sets,
            swapchain.images.len(),
        )?;

        let secondary_command_buffers = vec![vec![]; swapchain.images.len()];

        let (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        ) = g_utils::create_sync_objects(logical_device, swapchain.images.len())?;

        Ok(Self {
            // descriptor_sets,
            framebuffers,
            command_pool_sets,
            master_command_pool_set,
            depth_image,
            depth_image_memory,
            depth_image_view,
            color_image,
            color_image_memory,
            color_image_view,
            command_buffers,
            secondary_command_buffers,
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
        self.command_pool_sets.iter().for_each(|pool_set| {
            pool_set.destroy(logical_device);
        });
        self.master_command_pool_set.destroy(logical_device);
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
    reqs: Option<vk::MemoryRequirements>,
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
            reqs: None,
        })
    }

    pub fn get_buffer(&self) -> vk::Buffer {
        self.buffer
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

            self.reqs = Some(logical_device.get_buffer_memory_requirements(self.buffer));
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
    reqs: Option<vk::MemoryRequirements>,
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
            reqs: None,
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
            reqs: None,
        }
    }

    pub unsafe fn create_buffer(&mut self, logical_device: &Device) -> Result<()> {
        if self.buffer.is_null() && self.size.is_some() {
            self.buffer = g_utils::create_buffer(
                logical_device,
                self.size.unwrap(),
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            )?;

            self.reqs = Some(logical_device.get_buffer_memory_requirements(self.buffer));
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
    // pub vertex_buffers_to_allocate: Vec<VertexBuffer>,
    // pub index_buffer_to_allocate: IndexBuffer,
    pub uniform_buffers_to_allocate: Vec<UniformBuffer>,
    pub changed: bool,
}

impl BufferMemoryAllocator {
    pub unsafe fn create() -> Result<Self> {
        Ok(Self {
            vertex_index_memory: vk::DeviceMemory::null(),
            vertex_index_buffer: vk::Buffer::null(),
            uniform_memory: vk::DeviceMemory::null(),
            uniform_buffer: vk::Buffer::null(),
            uniform_memory_ptr: std::ptr::null_mut(),
            staging_buffer: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            stage_memory_ptr: std::ptr::null_mut(),
            // vertex_buffers_to_allocate: Vec::new(),
            // index_buffer_to_allocate: IndexBuffer::null(),
            uniform_buffers_to_allocate: Vec::new(),
            changed: true,
        })
    }

    pub unsafe fn add_uniform_buffer(&mut self, buffer: UniformBuffer) {
        self.uniform_buffers_to_allocate.push(buffer);
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
        models: &mut HashMap<String, Model>,
    ) -> Result<()> {
        // self.check_for_changes();
        if !self.changed {
            return Ok(());
        }
        // self.reset_changes();
        self.changed = false;

        let vertex_size = models
            .values()
            .map(|model| &model.vertex_buffer)
            .filter(|buffer| buffer.size.is_some())
            .fold(0, |acc, buffer| {
                acc + g_utils::align_up(buffer.reqs.unwrap().size, buffer.reqs.unwrap().alignment)
            });

        let index_size = models
            .values()
            .map(|model| &model.index_buffer)
            .filter(|buffer| buffer.size.is_some())
            .fold(0, |acc, buffer| {
                acc + g_utils::align_up(buffer.reqs.unwrap().size, buffer.reqs.unwrap().alignment)
            });

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
        models
            .values_mut()
            .map(|model| &mut model.vertex_buffer)
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

        models
            .values_mut()
            .map(|model| &mut model.index_buffer)
            .filter(|buffer| buffer.size.is_some())
            .for_each(|buffer: &mut IndexBuffer| {
                let alignment = logical_device
                    .get_buffer_memory_requirements(buffer.buffer)
                    .alignment;
                offset = g_utils::align_up(offset, alignment);
                let dst = self.stage_memory_ptr.add(offset as usize).cast();
                g_utils::memcpy(buffer.indices.as_ptr(), dst, buffer.indices.len());
                buffer.offset = Some(offset);
                offset += buffer.size.unwrap();
            });

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

        for buffer in models
            .values()
            .map(|model| &model.vertex_buffer)
            .filter(|buffer| buffer.offset.is_some())
        {
            logical_device.bind_buffer_memory(
                buffer.buffer,
                self.vertex_index_memory,
                buffer.offset.unwrap(),
            )?;
        }

        for buffer in models
            .values()
            .map(|model| &model.index_buffer)
            .filter(|buffer| buffer.offset.is_some())
        {
            logical_device.bind_buffer_memory(
                buffer.buffer,
                self.vertex_index_memory,
                buffer.offset.unwrap(),
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
        // self.vertex_buffers_to_allocate
        //     .iter_mut()
        //     .filter(|buffer| !buffer.buffer.is_null())
        //     .for_each(|buffer| buffer.destroy(logical_device));
        // if !self.index_buffer_to_allocate.buffer.is_null() {
        //     self.index_buffer_to_allocate.destroy(logical_device);
        // }
        self.uniform_buffers_to_allocate
            .iter_mut()
            .filter(|buffer| !buffer.buffer.is_null())
            .for_each(|buffer| buffer.destroy(logical_device));
    }

    pub unsafe fn create_buffers(
        &mut self,
        logical_device: &Device,
        models: &mut HashMap<String, Model>,
    ) -> Result<()> {
        for model in models.values_mut() {
            model.vertex_buffer.create_buffer(logical_device)?;
            model.index_buffer.create_buffer(logical_device)?;
        }

        for buffer in self.uniform_buffers_to_allocate.iter_mut() {
            buffer.create_buffer(logical_device)?;
        }
        Ok(())
    }
}

pub struct TextureMemoryAllocator {
    // pub textures: Vec<Texture>,
    pub staging_buffer: vk::Buffer,
    pub staging_memory: vk::DeviceMemory,
    pub staging_memory_ptr: *mut c_void,
    pub texture_memorys: HashMap<u32, vk::DeviceMemory>,
    changed: bool,
}

impl TextureMemoryAllocator {
    pub unsafe fn create() -> Result<Self> {
        Ok(Self {
            // textures: Vec::new(),
            staging_buffer: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            staging_memory_ptr: std::ptr::null_mut(),
            texture_memorys: HashMap::new(), //vk::DeviceMemory::null(),
            changed: true,
        })
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
        models: &mut HashMap<String, Model>,
    ) -> Result<()> {
        for model in models.values_mut() {
            model
                .texture
                .create_image_objects(instance, logical_device, physical_device)?;
        }

        Ok(())
    }

    // fn check_for_changes(&mut self) {
    //     for texture in self.textures.iter() {
    //         if texture.changed {
    //             self.changed = true;
    //             return;
    //         }
    //     }
    // }

    // fn reset_changes(&mut self) {
    //     for texture in self.textures.iter_mut() {
    //         texture.changed = false;
    //     }
    // }

    pub unsafe fn allocate_memory(
        &mut self,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
        command_pool_set: &g_types::CommandPoolSet,
        models: &mut HashMap<String, Model>,
    ) -> Result<()> {
        // self.check_for_changes();
        if !self.changed {
            return Ok(());
        }
        // self.reset_changes();
        self.changed = false;

        let memory_type_indexes = models
            .values()
            .map(|model| model.texture.memory_type_index)
            .collect::<HashSet<_>>();

        let total_size = models.values().fold(0, |acc, model| {
            acc + g_utils::align_up(
                model.texture.reqs.unwrap().size,
                model.texture.reqs.unwrap().alignment,
            )
        });

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
            let required_size = models
                .values()
                .filter(|model| model.texture.memory_type_index == memory_type_index)
                .fold(0, |acc, model| {
                    acc + g_utils::align_up(
                        model.texture.reqs.unwrap().size,
                        model.texture.reqs.unwrap().alignment,
                    )
                });

            let mut offset = 0;
            models
                .values_mut()
                .filter(|model| model.texture.memory_type_index == memory_type_index)
                .for_each(|model| {
                    offset = g_utils::align_up(offset, model.texture.reqs.unwrap().alignment);
                    let dst = self.staging_memory_ptr.add(offset as usize).cast();
                    g_utils::memcpy(
                        model.texture.pixels.as_ptr(),
                        dst,
                        model.texture.pixels.len(),
                    );
                    model.texture.offset = Some(offset);
                    offset += model.texture.reqs.unwrap().size;
                });

            let texture_memory = g_utils::create_memory_with_mem_type_index(
                logical_device,
                required_size,
                memory_type_index,
            )?;

            self.texture_memorys
                .insert(memory_type_index, texture_memory);

            for model in models.values_mut().filter(|model| {
                model.texture.memory_type_index == memory_type_index
                    && model.texture.offset.is_some()
            }) {
                logical_device.bind_image_memory(
                    model.texture.image,
                    *self.texture_memorys.get(&memory_type_index).unwrap(),
                    model.texture.offset.unwrap(),
                )?;

                g_utils::transition_image_layout(
                    logical_device,
                    command_pool_set.graphics,
                    queue_set.graphics,
                    model.texture.image,
                    model.texture.mip_levels,
                    model.texture.format,
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
                    model.texture.image,
                    model.texture.width,
                    model.texture.height,
                    model.texture.offset.unwrap(),
                )?;

                g_utils::generate_mipmaps(
                    instance,
                    logical_device,
                    physical_device,
                    command_pool_set.graphics,
                    queue_set.graphics,
                    model.texture.image,
                    model.texture.format,
                    model.texture.width,
                    model.texture.height,
                    model.texture.mip_levels,
                )?;

                model.texture.image_view = g_utils::create_texture_image_view(
                    logical_device,
                    model.texture.image,
                    model.texture.mip_levels,
                    model.texture.format,
                )?;
            }
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
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
    pub sampler: vk::Sampler,
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
    format: vk::Format,
}

impl Texture {
    pub unsafe fn create(path: &str) -> Result<Self> {
        let image = File::open(path)?;
        let mut decoder = png::Decoder::new(image);
        decoder.set_transformations(png::Transformations::ALPHA);
        let mut reader = decoder.read_info()?;

        let mut pixels = vec![0; reader.output_buffer_size()];
        reader.next_frame(&mut pixels)?;

        let pixels_size = reader.info().raw_bytes() as u64;
        let (width, height) = reader.info().size();

        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        Ok(Self {
            image: vk::Image::null(),
            sampler: vk::Sampler::null(),
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
            format: vk::Format::R8G8B8A8_SRGB,
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
                .format(self.format)
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
            self.sampler = g_utils::create_texture_sampler(logical_device, self.mip_levels)?;
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
        logical_device.destroy_sampler(self.sampler, None);
        logical_device.destroy_image_view(self.image_view, None);
        logical_device.destroy_image(self.image, None);
        // logical_device.free_memory(self.image_memory, None);
    }
}

pub struct Model {
    pub vertex_buffer: VertexBuffer,
    pub index_buffer: IndexBuffer,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub texture: Texture,
}

impl Model {
    pub unsafe fn create(
        path: &str,
        texture_path: &str,
        logical_device: &Device,
        // msaa_samples: vk::SampleCountFlags,
        // render_pass: vk::RenderPass,
        // swapchain_images_count: u32,
    ) -> Result<Self> {
        let (vertices, indices) = g_utils::load_model(path)?;

        let vertex_buffer = VertexBuffer::create(&vertices)?;

        let index_buffer = IndexBuffer::create(&indices)?;

        let texture = Texture::create(texture_path)?;

        let descriptor_set_layout = g_utils::create_descriptor_set_layout(logical_device, 1)?;

        let pipeline_layout =
            g_utils::create_pipeline_layout(logical_device, descriptor_set_layout)?;

        // let pipeline = g_utils::create_pipeline(
        //     logical_device,
        //     vert_shader_module,
        //     frag_shader_module,
        //     extent.width,
        //     extent.height,
        //     msaa_samples,
        //     pipeline_layout,
        //     render_pass,
        // )?;
        let pipeline = vk::Pipeline::null();

        // let descriptor_pool =
        //     g_utils::create_descriptor_pool(logical_device, swapchain_images_count, 1, 1)?;

        // let descriptor_sets = g_utils::create_descriptor_sets(
        //     logical_device,
        //     descriptor_set_layout,
        //     descriptor_pool,
        //     1,
        // )?;

        let descriptor_pool = vk::DescriptorPool::null();
        let descriptor_sets = Vec::new();

        Ok(Self {
            vertex_buffer,
            index_buffer,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            texture,
            descriptor_pool,
            descriptor_sets,
        })
    }

    pub unsafe fn create_descriptor_pool_and_sets(
        &mut self,
        logical_device: &Device,
        swapchain_images_count: usize,
    ) -> Result<()> {
        self.descriptor_pool =
            g_utils::create_descriptor_pool(logical_device, swapchain_images_count as u32, 1)?;

        self.descriptor_sets = g_utils::create_descriptor_sets(
            logical_device,
            self.descriptor_set_layout,
            self.descriptor_pool,
            swapchain_images_count,
        )?;

        Ok(())
    }

    pub unsafe fn update_descriptor_sets(
        &mut self,
        logical_device: &Device,
        swapchain_images_count: usize,
        uniform_buffers: &[UniformBuffer],
    ) -> Result<()> {
        (0..swapchain_images_count).enumerate().for_each(|(i, _)| {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(uniform_buffers[i].get_buffer())
                .offset(0)
                .range(uniform_buffers[i].get_size());
            let buffer_info = &[buffer_info];

            let image_info = &[vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.texture.image_view)
                .sampler(self.texture.sampler)
                .build()];

            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info);

            let image_info_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image_info);

            logical_device.update_descriptor_sets(
                &[ubo_write, image_info_write],
                &[] as &[vk::CopyDescriptorSet],
            );
        });

        Ok(())
    }

    pub unsafe fn create_pipeline(
        &mut self,
        logical_device: &Device,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Result<()> {
        let vert = include_bytes!("../../shaders/vert.spv");
        let frag = include_bytes!("../../shaders/frag.spv");

        let vert_shader_module = g_utils::create_shader_module(logical_device, &vert[..])?;
        let frag_shader_module = g_utils::create_shader_module(logical_device, &frag[..])?;
        self.pipeline = g_utils::create_pipeline(
            logical_device,
            vert_shader_module,
            frag_shader_module,
            extent.width,
            extent.height,
            msaa_samples,
            self.pipeline_layout,
            render_pass,
        )?;

        logical_device.destroy_shader_module(vert_shader_module, None);
        logical_device.destroy_shader_module(frag_shader_module, None);

        Ok(())
    }
    pub unsafe fn push_constants(
        &self,
        logical_device: &Device,
        command_buffer: vk::CommandBuffer,

        // model_index: usize,
        position: &g_types::Vec3,
    ) {
        // let y = (((model_index % 2) as f32) * 2.5) - 1.25;
        // let z = (((model_index / 2) as f32) * -2.0) + 1.0;

        // self.set_position(g_types::vec3(0.0, y, z));

        let model_mat = g_types::Mat4::from_translation(*position)
            * g_types::Mat4::from_axis_angle(g_types::vec3(0.0, 0.0, 1.0), g_types::Deg(90.0));
        let model_bytes = std::slice::from_raw_parts(
            &model_mat as *const g_types::Mat4 as *const u8,
            size_of::<g_types::Mat4>(),
        );

        logical_device.cmd_push_constants(
            command_buffer,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );

        // logical_device.cmd_push_constants(
        //     command_buffer,
        //     self.pipeline_layout,
        //     vk::ShaderStageFlags::FRAGMENT,
        //     64,
        //     &[
        //         (model_index % 2).to_ne_bytes(),
        //         (model_index % 2).to_ne_bytes(),
        //     ]
        //     .concat(),
        // );
    }

    pub unsafe fn draw_model(
        &self,
        logical_device: &Device,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        position: &g_types::Vec3,
    ) -> Result<()> {
        logical_device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );

        logical_device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[self.vertex_buffer.get_buffer()],
            &[0],
        );

        logical_device.cmd_bind_index_buffer(
            command_buffer,
            self.index_buffer.get_buffer(),
            0,
            vk::IndexType::UINT32,
        );

        logical_device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[self.descriptor_sets[image_index]],
            &[],
        );

        self.push_constants(logical_device, command_buffer, position);

        logical_device.cmd_draw_indexed(
            command_buffer,
            self.index_buffer.get_indice_count(),
            1,
            0,
            0,
            0,
        );

        Ok(())
    }

    // pub unsafe fn make_command_buffer(
    //     &mut self,
    //     logical_device: &Device,
    //     image_index: usize,
    //     model_index: usize,
    //     command_buffers: &mut Vec<vk::CommandBuffer>,
    //     command_pool_sets: &Vec<g_types::CommandPoolSet>,
    //     pipeline: &Pipeline,
    //     framebuffers: &Vec<vk::Framebuffer>,
    //     descriptor_sets: &Vec<vk::DescriptorSet>,
    //     start: std::time::Instant,
    // ) -> Result<vk::CommandBuffer> {
    //     // let command_buffers = &mut self.presenter.secondary_command_buffers[image_index];
    //     while model_index >= command_buffers.len() {
    //         println!("Allocating new secondary command buffer");
    //         let allocate_info = vk::CommandBufferAllocateInfo::builder()
    //             .command_pool(command_pool_sets[image_index].graphics)
    //             .level(vk::CommandBufferLevel::SECONDARY)
    //             .command_buffer_count(1);

    //         let command_buffer = logical_device.allocate_command_buffers(&allocate_info)?[0];
    //         command_buffers.push(command_buffer);
    //     }

    //     let command_buffer = command_buffers[model_index];

    //     let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
    //         .render_pass(pipeline.render_pass)
    //         .subpass(0)
    //         .framebuffer(framebuffers[image_index]);

    //     let info = vk::CommandBufferBeginInfo::builder()
    //         .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
    //         .inheritance_info(&inheritance_info);

    //     logical_device.begin_command_buffer(command_buffer, &info)?;

    //     let time = start.elapsed().as_secs_f32();

    //     let y = (((model_index % 2) as f32) * 2.5) - 1.25;
    //     let z = (((model_index / 2) as f32) * -2.0) + 1.0;

    //     logical_device.cmd_bind_pipeline(
    //         command_buffer,
    //         vk::PipelineBindPoint::GRAPHICS,
    //         pipeline.pipeline,
    //     );

    //     logical_device.cmd_bind_descriptor_sets(
    //         command_buffer,
    //         vk::PipelineBindPoint::GRAPHICS,
    //         pipeline.pipeline_layout,
    //         0,
    //         &[descriptor_sets[image_index]],
    //         &[],
    //     );

    //     // for (_name, model) in self.model_manager.models.iter() {
    //     // let model = self.model_manager.models.get_mut(model_name).unwrap();

    //     logical_device.cmd_bind_vertex_buffers(
    //         command_buffer,
    //         0,
    //         &[self.vertex_buffer.get_buffer()],
    //         &[0],
    //     );

    //     logical_device.cmd_bind_index_buffer(
    //         command_buffer,
    //         self.index_buffer.get_buffer(),
    //         0,
    //         vk::IndexType::UINT32,
    //     );

    //     let model_mat = g_types::Mat4::from_translation(self.position)
    //         * g_types::Mat4::from_axis_angle(
    //             g_types::vec3(0.0, 0.0, 1.0),
    //             g_types::Deg(90.0) * time,
    //         );
    //     let model_bytes = std::slice::from_raw_parts(
    //         &model_mat as *const g_types::Mat4 as *const u8,
    //         size_of::<g_types::Mat4>(),
    //     );

    //     logical_device.cmd_push_constants(
    //         command_buffer,
    //         pipeline.pipeline_layout,
    //         vk::ShaderStageFlags::VERTEX,
    //         0,
    //         model_bytes,
    //     );

    //     logical_device.cmd_push_constants(
    //         command_buffer,
    //         pipeline.pipeline_layout,
    //         vk::ShaderStageFlags::FRAGMENT,
    //         64,
    //         &[
    //             (model_index % 2).to_ne_bytes(),
    //             (model_index % 2).to_ne_bytes(),
    //         ]
    //         .concat(),
    //     );

    //     logical_device.cmd_draw_indexed(
    //         command_buffer,
    //         self.index_buffer.get_indice_count(),
    //         1,
    //         0,
    //         0,
    //         0,
    //     );
    //     // }

    //     logical_device.end_command_buffer(command_buffer)?;

    //     Ok(command_buffer)
    // }

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
        self.texture.destroy(logical_device);
        self.vertex_buffer.destroy(logical_device);
        self.index_buffer.destroy(logical_device);
        logical_device.destroy_descriptor_pool(self.descriptor_pool, None);
        logical_device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        logical_device.destroy_pipeline(self.pipeline, None);
        logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
    }
}

pub struct ModelManager {
    pub models: HashMap<String, Model>,
    pub buffer_allocator: BufferMemoryAllocator,
    pub texture_engine: TextureMemoryAllocator,
}

impl ModelManager {
    pub unsafe fn create() -> Result<Self> {
        let buffer_allocator = BufferMemoryAllocator::create()?;
        let texture_engine = TextureMemoryAllocator::create()?;

        Ok(Self {
            models: HashMap::new(),
            buffer_allocator,
            texture_engine,
        })
    }
    pub unsafe fn recreate_pipelines(
        &mut self,
        logical_device: &Device,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Result<()> {
        for model in self.models.values_mut() {
            logical_device.destroy_pipeline(model.pipeline, None);
            model.create_pipeline(logical_device, msaa_samples, render_pass, extent)?;
        }

        Ok(())
    }

    pub unsafe fn create_pipelines(
        &mut self,
        logical_device: &Device,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Result<()> {
        for model in self.models.values_mut() {
            model.create_pipeline(logical_device, msaa_samples, render_pass, extent)?;
        }

        Ok(())
    }

    pub unsafe fn create_descriptor_pools_and_sets(
        &mut self,
        logical_device: &Device,
        swapchain_images_count: usize,
    ) -> Result<()> {
        for model in self.models.values_mut() {
            model.create_descriptor_pool_and_sets(logical_device, swapchain_images_count)?;
        }

        Ok(())
    }

    pub unsafe fn update_descriptor_sets(
        &mut self,
        logical_device: &Device,
        swapchain_images_count: usize,
    ) -> Result<()> {
        for model in self.models.values_mut() {
            model.update_descriptor_sets(
                logical_device,
                swapchain_images_count,
                &self.buffer_allocator.uniform_buffers_to_allocate,
            )?;
        }

        Ok(())
    }

    pub unsafe fn allocate_texture_memory(
        &mut self,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
        command_pool_set: g_types::CommandPoolSet,
    ) -> Result<()> {
        self.texture_engine.allocate_memory(
            instance,
            logical_device,
            physical_device,
            queue_set,
            &command_pool_set,
            &mut self.models,
        )?;

        Ok(())
    }
    pub unsafe fn create_textures(
        &mut self,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<()> {
        self.texture_engine.create_textures(
            instance,
            logical_device,
            physical_device,
            &mut self.models,
        )?;

        Ok(())
    }

    pub unsafe fn load_models_from_scene(
        &mut self,
        scene: &Scene,
        logical_device: &Device,
        // msaa_samples: vk::SampleCountFlags,
        // render_pass: vk::RenderPass,
        // swapchain_images_count: u32,
    ) -> Result<()> {
        for (model_name, _) in scene.models.iter() {
            let model = Model::create(
                &format!("resources/{}/{}.obj", model_name, model_name),
                &format!("resources/{}/{}.png", model_name, model_name),
                logical_device,
                // msaa_samples,
                // render_pass,
                // swapchain_images_count,
            )?;
            self.add_model(model_name, model);
        }

        Ok(())
    }

    pub unsafe fn create_buffers(&mut self, logical_device: &Device) -> Result<()> {
        self.buffer_allocator
            .create_buffers(logical_device, &mut self.models)?;

        Ok(())
    }

    pub unsafe fn allocate_memory_for_buffers(
        &mut self,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
        command_pool_set: g_types::CommandPoolSet,
    ) -> Result<()> {
        self.buffer_allocator.allocate_memory(
            instance,
            logical_device,
            physical_device,
            queue_set,
            command_pool_set,
            &mut self.models,
        )?;

        Ok(())
    }

    pub unsafe fn add_model(&mut self, name: &str, model: Model) {
        self.models.insert(name.to_string(), model);
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
        self.models
            .values_mut()
            .for_each(|model| model.destroy(logical_device));
        self.texture_engine.destroy(logical_device);

        self.buffer_allocator.destroy(logical_device);
    }
}

pub struct Scene {
    models: HashMap<String, g_types::Vec3>,
}

impl Scene {
    pub unsafe fn create() -> Result<Self> {
        Ok(Self {
            models: [
                ("landscape", g_types::vec3(0.0, 0.0, 0.0)),
                ("viking_room", g_types::vec3(0.0, 0.0, 4.0)),
                ("sphere", g_types::vec3(3.0, 0.0, 4.0)),
            ]
            .iter()
            .map(|(name, pos)| (name.to_string(), *pos))
            .collect::<HashMap<_, _>>(),
        })
    }

    pub unsafe fn draw_models(
        &self,
        image_index: usize,
        model_manager: &ModelManager,
        logical_device: &Device,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        for (_model_index, (model_name, pos)) in self.models.iter().enumerate() {
            let model = model_manager.models.get(model_name).unwrap();

            model.draw_model(logical_device, command_buffer, image_index, pos)?;
        }

        Ok(())
    }
}
