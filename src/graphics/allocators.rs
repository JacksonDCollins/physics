use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    os::raw::c_void,
    sync::Arc,
};

use ash::vk;

use anyhow::Result;

use super::{
    objects as g_objects, types as g_types,
    utils::{self as g_utils, IsNull},
};

pub struct TextureMemoryAllocator {
    pub staging_buffer: vk::Buffer,
    pub staging_memory: vk::DeviceMemory,
    pub staging_memory_ptr: *mut c_void,
    pub texture_memorys: HashMap<u32, vk::DeviceMemory>,
    changed: bool,
}

impl TextureMemoryAllocator {
    pub unsafe fn create() -> Result<Self> {
        Ok(Self {
            staging_buffer: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            staging_memory_ptr: std::ptr::null_mut(),
            texture_memorys: HashMap::new(),
            changed: true,
        })
    }

    unsafe fn create_and_map_staging_buffer_and_memory(
        instance: &ash::Instance,
        logical_device: &ash::Device,
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
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        models: Arc<RefCell<Vec<g_objects::Model>>>,
    ) -> Result<()> {
        for model in models.borrow_mut().iter_mut() {
            model
                .texture
                .create_image_objects(instance, logical_device, physical_device)?;
        }

        Ok(())
    }

    pub unsafe fn allocate_memory(
        &mut self,
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
        command_pool_set: &g_types::CommandPoolSet,
        models: Arc<RefCell<Vec<g_objects::Model>>>,
    ) -> Result<()> {
        if !self.changed {
            return Ok(());
        }

        self.changed = false;

        let memory_type_indexes = models
            .borrow()
            .iter()
            .map(|model| model.texture.memory_type_index)
            .collect::<HashSet<_>>();

        let total_size = models.borrow().iter().fold(0, |acc, model| {
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
                .borrow()
                .iter()
                .filter(|model| model.texture.memory_type_index == memory_type_index)
                .fold(0, |acc, model| {
                    acc + g_utils::align_up(
                        model.texture.reqs.unwrap().size,
                        model.texture.reqs.unwrap().alignment,
                    )
                });

            let mut offset = 0;
            models
                .borrow_mut()
                .iter_mut()
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

            for model in models.borrow_mut().iter_mut().filter(|model| {
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

    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
        for (_, texture_memory) in self.texture_memorys.iter() {
            logical_device.free_memory(*texture_memory, None);
        }

        logical_device.unmap_memory(self.staging_memory);
        logical_device.destroy_buffer(self.staging_buffer, None);
        self.staging_memory_ptr = std::ptr::null_mut();
        logical_device.free_memory(self.staging_memory, None);
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

    pub uniform_buffers_to_allocate: Vec<g_objects::UniformBuffer>,
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

            uniform_buffers_to_allocate: Vec::new(),
            changed: true,
        })
    }

    pub unsafe fn add_uniform_buffer(&mut self, buffer: g_objects::UniformBuffer) {
        self.uniform_buffers_to_allocate.push(buffer);
    }

    unsafe fn create_and_map_staging_buffer_and_memory(
        instance: &ash::Instance,
        logical_device: &ash::Device,
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

    pub unsafe fn create_memories(
        &mut self,
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        size: u64,
    ) -> Result<()> {
        if self.stage_memory_ptr.is_null() {
            let (staging_buffer, staging_buffer_memory, memory_ptr) =
                Self::create_and_map_staging_buffer_and_memory(
                    instance,
                    logical_device,
                    physical_device,
                    size,
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
                size,
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

        if self.vertex_index_memory.is_null() {
            (self.vertex_index_buffer, self.vertex_index_memory, _) =
                g_utils::create_buffer_and_memory(
                    instance,
                    logical_device,
                    physical_device,
                    size,
                    vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::INDEX_BUFFER,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?;
        }
        Ok(())
    }

    pub unsafe fn allocate_memory(
        &mut self,
        logical_device: &ash::Device,
        queue_set: &g_utils::QueueSet,
        command_pool_set: g_types::CommandPoolSet,
        models: Arc<RefCell<Vec<g_objects::Model>>>,
        size: u64,
    ) -> Result<()> {
        if !self.changed {
            return Ok(());
        }

        self.changed = false;

        models.borrow_mut().iter_mut().for_each(|model| {
            g_utils::memcpy(
                model.vertex_buffer.vertices.as_ptr(),
                self.stage_memory_ptr
                    .add(model.vertex_buffer.offset.unwrap() as usize)
                    .cast(),
                model.vertex_buffer.vertices.len(),
            );

            g_utils::memcpy(
                model.index_buffer.indices.as_ptr(),
                self.stage_memory_ptr
                    .add(model.index_buffer.offset.unwrap() as usize)
                    .cast(),
                model.index_buffer.indices.len(),
            );

            g_utils::memcpy(
                model.instance_buffer.model_matrixes.as_ptr(),
                self.stage_memory_ptr
                    .add(model.instance_buffer.offset.unwrap() as usize)
                    .cast(),
                model.instance_buffer.model_matrixes.len(),
            );
        });

        g_utils::copy_buffer(
            logical_device,
            queue_set.transfer,
            command_pool_set.transfer,
            self.staging_buffer,
            self.vertex_index_buffer,
            size,
            0,
            0,
        )?;

        for buffer in models
            .borrow()
            .iter()
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
            .borrow()
            .iter()
            .map(|model| &model.index_buffer)
            .filter(|buffer| buffer.offset.is_some())
        {
            logical_device.bind_buffer_memory(
                buffer.buffer,
                self.vertex_index_memory,
                buffer.offset.unwrap(),
            )?;
        }

        for buffer in models
            .borrow()
            .iter()
            .map(|model| &model.instance_buffer)
            .filter(|buffer| buffer.offset.is_some())
        {
            logical_device.bind_buffer_memory(
                buffer.buffer,
                self.vertex_index_memory,
                buffer.offset.unwrap(),
            )?;
        }

        let mut offset = 0;
        for buffer in self.uniform_buffers_to_allocate.iter_mut() {
            let alignment = logical_device
                .get_buffer_memory_requirements(buffer.buffer)
                .alignment;
            offset = g_utils::align_up(offset, alignment);

            logical_device.bind_buffer_memory(buffer.buffer, self.uniform_memory, offset)?;
            buffer.offset = Some(offset);
            offset += buffer.size;
        }

        Ok(())
    }

    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
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

    pub unsafe fn destroy_buffers(&mut self, logical_device: &ash::Device) {
        self.uniform_buffers_to_allocate
            .iter_mut()
            .filter(|buffer| !buffer.buffer.is_null())
            .for_each(|buffer| buffer.destroy(logical_device));
    }

    pub unsafe fn create_buffers(
        &mut self,
        logical_device: &ash::Device,
        models: Arc<RefCell<Vec<g_objects::Model>>>,
    ) -> Result<()> {
        for model in models.borrow_mut().iter_mut() {
            model.vertex_buffer.create_buffer(logical_device)?;
            model.index_buffer.create_buffer(logical_device)?;
            model.instance_buffer.create_buffer(logical_device)?;
        }

        for buffer in self.uniform_buffers_to_allocate.iter_mut() {
            buffer.create_buffer(logical_device)?;
        }

        Ok(())
    }
}
