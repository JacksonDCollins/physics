use std::cell::RefCell;
use std::collections::HashMap;
// use std::collections::HashSet;
use std::ffi::CStr;
use std::fs;
use std::fs::File;
// use std::hash::Hash;
// use std::io::Cursor;
use std::mem::size_of;
use std::sync::Arc;
// use std::os::raw::c_void;

use crate::graphics::types as g_types;
use crate::graphics::utils as g_utils;
use anyhow::{anyhow, Result};

use ash::util::read_spv;
use cgmath::Euler;
use cgmath::Quaternion;
use cgmath::Rotation3;
use cgmath::Zero;
// use rand::distributions::uniform;
use rand::Rng;

use ash::vk;

use winit::window::Window;

use super::allocators::BufferMemoryAllocator;
use super::allocators::TextureMemoryAllocator;
use super::types::AttributeDescriptions;
use super::types::BindingDescription;

use super::types::render_info;
use super::types::Vertex;
// use super::types::Vertex;
use super::utils::IsNull;
use super::utils::SHADER_FILES;

pub struct Swapchain {
    pub swapchain_loader: ash::extensions::khr::Swapchain,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    pub unsafe fn create(
        window: &Window,
        instance: &ash::Instance,
        logical_device: &ash::Device,
        surface: vk::SurfaceKHR,
        queue_family_indices: &g_utils::QueueFamilyIndices,
        swapchain_support: &g_utils::SwapchainSupport,
    ) -> Result<Self> {
        let (swapchain_loader, swapchain, images, extent, format) = g_utils::create_swapchain(
            window,
            instance,
            logical_device,
            surface,
            queue_family_indices,
            swapchain_support,
        )?;

        let image_views = g_utils::create_swapchain_image_views(logical_device, &images, format)?;

        Ok(Self {
            swapchain_loader,
            swapchain,
            images,
            extent,
            format,
            image_views,
        })
    }

    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        self.image_views
            .iter()
            .for_each(|image_view| logical_device.destroy_image_view(*image_view, None));
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
    }
}

pub struct Pipeline {
    pub render_pass: vk::RenderPass,
}

impl Pipeline {
    pub unsafe fn create(
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        swapchain: &Swapchain,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<Self> {
        let render_pass = g_utils::create_render_pass(
            instance,
            logical_device,
            physical_device,
            swapchain,
            msaa_samples,
        )?;

        Ok(Self { render_pass })
    }

    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        logical_device.destroy_render_pass(self.render_pass, None);
    }
}

pub struct Presenter {
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
        logical_device: &ash::Device,
        swapchain: &Swapchain,
        pipeline: &Pipeline,
        queue_family_indices: &g_utils::QueueFamilyIndices,
        model_manager: &mut ModelManager,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<Self> {
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

        model_manager.create_textures(instance, logical_device, physical_device)?;

        model_manager.allocate_texture_memory(
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

    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
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

#[derive(Debug, Default)]
pub struct InstanceBuffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    pub offset: Option<u64>,
    pub changed: bool,
    pub reqs: Option<vk::MemoryRequirements>,
    pub model_matrixes: Vec<g_types::Mat4>,
}

impl InstanceBuffer {
    pub unsafe fn create(
        pos_and_rot: &[(g_types::Vec3, cgmath::Quaternion<f32>)],
        render_info: &g_types::RenderInfo,
    ) -> Result<Self> {
        let model_matrixes = pos_and_rot
            .iter()
            .map(|(position, rotation)| {
                g_types::Mat4::from_translation(*position)
                    * g_types::Mat4::from(*rotation)
                    * g_types::Mat4::from_nonuniform_scale(
                        render_info.model_info.scale.x,
                        render_info.model_info.scale.y,
                        render_info.model_info.scale.z,
                    )
            })
            .collect::<Vec<_>>();

        let size = std::mem::size_of::<g_types::Mat4>() as u64 * pos_and_rot.len() as u64;

        Ok(Self {
            buffer: vk::Buffer::null(),
            size,
            offset: None,
            changed: false,
            reqs: None,
            model_matrixes,
        })
    }

    pub fn get_required_size(&self) -> u64 {
        g_utils::align_up(self.reqs.unwrap().size, self.reqs.unwrap().alignment)
    }

    pub fn get_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn get_size(&self) -> u64 {
        self.size
    }

    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
        logical_device.destroy_buffer(self.buffer, None);
        self.changed = true;
    }

    pub unsafe fn create_buffer(&mut self, logical_device: &ash::Device) -> Result<()> {
        if self.buffer.is_null() {
            self.buffer = g_utils::create_buffer(
                logical_device,
                self.size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;

            self.reqs = Some(logical_device.get_buffer_memory_requirements(self.buffer));
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct VertexBuffer {
    pub buffer: vk::Buffer,
    pub vertices: Vec<g_types::Vertex>,
    size: u64,
    pub offset: Option<u64>,
    changed: bool,
    pub reqs: Option<vk::MemoryRequirements>,
}

impl VertexBuffer {
    pub unsafe fn create(vertices: &[g_types::Vertex]) -> Result<Self> {
        let size = std::mem::size_of_val(vertices) as u64;

        Ok(Self {
            buffer: vk::Buffer::null(),
            vertices: vertices.to_vec(),
            size,
            offset: None,
            changed: false,
            reqs: None,
        })
    }

    pub fn get_required_size(&self) -> u64 {
        g_utils::align_up(self.reqs.unwrap().size, self.reqs.unwrap().alignment)
    }
    pub fn get_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
        logical_device.destroy_buffer(self.buffer, None);
        self.changed = true;
    }

    pub unsafe fn create_buffer(&mut self, logical_device: &ash::Device) -> Result<()> {
        if self.buffer.is_null() {
            self.buffer = g_utils::create_buffer(
                logical_device,
                self.size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;

            self.reqs = Some(logical_device.get_buffer_memory_requirements(self.buffer));
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct UniformBuffer {
    pub buffer: vk::Buffer,
    pub ubo: g_types::UniformBufferObject,
    pub size: u64,
    pub offset: Option<u64>,
    changed: bool,
}

impl UniformBuffer {
    pub unsafe fn create(
        _logical_device: &ash::Device,
        ubo: g_types::UniformBufferObject,
    ) -> Result<Self> {
        let size = std::mem::size_of::<g_types::UniformBufferObject>() as u64;

        Ok(Self {
            buffer: vk::Buffer::null(),
            ubo,
            size,
            offset: None,
            changed: false,
        })
    }

    pub unsafe fn get_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub unsafe fn get_size(&self) -> u64 {
        self.size
    }

    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
        logical_device.destroy_buffer(self.buffer, None);
        self.changed = true;
    }

    pub unsafe fn update(&mut self, ubo: g_types::UniformBufferObject) {
        self.ubo = ubo;
    }

    pub unsafe fn create_buffer(&mut self, logical_device: &ash::Device) -> Result<()> {
        if self.buffer.is_null() {
            self.buffer = g_utils::create_buffer(
                logical_device,
                self.size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct IndexBuffer {
    pub buffer: vk::Buffer,
    pub indices: Vec<u32>,
    size: u64,
    pub offset: Option<u64>,
    changed: bool,
    pub reqs: Option<vk::MemoryRequirements>,
}

impl IndexBuffer {
    pub unsafe fn create(indices: &[u32]) -> Result<Self> {
        let size = std::mem::size_of_val(indices) as u64;

        Ok(Self {
            buffer: vk::Buffer::null(),
            indices: indices.to_vec(),
            size,
            offset: None,
            changed: false,
            reqs: None,
        })
    }

    pub fn get_required_size(&self) -> u64 {
        g_utils::align_up(self.reqs.unwrap().size, self.reqs.unwrap().alignment)
    }

    pub fn get_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn get_indice_count(&self) -> u32 {
        self.indices.len() as u32
    }

    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
        logical_device.destroy_buffer(self.buffer, None);
        self.changed = true;
    }

    pub unsafe fn create_buffer(&mut self, logical_device: &ash::Device) -> Result<()> {
        if self.buffer.is_null() {
            self.buffer = g_utils::create_buffer(
                logical_device,
                self.size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            )?;

            self.reqs = Some(logical_device.get_buffer_memory_requirements(self.buffer));
        }
        Ok(())
    }
}

pub struct ComputeBuffer<T> {
    pub buffer: vk::Buffer,
    pub data: Vec<T>,
    pub size: u64,
    pub offset: Option<u64>,
    pub changed: bool,
    pub reqs: Option<vk::MemoryRequirements>,
}

impl<T: Clone> ComputeBuffer<T> {
    pub unsafe fn create(data: &[T]) -> Result<Self> {
        let size = std::mem::size_of_val(data) as u64;

        Ok(Self {
            data: data.to_vec(),
            buffer: vk::Buffer::null(),
            size,
            offset: None,
            changed: false,
            reqs: None,
        })
    }

    pub fn get_required_size(&self) -> u64 {
        g_utils::align_up(self.reqs.unwrap().size, self.reqs.unwrap().alignment)
    }

    pub fn get_buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn get_size(&self) -> u64 {
        self.size
    }

    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        logical_device.destroy_buffer(self.buffer, None);
    }

    pub unsafe fn create_buffer(&mut self, logical_device: &ash::Device) -> Result<()> {
        if self.buffer.is_null() {
            self.buffer = g_utils::create_buffer(
                logical_device,
                self.size,
                vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;

            self.reqs = Some(logical_device.get_buffer_memory_requirements(self.buffer));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Texture {
    pub image: vk::Image,
    pub sampler: vk::Sampler,
    pub pixels: Vec<u8>,

    pub image_view: vk::ImageView,
    pub pixels_size: u64,
    pub reqs: Option<vk::MemoryRequirements>,
    pub offset: Option<u64>,
    pub mip_levels: u32,
    pub width: u32,
    pub height: u32,
    pub memory_type_index: u32,
    // changed: bool,
    pub format: vk::Format,
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
            // changed: false,
            format: vk::Format::R8G8B8A8_SRGB,
        })
    }

    pub unsafe fn create_image_objects(
        &mut self,
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<()> {
        if self.image.is_null() {
            let info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
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
                .samples(vk::SampleCountFlags::TYPE_1)
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

    pub unsafe fn destroy(&self, logical_device: &ash::Device) {
        logical_device.destroy_sampler(self.sampler, None);
        logical_device.destroy_image_view(self.image_view, None);
        logical_device.destroy_image(self.image, None);
    }
}

pub struct Terrain {
    pub compute_buffer: ComputeBuffer<Vertex>,
    pub index_buffer: IndexBuffer,
    pub instance_buffer: InstanceBuffer,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub render_info: g_types::RenderInfo,
    pub buffer_memory_allocator: BufferMemoryAllocator,
}

impl Terrain {
    pub unsafe fn create(logical_device: &ash::Device) -> Result<Self> {
        // let (vertices, indices) = g_utils::load_model("resources/landscape/landscape.obj")?;
        // let vertex_buffer = VertexBuffer::create(&vertices)?;
        // let index_buffer = IndexBuffer::create(&indices)?;

        let mut render_info = g_types::DEFAULT_RENDER_INFO;

        render_info.polygon_info.front_face = vk::FrontFace::CLOCKWISE;

        let mut rand = rand::thread_rng();

        let height_list = (0..10)
            .map(|_| (0..5).map(|_| 1.).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let vertex_list = (0..10)
            .flat_map(|i| {
                (0..5).map(move |j| (j, i)).flat_map(|(j, i)| {
                    vec![
                        Vertex::new(
                            g_types::vec3(
                                i as f32,
                                j as f32,
                                *height_list
                                    .get(i.max(1) - 1)
                                    .and_then(|v| v.get(j))
                                    .unwrap_or(&0.0),
                            ),
                            g_types::Vec3::unit_z(),
                            g_types::Vec2::zero(),
                        ),
                        Vertex::new(
                            g_types::vec3(
                                i as f32,
                                j as f32 + 1.,
                                *height_list
                                    .get(i)
                                    .and_then(|v| v.get(j.max(1) - 1))
                                    .unwrap_or(&0.0),
                            ),
                            g_types::Vec3::unit_z(),
                            g_types::Vec2::zero(),
                        ),
                        Vertex::new(
                            g_types::vec3(
                                i as f32 + 1.,
                                j as f32,
                                *height_list.get(i).and_then(|v| v.get(j)).unwrap_or(&0.0),
                            ),
                            g_types::Vec3::unit_z(),
                            g_types::Vec2::zero(),
                        ),
                        Vertex::new(
                            g_types::vec3(
                                i as f32,
                                j as f32 + 1.,
                                *height_list
                                    .get(i)
                                    .and_then(|v| v.get(j.max(1) - 1))
                                    .unwrap_or(&0.0),
                            ),
                            g_types::Vec3::unit_z(),
                            g_types::Vec2::zero(),
                        ),
                        Vertex::new(
                            g_types::vec3(
                                i as f32 + 1.,
                                j as f32 + 1.0,
                                *height_list
                                    .get(i.max(1) - 1)
                                    .and_then(|v| v.get(j.max(1) - 1))
                                    .unwrap_or(&0.0),
                            ),
                            g_types::Vec3::unit_z(),
                            g_types::Vec2::zero(),
                        ),
                        Vertex::new(
                            g_types::vec3(
                                i as f32 + 1.,
                                j as f32,
                                *height_list.get(i).and_then(|v| v.get(j)).unwrap_or(&0.0),
                            ),
                            g_types::Vec3::unit_z(),
                            g_types::Vec2::zero(),
                        ),
                    ]
                })
            })
            .collect::<Vec<_>>();

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for i in 0..vertex_list.len() {
            vertices.push(vertex_list[i]);
            indices.push(i as u32);
        }

        let compute_buffer = ComputeBuffer::create(&vertices)?;

        let index_buffer = IndexBuffer::create(&indices)?;

        let instance_buffer = InstanceBuffer::create(
            &[(
                g_types::vec3(0., 0., 0.),
                cgmath::Quaternion::from_axis_angle(g_types::Vec3::unit_y(), cgmath::Deg(0.)),
            )],
            &render_info,
        )?;

        let descriptor_set_layout = Self::create_descriptor_set_layout(logical_device)?;

        let pipeline_layout = Self::create_pipeline_layout(logical_device, descriptor_set_layout)?;

        let pipeline = vk::Pipeline::null();

        let descriptor_pool = vk::DescriptorPool::null();
        let descriptor_sets = Vec::new();

        let buffer_memory_allocator = BufferMemoryAllocator::create()?;

        Ok(Self {
            compute_buffer,
            index_buffer,
            instance_buffer,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            descriptor_sets,
            render_info,
            buffer_memory_allocator,
        })
    }

    pub unsafe fn create_buffers(&mut self, logical_device: &ash::Device) -> Result<()> {
        self.compute_buffer.create_buffer(logical_device)?;
        self.index_buffer.create_buffer(logical_device)?;
        self.instance_buffer.create_buffer(logical_device)?;

        Ok(())
    }

    pub unsafe fn push_constants(
        &self,
        logical_device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        view: &g_types::Mat4,
        proj: &g_types::Mat4,
    ) {
        let view_bytes = std::slice::from_raw_parts(
            view as *const g_types::Mat4 as *const u8,
            size_of::<g_types::Mat4>(),
        );

        let proj_bytes = std::slice::from_raw_parts(
            proj as *const g_types::Mat4 as *const u8,
            size_of::<g_types::Mat4>(),
        );

        logical_device.cmd_push_constants(
            command_buffer,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            &[view_bytes, proj_bytes].concat(),
        );
    }

    pub unsafe fn draw(
        &self,
        logical_device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        view: &g_types::Mat4,
        proj: &g_types::Mat4,
    ) {
        logical_device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );

        logical_device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[
                self.compute_buffer.get_buffer(),
                self.instance_buffer.get_buffer(),
            ],
            &[0, 0],
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

        self.push_constants(logical_device, command_buffer, view, proj);

        logical_device.cmd_draw_indexed(
            command_buffer,
            self.index_buffer.get_indice_count(),
            1,
            0,
            0,
            0,
        );
    }
    pub unsafe fn create_descriptor_set_layout(
        logical_device: &ash::Device,
    ) -> Result<vk::DescriptorSetLayout> {
        let in_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();

        let out_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();

        let bindings = &[in_binding, out_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        logical_device
            .create_descriptor_set_layout(&info, None)
            .map_err(|e| anyhow!("{}", e))
    }

    pub unsafe fn create_descriptor_pool_and_sets(
        &mut self,
        logical_device: &ash::Device,
        swapchain_images_count: usize,
    ) -> Result<()> {
        self.descriptor_pool =
            Self::create_descriptor_pool(logical_device, swapchain_images_count as u32)?;

        self.descriptor_sets = g_utils::create_descriptor_sets(
            logical_device,
            self.descriptor_set_layout,
            self.descriptor_pool,
            swapchain_images_count,
        )?;

        Ok(())
    }

    pub unsafe fn create_descriptor_pool(
        logical_device: &ash::Device,
        swapchain_images_count: u32,
    ) -> Result<vk::DescriptorPool> {
        let in_size = vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(swapchain_images_count)
            .build();

        let out_size = vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(swapchain_images_count)
            .build();

        let pool_sizes = &[in_size, out_size];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(swapchain_images_count);

        logical_device
            .create_descriptor_pool(&info, None)
            .map_err(|e| anyhow!("{}", e))
    }
    pub unsafe fn update_descriptor_sets(
        &mut self,
        logical_device: &ash::Device,
        swapchain_images_count: usize,
        // uniform_buffers: &[UniformBuffer],
    ) -> Result<()> {
        (0..swapchain_images_count).enumerate().for_each(|(i, _)| {
            let in_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.compute_buffer.get_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build();

            let out_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.compute_buffer.get_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build();

            let in_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[in_info])
                .build();

            let out_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[out_info])
                .build();

            logical_device
                .update_descriptor_sets(&[in_write, out_write], &[] as &[vk::CopyDescriptorSet]);
        });

        Ok(())
    }

    pub unsafe fn create_pipeline_layout(
        logical_device: &ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::PipelineLayout> {
        let vert_push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(128 /*2 *  16 × 4 byte floats */)
            .build();

        // let frag_push_constant_range = vk::PushConstantRange::builder()
        //     .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        //     .offset(64)
        //     .size(16 /* 2 x 4 byte ints */)
        //     .build();

        let set_layouts = &[descriptor_set_layout];
        let push_constant_ranges = &[vert_push_constant_range];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constant_ranges);
        logical_device
            .create_pipeline_layout(&layout_info, None)
            .map_err(|e| anyhow!("{}", e))
    }

    pub unsafe fn create_pipeline(
        &mut self,
        logical_device: &ash::Device,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Result<()> {
        let vert_code = read_spv(&mut fs::File::open(
            SHADER_FILES.get("terrain_vert").unwrap(),
        )?)?;
        let frag_code = read_spv(&mut fs::File::open(
            SHADER_FILES.get("terrain_frag").unwrap(),
        )?)?;
        let compute_code = read_spv(&mut fs::File::open(
            SHADER_FILES.get("terrain_compute").unwrap(),
        )?)?;

        let vert_shader_module = g_utils::create_shader_module(logical_device, &vert_code)?;
        let frag_shader_module = g_utils::create_shader_module(logical_device, &frag_code)?;
        let compute_shader_module = g_utils::create_shader_module(logical_device, &compute_code)?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
            .build();

        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
            .build();

        let compute_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(compute_shader_module)
            .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
            .build();

        let binding_descriptions = &[
            g_types::Vertex::binding_description(0, vk::VertexInputRate::VERTEX),
            g_types::Mat4::binding_description(1, vk::VertexInputRate::INSTANCE),
        ];
        let attribute_descriptions = &[
            g_types::Vertex::attribute_descriptions(0, 0),
            g_types::Mat4::attribute_descriptions(1, 3),
        ]
        .concat();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(attribute_descriptions)
            .build();

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false)
            .build();

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build();

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D {
                width: extent.width,
                height: extent.height,
            })
            .build();

        let viewports = &[viewport];
        let scissors = &[scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(self.render_info.polygon_info.cull_mode)
            .front_face(self.render_info.polygon_info.front_face)
            .depth_bias_enable(false);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .build();

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(msaa_samples)
            .build();

        let attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build();

        let attachments = &[attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let stages = &[vert_stage, frag_stage, compute_stage];
        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .layout(self.pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)
            .build();

        self.pipeline = logical_device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|e| anyhow!("{}", e.1))?[0];

        logical_device.destroy_shader_module(vert_shader_module, None);
        logical_device.destroy_shader_module(frag_shader_module, None);

        Ok(())
    }

    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
        self.compute_buffer.destroy(logical_device);
        self.index_buffer.destroy(logical_device);
        self.instance_buffer.destroy(logical_device);
        self.buffer_memory_allocator.destroy(logical_device);
        logical_device.destroy_descriptor_pool(self.descriptor_pool, None);
        logical_device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        logical_device.destroy_pipeline(self.pipeline, None);
        logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
    }
}

#[derive(Debug)]
pub struct Model {
    pub vertex_buffer: VertexBuffer,
    pub index_buffer: IndexBuffer,
    pub instance_buffer: InstanceBuffer,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub texture: Texture,
    instance_count: u32,
    pub render_info: g_types::RenderInfo,
}

impl Model {
    pub unsafe fn create(
        // path: &str,
        // texture_path: &str,
        model_name: &str,
        logical_device: &ash::Device,
        pos_and_rot: &[(g_types::Vec3, cgmath::Quaternion<f32>)],
    ) -> Result<Self> {
        let model_path = format!("resources/{}/{}.obj", model_name, model_name);
        let model_texture_path = format!("resources/{}/{}.png", model_name, model_name);
        let model_render_info = format!("resources/{}/render_info.toml", model_name);

        let render_info = g_types::RenderInfo::create(&model_render_info)?;

        let (vertices, indices) = g_utils::load_model(&model_path)?;

        let vertex_buffer = VertexBuffer::create(&vertices)?;

        let index_buffer = IndexBuffer::create(&indices)?;

        let instance_buffer = InstanceBuffer::create(pos_and_rot, &render_info)?;

        let texture = Texture::create(&model_texture_path)?;

        let descriptor_set_layout = Self::create_descriptor_set_layout(logical_device)?;

        let pipeline_layout = Self::create_pipeline_layout(logical_device, descriptor_set_layout)?;

        let pipeline = vk::Pipeline::null();

        let descriptor_pool = vk::DescriptorPool::null();
        let descriptor_sets = Vec::new();

        Ok(Self {
            vertex_buffer,
            index_buffer,
            instance_buffer,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            texture,
            descriptor_pool,
            descriptor_sets,
            instance_count: pos_and_rot.len() as u32,
            render_info,
        })
    }

    pub unsafe fn create_pipeline_layout(
        logical_device: &ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::PipelineLayout> {
        let vert_push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(128 /*2 *  16 × 4 byte floats */)
            .build();

        // let frag_push_constant_range = vk::PushConstantRange::builder()
        //     .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        //     .offset(64)
        //     .size(16 /* 2 x 4 byte ints */)
        //     .build();

        let set_layouts = &[descriptor_set_layout];
        let push_constant_ranges = &[vert_push_constant_range];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constant_ranges);
        logical_device
            .create_pipeline_layout(&layout_info, None)
            .map_err(|e| anyhow!("{}", e))
    }

    pub unsafe fn create_descriptor_set_layout(
        logical_device: &ash::Device,
    ) -> Result<vk::DescriptorSetLayout> {
        // let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        //     .binding(0)
        //     .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        //     .descriptor_count(1)
        //     .stage_flags(vk::ShaderStageFlags::VERTEX)
        //     .build();

        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let bindings = &[sampler_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        logical_device
            .create_descriptor_set_layout(&info, None)
            .map_err(|e| anyhow!("{}", e))
    }

    pub unsafe fn create_descriptor_pool_and_sets(
        &mut self,
        logical_device: &ash::Device,
        swapchain_images_count: usize,
    ) -> Result<()> {
        self.descriptor_pool =
            Self::create_descriptor_pool(logical_device, swapchain_images_count as u32)?;

        self.descriptor_sets = g_utils::create_descriptor_sets(
            logical_device,
            self.descriptor_set_layout,
            self.descriptor_pool,
            swapchain_images_count,
        )?;

        Ok(())
    }

    pub unsafe fn create_descriptor_pool(
        logical_device: &ash::Device,
        swapchain_images_count: u32,
    ) -> Result<vk::DescriptorPool> {
        let ubo_size = vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(swapchain_images_count)
            .build();

        let sampler_size = vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(swapchain_images_count)
            .build();

        let pool_sizes = &[ubo_size, sampler_size];
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(swapchain_images_count);

        logical_device
            .create_descriptor_pool(&info, None)
            .map_err(|e| anyhow!("{}", e))
    }
    pub unsafe fn update_descriptor_sets(
        &mut self,
        logical_device: &ash::Device,
        swapchain_images_count: usize,
        uniform_buffers: &[UniformBuffer],
    ) -> Result<()> {
        (0..swapchain_images_count).enumerate().for_each(|(i, _)| {
            let image_info = &[vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.texture.image_view)
                .sampler(self.texture.sampler)
                .build()];

            let image_info_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image_info)
                .build();

            logical_device
                .update_descriptor_sets(&[image_info_write], &[] as &[vk::CopyDescriptorSet]);
        });

        Ok(())
    }

    pub unsafe fn create_pipeline(
        &mut self,
        logical_device: &ash::Device,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Result<()> {
        let vert_code = read_spv(&mut fs::File::open(
            SHADER_FILES.get("instanced_vert").unwrap(),
        )?)?;
        let frag_code = read_spv(&mut fs::File::open(
            SHADER_FILES.get("instanced_frag").unwrap(),
        )?)?;

        let vert_shader_module = g_utils::create_shader_module(logical_device, &vert_code)?;
        let frag_shader_module = g_utils::create_shader_module(logical_device, &frag_code)?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
            .build();

        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
            .build();

        let binding_descriptions = &[
            g_types::Vertex::binding_description(0, vk::VertexInputRate::VERTEX),
            g_types::Mat4::binding_description(1, vk::VertexInputRate::INSTANCE),
        ];
        let attribute_descriptions = &[
            g_types::Vertex::attribute_descriptions(0, 0),
            g_types::Mat4::attribute_descriptions(1, 3),
        ]
        .concat();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(attribute_descriptions)
            .build();

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false)
            .build();

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build();

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D {
                width: extent.width,
                height: extent.height,
            })
            .build();

        let viewports = &[viewport];
        let scissors = &[scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(self.render_info.polygon_info.cull_mode)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .build();

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(msaa_samples)
            .build();

        let attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build();

        let attachments = &[attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let stages = &[vert_stage, frag_stage];
        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .layout(self.pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)
            .build();

        self.pipeline = logical_device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|e| anyhow!("{}", e.1))?[0];

        logical_device.destroy_shader_module(vert_shader_module, None);
        logical_device.destroy_shader_module(frag_shader_module, None);

        Ok(())
    }

    pub unsafe fn push_constants(
        &self,
        logical_device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        view: &g_types::Mat4,
        proj: &g_types::Mat4,
    ) {
        let view_bytes = std::slice::from_raw_parts(
            view as *const g_types::Mat4 as *const u8,
            size_of::<g_types::Mat4>(),
        );

        let proj_bytes = std::slice::from_raw_parts(
            proj as *const g_types::Mat4 as *const u8,
            size_of::<g_types::Mat4>(),
        );

        logical_device.cmd_push_constants(
            command_buffer,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            &[view_bytes, proj_bytes].concat(),
        );
    }

    pub unsafe fn draw_models(
        &self,
        logical_device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        view: &g_types::Mat4,
        proj: &g_types::Mat4,
    ) {
        logical_device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );

        logical_device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[
                self.vertex_buffer.get_buffer(),
                self.instance_buffer.get_buffer(),
            ],
            &[0, 0],
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

        self.push_constants(logical_device, command_buffer, view, proj);

        logical_device.cmd_draw_indexed(
            command_buffer,
            self.index_buffer.get_indice_count(),
            self.instance_count,
            0,
            0,
            0,
        );
    }

    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
        self.texture.destroy(logical_device);
        self.vertex_buffer.destroy(logical_device);
        self.index_buffer.destroy(logical_device);
        self.instance_buffer.destroy(logical_device);
        logical_device.destroy_descriptor_pool(self.descriptor_pool, None);
        logical_device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        logical_device.destroy_pipeline(self.pipeline, None);
        logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
    }
}

pub struct ModelManager {
    pub models: Arc<RefCell<Vec<Model>>>,
    pub buffer_allocator: BufferMemoryAllocator,
    pub texture_engine: TextureMemoryAllocator,
}

impl ModelManager {
    pub unsafe fn create() -> Result<Self> {
        let buffer_allocator = BufferMemoryAllocator::create()?;
        let texture_engine = TextureMemoryAllocator::create()?;

        Ok(Self {
            models: Arc::new(RefCell::new(Vec::new())),
            buffer_allocator,
            texture_engine,
        })
    }
    pub unsafe fn recreate_pipelines(
        &mut self,
        logical_device: &ash::Device,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Result<()> {
        for model in self.models.borrow_mut().iter_mut() {
            logical_device.destroy_pipeline(model.pipeline, None);
            model.create_pipeline(logical_device, msaa_samples, render_pass, extent)?;
        }

        Ok(())
    }

    pub unsafe fn create_pipelines(
        &mut self,

        logical_device: &ash::Device,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Result<()> {
        for model in self.models.borrow_mut().iter_mut() {
            model.create_pipeline(logical_device, msaa_samples, render_pass, extent)?;
        }

        Ok(())
    }

    pub unsafe fn create_descriptor_pools_and_sets(
        &mut self,
        logical_device: &ash::Device,
        swapchain_images_count: usize,
    ) -> Result<()> {
        for model in self.models.borrow_mut().iter_mut() {
            model.create_descriptor_pool_and_sets(logical_device, swapchain_images_count)?;
        }

        Ok(())
    }

    pub unsafe fn update_descriptor_sets(
        &mut self,

        logical_device: &ash::Device,
        swapchain_images_count: usize,
    ) -> Result<()> {
        for model in self.models.borrow_mut().iter_mut() {
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
        instance: &ash::Instance,
        logical_device: &ash::Device,
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
            self.models.clone(),
        )?;

        Ok(())
    }
    pub unsafe fn create_textures(
        &mut self,

        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<()> {
        self.texture_engine.create_textures(
            instance,
            logical_device,
            physical_device,
            self.models.clone(),
        )?;

        Ok(())
    }

    // pub unsafe fn load_models_from_scene(
    //     &mut self,
    //     scene: &Scene,
    //     logical_device: &ash::Device,
    // ) -> Result<()> {
    //     for model in scene.models.iter() {
    //         // let model = Model::create(model_name, logical_device, pos_and_rot)?;
    //         // self.add_model(&model);
    //     }

    //     Ok(())
    // }

    pub unsafe fn create_buffers(&mut self, logical_device: &ash::Device) -> Result<()> {
        self.buffer_allocator
            .create_buffers(logical_device, self.models.clone())?;

        Ok(())
    }

    pub unsafe fn calculate_buffer_offsets_and_size(&mut self) -> u64 {
        let mut acc = 0;

        for model in self.models.borrow_mut().iter_mut() {
            acc = g_utils::align_up(acc, model.vertex_buffer.reqs.unwrap().alignment);
            model.vertex_buffer.offset = Some(acc);
            acc += model.vertex_buffer.get_required_size();

            acc = g_utils::align_up(acc, model.index_buffer.reqs.unwrap().alignment);
            model.index_buffer.offset = Some(acc);
            acc += model.index_buffer.get_required_size();

            acc = g_utils::align_up(acc, model.instance_buffer.reqs.unwrap().alignment);
            model.instance_buffer.offset = Some(acc);
            acc += model.instance_buffer.get_required_size();
        }

        acc
    }

    pub unsafe fn allocate_memory_for_buffers(
        &mut self,
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        queue_set: &g_utils::QueueSet,
        command_pool_set: g_types::CommandPoolSet,
    ) -> Result<()> {
        let size = self.calculate_buffer_offsets_and_size();

        println!("size: {}", size);

        self.buffer_allocator
            .create_memories(instance, logical_device, physical_device, size)?;

        self.buffer_allocator.allocate_memory(
            logical_device,
            queue_set,
            command_pool_set,
            self.models.clone(),
            size,
        )?;

        Ok(())
    }

    pub fn add_models(&mut self, models: Arc<RefCell<Vec<Model>>>) {
        self.models = models;
    }

    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
        // self.models
        //     .values_mut()
        //     .for_each(|model| model.destroy(logical_device));

        self.texture_engine.destroy(logical_device);

        self.buffer_allocator.destroy(logical_device);
    }
}

pub struct Scene {
    models: Arc<RefCell<Vec<Model>>>, //, Vec<(g_types::Vec3, cgmath::Quaternion<f32>)>>,
    pub terrain: Terrain,
    pub model_manager: ModelManager,
}

impl Scene {
    pub unsafe fn create(terrain: Terrain, logical_device: &ash::Device) -> Result<Self> {
        let model_manager = ModelManager::create()?;

        // model_manager.load_models_from_scene(&logical_device)?;

        let mut rng = rand::thread_rng();
        Ok(Self {
            terrain,
            model_manager,
            models: Arc::new(RefCell::new(vec![
                // Model::create(
                //     "landscape",
                //     logical_device,
                //     &[(g_types::vec3(0.0, 0.0, 0.0), Quaternion::zero())],
                // )?,
                Model::create(
                    "viking_room",
                    logical_device,
                    &[(g_types::vec3(0.0, 0.0, 4.0), Quaternion::zero())],
                )?,
                Model::create(
                    "grass_blade",
                    logical_device,
                    &(0..10000)
                        .map(|_| {
                            (
                                g_types::vec3(
                                    rng.gen_range(-100.0..=100.0),
                                    rng.gen_range(-100.0..=100.0),
                                    rng.gen_range(5.0..=150.0),
                                ),
                                Quaternion::from(Euler {
                                    x: g_types::Deg(rng.gen_range(0.0..=360.0)),
                                    y: g_types::Deg(rng.gen_range(0.0..=360.0)),
                                    z: g_types::Deg(rng.gen_range(0.0..=360.0)),
                                }),
                            )
                        })
                        .collect::<Vec<_>>(),
                )?,
            ])),
        })
    }

    pub fn load_models(&mut self) {
        self.model_manager.add_models(self.models.clone());
    }
    pub unsafe fn draw_terrain(
        &self,
        logical_device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        view: &g_types::Mat4,
        proj: &g_types::Mat4,
    ) {
        self.terrain
            .draw(logical_device, command_buffer, image_index, view, proj)
    }

    pub unsafe fn draw_instanced_models(
        &self,
        image_index: usize,
        logical_device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        view: &g_types::Mat4,
        proj: &g_types::Mat4,
    ) {
        for (_model_index, model) in self.models.borrow().iter().enumerate() {
            // let model = model_manager.models.get(model_name).unwrap();

            model.draw_models(logical_device, command_buffer, image_index, view, proj);
        }
    }

    pub unsafe fn destroy(&mut self, logical_device: &ash::Device) {
        self.terrain.destroy(logical_device);
        self.models
            .borrow_mut()
            .iter_mut()
            .for_each(|model| model.destroy(logical_device));
        self.model_manager.destroy(logical_device);
    }
}
