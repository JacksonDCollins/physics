pub mod model;
pub mod texture;
pub mod utils;
// pub mod vk_objects;

use std::fs::File;
use std::mem::size_of;
use std::time::Instant;

use anyhow::anyhow;
use anyhow::Result;
use std::ptr::copy_nonoverlapping as memcpy;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;

use winit::window::{Window, WindowBuilder};

use cgmath::{point3, Deg};
use cgmath::{vec2, vec3};

pub type Mat4 = cgmath::Matrix4<f32>;

pub type Vec2 = cgmath::Vector2<f32>;
pub type Vec3 = cgmath::Vector3<f32>;

use crate::SuitabilityError;
use crate::MAX_FRAMES_IN_FLIGHT;

use std::hash::Hash;
use std::hash::Hasher;

#[derive(Clone, Debug)]
pub struct GraphicsData {
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub msaa_samples: vk::SampleCountFlags,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub command_pool: vk::CommandPool,
    pub command_pools: Vec<vk::CommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffers_memory: Vec<vk::DeviceMemory>,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    // pub texture_image: texture::Texture,
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    pub color_image: vk::Image,
    pub color_image_memory: vk::DeviceMemory,
    pub color_image_view: vk::ImageView,
    // pub model: model::Model,
    pub wall: model::Wall,
}

impl GraphicsData {
    pub unsafe fn new(
        instance: &Instance,
        window: &Window,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        logical_device: &Device,
        graphics_queue: vk::Queue,
        present_queue: vk::Queue,
    ) -> Result<Self> {
        let logical_device_ref = &logical_device;
        let msaa_samples = crate::graphics::utils::get_max_msaa_samples(instance, physical_device);
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) =
            crate::graphics::utils::create_swapchain(
                window,
                instance,
                logical_device,
                surface,
                physical_device,
            )?;
        let swapchain_image_views = crate::graphics::utils::create_swapchain_image_views(
            logical_device_ref,
            &swapchain_images,
            swapchain_format,
        )?;
        let render_pass = crate::graphics::utils::create_render_pass(
            instance,
            logical_device_ref,
            physical_device,
            swapchain_format,
            msaa_samples,
        )?;
        let descriptor_set_layout =
            crate::graphics::utils::create_descriptor_set_layout(logical_device_ref)?;
        let (pipeline_layout, pipeline) = crate::graphics::utils::create_pipeline(
            logical_device_ref,
            swapchain_extent,
            msaa_samples,
            descriptor_set_layout,
            render_pass,
        )?;
        let (command_pool, command_pools) = crate::graphics::utils::create_command_pools(
            instance,
            logical_device_ref,
            &swapchain_images,
            surface,
            physical_device,
        )?;
        let (color_image, color_image_memory, color_image_view) =
            crate::graphics::utils::create_color_objects(
                instance,
                logical_device_ref,
                physical_device,
                swapchain_extent,
                msaa_samples,
                swapchain_format,
            )?;
        let (depth_image, depth_image_memory, depth_image_view) =
            crate::graphics::utils::create_depth_objects(
                instance,
                logical_device_ref,
                physical_device,
                swapchain_extent,
                msaa_samples,
                command_pool,
                graphics_queue,
            )?;

        let framebuffers = crate::graphics::utils::create_framebuffers(
            logical_device_ref,
            &swapchain_image_views,
            color_image_view,
            depth_image_view,
            render_pass,
            swapchain_extent,
        )?;
        let texture_image = texture::Texture::new(
            "resources/viking_room.png",
            instance,
            logical_device_ref,
            physical_device,
            command_pool,
            graphics_queue,
        )?; //create_texture_image(&instance, logical_device_ref, &mut data)?;

        // let model = model::Model::new(
        //     "resources/viking_room.obj",
        //     instance,
        //     logical_device_ref,
        //     physical_device,
        //     command_pool,
        //     graphics_queue,
        // )?; //load_model(&mut data)?;

        let wall = model::Wall::new(
            2.0,
            texture_image,
            instance,
            logical_device,
            physical_device,
            command_pool,
            graphics_queue,
        )?;

        let (uniform_buffers, uniform_buffers_memory) =
            crate::graphics::utils::create_uniform_buffers(
                instance,
                logical_device_ref,
                physical_device,
                &swapchain_images,
            )?;

        let descriptor_pool =
            crate::graphics::utils::create_descriptor_pool(logical_device_ref, &swapchain_images)?;

        let descriptor_sets = crate::graphics::utils::create_descriptor_sets(
            logical_device_ref,
            descriptor_set_layout,
            &swapchain_images,
            descriptor_pool,
            &uniform_buffers,
            &wall.texture,
        )?;

        let (command_buffers, secondary_command_buffers) =
            crate::graphics::utils::create_command_buffers(
                logical_device_ref,
                &swapchain_images,
                &command_pools,
            )?;

        let (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        ) = crate::graphics::utils::create_sync_objects(logical_device_ref, &swapchain_images)?;

        Ok(Self {
            surface,
            physical_device,
            swapchain,
            swapchain_images,
            swapchain_format,
            swapchain_extent,
            swapchain_image_views,
            render_pass,
            msaa_samples,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            framebuffers,
            command_pool,
            command_pools,
            command_buffers,
            secondary_command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
            graphics_queue,
            present_queue,
            uniform_buffers,
            descriptor_pool,
            descriptor_sets,
            // texture_image,
            depth_image,
            depth_image_memory,
            depth_image_view,
            color_image,
            color_image_memory,
            color_image_view,
            uniform_buffers_memory,
            wall,
        })
    }

    pub unsafe fn recreate_swapchain(
        &mut self,
        window: &Window,
        logical_device: &Device,
        instance: &Instance,
    ) -> Result<()> {
        logical_device.device_wait_idle()?;
        self.destroy_swapchain(logical_device);
        (
            self.swapchain,
            self.swapchain_images,
            self.swapchain_format,
            self.swapchain_extent,
        ) = crate::graphics::utils::create_swapchain(
            window,
            instance,
            logical_device,
            self.surface,
            self.physical_device,
        )?;
        self.swapchain_image_views = crate::graphics::utils::create_swapchain_image_views(
            logical_device,
            &self.swapchain_images,
            self.swapchain_format,
        )?;
        self.render_pass = crate::graphics::utils::create_render_pass(
            instance,
            logical_device,
            self.physical_device,
            self.swapchain_format,
            self.msaa_samples,
        )?;
        (self.pipeline_layout, self.pipeline) = crate::graphics::utils::create_pipeline(
            logical_device,
            self.swapchain_extent,
            self.msaa_samples,
            self.descriptor_set_layout,
            self.render_pass,
        )?;
        (
            self.color_image,
            self.color_image_memory,
            self.color_image_view,
        ) = crate::graphics::utils::create_color_objects(
            instance,
            logical_device,
            self.physical_device,
            self.swapchain_extent,
            self.msaa_samples,
            self.swapchain_format,
        )?;
        (
            self.depth_image,
            self.depth_image_memory,
            self.depth_image_view,
        ) = crate::graphics::utils::create_depth_objects(
            instance,
            logical_device,
            self.physical_device,
            self.swapchain_extent,
            self.msaa_samples,
            self.command_pool,
            self.graphics_queue,
        )?;
        self.framebuffers = crate::graphics::utils::create_framebuffers(
            logical_device,
            &self.swapchain_image_views,
            self.color_image_view,
            self.depth_image_view,
            self.render_pass,
            self.swapchain_extent,
        )?;
        (self.uniform_buffers, self.uniform_buffers_memory) =
            crate::graphics::utils::create_uniform_buffers(
                instance,
                logical_device,
                self.physical_device,
                &self.swapchain_images,
            )?;
        self.descriptor_pool =
            crate::graphics::utils::create_descriptor_pool(logical_device, &self.swapchain_images)?;

        self.descriptor_sets = crate::graphics::utils::create_descriptor_sets(
            logical_device,
            self.descriptor_set_layout,
            &self.swapchain_images,
            self.descriptor_pool,
            &self.uniform_buffers,
            &self.wall.texture,
        )?;
        (self.command_buffers, self.secondary_command_buffers) =
            crate::graphics::utils::create_command_buffers(
                logical_device,
                &self.swapchain_images,
                &self.command_pools,
            )?;
        self.images_in_flight
            .resize(self.swapchain_images.len(), vk::Fence::null());

        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self, logical_device: &Device) {
        logical_device.destroy_image_view(self.color_image_view, None);
        logical_device.free_memory(self.color_image_memory, None);
        logical_device.destroy_image(self.color_image, None);
        logical_device.destroy_image_view(self.depth_image_view, None);
        logical_device.free_memory(self.depth_image_memory, None);
        logical_device.destroy_image(self.depth_image, None);
        logical_device.destroy_descriptor_pool(self.descriptor_pool, None);
        self.uniform_buffers
            .iter()
            .for_each(|b| logical_device.destroy_buffer(*b, None));
        self.uniform_buffers_memory
            .iter()
            .for_each(|m| logical_device.free_memory(*m, None));
        self.framebuffers
            .iter()
            .for_each(|f| logical_device.destroy_framebuffer(*f, None));
        logical_device.destroy_pipeline(self.pipeline, None);
        logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
        logical_device.destroy_render_pass(self.render_pass, None);
        self.swapchain_image_views
            .iter()
            .for_each(|v| logical_device.destroy_image_view(*v, None));
        logical_device.destroy_swapchain_khr(self.swapchain, None);
    }

    pub unsafe fn update_uniform_buffer(
        &self,
        image_index: usize,
        start: Instant,
        device: &Device,
    ) -> Result<()> {
        let time = start.elapsed().as_secs_f32();

        let view = Mat4::look_at_rh(
            point3(6.0, 0.0, 2.0),
            point3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 1.0),
        );

        #[rustfmt::skip]
        let correction = Mat4::new(
            1.0,  0.0,       0.0, 0.0,
            // We're also flipping the Y-axis with this line's `-1.0`.
            0.0, -1.0,       0.0, 0.0,
            0.0,  0.0, 1.0 / 2.0, 0.0,
            0.0,  0.0, 1.0 / 2.0, 1.0,
        );

        let proj = correction
            * cgmath::perspective(
                Deg(45.0),
                self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
                0.1,
                10.0,
            );

        let ubo = UniformBufferObject { view, proj };

        let memory = device.map_memory(
            self.uniform_buffers_memory[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);

        device.unmap_memory(self.uniform_buffers_memory[image_index]);

        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        self.destroy_swapchain(device);
        // self.texture_image.destroy(device);
        self.wall.destroy(device);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pos: Vec3,
    color: Vec3,
    tex_coord: Vec2,
}

impl Vertex {
    const fn new(pos: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
        Self {
            pos,
            color,
            tex_coord,
        }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec3>() as u32)
            .build();
        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32)
            .build();
        [pos, color, tex_coord]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.color && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub present: u32,
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                surface,
            )? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
    }
}

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, surface)?,
            formats: instance.get_physical_device_surface_formats_khr(physical_device, surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, surface)?,
        })
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    view: Mat4,
    proj: Mat4,
}
