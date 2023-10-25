use crate::graphics::objects as g_objects;
use crate::graphics::types as g_types;
use crate::graphics::utils as g_utils;

use anyhow::{anyhow, Result};
use std::time::Instant;
use vk::SurfaceKHR;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;

use winit::window::Window;

pub struct RenderEngine {
    surface: SurfaceKHR,
    queue_set: g_utils::QueueSet,
    swapchain: g_objects::Swapchain,
    pipeline: g_objects::Pipeline,
    presenter: g_objects::Presenter,

    buffer_memory_allocator: g_objects::BufferMemoryAllocator,
    texture_engine: g_objects::TextureMemoryAllocator,
    queue_family_indices: g_utils::QueueFamilyIndices,
    swapchain_support: g_utils::SwapchainSupport,
    msaa_samples: vk::SampleCountFlags,

    start: Instant,
}

impl RenderEngine {
    pub unsafe fn create(
        window: &Window,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        surface: SurfaceKHR,
        queue_set: g_utils::QueueSet,
        queue_family_indices: g_utils::QueueFamilyIndices,
        swapchain_support: g_utils::SwapchainSupport,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<Self> {
        let swapchain = g_objects::Swapchain::create(
            window,
            logical_device,
            surface,
            &queue_family_indices,
            &swapchain_support,
        )?;

        let pipeline = g_objects::Pipeline::create(
            instance,
            logical_device,
            physical_device,
            &swapchain,
            msaa_samples,
        )?;

        let mut buffer_memory_allocator = g_objects::BufferMemoryAllocator::create()?;

        let model = g_objects::Model::create()?;

        buffer_memory_allocator.add_vertex_buffer(model.vertex_buffer);

        buffer_memory_allocator.set_index_buffer(model.index_buffer);

        for _ in 0..swapchain.images.len() {
            let uniform_buffer = g_objects::UniformBuffer::create(
                logical_device,
                g_types::UniformBufferObject::default(),
            )?;
            buffer_memory_allocator.add_uniform_buffer(uniform_buffer);
        }

        let mut texture_engine = g_objects::TextureMemoryAllocator::create()?;

        let texture = g_objects::Texture::create();

        texture_engine.add_texture(texture);

        let presenter = g_objects::Presenter::create(
            logical_device,
            &swapchain,
            &pipeline,
            &queue_family_indices,
            &mut buffer_memory_allocator,
            &mut texture_engine,
            instance,
            physical_device,
            &queue_set,
            msaa_samples,
        )?;

        Ok(Self {
            surface,
            queue_set,
            swapchain,
            pipeline,
            presenter,

            buffer_memory_allocator,
            texture_engine,
            queue_family_indices,
            swapchain_support,
            msaa_samples,

            start: Instant::now(),
        })
    }

    pub unsafe fn recreate_sawpchain(
        &mut self,
        window: &Window,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<()> {
        logical_device.device_wait_idle()?;

        self.swapchain.destroy(logical_device);
        self.pipeline.destroy(logical_device);
        self.presenter.destroy(logical_device);

        self.swapchain_support =
            g_utils::query_swapchain_support(instance, physical_device, self.surface)?;

        self.swapchain = g_objects::Swapchain::create(
            window,
            logical_device,
            self.surface,
            &self.queue_family_indices,
            &self.swapchain_support,
        )?;

        self.pipeline = g_objects::Pipeline::create(
            instance,
            logical_device,
            physical_device,
            &self.swapchain,
            self.msaa_samples,
        )?;

        self.presenter = g_objects::Presenter::create(
            logical_device,
            &self.swapchain,
            &self.pipeline,
            &self.queue_family_indices,
            &mut self.buffer_memory_allocator,
            &mut self.texture_engine,
            instance,
            physical_device,
            &self.queue_set,
            self.msaa_samples,
        )?;

        Ok(())
    }

    pub unsafe fn render(
        &mut self,
        window: &Window,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        instance: &Instance,
        frame: usize,
        resized: &mut bool,
    ) -> Result<()> {
        logical_device.wait_for_fences(
            &[self.presenter.in_flight_fences[frame]],
            true,
            u64::MAX,
        )?;

        let result = logical_device.acquire_next_image_khr(
            self.swapchain.swapchain,
            u64::MAX,
            self.presenter.image_available_semaphores[frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok(res) => match res {
                (_, vk::SuccessCode::SUBOPTIMAL_KHR) => {
                    return self.recreate_sawpchain(
                        window,
                        instance,
                        logical_device,
                        physical_device,
                    )
                }
                (index, _) => index as usize,
            },
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                return self.recreate_sawpchain(window, instance, logical_device, physical_device)
            }
            Err(error) => return Err(anyhow!(error)),
        };

        if !self.presenter.images_in_flight[image_index].is_null() {
            logical_device.wait_for_fences(
                &[self.presenter.images_in_flight[image_index]],
                true,
                u64::MAX,
            )?;
        }

        self.presenter.images_in_flight[image_index] = self.presenter.in_flight_fences[frame];

        self.update_uniform_buffer(image_index)?;

        let wait_semaphores = &[self.presenter.image_available_semaphores[frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.presenter.command_buffers[image_index]];
        let signal_semaphores = &[self.presenter.render_finished_semaphores[frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        logical_device.reset_fences(&[self.presenter.in_flight_fences[frame]])?;

        logical_device.queue_submit(
            self.queue_set.graphics,
            &[submit_info],
            self.presenter.in_flight_fences[frame],
        )?;

        let swapchains = &[self.swapchain.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = logical_device.queue_present_khr(self.queue_set.present, &present_info);

        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if *resized || changed {
            *resized = false;
            self.recreate_sawpchain(window, instance, logical_device, physical_device)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        Ok(())
    }

    pub unsafe fn update_uniform_buffer(&mut self, image_index: usize) -> Result<()> {
        let time = self.start.elapsed().as_secs_f32();

        let model =
            g_types::Mat4::from_axis_angle(g_types::vec3(0.0, 0.0, 1.0), g_types::Deg(90.0) * time);

        let view = g_types::Mat4::look_at_rh(
            g_types::point3(2.0, 2.0, 2.0),
            g_types::point3(0.0, 0.0, 0.0),
            g_types::vec3(0.0, 0.0, 1.0),
        );

        #[rustfmt::skip]
        let correction = g_types::Mat4::new(
            1.0,  0.0,       0.0, 0.0,
            // We're also flipping the Y-axis with this line's `-1.0`.
            0.0, -1.0,       0.0, 0.0,
            0.0,  0.0, 1.0 / 2.0, 0.0,
            0.0,  0.0, 1.0 / 2.0, 1.0,
        );

        let proj = correction
            * cgmath::perspective(
                g_types::Deg(45.0),
                self.swapchain.extent.width as f32 / self.swapchain.extent.height as f32,
                0.1,
                10.0,
            );

        let ubo = g_types::UniformBufferObject { model, view, proj };

        self.buffer_memory_allocator
            .update_uniform_buffer(ubo, image_index)?;

        Ok(())
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device, instance: &Instance) {
        self.buffer_memory_allocator.destroy(logical_device);
        self.texture_engine.destroy(logical_device);
        self.presenter.destroy(logical_device);
        self.pipeline.destroy(logical_device);
        self.swapchain.destroy(logical_device);
        instance.destroy_surface_khr(self.surface, None);
    }
}
