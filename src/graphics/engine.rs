use crate::graphics::objects as g_objects;
use crate::graphics::types as g_types;
use crate::graphics::utils as g_utils;

use anyhow::{anyhow, Result};
use vk::SurfaceKHR;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;
use vulkanalia::window;
use winit::window::Window;

pub struct RenderEngine {
    surface: SurfaceKHR,
    queue_set: g_utils::QueueSet,
    swapchain: g_objects::Swapchain,
    pipeline: g_objects::Pipeline,
    presenter: g_objects::Presenter,

    buffer_memory_allocator: g_objects::BufferMemoryAllocator,
    queue_family_indices: g_utils::QueueFamilyIndices,
    swapchain_support: g_utils::SwapchainSupport,
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
    ) -> Result<Self> {
        let swapchain = g_objects::Swapchain::create(
            window,
            instance,
            logical_device,
            physical_device,
            surface,
            &queue_family_indices,
            &swapchain_support,
        )?;

        let pipeline = g_objects::Pipeline::create(logical_device, &swapchain)?;

        let mut buffer_memory_allocator = g_objects::BufferMemoryAllocator::create()?;

        for i in 0..1 {
            let vertex_buffer = g_objects::VertexBuffer::create(
                // instance,
                // physical_device,
                logical_device,
                &g_types::VERTICES,
            )?;
            buffer_memory_allocator.add_vertex_buffer(vertex_buffer);
        }

        let index_buffer = g_objects::IndexBuffer::create(
            // instance,
            // physical_device,
            logical_device,
            g_types::INDICES,
        )?;

        buffer_memory_allocator.set_index_buffer(index_buffer);

        let presenter = g_objects::Presenter::create(
            logical_device,
            &swapchain,
            &pipeline,
            &queue_family_indices,
            &mut buffer_memory_allocator,
            instance,
            physical_device,
            &queue_set,
        )?;

        Ok(Self {
            surface,
            queue_set,
            swapchain,
            pipeline,
            presenter,

            buffer_memory_allocator,
            queue_family_indices,
            swapchain_support,
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
        self.buffer_memory_allocator.destroy(logical_device);
        self.presenter.destroy(logical_device);

        self.swapchain_support =
            g_utils::query_swapchain_support(instance, physical_device, self.surface)?;

        self.swapchain = g_objects::Swapchain::create(
            window,
            instance,
            logical_device,
            physical_device,
            self.surface,
            &self.queue_family_indices,
            &self.swapchain_support,
        )?;

        self.pipeline = g_objects::Pipeline::create(logical_device, &self.swapchain)?;

        self.buffer_memory_allocator = g_objects::BufferMemoryAllocator::create()?;
        for i in 0..1 {
            let vertex_buffer = g_objects::VertexBuffer::create(
                // instance,
                // physical_device,
                logical_device,
                &g_types::VERTICES,
            )?;
            self.buffer_memory_allocator
                .add_vertex_buffer(vertex_buffer);
        }

        let index_buffer = g_objects::IndexBuffer::create(
            // instance,
            // physical_device,
            logical_device,
            g_types::INDICES,
        )?;

        self.buffer_memory_allocator.set_index_buffer(index_buffer);

        self.presenter = g_objects::Presenter::create(
            logical_device,
            &self.swapchain,
            &self.pipeline,
            &self.queue_family_indices,
            &mut self.buffer_memory_allocator,
            instance,
            physical_device,
            &self.queue_set,
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
            Ok((image_index, _)) => image_index as usize,
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

    pub unsafe fn destroy(&mut self, logical_device: &Device, instance: &Instance) {
        self.buffer_memory_allocator.destroy(logical_device);
        self.presenter.destroy(logical_device);
        self.pipeline.destroy(logical_device);
        self.swapchain.destroy(logical_device);
        instance.destroy_surface_khr(self.surface, None);
    }
}
