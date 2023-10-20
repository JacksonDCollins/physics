use crate::graphics::objects as g_objects;
use crate::graphics::utils as g_utils;

use anyhow::Result;
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
}

impl RenderEngine {
    pub unsafe fn create(
        window: &Window,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        surface: SurfaceKHR,
        queue_set: g_utils::QueueSet,
        queue_family_indices: &g_utils::QueueFamilyIndices,
        swapchain_support: &g_utils::SwapchainSupport,
    ) -> Result<Self> {
        let swapchain = g_objects::Swapchain::create(
            window,
            instance,
            logical_device,
            physical_device,
            surface,
            queue_family_indices,
            swapchain_support,
        )?;

        let pipeline = g_objects::Pipeline::create(logical_device, &swapchain)?;

        let presenter = g_objects::Presenter::create(
            logical_device,
            &swapchain,
            &pipeline,
            queue_family_indices,
        )?;

        Ok(Self {
            surface,
            queue_set,
            swapchain,
            pipeline,
            presenter,
        })
    }

    pub unsafe fn render(&mut self, logical_device: &Device, frame: usize) -> Result<()> {
        logical_device.wait_for_fences(
            &[self.presenter.in_flight_fences[frame]],
            true,
            u64::MAX,
        )?;

        let image_index = logical_device
            .acquire_next_image_khr(
                self.swapchain.swapchain,
                u64::MAX,
                self.presenter.image_available_semaphores[frame],
                vk::Fence::null(),
            )?
            .0 as usize;

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

        logical_device.queue_present_khr(self.queue_set.present, &present_info)?;

        Ok(())
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device, instance: &Instance) {
        self.presenter.destroy(logical_device);
        self.pipeline.destroy(logical_device);
        self.swapchain.destroy(logical_device);
        instance.destroy_surface_khr(self.surface, None);
    }
}
