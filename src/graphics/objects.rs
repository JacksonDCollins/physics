use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;

use crate::graphics::objects as g_objects;
use crate::graphics::utils as g_utils;
use anyhow::{anyhow, Error, Result};
use thiserror::Error;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension};
use vulkanalia::{Entry, Version};
use winit::window::Window;

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

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
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

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
        logical_device.destroy_pipeline(self.pipeline, None);
        logical_device.destroy_render_pass(self.render_pass, None);
        logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
    }
}

pub struct Presenter {
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
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
    ) -> Result<Self> {
        let framebuffers =
            g_utils::create_framebuffers(logical_device, pipeline.render_pass, swapchain)?;

        let command_pool = g_utils::create_command_pool(logical_device, queue_family_indices)?;

        let command_buffers = g_utils::create_command_buffers(
            logical_device,
            command_pool,
            &framebuffers,
            swapchain,
            pipeline,
        )?;

        let (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        ) = g_utils::create_sync_objects(logical_device, swapchain.images.len())?;

        Ok(Self {
            framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        })
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
        self.in_flight_fences
            .iter()
            .for_each(|fence| logical_device.destroy_fence(*fence, None));
        self.image_available_semaphores
            .iter()
            .for_each(|semaphore| logical_device.destroy_semaphore(*semaphore, None));
        self.render_finished_semaphores
            .iter()
            .for_each(|semaphore| logical_device.destroy_semaphore(*semaphore, None));
        logical_device.destroy_command_pool(self.command_pool, None);
        self.framebuffers
            .iter()
            .for_each(|framebuffer| logical_device.destroy_framebuffer(*framebuffer, None));
    }
}
