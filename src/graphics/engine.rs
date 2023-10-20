use crate::graphics::objects as g_objects;
use crate::graphics::utils as g_utils;

use anyhow::Result;
use vk::SurfaceKHR;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::KhrSurfaceExtension;
use winit::window::Window;

pub struct RenderEngine {
    surface: SurfaceKHR,
    queue_set: g_utils::QueueSet,
    swapchain: g_objects::Swapchain,
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
            queue_family_indices,
            swapchain_support,
        )?;

        let pipeline = g_objects::Pipeline::create(logical_device, &swapchain)?;

        Ok(Self {
            surface,
            queue_set,
            swapchain,
        })
    }

    pub fn render(&mut self, window: &Window) -> Result<()> {
        // render
        Ok(())
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device, instance: &Instance) {
        self.swapchain.destroy(logical_device);
        instance.destroy_surface_khr(self.surface, None);
    }
}
