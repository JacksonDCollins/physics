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
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    format: vk::Format,
    image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    pub unsafe fn create(
        window: &Window,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        queue_family_indices: g_utils::QueueFamilyIndices,
        swapchain_support: g_utils::SwapchainSupport,
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
    pipeline: vk::Pipeline,
}

impl Pipeline {
    pub unsafe fn create(logical_device: &Device, swapchain: &Swapchain) -> Result<Self> {
        let pipeline = g_utils::create_pipeline(logical_device, swapchain)?;
        Ok(Self { pipeline })
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device) {
        logical_device.destroy_pipeline(self.pipeline, None);
    }
}
