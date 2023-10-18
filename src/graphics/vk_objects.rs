use std::fs::File;
use std::mem::size_of;
use std::ops::Deref;
use std::ops::DerefMut;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use anyhow::anyhow;
use anyhow::Result;
use log::info;
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

#[derive(Clone, Debug, Copy)]
pub struct SwapchainKHR(vk::SwapchainKHR);

impl Deref for SwapchainKHR {
    type Target = vk::SwapchainKHR;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SwapchainKHR {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Create<SwapchainKHR> for Device {
    unsafe fn create_swapchain_khr_vko(&self, create_info: &vk::SwapchainCreateInfoKHR, allocator: Option<&vk::AllocationCallbacks>) -> Result<SwapchainKHR> {
        let swapchain = self.create_swapchain_khr(create_info, allocator)?;
        Ok(SwapchainKHR(swapchain))
    }
}

impl Destroy<SwapchainKHR> for Device {
    unsafe fn destroy_swapchain_khr(&self, swapchain: SwapchainKHR, allocator: Option<&vk::AllocationCallbacks>) {
        KhrSwapchainExtension::destroy_swapchain_khr(self, *swapchain, allocator);
    }
}

pub trait Create<SwapchainKHR> {
    unsafe fn create_swapchain_khr_vko(
        &self,
        create_info: &vk::SwapchainCreateInfoKHR,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> Result<SwapchainKHR>;
}

pub trait Destroy<SwapchainKHR> {
    unsafe fn destroy_swapchain_khr(&self, swapchain: SwapchainKHR, allocator: Option<&vk::AllocationCallbacks>);
}