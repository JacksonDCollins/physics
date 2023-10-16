use std::fs::File;

use anyhow::anyhow;
use anyhow::Result;
use std::ptr::copy_nonoverlapping as memcpy;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_0::*;
use winit::window::{Window, WindowBuilder};

#[derive(Clone, Debug, Default)]
pub struct Texture {
    pub image_path: String,
    pub mip_levels: u32,
    pub texture_image: vk::Image,
    pub texture_image_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,
}

impl Texture {
    pub unsafe fn new(
        path: &str,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        let image = File::open(path)?;

        let decoder = png::Decoder::new(image);
        let mut reader = decoder.read_info()?;

        let mut pixels = vec![0; reader.info().raw_bytes()];
        reader.next_frame(&mut pixels)?;

        let size = reader.info().raw_bytes() as u64;
        let (width, height) = reader.info().size();

        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        let (staging_buffer, staging_buffer_memory) = crate::graphics::utils::create_buffer(
            instance,
            device,
            physical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

        memcpy(pixels.as_ptr(), memory.cast(), pixels.len());

        device.unmap_memory(staging_buffer_memory);

        let (texture_image, texture_image_memory) = crate::graphics::utils::create_image(
            instance,
            device,
            physical_device,
            width,
            height,
            mip_levels,
            vk::SampleCountFlags::_1,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        crate::graphics::utils::transition_image_layout(
            device,
            texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
            command_pool,
            queue,
        )?;

        crate::graphics::utils::copy_buffer_to_image(
            device,
            command_pool,
            queue,
            staging_buffer,
            texture_image,
            width,
            height,
        )?;

        crate::graphics::utils::generate_mipmaps(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            texture_image,
            vk::Format::R8G8B8A8_SRGB,
            width,
            height,
            mip_levels,
        )?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        let texture_image_view = crate::graphics::utils::create_image_view(
            device,
            texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?;

        let info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0) //Optional.
            .max_lod(mip_levels as f32)
            .mip_lod_bias(0.0); // Optional.

        let texture_sampler = device.create_sampler(&info, None)?;

        Ok(Self {
            image_path: path.to_owned(),
            mip_levels,
            texture_image,
            texture_image_memory,
            texture_image_view,
            texture_sampler,
        })
    }
}
