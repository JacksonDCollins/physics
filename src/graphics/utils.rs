use crate::graphics::objects as g_objects;
use anyhow::{anyhow, Error, Result};
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;
use thiserror::Error;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension};
use vulkanalia::{Entry, Version};
use winit::window::Window;

pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
pub const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
pub const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
pub const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
) -> Result<(Instance, Option<vk::DebugUtilsMessengerEXT>)> {
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Physics")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|layer| layer.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer not available"));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    let mut extensions = vulkanalia::window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // Required by Vulkan SDK on macOS since 1.3.216.
    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        log::info!("Enabling extensions for macOS portability");
        extensions.push(
            vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                .name
                .as_ptr(),
        );
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    let mut features = vk::ValidationFeaturesEXT::builder()
        .enabled_validation_features(&[vk::ValidationFeatureEnableEXT::BEST_PRACTICES])
        .build();

    let mut instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        instance_info = instance_info.push_next(&mut debug_info);
        instance_info = instance_info.push_next(&mut features);
    }

    let instance = entry.create_instance(&instance_info, None)?;

    let messenger = if VALIDATION_ENABLED {
        Some(instance.create_debug_utils_messenger_ext(&debug_info, None)?)
    } else {
        None
    };

    Ok((instance, messenger))
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        log::error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        log::warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        log::debug!("({:?}) {}", type_, message);
    } else {
        log::trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

pub unsafe fn pick_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, QueueFamilyIndices, SwapchainSupport)> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        match check_physical_device(instance, surface, physical_device) {
            Err(error) => log::warn!(
                "Skipping physical device (`{}`): {}",
                properties.device_name,
                error
            ),
            Ok((indices, swapchain_support)) => {
                log::info!("Selected physical device (`{}`).", properties.device_name);
                return Ok((physical_device, indices, swapchain_support));
            }
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

unsafe fn check_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<(QueueFamilyIndices, SwapchainSupport)> {
    let properties = instance.get_physical_device_properties(physical_device);
    if properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        return Err(anyhow!(SuitabilityError(
            "Only discrete GPUs are supported."
        )));
    }

    let features = instance.get_physical_device_features(physical_device);
    if features.geometry_shader != vk::TRUE {
        return Err(anyhow!(SuitabilityError(
            "Missing geometry shader support."
        )));
    }
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError("No sampler anisotropy.")));
    }

    let indices = QueueFamilyIndices::get(instance, surface, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, surface, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    Ok((indices, support))
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError(
            "Missing required device extensions."
        )))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub present: u32,
    pub transfer: u32,
    pub compute: u32,
}

impl QueueFamilyIndices {
    pub fn to_render_vec(&self) -> Vec<u32> {
        let set = HashSet::from([self.graphics, self.present, self.transfer]);
        set.into_iter().collect::<Vec<_>>()
    }
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

        let transfer = properties
            .iter()
            .position(|p| {
                p.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .map(|i| i as u32);

        let compute = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
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

        if let (Some(graphics), Some(present), Some(transfer), Some(compute)) =
            (graphics, present, transfer, compute)
        {
            Ok(Self {
                graphics,
                present,
                transfer,
                compute,
            })
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

pub struct QueueSet {
    pub graphics: vk::Queue,
    pub present: vk::Queue,
    pub transfer: vk::Queue,
    pub compute: vk::Queue,
}

pub unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    indices: QueueFamilyIndices,
) -> Result<(Device, QueueSet)> {
    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);
    unique_indices.insert(indices.transfer);
    unique_indices.insert(indices.compute);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    let features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(physical_device, &info, None)?;

    let graphics_queue = device.get_device_queue(indices.graphics, 0);
    let present_queue = device.get_device_queue(indices.present, 0);
    let transfer_queue = device.get_device_queue(indices.transfer, 0);
    let compute_queue = device.get_device_queue(indices.compute, 0);
    Ok((
        device,
        QueueSet {
            graphics: graphics_queue,
            present: present_queue,
            transfer: transfer_queue,
            compute: compute_queue,
        },
    ))
}

fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        let size = window.inner_size();
        let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
        vk::Extent2D::builder()
            .width(clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
                size.width,
            ))
            .height(clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
                size.height,
            ))
            .build()
    }
}

pub unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    logical_device: &Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    queue_family_indices: &QueueFamilyIndices,
    swapchain_support: &SwapchainSupport,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Extent2D, vk::Format)> {
    let surface_format = get_swapchain_surface_format(&swapchain_support.formats);
    let present_mode = get_swapchain_present_mode(&swapchain_support.present_modes);
    let extent = get_swapchain_extent(window, swapchain_support.capabilities);

    let mut image_count = swapchain_support.capabilities.min_image_count + 1;
    if swapchain_support.capabilities.max_image_count != 0
        && image_count > swapchain_support.capabilities.max_image_count
    {
        image_count = swapchain_support.capabilities.max_image_count;
    }

    let image_sharing_mode = vk::SharingMode::CONCURRENT;

    let indices = queue_family_indices.to_render_vec();
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&indices)
        .pre_transform(swapchain_support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain = logical_device.create_swapchain_khr(&swapchain_create_info, None)?;
    let images = logical_device.get_swapchain_images_khr(swapchain)?;
    let extent = swapchain_create_info.image_extent;
    let format = swapchain_create_info.image_format;
    Ok((swapchain, images, extent, format))
}

pub unsafe fn create_swapchain_image_views(
    logical_device: &Device,
    images: &Vec<vk::Image>,
    swapchain_format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    images
        .iter()
        .map(|i| {
            let components = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY);

            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let info = vk::ImageViewCreateInfo::builder()
                .image(*i)
                .view_type(vk::ImageViewType::_2D)
                .format(swapchain_format)
                .components(components)
                .subresource_range(subresource_range);

            logical_device
                .create_image_view(&info, None)
                .map_err(|e| anyhow!("{}", e))
        })
        .collect::<Result<Vec<_>, _>>()
}

pub unsafe fn create_pipeline_and_renderpass(
    logical_device: &Device,
    swapchain: &g_objects::Swapchain,
) -> Result<(vk::PipelineLayout, vk::RenderPass, vk::Pipeline)> {
    let vert = include_bytes!("../../shaders/vert.spv");
    let frag = include_bytes!("../../shaders/frag.spv");

    let vert_shader_module = create_shader_module(logical_device, &vert[..])?;
    let frag_shader_module = create_shader_module(logical_device, &frag[..])?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(b"main\0");

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(swapchain.extent.width as f32)
        .height(swapchain.extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(swapchain.extent);

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
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::_1);

    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let layout_info = vk::PipelineLayoutCreateInfo::builder();

    let pipeline_layout = logical_device.create_pipeline_layout(&layout_info, None)?;

    let render_pass = create_render_pass(logical_device, swapchain)?;

    let stages = &[vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = logical_device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0];

    logical_device.destroy_shader_module(vert_shader_module, None);
    logical_device.destroy_shader_module(frag_shader_module, None);

    Ok((pipeline_layout, render_pass, pipeline))
}

unsafe fn create_shader_module(
    logical_device: &Device,
    bytecode: &[u8],
) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    logical_device
        .create_shader_module(&info, None)
        .map_err(|e| anyhow!("{}", e))
}

pub unsafe fn create_render_pass(
    logical_device: &Device,
    swapchain: &g_objects::Swapchain,
) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(swapchain.format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments);

    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    let attachments = &[color_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    logical_device
        .create_render_pass(&info, None)
        .map_err(|e| anyhow!("{}", e))
}

pub unsafe fn create_framebuffers(
    device: &Device,

    render_pass: vk::RenderPass,
    swapchain: &g_objects::Swapchain,
) -> Result<Vec<vk::Framebuffer>> {
    swapchain
        .image_views
        .iter()
        .map(|i| {
            let attachments = &[*i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(attachments)
                .width(swapchain.extent.width)
                .height(swapchain.extent.height)
                .layers(1);

            device
                .create_framebuffer(&create_info, None)
                .map_err(|e| anyhow!("{}", e))
        })
        .collect::<Result<Vec<_>, _>>()
}

pub unsafe fn create_command_pool(
    logical_device: &Device,
    queue_family_indices: &QueueFamilyIndices,
) -> Result<vk::CommandPool> {
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::empty()) // Optional.
        .queue_family_index(queue_family_indices.graphics);

    logical_device
        .create_command_pool(&info, None)
        .map_err(|e| anyhow!("{}", e))
}

pub unsafe fn create_command_buffers(
    device: &Device,
    command_pool: vk::CommandPool,
    framebuffers: &Vec<vk::Framebuffer>,
    swapchain: &g_objects::Swapchain,
    pipeline: &g_objects::Pipeline,
) -> Result<Vec<vk::CommandBuffer>> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(framebuffers.len() as u32);

    let command_buffers = device.allocate_command_buffers(&allocate_info)?;

    for (i, command_buffer) in command_buffers.iter().enumerate() {
        let inheritance = vk::CommandBufferInheritanceInfo::builder();

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::empty()) // Optional.
            .inheritance_info(&inheritance); // Optional.

        device.begin_command_buffer(*command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(swapchain.extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let clear_values = &[color_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(pipeline.render_pass)
            .framebuffer(framebuffers[i])
            .render_area(render_area)
            .clear_values(clear_values);

        device.cmd_begin_render_pass(*command_buffer, &info, vk::SubpassContents::INLINE);

        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline,
        );

        device.cmd_draw(*command_buffer, 3, 1, 0, 0);

        device.cmd_end_render_pass(*command_buffer);

        device.end_command_buffer(*command_buffer)?;
    }

    Ok(command_buffers)
}

pub unsafe fn create_sync_objects(
    logical_device: &Device,
    swapchain_images_count: usize,
) -> Result<(
    Vec<vk::Semaphore>,
    Vec<vk::Semaphore>,
    Vec<vk::Fence>,
    Vec<vk::Fence>,
)> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();

    let image_available_semaphores = (0..MAX_FRAMES_IN_FLIGHT)
        .map(|_| {
            logical_device
                .create_semaphore(&semaphore_info, None)
                .map_err(|e| anyhow!("{}", e))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let render_finished_semaphores = (0..MAX_FRAMES_IN_FLIGHT)
        .map(|_| {
            logical_device
                .create_semaphore(&semaphore_info, None)
                .map_err(|e| anyhow!("{}", e))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    let in_flight_fences = (0..MAX_FRAMES_IN_FLIGHT)
        .map(|_| {
            logical_device
                .create_fence(&fence_info, None)
                .map_err(|e| anyhow!("{}", e))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let images_in_flight = (0..swapchain_images_count)
        .map(|_| vk::Fence::null())
        .collect();

    Ok((
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
        images_in_flight,
    ))
}
