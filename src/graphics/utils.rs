use crate::graphics::objects as g_objects;
use crate::graphics::types as g_types;
use anyhow::{anyhow, Result};

use vulkano::device::physical::PhysicalDevice;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::Device;
use vulkano::device::DeviceCreateInfo;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;
use vulkano::device::QueueCreateInfo;
use vulkano::device::QueueFlags;
use vulkano::format::Format;
use vulkano::image::SampleCounts;
use vulkano::instance::debug::DebugUtilsMessageSeverity;
use vulkano::instance::debug::DebugUtilsMessageType;
use vulkano::instance::debug::DebugUtilsMessenger;
use vulkano::instance::debug::DebugUtilsMessengerCallback;
use vulkano::instance::debug::DebugUtilsMessengerCallbackData;
use vulkano::instance::debug::DebugUtilsMessengerCreateInfo;
use vulkano::instance::debug::ValidationFeatureEnable;
use vulkano::instance::Instance;
use vulkano::instance::InstanceCreateFlags;
use vulkano::instance::InstanceCreateInfo;
use vulkano::instance::InstanceExtensions;
use vulkano::library;
use vulkano::swapchain::ColorSpace;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::Surface;
use vulkano::swapchain::SurfaceCapabilities;
use vulkano::swapchain::SurfaceInfo;
use vulkano::ExtensionProperties;
use vulkano::Version;
use vulkano::VulkanLibrary;

use std::collections::HashSet;

use std::fmt::Debug;

use std::fs::File;

use std::io::BufReader;
use std::sync::Arc;
use thiserror::Error;
// use vulkanalia::bytecode::Bytecode;
// use vulkanalia::prelude::v1_2::*;
// use vulkanalia::vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension};
// use vulkanalia::{Entry, Version};
use winit::window::Window;

pub use std::ptr::copy_nonoverlapping as memcpy;

pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
pub const VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";
pub const PORTABILITY_MACOS_VERSION: Version = Version {
    major: 1,
    minor: 3,
    patch: 216,
};
pub const DEVICE_EXTENSIONS: DeviceExtensions = DeviceExtensions {
    khr_swapchain: true,
    khr_shader_non_semantic_info: true,
    ..DeviceExtensions::empty()
};
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub unsafe fn create_instance(
    // window: &Window,
    // entry: &Entry,
    library: Arc<VulkanLibrary>,
    mut extensions: InstanceExtensions,
) -> Result<(Arc<Instance>, Option<DebugUtilsMessenger>)> {
    let available_layers = library
        .layer_properties()?
        .into_iter()
        .map(|l| l.name())
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer not available"));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.to_string()]
    } else {
        vec![]
    };

    if VALIDATION_ENABLED {
        extensions.ext_debug_utils = true;
    }

    // // Required by Vulkan SDK on macOS since 1.3.216.
    let flags = if cfg!(target_os = "macos") && library.api_version() >= PORTABILITY_MACOS_VERSION {
        log::info!("Enabling extensions for macOS portability");
        extensions.khr_get_physical_device_properties2 = true;
        extensions.khr_portability_enumeration = true;
        InstanceCreateFlags::ENUMERATE_PORTABILITY
    } else {
        InstanceCreateFlags::empty()
    };

    let mut instance_info = InstanceCreateInfo {
        enabled_layers: layers,
        enabled_extensions: extensions,
        engine_name: Some("No Engine".to_string()),
        engine_version: Version {
            major: 1,
            minor: 0,
            patch: 0,
        },
        flags,
        ..InstanceCreateInfo::application_from_cargo_toml()
    };

    let mut debug_info = DebugUtilsMessengerCreateInfo {
        message_severity: DebugUtilsMessageSeverity::ERROR
            | DebugUtilsMessageSeverity::WARNING
            | DebugUtilsMessageSeverity::INFO
            | DebugUtilsMessageSeverity::VERBOSE,
        message_type: DebugUtilsMessageType::GENERAL
            | DebugUtilsMessageType::VALIDATION
            | DebugUtilsMessageType::PERFORMANCE,
        ..DebugUtilsMessengerCreateInfo::user_callback(DebugUtilsMessengerCallback::new(
            debug_callback,
        ))
    };

    let features = vec![
        ValidationFeatureEnable::BestPractices,
        ValidationFeatureEnable::DebugPrintf,
    ];
    if VALIDATION_ENABLED {
        instance_info.debug_utils_messengers = vec![debug_info];
        instance_info.enabled_validation_features = features
    }

    let instance = Instance::new(library, instance_info)?;

    let messenger = if VALIDATION_ENABLED {
        Some(DebugUtilsMessenger::new(instance, debug_info)?)
    } else {
        None
    };

    Ok((instance, messenger))
}

fn debug_callback<'a>(
    severity: DebugUtilsMessageSeverity,
    type_: DebugUtilsMessageType,
    data: DebugUtilsMessengerCallbackData<'a>,
) {
    let message = data.message;

    match severity {
        DebugUtilsMessageSeverity::ERROR => log::error!("({:?}) {}", type_, message),
        DebugUtilsMessageSeverity::WARNING => log::warn!("({:?}) {}", type_, message),
        DebugUtilsMessageSeverity::INFO => log::debug!("({:?}) {}", type_, message),
        DebugUtilsMessageSeverity::VERBOSE => log::trace!("({:?}) {}", type_, message),
        _ => {}
    }
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

pub unsafe fn pick_physical_device(
    instance: Arc<Instance>,
    surface: Arc<Surface>,
) -> Result<(
    Arc<PhysicalDevice>,
    QueueFamilyIndices,
    SwapchainSupport,
    SampleCounts,
)> {
    for physical_device in instance.enumerate_physical_devices()? {
        match check_physical_device(instance, surface, physical_device) {
            Err(error) => log::warn!(
                "Skipping physical device (`{}`): {}",
                physical_device.properties().device_name,
                error
            ),
            Ok((indices, swapchain_support)) => {
                log::info!(
                    "Selected physical device (`{}`).",
                    physical_device.properties().device_name
                );
                let msaa_samples = get_max_msaa_samples(instance, physical_device);
                return Ok((physical_device, indices, swapchain_support, msaa_samples));
            }
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

unsafe fn check_physical_device(
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
) -> Result<(QueueFamilyIndices, SwapchainSupport)> {
    let properties = physical_device.properties();
    if properties.device_type != PhysicalDeviceType::DiscreteGpu {
        return Err(anyhow!(SuitabilityError(
            "Only discrete GPUs are supported."
        )));
    }

    let features = physical_device.supported_features();
    if !features.geometry_shader {
        return Err(anyhow!(SuitabilityError(
            "Missing geometry shader support."
        )));
    }
    if !features.sampler_anisotropy {
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
    instance: Arc<Instance>,
    physical_device: Arc<PhysicalDevice>,
) -> Result<()> {
    let extensions = physical_device.supported_extensions();
    if DEVICE_EXTENSIONS.contains(extensions) {
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
        let set = HashSet::from([self.graphics, self.transfer]);
        set.into_iter().collect::<Vec<_>>()
    }

    pub unsafe fn get_unique_indices(&self) -> Vec<u32> {
        let set = HashSet::from([self.graphics, self.present, self.transfer, self.compute]);
        set.into_iter().collect::<Vec<_>>()
    }

    pub unsafe fn get(
        instance: Arc<Instance>,
        surface: Arc<Surface>,
        physical_device: Arc<PhysicalDevice>,
    ) -> Result<Self> {
        let properties = physical_device.queue_family_properties();

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let transfer = properties
            .iter()
            .position(|p| {
                p.queue_flags.contains(QueueFlags::TRANSFER)
                    && !p.queue_flags.contains(QueueFlags::GRAPHICS)
            })
            .map(|i| i as u32);

        let compute = properties
            .iter()
            .position(|p| {
                p.queue_flags.contains(QueueFlags::COMPUTE)
                    && !p.queue_flags.contains(QueueFlags::GRAPHICS)
            })
            .map(|i| i as u32);

        let mut present = None;
        for (index, _properties) in properties.iter().enumerate() {
            if physical_device.surface_support(index as u32, &surface)? {
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
    pub capabilities: SurfaceCapabilities,
    pub formats: Vec<(Format, ColorSpace)>,
    pub present_modes: Vec<PresentMode>,
}

impl SwapchainSupport {
    pub unsafe fn get(
        instance: Arc<Instance>,
        surface: Arc<Surface>,
        physical_device: Arc<PhysicalDevice>,
    ) -> Result<Self> {
        let surface_info = SurfaceInfo::default();
        Ok(Self {
            // capabilities: instance
            //     .get_physical_device_surface_capabilities_khr(physical_device, surface)?,
            capabilities: physical_device.surface_capabilities(&surface, surface_info)?,
            // formats: instance.get_physical_device_surface_formats_khr(physical_device, surface)?,
            formats: physical_device.surface_formats(&surface, surface_info)?,
            // present_modes: instance
            //     .get_physical_device_surface_present_modes_khr(physical_device, surface)?,
            present_modes: physical_device.surface_present_modes(&surface, surface_info)?,
        })
    }
}

pub struct QueueSet {
    pub graphics: Arc<Queue>,
    pub present: Arc<Queue>,
    pub transfer: Arc<Queue>,
    pub compute: Arc<Queue>,
}

pub unsafe fn create_logical_device(
    instance: Arc<Instance>,
    physical_device: Arc<PhysicalDevice>,
    library: Arc<VulkanLibrary>,
    indices: QueueFamilyIndices,
) -> Result<(Device, QueueSet)> {
    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);
    unique_indices.insert(indices.transfer);
    unique_indices.insert(indices.compute);

    let queue_infos = unique_indices
        .iter()
        .map(|i| QueueCreateInfo {
            queue_family_index: *i,
            ..Default::default()
        })
        .collect::<Vec<_>>();

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    let mut extensions = DEVICE_EXTENSIONS;

    // Required by Vulkan SDK on macOS since 1.3.216.
    if cfg!(target_os = "macos") && library.api_version() >= PORTABILITY_MACOS_VERSION {
        extensions.khr_portability_subset = true;
    }

    let features = Features {
        sampler_anisotropy: true,
        ..Features::empty()
    };
    // let info = vk::DeviceCreateInfo::builder()
    //     .queue_create_infos(&queue_infos)
    //     .enabled_layer_names(&layers)
    //     .enabled_extension_names(&extensions)
    //     .enabled_features(&features);
    let info = DeviceCreateInfo {
        enabled_features: features,
        queue_create_infos: queue_infos,
        enabled_extensions: extensions,
        ..DeviceCreateInfo::default()
    };

    // let device = instance.create_device(physical_device, &info, None)?;
    let (device, mut queues) = Device::new(physical_device, info)?;

    let graphics_queue = queues
        .find(|q| q.id_within_family() == indices.graphics)
        .unwrap();
    let present_queue = queues
        .find(|q| q.id_within_family() == indices.present)
        .unwrap();
    let transfer_queue = queues
        .find(|q| q.id_within_family() == indices.transfer)
        .unwrap();
    let compute_queue = queues
        .find(|q| q.id_within_family() == indices.compute)
        .unwrap();

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
    logical_device: Arc<Device>,
    surface: vk::Arc<Surface>,
    queue_family_indices: &QueueFamilyIndices,
    swapchain_support: &SwapchainSupport,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Extent2D, vk::Format)> {
    let surface_format = get_swapchain_surface_format(&swapchain_support.formats);
    let present_mode = get_swapchain_present_mode(&swapchain_support.present_modes);
    let extent = get_swapchain_extent(window, swapchain_support.capabilities);

    let mut image_count = (swapchain_support.capabilities.min_image_count + 1).max(3);
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
    logical_device: Arc<Device>,
    images: &[vk::Image],
    swapchain_format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    images
        .iter()
        .map(|i| {
            create_image_view(
                logical_device,
                *i,
                1,
                swapchain_format,
                vk::ImageAspectFlags::COLOR,
            )
        })
        .collect::<Result<Vec<_>, _>>()
}

pub unsafe fn create_pipeline_layout(
    logical_device: Arc<Device>,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let vert_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(64 /* 16 × 4 byte floats */);

    let frag_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(64)
        .size(16 /* 2 x 4 byte ints */);

    let set_layouts = &[descriptor_set_layout];
    let push_constant_ranges = &[vert_push_constant_range, frag_push_constant_range];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(set_layouts)
        .push_constant_ranges(push_constant_ranges);
    logical_device
        .create_pipeline_layout(&layout_info, None)
        .map_err(|e| anyhow!("{}", e))
}

pub unsafe fn create_shader_module(
    logical_device: Arc<Device>,
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
    instance: Arc<Instance>,
    logical_device: Arc<Device>,
    physical_device: PhysicalDevice,
    swapchain: &g_objects::Swapchain,
    msaa_samples: SampleCounts,
) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(swapchain.format)
        .samples(msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_stencil_attachment = vk::AttachmentDescription::builder()
        .format(get_depth_format(instance, physical_device)?)
        .samples(msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_resolve_attachment = vk::AttachmentDescription::builder()
        .format(swapchain.format)
        .samples(SampleCounts::_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_resolve_attachment_ref = vk::AttachmentReference::builder()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
    let resolve_attachments = &[color_resolve_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments)
        .depth_stencil_attachment(&depth_stencil_attachment_ref)
        .resolve_attachments(resolve_attachments);

    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    let attachments = &[
        color_attachment,
        depth_stencil_attachment,
        color_resolve_attachment,
    ];
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
    device: Arc<Device>,
    render_pass: vk::RenderPass,
    swapchain: &g_objects::Swapchain,
    depth_image_view: vk::ImageView,
    color_image_view: vk::ImageView,
) -> Result<Vec<vk::Framebuffer>> {
    swapchain
        .image_views
        .iter()
        .map(|i| {
            let attachments = &[color_image_view, depth_image_view, *i];
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

pub unsafe fn create_command_pool_set(
    logical_device: Arc<Device>,
    queue_family_indices: &QueueFamilyIndices,
) -> Result<g_types::CommandPoolSet> {
    let present = create_command_pool(logical_device, queue_family_indices.present)?;
    let graphics = create_command_pool(logical_device, queue_family_indices.graphics)?;
    let transfer = create_command_pool(logical_device, queue_family_indices.transfer)?;
    let compute = create_command_pool(logical_device, queue_family_indices.compute)?;

    Ok(g_types::CommandPoolSet::create(
        present, graphics, transfer, compute,
    ))
}

pub unsafe fn create_command_pool_sets(
    logical_device: Arc<Device>,
    swapchain_images_count: u32,
    queue_family_indices: &QueueFamilyIndices,
) -> Result<Vec<g_types::CommandPoolSet>> {
    (0..swapchain_images_count)
        .map(|_| create_command_pool_set(logical_device, queue_family_indices))
        .collect::<Result<Vec<_>>>()
}

unsafe fn create_command_pool(
    logical_device: Arc<Device>,
    queue_family_index: u32,
) -> Result<vk::CommandPool> {
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(queue_family_index);

    Ok(logical_device.create_command_pool(&info, None)?)
}
pub unsafe fn create_command_buffers(
    device: Arc<Device>,
    command_pool_sets: &[g_types::CommandPoolSet],
    swapchain_images_count: usize,
) -> Result<Vec<vk::CommandBuffer>> {
    (0..swapchain_images_count)
        .map(|index| {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(command_pool_sets[index].graphics)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];

            Ok(command_buffer)
        })
        .collect::<Result<Vec<_>>>()
}

pub unsafe fn create_sync_objects(
    logical_device: Arc<Device>,
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

pub unsafe fn query_swapchain_support(
    instance: Arc<Instance>,
    physical_device: PhysicalDevice,
    surface: vk::Arc<Surface>,
) -> Result<SwapchainSupport> {
    let capabilities =
        instance.get_physical_device_surface_capabilities_khr(physical_device, surface)?;
    let formats = instance.get_physical_device_surface_formats_khr(physical_device, surface)?;
    let present_modes =
        instance.get_physical_device_surface_present_modes_khr(physical_device, surface)?;

    Ok(SwapchainSupport {
        capabilities,
        formats,
        present_modes,
    })
}

pub unsafe fn get_memory_type_index(
    instance: Arc<Instance>,
    physical_device: PhysicalDevice,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    let memory = instance.get_physical_device_memory_properties(physical_device);

    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}

pub unsafe fn create_buffer(
    logical_device: Arc<Device>,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<vk::Buffer> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    logical_device
        .create_buffer(&buffer_info, None)
        .map_err(|e| anyhow!("{}", e))
}

pub unsafe fn create_buffer_and_memory(
    instance: Arc<Instance>,
    device: Arc<Device>,
    physical_device: PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, vk::MemoryRequirements)> {
    let buffer = create_buffer(device, size, usage)?; //device.create_buffer(&buffer_info, None)?;

    let requirements = device.get_buffer_memory_requirements(buffer);

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            physical_device,
            properties,
            requirements,
        )?);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory, requirements))
}

pub unsafe fn create_memory_with_mem_type_index(
    device: Arc<Device>,
    size: vk::DeviceSize,
    memory_type_index: u32,
) -> Result<vk::DeviceMemory> {
    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(size)
        .memory_type_index(memory_type_index);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    Ok(buffer_memory)
}

pub unsafe fn copy_buffer(
    logical_device: Arc<Device>,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
    src_offset: u64,
    dst_offset: u64,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(logical_device, command_pool)?;

    let regions = vk::BufferCopy::builder()
        .size(size)
        .src_offset(src_offset)
        .dst_offset(dst_offset);
    logical_device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

    end_single_time_commands(logical_device, command_pool, queue, command_buffer)?;

    Ok(())
}

pub unsafe fn get_memory_info(
    instance: Arc<Instance>,
    physical_device: PhysicalDevice,
    logical_device: Arc<Device>,
    buffer: vk::Buffer,
    size: u64,
    properties: vk::MemoryPropertyFlags,
) -> Result<vk::MemoryAllocateInfo> {
    let requirements = logical_device.get_buffer_memory_requirements(buffer);

    Ok(vk::MemoryAllocateInfo::builder()
        .allocation_size(size)
        .memory_type_index(get_memory_type_index(
            instance,
            physical_device,
            properties,
            requirements,
        )?)
        .build())
}

pub fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

pub unsafe fn create_descriptor_sets(
    logical_device: Arc<Device>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    swapchain_images_count: usize,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = vec![descriptor_set_layout; swapchain_images_count];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    let descriptor_sets = logical_device.allocate_descriptor_sets(&info)?;

    Ok(descriptor_sets)
}

pub unsafe fn create_image(
    instance: Arc<Instance>,
    logical_device: Arc<Device>,
    physical_device: PhysicalDevice,
    width: u32,
    height: u32,
    mip_levels: u32,
    samples: SampleCounts,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .samples(samples)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let image = logical_device.create_image(&info, None)?;

    let requirements = logical_device.get_image_memory_requirements(image);

    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            physical_device,
            properties,
            requirements,
        )?);

    let image_memory = logical_device.allocate_memory(&info, None)?;

    logical_device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

unsafe fn begin_single_time_commands(
    logical_device: Arc<Device>,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);

    let command_buffer = logical_device.allocate_command_buffers(&info)?[0];

    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    logical_device.begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

unsafe fn end_single_time_commands(
    logical_device: Arc<Device>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    logical_device.end_command_buffer(command_buffer)?;

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

    logical_device.queue_submit(queue, &[info], vk::Fence::null())?;
    logical_device.queue_wait_idle(queue)?;

    logical_device.free_command_buffers(command_pool, &[command_buffer]);

    Ok(())
}

pub unsafe fn transition_image_layout(
    logical_device: Arc<Device>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    mip_levels: u32,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
) -> Result<()> {
    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => return Err(anyhow!("Unsupported image layout transition!")),
        };

    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
        match format {
            vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            _ => vk::ImageAspectFlags::DEPTH,
        }
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let command_buffer = begin_single_time_commands(logical_device, command_pool)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(src_queue_family_index)
        .dst_queue_family_index(dst_queue_family_index)
        .image(image)
        .subresource_range(subresource)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    logical_device.cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask,
        dst_stage_mask,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(logical_device, command_pool, queue, command_buffer)
}

pub unsafe fn copy_buffer_to_image(
    logical_device: Arc<Device>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
    offset: u64,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(logical_device, command_pool)?;

    let subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(offset)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(subresource)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    logical_device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    end_single_time_commands(logical_device, command_pool, queue, command_buffer)
}

pub unsafe fn create_image_view(
    logical_device: Arc<Device>,
    image: vk::Image,
    mip_levels: u32,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource_range);

    logical_device
        .create_image_view(&info, None)
        .map_err(|e| anyhow!("{}", e))
}

pub unsafe fn create_texture_image_view(
    logical_device: Arc<Device>,
    texture_image: vk::Image,
    mip_levels: u32,
    format: vk::Format,
) -> Result<vk::ImageView> {
    create_image_view(
        logical_device,
        texture_image,
        mip_levels,
        format,
        vk::ImageAspectFlags::COLOR,
    )
}

pub unsafe fn create_texture_sampler(device: Arc<Device>, mip_levels: u32) -> Result<vk::Sampler> {
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
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(mip_levels as f32);

    device
        .create_sampler(&info, None)
        .map_err(|e| anyhow!("{}", e))
}

pub unsafe fn create_depth_objects(
    instance: Arc<Instance>,
    logical_device: Arc<Device>,
    physical_device: PhysicalDevice,
    extent: vk::Extent2D,
    command_pool_set: &g_types::CommandPoolSet,
    queue_set: &QueueSet,
    msaa_samples: SampleCounts,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let format = get_depth_format(instance, physical_device)?;

    let (depth_image, depth_image_memory) = create_image(
        instance,
        logical_device,
        physical_device,
        extent.width,
        extent.height,
        1,
        msaa_samples,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Image View

    let depth_image_view = create_image_view(
        logical_device,
        depth_image,
        1,
        format,
        vk::ImageAspectFlags::DEPTH,
    )?;

    transition_image_layout(
        logical_device,
        command_pool_set.graphics,
        queue_set.graphics,
        depth_image,
        1,
        format,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        vk::QUEUE_FAMILY_IGNORED,
        vk::QUEUE_FAMILY_IGNORED,
    )?;

    Ok((depth_image, depth_image_memory, depth_image_view))
}

unsafe fn get_supported_format(
    instance: Arc<Instance>,
    physical_device: PhysicalDevice,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    candidates
        .iter()
        .cloned()
        .find(|f| {
            let properties = instance.get_physical_device_format_properties(physical_device, *f);
            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .ok_or_else(|| anyhow!("Failed to find supported format!"))
}

unsafe fn get_depth_format(
    instance: Arc<Instance>,
    physical_device: PhysicalDevice,
) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    get_supported_format(
        instance,
        physical_device,
        candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

pub fn load_model(path: &str) -> Result<(Vec<g_types::Vertex>, Vec<u32>)> {
    let mut reader = BufReader::new(File::open(path)?);

    let (models, _) = tobj::load_obj_buf(&mut reader, &tobj::GPU_LOAD_OPTIONS, |_| {
        Ok(Default::default())
    })?;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for model in models {
        let vertex_triples = model
            .mesh
            .positions
            .chunks(3)
            .map(|p| g_types::vec3(p[0], p[1], p[2]))
            .collect::<Vec<_>>();

        let tex_coord_pairs = model
            .mesh
            .texcoords
            .chunks(2)
            .map(|t| g_types::vec2(t[0], 1.0 - t[1]))
            .collect::<Vec<_>>();

        let normals = model
            .mesh
            .normals
            .chunks(3)
            .map(|n| g_types::vec3(n[0], n[1], n[2]));

        vertex_triples
            .into_iter()
            .zip(tex_coord_pairs)
            .zip(normals)
            .for_each(|((vertex, tex_coord), normal)| {
                vertices.push(g_types::Vertex::new(vertex, normal, tex_coord));
            });

        let indice_len = indices.len();
        indices.extend(model.mesh.indices.iter().map(|i| i + indice_len as u32));
    }

    Ok((vertices, indices))
}

pub unsafe fn generate_mipmaps(
    instance: Arc<Instance>,
    logical_device: Arc<Device>,
    physical_device: PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    if !instance
        .get_physical_device_format_properties(physical_device, format)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        return Err(anyhow!(
            "Texture image format does not support linear blitting!"
        ));
    }

    let command_buffer = begin_single_time_commands(logical_device, command_pool)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let mut barrier = vk::ImageMemoryBarrier::builder()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);

    let mut mip_width = width;
    let mut mip_height = height;

    for i in 1..mip_levels {
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        logical_device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        let blit = vk::ImageBlit::builder()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,
                },
            ])
            .src_subresource(src_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource);

        logical_device.cmd_blit_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        logical_device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        if mip_width > 1 {
            mip_width /= 2;
        }

        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    logical_device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(logical_device, command_pool, queue, command_buffer)?;

    Ok(())
}

pub unsafe fn get_max_msaa_samples(
    instance: Arc<Instance>,
    physical_device: Arc<PhysicalDevice>,
) -> SampleCounts {
    // let properties = instance.get_physical_device_properties(physical_device);
    let properties = physical_device.properties();

    let counts =
        properties.framebuffer_color_sample_counts & properties.framebuffer_depth_sample_counts;
    [
        SampleCounts::SAMPLE_64,
        SampleCounts::SAMPLE_32,
        SampleCounts::SAMPLE_16,
        SampleCounts::SAMPLE_8,
        SampleCounts::SAMPLE_4,
        SampleCounts::SAMPLE_2,
        SampleCounts::SAMPLE_1,
    ]
    .iter()
    .cloned()
    .find(|c| counts.contains(*c))
    .unwrap_or(SampleCounts::SAMPLE_1)
}

pub unsafe fn create_color_objects(
    instance: Arc<Instance>,
    logical_device: Arc<Device>,
    physical_device: PhysicalDevice,
    swapchain: &g_objects::Swapchain,
    msaa_samples: SampleCounts,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let (color_image, color_image_memory) = create_image(
        instance,
        logical_device,
        physical_device,
        swapchain.extent.width,
        swapchain.extent.height,
        1,
        msaa_samples,
        swapchain.format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let color_image_view = create_image_view(
        logical_device,
        color_image,
        1,
        swapchain.format,
        vk::ImageAspectFlags::COLOR,
    )?;

    Ok((color_image, color_image_memory, color_image_view))
}
