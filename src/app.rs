use crate::graphics::objects::UniformBuffer;
use crate::graphics::types::UniformBufferObject;
use crate::graphics::{objects, utils as g_utils};
use crate::{controller, graphics, input};
use anyhow::{anyhow, Result};
use ash::extensions::khr::Surface;
// use vulkanalia::prelude::v1_2::*;
// use vulkanalia::vk::ExtDebugUtilsExtension;
// use vulkanalia::{
//     loader::{LibloadingLoader, LIBRARY},
//     Entry,
// };
use ash::{vk, Entry};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
// use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::event::WindowEvent;
use winit::event_loop::EventLoop;
use winit::window::Window;

pub struct App {
    instance: ash::Instance,
    logical_device: ash::Device,
    physical_device: vk::PhysicalDevice,
    dbg_messenger: Option<vk::DebugUtilsMessengerEXT>,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    render_engine: graphics::engine::RenderEngine,
    pub input_engine: input::engine::InputEngine,
    // model_manager: graphics::objects::ModelManager,
    camera: controller::camera::Camera,
    frame: usize,
    pub resized: bool,
    scene: graphics::objects::Scene,
    dt: std::time::Duration,
    last_tick: std::time::Instant,
}

impl App {
    pub async unsafe fn create(window: &Window, event_loop: &EventLoop<()>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        log::info!(
            "Using device {} and backend {:?}",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let msaa_sample_state = wgpu::MultisampleState {
            count: 16,
            mask: !0,
            alpha_to_coverage_enabled: false,
        };

        // let swapchain_support = graphics::utils::SwapchainSupport::get(surface, &adapter)?;

        // let loader = LibloadingLoader::new(LIBRARY)?;
        // let entry = Entry::new(loader).map_err(|e| anyhow!("{}", e))?;
        // let entry = Entry::linked();
        // let (instance, dbg_messenger, debug_utils_loader) =
        //     graphics::utils::create_instance(&entry, event_loop)?;
        // let surface = ash_window::create_surface(
        //     &entry,
        //     &instance,
        //     window.raw_display_handle(),
        //     window.raw_window_handle(),
        //     None,
        // )?;
        // let surface_loader = Surface::new(&entry, &instance);

        // let (physical_device, queue_family_indices, swapchain_support, msaa_samples) =
        //     graphics::utils::pick_physical_device(&instance, surface, &surface_loader)?;

        // let (logical_device, queue_set) = graphics::utils::create_logical_device(
        //     &entry,
        //     &instance,
        //     physical_device,
        //     queue_family_indices,
        // )?;

        // let mut model_manager = graphics::objects::ModelManager::create()?;

        let terrain = graphics::objects::Terrain::create(&device)?;
        let mut scene = graphics::objects::Scene::create(terrain, &device)?;

        scene.load_models();

        // model_manager.load_models_from_scene(&scene, &logical_device)?;

        let render_engine = graphics::engine::RenderEngine::create(
            window,
            &instance,
            // &logical_device,
            // physical_device,
            &device,
            surface,
            // surface_loader,
            // queue_set,
            // queue_family_indices,
            &queue,
            // swapchain_support,
            msaa_sample_state,
            &mut scene.model_manager,
        )?;

        //HERE

        // for _ in 0..render_engine.swapchain.images.len() {
        //     let uniform_buffer =
        //         UniformBuffer::create(&logical_device, UniformBufferObject::default())?;
        //     scene
        //         .terrain
        //         .buffer_memory_allocator
        //         .add_uniform_buffer(uniform_buffer);
        // }

        // scene.terrain.create_descriptor_pool_and_sets(
        //     &logical_device,
        //     render_engine.swapchain.images.len(),
        // )?;

        // scene.terrain.create_pipeline(
        //     &logical_device,
        //     msaa_samples,
        //     render_engine.pipeline.render_pass,
        //     render_engine.swapchain.extent,
        // )?;

        // scene.terrain.create_buffers(&logical_device)?;

        // let mut acc = 0;
        // acc = g_utils::align_up(acc, scene.terrain.compute_buffer.reqs.unwrap().alignment);
        // scene.terrain.compute_buffer.offset = Some(acc);
        // acc += scene.terrain.compute_buffer.get_required_size();

        // acc = g_utils::align_up(acc, scene.terrain.index_buffer.reqs.unwrap().alignment);
        // scene.terrain.index_buffer.offset = Some(acc);
        // acc += scene.terrain.index_buffer.get_required_size();

        // acc = g_utils::align_up(acc, scene.terrain.instance_buffer.reqs.unwrap().alignment);
        // scene.terrain.instance_buffer.offset = Some(acc);
        // acc += scene.terrain.instance_buffer.get_required_size();

        // println!("acc: {}", acc);

        // scene.terrain.buffer_memory_allocator.create_memories(
        //     &instance,
        //     &logical_device,
        //     physical_device,
        //     acc,
        // )?;

        // g_utils::memcpy(
        //     scene.terrain.compute_buffer.data.as_ptr(),
        //     scene
        //         .terrain
        //         .buffer_memory_allocator
        //         .stage_memory_ptr
        //         .add(scene.terrain.compute_buffer.offset.unwrap() as usize)
        //         .cast(),
        //     scene.terrain.compute_buffer.data.len(),
        // );

        // g_utils::memcpy(
        //     scene.terrain.index_buffer.indices.as_ptr(),
        //     scene
        //         .terrain
        //         .buffer_memory_allocator
        //         .stage_memory_ptr
        //         .add(scene.terrain.index_buffer.offset.unwrap() as usize)
        //         .cast(),
        //     scene.terrain.index_buffer.indices.len(),
        // );

        // g_utils::memcpy(
        //     scene.terrain.instance_buffer.model_matrixes.as_ptr(),
        //     scene
        //         .terrain
        //         .buffer_memory_allocator
        //         .stage_memory_ptr
        //         .add(scene.terrain.instance_buffer.offset.unwrap() as usize)
        //         .cast(),
        //     scene.terrain.instance_buffer.model_matrixes.len(),
        // );

        // g_utils::copy_buffer(
        //     &logical_device,
        //     render_engine.queue_set.transfer,
        //     render_engine.presenter.master_command_pool_set.transfer,
        //     scene.terrain.buffer_memory_allocator.staging_buffer,
        //     scene.terrain.buffer_memory_allocator.vertex_index_buffer,
        //     acc,
        //     0,
        //     0,
        // )?;

        // logical_device.bind_buffer_memory(
        //     scene.terrain.compute_buffer.buffer,
        //     scene.terrain.buffer_memory_allocator.vertex_index_memory,
        //     scene.terrain.compute_buffer.offset.unwrap(),
        // )?;

        // logical_device.bind_buffer_memory(
        //     scene.terrain.index_buffer.buffer,
        //     scene.terrain.buffer_memory_allocator.vertex_index_memory,
        //     scene.terrain.index_buffer.offset.unwrap(),
        // )?;

        // logical_device.bind_buffer_memory(
        //     scene.terrain.instance_buffer.buffer,
        //     scene.terrain.buffer_memory_allocator.vertex_index_memory,
        //     scene.terrain.instance_buffer.offset.unwrap(),
        // )?;

        // scene
        //     .terrain
        //     .update_descriptor_sets(&logical_device, render_engine.swapchain.images.len())?;

        // END HERE
        let input_engine = input::engine::InputEngine::create();

        let camera = controller::camera::Camera::create();

        Ok(Self {
            instance,
            logical_device,
            physical_device,
            dbg_messenger,
            debug_utils_loader,
            render_engine,
            input_engine,
            // model_manager,
            camera,
            frame: 0,
            resized: false,
            scene,
            dt: std::time::Duration::from_secs(0),
            last_tick: std::time::Instant::now(),
        })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        self.render_engine.render(
            window,
            &self.logical_device,
            self.physical_device,
            &self.instance,
            self.frame,
            &mut self.resized,
            &self.camera,
            &mut self.scene,
            // &mut self.model_manager,
        )?;

        self.frame = (self.frame + 1) % g_utils::MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    pub fn tick(&mut self, now: std::time::Instant) {
        self.camera
            .update_mouse_motion(self.input_engine.recent_delta);

        self.input_engine.clear_old_input();

        self.dt = now - self.last_tick;
        if self.dt < *crate::ALLOWED_DT {
            return;
        }
        self.last_tick = now;

        self.camera.update(&self.input_engine.keydata, self.dt);
    }

    pub fn window_input(&mut self, event: &WindowEvent) {
        self.input_engine.update(event)
    }

    pub fn update_mouse_motion(&mut self, delta: (f64, f64)) {
        self.input_engine.update_mouse_motion(delta)
    }

    pub unsafe fn device_wait_idle(&self) -> Result<()> {
        self.logical_device
            .device_wait_idle()
            .map_err(|e| anyhow!("{}", e))
    }
    pub unsafe fn destroy(&mut self) {
        // self.model_manager.destroy(&self.logical_device);
        self.scene.destroy(&self.logical_device);

        self.render_engine.destroy(&self.logical_device);

        self.logical_device.destroy_device(None);
        self.dbg_messenger.iter().for_each(|msger| {
            self.debug_utils_loader
                .destroy_debug_utils_messenger(*msger, None)
        });
        self.instance.destroy_instance(None);
    }
}

// refactor
