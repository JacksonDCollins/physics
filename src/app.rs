use crate::graphics::utils as g_utils;
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
    model_manager: graphics::objects::ModelManager,
    camera: controller::camera::Camera,
    frame: usize,
    pub resized: bool,
    scene: graphics::objects::Scene,
}

impl App {
    pub unsafe fn create(window: &Window, event_loop: &EventLoop<()>) -> Result<Self> {
        // let loader = LibloadingLoader::new(LIBRARY)?;
        // let entry = Entry::new(loader).map_err(|e| anyhow!("{}", e))?;
        let entry = Entry::linked();
        let (instance, dbg_messenger, debug_utils_loader) =
            graphics::utils::create_instance(&entry, event_loop)?;
        let surface = ash_window::create_surface(
            &entry,
            &instance,
            window.raw_display_handle(),
            window.raw_window_handle(),
            None,
        )?;
        let surface_loader = Surface::new(&entry, &instance);

        let (physical_device, queue_family_indices, swapchain_support, msaa_samples) =
            graphics::utils::pick_physical_device(&instance, surface, &surface_loader)?;

        let (logical_device, queue_set) = graphics::utils::create_logical_device(
            &entry,
            &instance,
            physical_device,
            queue_family_indices,
        )?;

        let mut model_manager = graphics::objects::ModelManager::create()?;

        let scene = graphics::objects::Scene::create()?;

        model_manager.load_models_from_scene(&scene, &logical_device)?;

        let render_engine = graphics::engine::RenderEngine::create(
            window,
            &instance,
            &logical_device,
            physical_device,
            surface,
            surface_loader,
            queue_set,
            queue_family_indices,
            swapchain_support,
            msaa_samples,
            &mut model_manager,
        )?;

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
            model_manager,
            camera,
            frame: 0,
            resized: false,
            scene,
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
            &self.scene,
            &mut self.model_manager,
        )?;

        self.frame = (self.frame + 1) % g_utils::MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    pub fn tick(&mut self) {
        self.camera.update(&self.input_engine.keydata);
        self.camera
            .update_mouse_motion(self.input_engine.recent_delta);
        self.input_engine.clear_old_input();
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
        self.model_manager.destroy(&self.logical_device);

        self.render_engine
            .destroy(&self.logical_device, &self.instance);

        self.logical_device.destroy_device(None);
        self.dbg_messenger.iter().for_each(|msger| {
            self.debug_utils_loader
                .destroy_debug_utils_messenger(*msger, None)
        });
        self.instance.destroy_instance(None);
    }
}

// put view proj into push constants
//refmorat vertex attrbviute bindinghs
