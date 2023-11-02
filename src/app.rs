use crate::graphics;
use crate::graphics::utils as g_utils;
use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::{
    loader::{LibloadingLoader, LIBRARY},
    Entry,
};
use winit::event::WindowEvent;
use winit::window::Window;

pub struct App {
    entry: Entry,
    instance: Instance,
    logical_device: Device,
    physical_device: vk::PhysicalDevice,
    dbg_messenger: Option<vk::DebugUtilsMessengerEXT>,
    render_engine: graphics::engines::RenderEngine,
    pub input_engine: graphics::engines::InputEngine,
    frame: usize,
    pub resized: bool,
}

impl App {
    pub unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|e| anyhow!("{}", e))?;
        let (instance, dbg_messenger) = graphics::utils::create_instance(window, &entry)?;
        let surface = vulkanalia::window::create_surface(&instance, &window, &window)?;
        let (physical_device, queue_family_indices, swapchain_support, msaa_samples) =
            graphics::utils::pick_physical_device(&instance, surface)?;

        let (logical_device, queue_set) = graphics::utils::create_logical_device(
            &entry,
            &instance,
            physical_device,
            queue_family_indices,
        )?;

        let render_engine = graphics::engines::RenderEngine::create(
            window,
            &instance,
            &logical_device,
            physical_device,
            surface,
            queue_set,
            queue_family_indices,
            swapchain_support,
            msaa_samples,
        )?;

        let input_engine = graphics::engines::InputEngine::create();
        Ok(Self {
            entry,
            instance,
            logical_device,
            physical_device,
            dbg_messenger,
            render_engine,
            input_engine,
            frame: 0,
            resized: false,
        })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        println!("{:?}", self.input_engine.keydata);

        self.render_engine.render(
            window,
            &self.logical_device,
            self.physical_device,
            &self.instance,
            self.frame,
            &mut self.resized,
        )?;

        self.frame = (self.frame + 1) % g_utils::MAX_FRAMES_IN_FLIGHT;

        // log::info!("frame: {}", self.frame);

        Ok(())
    }

    pub fn input(&mut self, event: &WindowEvent) {
        self.input_engine.update(event)
    }

    pub fn clear_old_input(&mut self) {
        self.input_engine.clear_old_input();
    }

    pub unsafe fn device_wait_idle(&self) -> Result<()> {
        self.logical_device
            .device_wait_idle()
            .map_err(|e| anyhow!("{}", e))
    }
    pub unsafe fn destroy(&mut self) {
        self.render_engine
            .destroy(&self.logical_device, &self.instance);

        self.logical_device.destroy_device(None);
        self.dbg_messenger.iter().for_each(|msger| {
            self.instance
                .destroy_debug_utils_messenger_ext(*msger, None)
        });
        self.instance.destroy_instance(None);
    }
}
