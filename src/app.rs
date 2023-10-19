use crate::graphics;
use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::{
    loader::{LibloadingLoader, LIBRARY},
    Entry,
};
use winit::window::Window;

pub struct App {
    entry: Entry,
    instance: Instance,
    logical_device: Device,
    dbg_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl App {
    pub unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|e| anyhow!("{}", e))?;
        let (instance, dbg_messenger) = graphics::utils::create_instance(window, &entry)?;
        let surface = vulkanalia::window::create_surface(&instance, &window, &window)?;
        let physical_device = graphics::utils::pick_physical_device(&instance, surface)?;

        let (logical_device, graphics_queue, present_queue) =
            graphics::utils::create_logical_device(&entry, &instance, surface, physical_device)?;

        Ok(Self {
            entry,
            instance,
            logical_device,
            dbg_messenger,
        })
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        self.dbg_messenger
            .map(|msger| self.instance.destroy_debug_utils_messenger_ext(msger, None));

        self.instance.destroy_instance(None);
    }
}
