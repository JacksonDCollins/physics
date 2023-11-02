use crate::graphics::objects as g_objects;
use crate::graphics::types as g_types;
use crate::graphics::utils as g_utils;

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;
use vk::SurfaceKHR;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;
use winit::event::ElementState;
use winit::event::KeyboardInput;
use winit::event::VirtualKeyCode;
use winit::event::WindowEvent;

use winit::window::Window;

pub struct RenderEngine {
    surface: SurfaceKHR,
    queue_set: g_utils::QueueSet,
    swapchain: g_objects::Swapchain,
    pipeline: g_objects::Pipeline,
    presenter: g_objects::Presenter,

    model_manager: g_objects::ModelManager,
    texture_engine: g_objects::TextureMemoryAllocator,
    queue_family_indices: g_utils::QueueFamilyIndices,
    swapchain_support: g_utils::SwapchainSupport,
    msaa_samples: vk::SampleCountFlags,

    start: Instant,
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
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<Self> {
        let swapchain = g_objects::Swapchain::create(
            window,
            logical_device,
            surface,
            &queue_family_indices,
            &swapchain_support,
        )?;

        // let mut buffer_memory_allocator = g_objects::BufferMemoryAllocator::create()?;
        let mut model_manager = g_objects::ModelManager::create()?;

        let sphere = g_objects::Model::create("resources/sphere.obj")?;

        model_manager.add_model("sphere", sphere);

        // buffer_memory_allocator.add_vertex_buffer(sphere.vertex_buffer);

        // buffer_memory_allocator.set_index_buffer(sphere.index_buffer);

        let room = g_objects::Model::create("resources/viking_room.obj")?;

        model_manager.add_model("room", room);
        // buffer_memory_allocator.add_vertex_buffer(room.vertex_buffer);

        // buffer_memory_allocator.set_index_buffer(room.index_buffer);

        for _ in 0..swapchain.images.len() {
            let uniform_buffer = g_objects::UniformBuffer::create(
                logical_device,
                g_types::UniformBufferObject::default(),
            )?;
            model_manager
                .buffer_allocator
                .add_uniform_buffer(uniform_buffer);
        }

        let mut texture_engine = g_objects::TextureMemoryAllocator::create()?;

        let texture = g_objects::Texture::create("resources/viking_room.png")?;

        let texture2 = g_objects::Texture::create("resources/texture.png")?;

        texture_engine.add_texture(texture);
        texture_engine.add_texture(texture2);

        texture_engine.prepare_samplers(logical_device)?;

        let pipeline = g_objects::Pipeline::create(
            instance,
            logical_device,
            physical_device,
            &swapchain,
            msaa_samples,
            &texture_engine,
        )?;

        let presenter = g_objects::Presenter::create(
            logical_device,
            &swapchain,
            &pipeline,
            &queue_family_indices,
            &mut model_manager,
            &mut texture_engine,
            instance,
            physical_device,
            &queue_set,
            msaa_samples,
        )?;

        Ok(Self {
            surface,
            queue_set,
            swapchain,
            pipeline,
            presenter,

            model_manager,
            texture_engine,
            queue_family_indices,
            swapchain_support,
            msaa_samples,

            start: Instant::now(),
        })
    }

    pub unsafe fn recreate_sawpchain(
        &mut self,
        window: &Window,
        instance: &Instance,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<()> {
        logical_device.device_wait_idle()?;

        self.swapchain.destroy(logical_device);
        self.pipeline.destroy(logical_device);
        self.presenter.destroy(logical_device);

        self.swapchain_support =
            g_utils::query_swapchain_support(instance, physical_device, self.surface)?;

        self.swapchain = g_objects::Swapchain::create(
            window,
            logical_device,
            self.surface,
            &self.queue_family_indices,
            &self.swapchain_support,
        )?;

        self.pipeline = g_objects::Pipeline::create(
            instance,
            logical_device,
            physical_device,
            &self.swapchain,
            self.msaa_samples,
            &self.texture_engine,
        )?;

        self.presenter = g_objects::Presenter::create(
            logical_device,
            &self.swapchain,
            &self.pipeline,
            &self.queue_family_indices,
            &mut self.model_manager,
            &mut self.texture_engine,
            instance,
            physical_device,
            &self.queue_set,
            self.msaa_samples,
        )?;

        Ok(())
    }

    pub unsafe fn render(
        &mut self,
        window: &Window,
        logical_device: &Device,
        physical_device: vk::PhysicalDevice,
        instance: &Instance,
        frame: usize,
        resized: &mut bool,
    ) -> Result<()> {
        logical_device.wait_for_fences(
            &[self.presenter.in_flight_fences[frame]],
            true,
            u64::MAX,
        )?;

        let result = logical_device.acquire_next_image_khr(
            self.swapchain.swapchain,
            u64::MAX,
            self.presenter.image_available_semaphores[frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok(res) => match res {
                (_, vk::SuccessCode::SUBOPTIMAL_KHR) => {
                    return self.recreate_sawpchain(
                        window,
                        instance,
                        logical_device,
                        physical_device,
                    )
                }
                (index, _) => index as usize,
            },
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                return self.recreate_sawpchain(window, instance, logical_device, physical_device)
            }
            Err(error) => return Err(anyhow!(error)),
        };

        if !self.presenter.images_in_flight[image_index].is_null() {
            logical_device.wait_for_fences(
                &[self.presenter.images_in_flight[image_index]],
                true,
                u64::MAX,
            )?;
        }

        self.presenter.images_in_flight[image_index] = self.presenter.in_flight_fences[frame];

        self.update_uniform_buffer(image_index)?;
        self.update_command_buffer(logical_device, image_index)?;

        let wait_semaphores = &[self.presenter.image_available_semaphores[frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.presenter.command_buffers[image_index]];
        let signal_semaphores = &[self.presenter.render_finished_semaphores[frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        logical_device.reset_fences(&[self.presenter.in_flight_fences[frame]])?;

        logical_device.queue_submit(
            self.queue_set.graphics,
            &[submit_info],
            self.presenter.in_flight_fences[frame],
        )?;

        let swapchains = &[self.swapchain.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = logical_device.queue_present_khr(self.queue_set.present, &present_info);

        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if *resized || changed {
            *resized = false;
            self.recreate_sawpchain(window, instance, logical_device, physical_device)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        Ok(())
    }

    pub unsafe fn update_uniform_buffer(&mut self, image_index: usize) -> Result<()> {
        // let time = self.start.elapsed().as_secs_f32();

        // let model =
        //     g_types::Mat4::from_axis_angle(g_types::vec3(0.0, 0.0, 1.0), g_types::Deg(90.0) * time);

        let view = g_types::Mat4::look_at_rh(
            g_types::point3(6.0, 0.0, 2.0),
            g_types::point3(0.0, 0.0, 0.0),
            g_types::vec3(0.0, 0.0, 1.0),
        );

        #[rustfmt::skip]
        let correction = g_types::Mat4::new(
            1.0,  0.0,       0.0, 0.0,
            // We're also flipping the Y-axis with this line's `-1.0`.
            0.0, -1.0,       0.0, 0.0,
            0.0,  0.0, 1.0 / 2.0, 0.0,
            0.0,  0.0, 1.0 / 2.0, 1.0,
        );

        let proj = correction
            * cgmath::perspective(
                g_types::Deg(45.0),
                self.swapchain.extent.width as f32 / self.swapchain.extent.height as f32,
                0.1,
                10.0,
            );

        let ubo = g_types::UniformBufferObject { view, proj };

        self.model_manager
            .buffer_allocator
            .update_uniform_buffer(ubo, image_index)?;

        Ok(())
    }

    unsafe fn update_command_buffer(
        &mut self,
        logical_device: &Device,
        image_index: usize,
    ) -> Result<()> {
        let command_pool = self.presenter.command_pool_sets[image_index].graphics;
        logical_device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.presenter.command_buffers[image_index];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        logical_device.begin_command_buffer(command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.swapchain.extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.pipeline.render_pass)
            .framebuffer(self.presenter.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        logical_device.cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::INLINE);

        // let secondary_command_buffers = (0..4)
        //     .map(|i| {
        //         self.update_secondary_command_buffer(
        //             logical_device,
        //             image_index,
        //             i,
        //             if i % 3 == 0 { "sphere" } else { "room" },
        //         )
        //     })
        //     .collect::<Result<Vec<_>, _>>()?;

        // let secondary_command_buffers = self
        //     .model_manager
        //     .models
        //     .values_mut()
        //     .enumerate()
        //     .map(|(index, model)| {
        //         model.make_command_buffer(
        //             logical_device,
        //             image_index,
        //             index,
        //             &mut self.presenter.secondary_command_buffers[image_index],
        //             &self.presenter.command_pool_sets,
        //             &self.pipeline,
        //             &self.presenter.framebuffers,
        //             &self.presenter.descriptor_sets,
        //             self.start,
        //         )
        //     })
        //     .collect::<Result<Vec<_>>>()?;

        // logical_device.cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);

        logical_device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.pipeline,
        );

        logical_device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.pipeline_layout,
            0,
            &[self.presenter.descriptor_sets[image_index]],
            &[],
        );

        for (model_index, model) in self.model_manager.models.values_mut().enumerate() {
            model.push_constants(
                logical_device,
                command_buffer,
                &self.pipeline,
                self.start,
                model_index,
            );

            model.draw_model(logical_device, command_buffer)?
        }

        logical_device.cmd_end_render_pass(command_buffer);

        logical_device.end_command_buffer(command_buffer)?;
        Ok(())
    }

    unsafe fn update_secondary_command_buffer(
        &mut self,
        logical_device: &Device,
        image_index: usize,
        model_index: usize,
        model_name: &str,
    ) -> Result<vk::CommandBuffer> {
        let command_buffers = &mut self.presenter.secondary_command_buffers[image_index];
        while model_index >= command_buffers.len() {
            println!("Allocating new secondary command buffer");
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.presenter.command_pool_sets[image_index].graphics)
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = logical_device.allocate_command_buffers(&allocate_info)?[0];
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.pipeline.render_pass)
            .subpass(0)
            .framebuffer(self.presenter.framebuffers[image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        logical_device.begin_command_buffer(command_buffer, &info)?;

        let time = self.start.elapsed().as_secs_f32();

        let y = (((model_index % 2) as f32) * 2.5) - 1.25;
        let z = (((model_index / 2) as f32) * -2.0) + 1.0;

        logical_device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.pipeline,
        );

        logical_device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.pipeline_layout,
            0,
            &[self.presenter.descriptor_sets[image_index]],
            &[],
        );

        // for (_name, model) in self.model_manager.models.iter() {
        let model = self.model_manager.models.get_mut(model_name).unwrap();
        model.set_position(g_types::vec3(0.0, y, z));

        logical_device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[model.vertex_buffer.get_buffer()],
            &[0],
        );

        logical_device.cmd_bind_index_buffer(
            command_buffer,
            model.index_buffer.get_buffer(),
            0,
            vk::IndexType::UINT32,
        );

        let model_mat = g_types::Mat4::from_translation(model.position)
            * g_types::Mat4::from_axis_angle(
                g_types::vec3(0.0, 0.0, 1.0),
                g_types::Deg(90.0) * time,
            );
        let model_bytes = std::slice::from_raw_parts(
            &model_mat as *const g_types::Mat4 as *const u8,
            size_of::<g_types::Mat4>(),
        );

        logical_device.cmd_push_constants(
            command_buffer,
            self.pipeline.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );

        logical_device.cmd_push_constants(
            command_buffer,
            self.pipeline.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            &[
                (model_index % 2).to_ne_bytes(),
                (model_index % 2).to_ne_bytes(),
            ]
            .concat(),
        );

        logical_device.cmd_draw_indexed(
            command_buffer,
            model.index_buffer.get_indice_count(),
            1,
            0,
            0,
            0,
        );
        // }

        logical_device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }

    pub unsafe fn destroy(&mut self, logical_device: &Device, instance: &Instance) {
        self.model_manager.destroy(logical_device);
        self.texture_engine.destroy(logical_device);
        self.presenter.destroy(logical_device);
        self.pipeline.destroy(logical_device);
        self.swapchain.destroy(logical_device);
        instance.destroy_surface_khr(self.surface, None);
    }
}

pub struct InputEngine {
    pub keydata: HashMap<VirtualKeyCode, ElementState>,
}

impl InputEngine {
    pub fn create() -> Self {
        Self {
            keydata: HashMap::new(),
        }
    }

    pub fn update(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                self.keydata
                    .entry(*keycode)
                    .and_modify(|e| *e = *state)
                    .or_insert(*state);
            }
            _ => {}
        }
    }

    pub fn clear_old_input(&mut self) {
        self.keydata.retain(|_, v| *v == ElementState::Pressed);
    }
}
