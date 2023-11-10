use crate::graphics::objects as g_objects;
use crate::graphics::types as g_types;
use crate::graphics::utils as g_utils;

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::Device;
use vulkano::image::SampleCounts;
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;
// use vk::Arc<Surface>;
// use vulkanalia::prelude::v1_2::*;
// use vulkanalia::vk::KhrSurfaceExtension;
// use vulkanalia::vk::KhrSwapchainExtension;
use winit::event::ElementState;
use winit::event::KeyboardInput;
use winit::event::VirtualKeyCode;
use winit::event::WindowEvent;

use winit::window::Window;

pub struct RenderEngine {
    surface: Arc<Surface>,
    queue_set: g_utils::QueueSet,
    swapchain: g_objects::Swapchain,
    pipeline: g_objects::Pipeline,
    presenter: g_objects::Presenter,

    queue_family_indices: g_utils::QueueFamilyIndices,
    swapchain_support: g_utils::SwapchainSupport,
    msaa_samples: SampleCounts,

    start: Instant,
}

impl RenderEngine {
    pub unsafe fn create(
        window: &Window,
        instance: Arc<Instance>,
        logical_device: Arc<Device>,
        physical_device: PhysicalDevice,
        surface: Arc<Surface>,
        queue_set: g_utils::QueueSet,
        queue_family_indices: g_utils::QueueFamilyIndices,
        swapchain_support: g_utils::SwapchainSupport,
        msaa_samples: SampleCounts,
        model_manager: &mut g_objects::ModelManager,
    ) -> Result<Self> {
        let swapchain = g_objects::Swapchain::create(
            window,
            logical_device,
            surface,
            &queue_family_indices,
            &swapchain_support,
        )?;

        for _ in 0..swapchain.images.len() {
            let uniform_buffer = g_objects::UniformBuffer::create(
                logical_device,
                g_types::UniformBufferObject::default(),
            )?;
            model_manager
                .buffer_allocator
                .add_uniform_buffer(uniform_buffer);
        }

        let pipeline = g_objects::Pipeline::create(
            instance,
            logical_device,
            physical_device,
            &swapchain,
            msaa_samples,
            // &model_manager.texture_engine,
        )?;

        model_manager.create_descriptor_pools_and_sets(logical_device, swapchain.images.len())?;

        model_manager.create_pipelines(
            logical_device,
            msaa_samples,
            pipeline.render_pass,
            swapchain.extent,
        )?;

        let presenter = g_objects::Presenter::create(
            logical_device,
            &swapchain,
            &pipeline,
            &queue_family_indices,
            model_manager,
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

            queue_family_indices,
            swapchain_support,
            msaa_samples,

            start: Instant::now(),
        })
    }

    pub unsafe fn recreate_sawpchain(
        &mut self,
        window: &Window,
        instance: Arc<Instance>,
        logical_device: Arc<Device>,
        physical_device: PhysicalDevice,
        model_manager: &mut g_objects::ModelManager,
    ) -> Result<()> {
        logical_device.wait_idle()?;

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
        )?;

        model_manager.recreate_pipelines(
            logical_device,
            self.msaa_samples,
            self.pipeline.render_pass,
            self.swapchain.extent,
        )?;

        self.presenter = g_objects::Presenter::create(
            logical_device,
            &self.swapchain,
            &self.pipeline,
            &self.queue_family_indices,
            model_manager,
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
        logical_device: Arc<Device>,
        physical_device: PhysicalDevice,
        instance: Arc<Instance>,
        frame: usize,
        resized: &mut bool,
        camera: &crate::controller::camera::Camera,
        scene: &g_objects::Scene,
        model_manager: &mut g_objects::ModelManager,
    ) -> Result<()> {
        self.presenter.in_flight_fences[frame].wait(None)?;

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
                        model_manager,
                    )
                }
                (index, _) => index as usize,
            },
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                return self.recreate_sawpchain(
                    window,
                    instance,
                    logical_device,
                    physical_device,
                    model_manager,
                )
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

        self.update_uniform_buffer(image_index, camera, model_manager)?;
        self.update_command_buffer(logical_device, image_index, scene, model_manager)?;

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
            self.recreate_sawpchain(
                window,
                instance,
                logical_device,
                physical_device,
                model_manager,
            )?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        Ok(())
    }

    pub unsafe fn update_uniform_buffer(
        &mut self,
        image_index: usize,
        camera: &crate::controller::camera::Camera,
        model_manager: &mut g_objects::ModelManager,
    ) -> Result<()> {
        let view = camera.get_view_matrix();

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
                100.0,
            );

        let ubo = g_types::UniformBufferObject { view, proj };

        model_manager
            .buffer_allocator
            .update_uniform_buffer(ubo, image_index)?;

        Ok(())
    }

    unsafe fn update_command_buffer(
        &mut self,
        logical_device: Arc<Device>,
        image_index: usize,
        scene: &g_objects::Scene,
        model_manager: &g_objects::ModelManager,
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
                float32: [0.0, 0.0, 0.8, 1.0],
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

        scene.draw_instanced_models(image_index, model_manager, logical_device, command_buffer)?;

        // scene.draw_instanced_models(image_index, model_manager, logical_device, command_buffer)?;

        logical_device.cmd_end_render_pass(command_buffer);

        logical_device.end_command_buffer(command_buffer)?;
        Ok(())
    }

    // unsafe fn update_secondary_command_buffer(
    //     &mut self,
    //     logical_device: Arc<Device>,
    //     image_index: usize,
    //     model_index: usize,
    //     model_name: &str,
    // ) -> Result<vk::CommandBuffer> {
    //     let command_buffers = &mut self.presenter.secondary_command_buffers[image_index];
    //     while model_index >= command_buffers.len() {
    //         println!("Allocating new secondary command buffer");
    //         let allocate_info = vk::CommandBufferAllocateInfo::builder()
    //             .command_pool(self.presenter.command_pool_sets[image_index].graphics)
    //             .level(vk::CommandBufferLevel::SECONDARY)
    //             .command_buffer_count(1);

    //         let command_buffer = logical_device.allocate_command_buffers(&allocate_info)?[0];
    //         command_buffers.push(command_buffer);
    //     }

    //     let command_buffer = command_buffers[model_index];

    //     let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
    //         .render_pass(self.pipeline.render_pass)
    //         .subpass(0)
    //         .framebuffer(self.presenter.framebuffers[image_index]);

    //     let info = vk::CommandBufferBeginInfo::builder()
    //         .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
    //         .inheritance_info(&inheritance_info);

    //     logical_device.begin_command_buffer(command_buffer, &info)?;

    //     let time = self.start.elapsed().as_secs_f32();

    //     let y = (((model_index % 2) as f32) * 2.5) - 1.25;
    //     let z = (((model_index / 2) as f32) * -2.0) + 1.0;

    //     logical_device.cmd_bind_pipeline(
    //         command_buffer,
    //         vk::PipelineBindPoint::GRAPHICS,
    //         self.pipeline.pipeline,
    //     );

    //     logical_device.cmd_bind_descriptor_sets(
    //         command_buffer,
    //         vk::PipelineBindPoint::GRAPHICS,
    //         self.pipeline.pipeline_layout,
    //         0,
    //         &[self.presenter.descriptor_sets[image_index]],
    //         &[],
    //     );

    //     // for (_name, model) in self.model_manager.models.iter() {
    //     let model = model_manager.models.get_mut(model_name).unwrap();
    //     model.set_position(g_types::vec3(0.0, y, z));

    //     logical_device.cmd_bind_vertex_buffers(
    //         command_buffer,
    //         0,
    //         &[model.vertex_buffer.get_buffer()],
    //         &[0],
    //     );

    //     logical_device.cmd_bind_index_buffer(
    //         command_buffer,
    //         model.index_buffer.get_buffer(),
    //         0,
    //         vk::IndexType::UINT32,
    //     );

    //     let model_mat = g_types::Mat4::from_translation(model.position)
    //         * g_types::Mat4::from_axis_angle(
    //             g_types::vec3(0.0, 0.0, 1.0),
    //             g_types::Deg(90.0) * time,
    //         );
    //     let model_bytes = std::slice::from_raw_parts(
    //         &model_mat as *const g_types::Mat4 as *const u8,
    //         size_of::<g_types::Mat4>(),
    //     );

    //     logical_device.cmd_push_constants(
    //         command_buffer,
    //         self.pipeline.pipeline_layout,
    //         vk::ShaderStageFlags::VERTEX,
    //         0,
    //         model_bytes,
    //     );

    //     logical_device.cmd_push_constants(
    //         command_buffer,
    //         self.pipeline.pipeline_layout,
    //         vk::ShaderStageFlags::FRAGMENT,
    //         64,
    //         &[
    //             (model_index % 2).to_ne_bytes(),
    //             (model_index % 2).to_ne_bytes(),
    //         ]
    //         .concat(),
    //     );

    //     logical_device.cmd_draw_indexed(
    //         command_buffer,
    //         model.index_buffer.get_indice_count(),
    //         1,
    //         0,
    //         0,
    //         0,
    //     );
    //     // }

    //     logical_device.end_command_buffer(command_buffer)?;

    //     Ok(command_buffer)
    // }

    pub unsafe fn destroy(&mut self, logical_device: Arc<Device>, instance: Arc<Instance>) {
        // self.model_manager.destroy(logical_device);
        // self.texture_engine.destroy(logical_device);
        self.presenter.destroy(logical_device);
        self.pipeline.destroy(logical_device);
        self.swapchain.destroy(logical_device);
        instance.destroy_surface_khr(self.surface, None);
    }
}
