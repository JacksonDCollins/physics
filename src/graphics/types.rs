use anyhow::Result;
use ash::vk;
use cgmath::SquareMatrix;
pub use cgmath::{point3, vec2, vec3, Deg};
use serde::{de::Visitor, Deserialize, Deserializer, Serialize};
use std::hash::{Hash, Hasher};

pub type Vec2 = cgmath::Vector2<f32>;
pub type Vec3 = cgmath::Vector3<f32>;
pub type Vec4 = cgmath::Vector4<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;
pub type Point3 = cgmath::Point3<f32>;
pub type Degf32 = cgmath::Deg<f32>;

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Serialize)]
pub struct Vertex {
    pub pos: Vec3,
    _padding: u32,
    normal: Vec3,
    _padding2: u32,
    tex_coord: Vec2,
}

impl Vertex {
    pub const fn new(pos: Vec3, normal: Vec3, tex_coord: Vec2) -> Self {
        Self {
            pos,
            _padding: 0,
            normal,
            _padding2: 0,
            tex_coord,
        }
    }

    pub fn offset_at(location: u32) -> u32 {
        match location {
            0 => 0,
            1 => std::mem::size_of::<Vec3>() as u32 + std::mem::size_of::<u32>() as u32,
            2 => std::mem::size_of::<Vec3>() as u32 * 2 + std::mem::size_of::<u32>() as u32 * 2,
            _ => panic!("Invalid location: {}", location),
        }
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.normal == other.normal && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.normal[0].to_bits().hash(state);
        self.normal[1].to_bits().hash(state);
        self.normal[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}

#[derive(Clone, Debug, Default, Copy)]
pub struct CommandPoolSet {
    // data: HashMap<u32, vk::CommandPool>,
    pub present: vk::CommandPool,
    pub graphics: vk::CommandPool,
    pub transfer: vk::CommandPool,
    pub compute: vk::CommandPool,
}

impl CommandPoolSet {
    pub fn create(
        present: vk::CommandPool,
        graphics: vk::CommandPool,
        transfer: vk::CommandPool,
        compute: vk::CommandPool,
    ) -> Self {
        Self {
            present,
            graphics,
            transfer,
            compute,
        }
    }

    pub fn destroy(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.destroy_command_pool(self.present, None);
            logical_device.destroy_command_pool(self.graphics, None);
            logical_device.destroy_command_pool(self.transfer, None);
            logical_device.destroy_command_pool(self.compute, None);
        }
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    // pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

impl Default for UniformBufferObject {
    fn default() -> Self {
        Self {
            // model: Mat4::identity(),
            view: Mat4::identity(),
            proj: Mat4::identity(),
        }
    }
}

pub trait BindingDescription {
    fn binding_description(
        binding: u32,
        rate: vk::VertexInputRate,
    ) -> vk::VertexInputBindingDescription;
}

impl BindingDescription for Mat4 {
    fn binding_description(
        binding: u32,
        rate: vk::VertexInputRate,
    ) -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(binding)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(rate)
            .build()
    }
}

impl BindingDescription for Vertex {
    fn binding_description(
        binding: u32,
        rate: vk::VertexInputRate,
    ) -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(binding)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(rate)
            .build()
    }
}

pub trait AttributeDescriptions {
    fn attribute_descriptions(
        binding: u32,
        location: u32,
    ) -> Vec<vk::VertexInputAttributeDescription>;
}

impl AttributeDescriptions for Mat4 {
    fn attribute_descriptions(
        binding: u32,
        location: u32,
    ) -> Vec<vk::VertexInputAttributeDescription> {
        let mut descriptions = Vec::new();
        for i in 0..4 {
            descriptions.push(
                vk::VertexInputAttributeDescription::builder()
                    .binding(binding)
                    .location(location + i)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .offset(i * std::mem::size_of::<Vec4>() as u32)
                    .build(),
            );
        }
        descriptions
    }
}

impl AttributeDescriptions for Vertex {
    fn attribute_descriptions(
        binding: u32,
        location: u32,
    ) -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription::builder()
                .binding(binding)
                .location(location)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(Vertex::offset_at(location))
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(binding)
                .location(location + 1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(Vertex::offset_at(location + 1))
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(binding)
                .location(location + 2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(Vertex::offset_at(location + 2))
                .build(),
        ]
    }
}

pub const DEFAULT_RENDER_INFO: RenderInfo = RenderInfo {
    model_info: render_info::ModelInfo {
        scale: render_info::Scale {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
    },
    polygon_info: render_info::PolygonInfo {
        cull_mode: vk::CullModeFlags::BACK,
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
    },
};
#[derive(Deserialize, Debug)]
pub struct RenderInfo {
    pub model_info: render_info::ModelInfo,
    pub polygon_info: render_info::PolygonInfo,
}

impl RenderInfo {
    pub fn create(path: &str) -> Result<Self> {
        if std::path::Path::new(path).exists() {
            let toml_string = std::fs::read_to_string(path).map_err(|err| {
                log::info!("Failed to read default render info: {} - {}", path, err);
                err
            })?;

            toml::from_str(&toml_string).map_err(|err| {
                log::info!(
                    "{} {}: {} Failed to parse render info: {}",
                    file!(),
                    line!(),
                    path,
                    err
                );
                err.into()
            })
        } else {
            log::info!("No render info found at {}", path);
            Ok(DEFAULT_RENDER_INFO)
        }
    }
}

pub mod render_info {
    use ash::vk;
    use serde::{Deserialize, Deserializer};

    #[derive(Deserialize, Debug)]
    pub struct ModelInfo {
        pub scale: Scale,
    }
    #[derive(Deserialize, Debug)]
    pub struct Scale {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    #[derive(Deserialize, Debug)]
    pub struct PolygonInfo {
        #[serde(deserialize_with = "deserialize_cull_flags")]
        pub cull_mode: vk::CullModeFlags,
        #[serde(deserialize_with = "deserialize_front_face")]
        pub front_face: vk::FrontFace,
    }

    fn deserialize_cull_flags<'de, D>(deserializer: D) -> Result<vk::CullModeFlags, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s: String = Deserialize::deserialize(deserializer)?;
        match s.as_str() {
            "none" => Ok(vk::CullModeFlags::NONE),
            "front" => Ok(vk::CullModeFlags::FRONT),
            "back" => Ok(vk::CullModeFlags::BACK),
            "both" => Ok(vk::CullModeFlags::FRONT_AND_BACK),
            _ => Err(serde::de::Error::custom(format!(
                "Invalid cull mode: {}",
                s
            ))),
        }
    }

    fn deserialize_front_face<'de, D>(deserializer: D) -> Result<vk::FrontFace, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s: String = Deserialize::deserialize(deserializer)?;
        match s.as_str() {
            "cw" => Ok(vk::FrontFace::CLOCKWISE),
            "ccw" => Ok(vk::FrontFace::COUNTER_CLOCKWISE),
            _ => Err(serde::de::Error::custom(format!(
                "Invalid front face: {}",
                s
            ))),
        }
    }
}
