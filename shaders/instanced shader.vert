#version 450

struct Render_Parameters {
    mat4 model;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(binding = 2) buffer RenderParameters {
    Render_Parameters pcs[];
} render_parameters;


layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    Render_Parameters pcs = render_parameters.pcs[gl_InstanceIndex];
    

    gl_Position = ubo.proj * ubo.view * pcs.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}
