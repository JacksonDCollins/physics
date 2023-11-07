#version 450
#extension GL_EXT_debug_printf : enable


layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
} pcs;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {

    // debugPrintfEXT("pcs: %f %f %f %f\n", pcs.model[0][0], pcs.model[0][1], pcs.model[0][2], pcs.model[0][3]);


    gl_Position = ubo.proj * ubo.view * pcs.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}
