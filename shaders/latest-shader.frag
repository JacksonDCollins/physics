#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(push_constant) uniform PushConstants {
    layout(offset = 64) float opacity;
} pcs;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    // outColor = vec4(fragColor, 1.0);
    // outColor = vec4(fragTexCoord, 0.0, 1.0);
    outColor = vec4(texture(texSampler, fragTexCoord).rgb, pcs.opacity);}