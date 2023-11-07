#version 450

layout(binding = 1) uniform sampler2D texSampler;
// layout(binding = 2) uniform texture2D tex;

// layout(push_constant) uniform PushConstants {
//     layout(offset = 64) int textureIdx;
//     layout(offset = 72) int samplerIdx;
// } pcs;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, fragTexCoord);
    // outColor = vec4(fragColor * texture(texSampler, fragTexCoord).rgb, 1.0);
    // outColor = texture(sampler2D(tex, texSampler), fragTexCoord);
    // outColor.a = 0.5;
}
