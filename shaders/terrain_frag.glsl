#version 450

// layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    highp float pixelXPos = gl_FragCoord.x * fragNormal.x;
    highp float pixelYPos = gl_FragCoord.y * fragNormal.y;
    // outColor = texture(texSampler, fragTexCoord);
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
