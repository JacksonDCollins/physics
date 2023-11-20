#version 450
layout(push_constant) uniform PushConstants {
    layout(offset = 0) mat4 view;
    layout(offset = 64) mat4 proj;
} pcs;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 3) in mat4 model;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;

void main () {

    gl_Position = pcs.proj * pcs.view * model *  vec4(inPosition.xy, inPosition.z + 10.0, 1.0);
    fragNormal = inNormal;
    fragTexCoord = inTexCoord;

}
