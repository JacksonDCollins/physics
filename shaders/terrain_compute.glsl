#version 450
// #extension GL_EXT_debug_printf : enable

struct Vertex {
  vec3 position;
  vec3 normal;
  vec2 texcoord;
};

layout(std140, binding = 0) buffer VertexSSBOIn {
   Vertex VertexesIn[ ];
};

// layout(std140, binding = 1) buffer VertexSSBOOut {
//    Vertex VertexesOut[ ];
// };

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
   uint index = gl_GlobalInvocationID.x;  

   // VertexesOut[gl_GlobalInvocationID.x] = VertexesIn[gl_GlobalInvocationID.x];
   Vertex vertex = VertexesIn[index];
   vertex.position = vec3(0.0, 100.0, 0.0);

   // printf("My float is %f", vertex.position.x);
}
