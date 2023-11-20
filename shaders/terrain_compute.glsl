#version 450

struct Vertex {
  vec3 position;
  vec3 normal;
  vec2 texcoord;
};

layout(std140, binding = 1) readonly buffer VertexSSBOIn {
   Vertex VertexesIn[ ];
};

layout(std140, binding = 2) buffer VertexSSBOOut {
   Vertex VertexesOut[ ];


};


void main() {
   VertexesOut[gl_GlobalInvocationID.x] = VertexesIn[gl_GlobalInvocationID.x];
}
