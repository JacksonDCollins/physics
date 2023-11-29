#version 450
#extension GL_EXT_debug_printf : enable

struct Vertex {
  vec3 position;
  vec3 normal;
  vec2 texcoord;
};

layout( binding = 0) buffer VertexSSBOIn {
   Vertex VertexesIn[ ];
};

// layout(std140, binding = 1) buffer VertexSSBOOut {
//    Vertex VertexesOut[ ];
// };

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
   uint index = gl_GlobalInvocationID.x;  

   // debugPrintfEXT("index = %d\n", index);


   // debugPrintfEXT("leng = %d\n", VertexesIn.length());


   
   // Vertex vertexIn = VertexesIn[index];
   // vertexIn.position += vec3(0.0, 0.0, 1.0);

   // VertexesOut[index] = vertexIn;
   // VertexesOut[index].position += vec3(0.0, 0.0, 0.0001);


   
   VertexesIn[0].position += vec3(0.0001, 0.0001, 0.0001);
   VertexesIn[1].position += vec3(0.0001, 0.0001, 0.0001);
   VertexesIn[2].position += vec3(0.0001, 0.0001, 0.0001);
   VertexesIn[3].position += vec3(0.0001, 0.0001, 0.0001);
   VertexesIn[4].position += vec3(0.0001, 0.0001, 0.0001);
   VertexesIn[5].position += vec3(0.0001, 0.0001, 0.0001);
   VertexesIn[6].position += vec3(0.0001, 0.0001, 0.0001);

   
   



   // debugPrintfEXT("vertex.position = %f %f %f\n", vertex.position.x, vertex.position.y, vertex.position.z);
}
