#include "Mesh.h"
#include <string>
#include <vector>
#include <ObjLoad.h>
#include "Render.cuh"

void initMesh(float*& d_vert, float*& d_normal, float*& d_uv, Object*& d_obj) {
	const std::string mesh_path = "asset/model/scene.obj";
	obj::Model data = obj::loadModelFromFile(mesh_path);
	//Upload Vertex
	size_t vert_size = data.vertex.size() * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&d_vert, vert_size));
	gpuErrchk(cudaMemcpy(d_vert, data.vertex.data(), vert_size, cudaMemcpyHostToDevice));
	//Upload Normal
	size_t normal_size = data.normal.size() * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&d_normal, normal_size));
	gpuErrchk(cudaMemcpy(d_normal, data.normal.data(), normal_size, cudaMemcpyHostToDevice));
	//Upload TexCoord
	size_t uv_size = data.texCoord.size() * sizeof(float);
	gpuErrchk(cudaMalloc((void**)&d_uv, uv_size));
	gpuErrchk(cudaMemcpy(d_uv, data.texCoord.data(), uv_size, cudaMemcpyHostToDevice));
	//Upload Index
	std::vector<Object> h_obj;
	for (auto object : data.faces) {
		if (object.first == "default")
			continue;
		glm::vec3 minAABB, maxAABB;
		memcpy(&minAABB[0], &data.AABB[object.first][0], 3 * sizeof(float));
		memcpy(&maxAABB[0], &data.AABB[object.first][3], 3 * sizeof(float));
		Object obj(0x00FF0000, minAABB - 0.01f, maxAABB + 0.01f);

		size_t idx_size = object.second.size() * sizeof(unsigned);
		gpuErrchk(cudaMalloc((void**)&obj.d_idx, idx_size));
		gpuErrchk(cudaMemcpy(obj.d_idx, object.second.data(), idx_size, cudaMemcpyHostToDevice));

		h_obj.push_back(obj);
	}
	gpuErrchk(cudaMalloc((void**)&d_obj, h_obj.size() * sizeof(Object)));
	gpuErrchk(cudaMemcpy(d_obj, h_obj.data(), h_obj.size() * sizeof(Object), cudaMemcpyHostToDevice));
}