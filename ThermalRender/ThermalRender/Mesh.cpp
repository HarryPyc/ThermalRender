#include "Mesh.h"
#include <string>
#include <ObjLoad.h>
#include "Render.cuh"

MeshInfo initMesh(std::vector<Object>& h_obj) {
	float* d_vert, * d_normal, * d_uv;
	MeshInfo info;

	const std::string mesh_path = "asset/model/human_and_car.obj";
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

	for (auto object : data.faces) {
		if (object.first == "default")
			continue;
		glm::vec3 minAABB, maxAABB;
		memcpy(&minAABB[0], &data.AABB[object.first][0], 3 * sizeof(float));
		memcpy(&maxAABB[0], &data.AABB[object.first][3], 3 * sizeof(float));
		Object obj;
		if (object.first[0] == '9' || object.first[0] == '6') {
			obj.color = glm::vec3(0.5f);
			obj.useTex = true;
		}
		else {
			obj.color = glm::vec3(0.5f);
			obj.useTex = false;
		}
		obj.emission = glm::vec3(0.f);
		obj.minAABB = minAABB - 0.01f;
		obj.maxAABB = maxAABB + 0.01f;
		obj.N = object.second.size();
		obj.Refl = 1;

		size_t idx_size = object.second.size() * sizeof(unsigned int);
		gpuErrchk(cudaMalloc((void**)&obj.d_idx, idx_size));
		gpuErrchk(cudaMemcpy(obj.d_idx, object.second.data(), idx_size, cudaMemcpyHostToDevice));

		h_obj.push_back(obj);
	}
	

	info.init(d_vert, d_normal, d_uv, h_obj.size());
	return info;
}