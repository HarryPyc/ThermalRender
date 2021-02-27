#include "Mesh.h"
#include <string>
#include <vector>
#include <ObjLoad.h>
#include "Render.cuh"

MeshInfo initMesh() {
	float* d_vert, * d_normal, * d_uv;
	Object* d_obj;
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
	std::vector<Object> h_obj;
	for (auto object : data.faces) {
		if (object.first == "default")
			continue;
		glm::vec3 minAABB, maxAABB;
		memcpy(&minAABB[0], &data.AABB[object.first][0], 3 * sizeof(float));
		memcpy(&maxAABB[0], &data.AABB[object.first][3], 3 * sizeof(float));
		Object obj;
		obj.color = glm::vec3(0.5f);
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
	h_obj[5].color = glm::vec3(0.75f, 0.25f, 0.25f);
	//h_obj[1].color = glm::vec3(0.25f, 0.75f, 0.25f);
	////h_obj[2].Refl = 0;
	//h_obj[2].color = glm::vec3(0.75f);
	gpuErrchk(cudaMalloc((void**)&d_obj, h_obj.size() * sizeof(Object)));
	gpuErrchk(cudaMemcpy(d_obj, h_obj.data(), h_obj.size() * sizeof(Object), cudaMemcpyHostToDevice));

	info.init(d_vert, d_normal, d_uv, d_obj, h_obj.size());
	return info;
}