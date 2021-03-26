#include "Mesh.h"
#include <string>
#include <ObjLoad.h>
#include "Render.cuh"
#include "ThermalData.h"

MeshInfo initMesh(std::vector<Object>& h_obj) {
	float* d_vert, * d_normal, * d_uv;
	MeshInfo info;

	const std::string mesh_path = "asset/model/human_and_car01.obj";
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
	
	//Upload Index & Object info
	for (auto object : data.faces) {
		if (object.first == "default")
			continue;
		glm::vec3 minAABB, maxAABB;
		memcpy(&minAABB[0], &data.AABB[object.first][0], 3 * sizeof(float));
		memcpy(&maxAABB[0], &data.AABB[object.first][3], 3 * sizeof(float));
		Object obj;

		std::string name = object.first;
		int mat = name[0] - '0';
		float temp = (name[2] - '0') * 10.f + (name[3] - '0');
		FetchThermalData(mat, temp, obj);
		if (mat == 6 || mat == 9)
			obj.useTex = true;
		else
			obj.useTex = false;

		obj.minAABB = minAABB - 0.01f;
		obj.maxAABB = maxAABB + 0.01f;
		obj.N = object.second.size();
		obj.refl_type = 1;//Diffuse

		size_t idx_size = object.second.size() * sizeof(unsigned int);
		gpuErrchk(cudaMalloc((void**)&obj.d_idx, idx_size));
		gpuErrchk(cudaMemcpy(obj.d_idx, object.second.data(), idx_size, cudaMemcpyHostToDevice));

		h_obj.push_back(obj);
	}
	h_obj[7].useTex = false;

	info.init(d_vert, d_normal, d_uv, h_obj.size());
	return info;
}