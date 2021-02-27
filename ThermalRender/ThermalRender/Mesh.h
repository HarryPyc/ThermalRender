#pragma once
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct Object {
	glm::vec3 color;
	unsigned int N;//index count
	glm::vec3 emission;
	unsigned int Refl;//Material Reflection Type
	glm::vec3 minAABB;
	unsigned* d_idx;
	glm::vec3 maxAABB;

};

struct MeshInfo {
	float* d_v, * d_n, * d_uv;
	Object* d_o;
	int N;//Object Count;

	void init(float* v, float* n, float* uv, Object* o, int _N) {
		d_v = v, d_n = n, d_uv = uv, d_o = o;
		N = _N;
	};
};



MeshInfo initMesh();

