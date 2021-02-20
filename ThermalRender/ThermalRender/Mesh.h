#pragma once
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>


struct Object {
	unsigned int  color;
	glm::vec3 minAABB, maxAABB;
	unsigned* d_idx;

	Object() {};
	Object(unsigned c, glm::vec3 _minAABB, glm::vec3 _maxAABB) 
	: color(c), minAABB(_minAABB), maxAABB(_maxAABB) {};
};

void initMesh(float*& d_vert, float*& d_normal, float*& d_uv, Object*& d_obj);

