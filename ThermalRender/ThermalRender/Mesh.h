#pragma once
#include <vector>

struct Object {
	unsigned* d_idx;
};

void initMesh(float*& d_vert, float*& d_normal, float*& d_uv, std::vector<Object> h_obj);

