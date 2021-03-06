#pragma once
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

struct Wave {
	float w[11];
	__host__ __device__ inline Wave operator*(const Wave& other) const{
		Wave res;
		for (int i = 0; i < 11; i++) {
			res[i] = w[i] * other.w[i];
		}
		return res;
	}
	__host__ __device__ inline Wave operator*(const float& f) const {
		Wave res;
		for (int i = 0; i < 11; i++) {
			res[i] = w[i] * f;
		}
		return res;
	}
	__host__ __device__ inline Wave operator/(const float& f) const {
		Wave res;
		for (int i = 0; i < 11; i++) {
			res[i] = w[i] / f;
		}
		return res;
	}
	__host__ __device__ inline Wave operator+(const Wave& other) const{
		Wave res;
		for (int i = 0; i < 11; i++) {
			res[i] = w[i] + other.w[i];
		}
		return res;
	}
	__host__ __device__ inline float& operator[](const unsigned& i) {
		return w[i];
	}
	__host__ __device__ static Wave GetWave(const float& val) {
		Wave w;
		memset(&w[0], val, 11 * sizeof(float));
		return w;
	}
};
struct Object {
	glm::vec3 color;
	unsigned int N;//index count
	Wave emis, refl;
	unsigned int refl_type;//Material Reflection Type
	glm::vec3 minAABB;
	unsigned* d_idx;
	glm::vec3 maxAABB;

};

struct MeshInfo {
	float* d_v, * d_n, * d_uv;
	int N;//Object Count;

	void init(float* v, float* n, float* uv, int _N) {
		d_v = v, d_n = n, d_uv = uv;
		N = _N;
	};
};



MeshInfo initMesh(std::vector<Object>& h_obj);

