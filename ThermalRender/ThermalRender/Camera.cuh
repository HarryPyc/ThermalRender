#include "cuda.h"
#include <cuda_runtime.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_CUDA
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
	glm::vec3 pos;
	__host__ void init(glm::vec3 position, glm::vec3 target, glm::vec3 up) {
		pos = position;
		P = glm::perspective(glm::radians(60.f), 4.f / 3.f, 0.01f, 100.f);
		V = glm::lookAt(pos, target, up);
		invPV = glm::inverse(P * V);
	}
	__device__ glm::vec3 UnProject(float u, float v) {
		u = u * 2.f - 1.f, v = v * 2.f - 1.f;
		glm::vec4 p(u, v, 0.f, 1.0f);
		p = invPV * p;
		p /= p.w;
		return glm::normalize(glm::vec3(p) - pos);
	}
private:
	glm::mat4 P, V, invPV;
};