#include "Render.cuh"
#include "Mesh.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include "Camera.cuh"
#include <glm/gtc/constants.hpp>
#include "ThermalData.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using namespace glm;

texture<float4, cudaTextureType2D, cudaReadModeElementType> emisMap, normalMap;

__constant__ MeshInfo m;
__constant__ Object objList[10];
__constant__ Wave wave_sky, wave_zero;
__constant__ int w, h, MAX_DEPTH = 1, d_Samples;
__constant__ curandState_t* state;
__constant__ Camera cam;
__constant__ float EPSILON = 1e-4, float PI;

__global__ void cuRand_Setup_Kernel(int seed) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= w || y >= h) return;

	curand_init(seed, x + y * w, 0, &state[x + y * w]);
}

void initTexture(textureReference& tex, const char* path) {
	int w, h, comp;
	float* h_image = stbi_loadf(path, &w, &h, &comp, 4);

	cudaChannelFormatDesc format = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	
	cudaArray_t cuArray;
	gpuErrchk(cudaMallocArray(&cuArray, &format, w, h));
	gpuErrchk(cudaMemcpyToArray(cuArray, 0, 0, h_image, w * h * 4 * sizeof(float), cudaMemcpyHostToDevice));

	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = true;

	gpuErrchk(cudaBindTextureToArray(&tex, cuArray, &format));
	delete[] h_image;
}

void initRender(int width, int height) {
	std::vector<Object> h_obj;
	MeshInfo h_m = initMesh(h_obj);
	gpuErrchk(cudaMemcpyToSymbol(objList, h_obj.data(), h_obj.size() * sizeof(Object)));
	gpuErrchk(cudaMemcpyToSymbol(m, &h_m, sizeof(MeshInfo)));

	gpuErrchk(cudaMemcpyToSymbol(w, &width, sizeof(unsigned)));
	gpuErrchk(cudaMemcpyToSymbol(h, &height, sizeof(unsigned)));

	curandState_t* d_randState;
	gpuErrchk(cudaMalloc((void**)&d_randState, width * height * sizeof(curandState_t)));
	gpuErrchk(cudaMemcpyToSymbol(state, &d_randState, sizeof(d_randState)));

	srand(time(0));
	int seed = rand();
	dim3 blockDim(16, 16, 1), gridDim(width / blockDim.x + 1, height / blockDim.y + 1, 1);
	cuRand_Setup_Kernel << < gridDim, blockDim >> > (seed);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//Setup Camera
	Camera h_cam;
	h_cam.init(vec3(-1.f, 1.f, 1.f), vec3(0.f, 0.f, -5.f), vec3(0.f, 1.0f, 0.f));
	gpuErrchk(cudaMemcpyToSymbol(cam, &h_cam, sizeof(Camera)));

	float h_pi = pi<float>();
	gpuErrchk(cudaMemcpyToSymbol(PI, &h_pi, sizeof(float)));
	gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 8));

	initTexture(normalMap, "asset/texture/NormalMap.jpg");
	initTexture(emisMap, "asset/texture/cube_emi.jpg");
}
struct Ray {
	vec3 o, d, invD;

	__device__ Ray(vec3 origin, vec3 dir) : o(origin), d(dir) {
		invD = 1.f / d;
	};
	__device__ bool RayTriangleIntersection(const vec3& v0, const vec3& v1, const vec3& v2, float &t, float& u, float& v) {
		vec3 edge1, edge2, h, s, q;
		float a, f;
		edge1 = v1 - v0;
		edge2 = v2 - v0;
		h = cross(d, edge2);
		a = dot(edge1, h);
		if (a > -EPSILON && a < EPSILON)
			return false;    // This ray is parallel to this triangle.
		f = 1.0 / a;
		s = o - v0;
		u = f * dot(s, h);
		if (u < 0.0 || u > 1.0)
			return false;
		q = cross(s, edge1);
		v = f * dot(d, q);
		if (v < 0.0 || u + v > 1.0)
			return false;
		// At this stage we can compute t to find out where the intersection point is on the line.
		t = f * dot(edge2, q);
		return t > EPSILON;
	}
	__device__ bool RayAABBIntersection(const glm::vec3& minAABB, const glm::vec3& maxAABB) {
		glm::vec3 t0s = (minAABB - o) * invD;
		glm::vec3 t1s = (maxAABB - o) * invD;

		glm::vec3 tsmaller = glm::min(t0s, t1s);
		glm::vec3 tbigger = glm::max(t0s, t1s);

		float tmin = glm::max(-9999.f, glm::max(tsmaller[0], glm::max(tsmaller[1], tsmaller[2])));
		float tmax = glm::min(9999.f, glm::min(tbigger[0], glm::min(tbigger[1], tbigger[2])));
		//t = (tmin + tmax) / 2.f;
		return (tmin < tmax) && tmax > 0.f;
	}
};

__device__ void FetchMesh(vec3& n, vec2& uv, int A, int B, int C, float u, float v) {
	
	vec3 n0(m.d_n[3 * A], m.d_n[3 * A + 1], m.d_n[3 * A + 2]), n1(m.d_n[3 * B], m.d_n[3 * B + 1], m.d_n[3 * B + 2]),
		n2(m.d_n[3 * C], m.d_n[3 * C + 1], m.d_n[3 * C + 2]);
	vec2 uv0(m.d_uv[2 * A], m.d_uv[2 * A + 1]), uv1(m.d_uv[2 * B], m.d_uv[2 * B + 1]), uv2(m.d_uv[2 * C], m.d_uv[2 * C + 1]);
	n = (1.f - u - v) * n0 + u * n1 + v * n2;
	n = normalize(n);

	uv = (1.f - u - v) * uv0 + u * uv1 + v * uv2;
	uv.y = 1.0f - uv.y;
}

__device__ mat3 TBN(int A, int B, int C) {
	vec3 v0(m.d_v[3 * A], m.d_v[3 * A + 1], m.d_v[3 * A + 2]), v1(m.d_v[3 * B], m.d_v[3 * B + 1], m.d_v[3 * B + 2]),
		v2(m.d_v[3 * C], m.d_v[3 * C + 1], m.d_v[3 * C + 2]);
	vec2 uv0(m.d_uv[2 * A], m.d_uv[2 * A + 1]), uv1(m.d_uv[2 * B], m.d_uv[2 * B + 1]), uv2(m.d_uv[2 * C], m.d_uv[2 * C + 1]);
	vec3 e0 = v1 - v0, e1 = v2 - v0;
	vec2 deltaUV0 = uv1 - uv0, deltaUV1 = uv2 - uv0;

	vec3 t, b, n = normalize(cross(e0, e1));
	float f = 1.0f / (deltaUV0.x * deltaUV1.y - deltaUV1.x * deltaUV0.y);

	t.x = f * (deltaUV1.y * e0.x - deltaUV0.y * e1.x);
	t.y = f * (deltaUV1.y * e0.y - deltaUV0.y * e1.y);
	t.z = f * (deltaUV1.y * e0.z - deltaUV0.y * e1.z);

	b.x = f * (-deltaUV1.x * e0.x + deltaUV0.x * e1.x);
	b.y = f * (-deltaUV1.x * e0.y + deltaUV0.x * e1.y);
	b.z = f * (-deltaUV1.x * e0.z + deltaUV0.x * e1.z);

	t = normalize(t);
	b = normalize(b);
	return mat3(t, b, n);
}

__device__ vec3 trace(Ray ray, int depth, curandState_t& state) {
	if (depth > MAX_DEPTH) return vec3(EPSILON);
	float u, v, t = 9999.f;
	int A, B, C, i_obj = -1;
	//Find the nearest triangle
	for (int i = 0; i < m.N; i++) {
		Object obj = objList[i];
		if (ray.RayAABBIntersection(obj.minAABB, obj.maxAABB)) {
			for (int j = 0; j < obj.N / 3; j++) {
				unsigned int idx0 = obj.d_idx[3 * j], idx1 = obj.d_idx[3 * j + 1], idx2 = obj.d_idx[3 * j + 2];
				vec3 v0(m.d_v[3 * idx0], m.d_v[3 * idx0 + 1], m.d_v[3 * idx0 + 2]), v1(m.d_v[3 * idx1], m.d_v[3 * idx1 + 1], m.d_v[3 * idx1 + 2]),
					v2(m.d_v[3 * idx2], m.d_v[3 * idx2 + 1], m.d_v[3 * idx2 + 2]);
				float _t, _u, _v;
				if (ray.RayTriangleIntersection(v0, v1, v2, _t, _u, _v) &&  _t < t) {
					t = _t, u = _u, v = _v;
					A = idx0, B = idx1, C = idx2;
					i_obj = i;
				}
			}
		}
	}

	if (i_obj == -1) return wave_zero;

	//Fetch vertex position, normal and texture coordinates
	const Object obj = objList[i_obj];
	vec3 p, n; vec2 uv;
	FetchMesh(n, uv, A, B, C, u, v);

	if (obj.useTex) {
		float4 n_sample = tex2D(normalMap, uv.x, uv.y);
		//float4 c_sample = tex2D(emisMap, uv.x, uv.y);
		vec3 nt;
		memcpy(&nt[0], &n_sample, 3 * sizeof(float));
		//memcpy(&color[0], &c_sample, 3 * sizeof(float));
		nt = normalize(nt * 2.0f - 1.0f);
		n = TBN(A, B, C) * nt;
	}
	p = ray.o + ray.d * t + EPSILON * n;

	if (obj.Refl == 0)//Specular
	{
		vec3 r = reflect(ray.d, n);
		return obj.emission + obj.color * trace(Ray(p, r), depth + 1, state);
	}
	else if (obj.Refl == 1) //Diffuse;
	{
		vec3 a = normalize(abs(n.x) < 1 - EPSILON ? cross(vec3(1, 0, 0), n) : cross(vec3(0, 1, 0), n)), b = cross(a, n);
		float alpha = 2.f * PI * curand_uniform(&state), beta = curand_uniform(&state);
		vec3 newDir = (glm::cos(alpha ) * a + glm::sin(alpha) * b) * sqrt(1.f - beta * beta) + beta * n;

		//if (depth == 0)
		//	return obj.refl * trace(Ray(p, newDir), depth + 1, state); //only reflection
		return obj.emis + obj.refl * trace(Ray(p, newDir), depth + 1, state);

	}

}


__global__ void RayTracingKernel(float* d_pbo) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= w || y >= h) return;
	const int idx = x + y * w;
	curandState_t localState = state[idx];//use register for local efficiency
	
	float u = float(x) + curand_uniform(&localState), v = float(y) + curand_uniform(&localState);//Anti-alising
	Ray ray(cam.pos, cam.UnProject(u / float(w), v / float(h)));
	vec3 c = trace(ray, 0, localState);
	//c = pow(c, vec3(1.f / 2.2f));//Gamma Correction

	vec3 preColor;
	memcpy(&preColor[0], d_pbo + 4 * idx, 3 * sizeof(float));
	c = (preColor * float(d_Samples - 1) + c) / float(d_Samples);
	memcpy(d_pbo + 4 * idx, &c[0], 3 * sizeof(float));
	state[idx] = localState;
}

void render(float* d_pbo, int _w, int _h, int& h_Samples)
{
	h_Samples++; 
	gpuErrchk(cudaMemcpyToSymbol(d_Samples, &h_Samples, sizeof(int)));
	dim3 blockDim(16, 16, 1), gridDim(_w / blockDim.x + 1, _h / blockDim.y + 1, 1);
	RayTracingKernel << <gridDim, blockDim >> > (d_pbo);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}
