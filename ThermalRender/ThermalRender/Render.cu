#include "Render.cuh"
#include "Mesh.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
struct MeshInfo {
	float* d_v, * d_n, *d_uv;
	Object* d_o;

	__host__ void init(float* v, float* n, float* uv, Object* o){
		d_v = v, d_n = n, d_uv = uv, d_o = o;
	};
};
__constant__ MeshInfo m;
__constant__ int w, h;
__constant__ curandState_t* state;

__global__ void cuRand_Setup_Kernel(int seed) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= w || y >= h) return;

	curand_init(seed, x + y * w, 0, &state[x + y * w]);
}

void initRender(int width, int height) {
	float* d_vert, * d_normal, * d_uv;
	Object* d_obj;
	initMesh(d_vert, d_normal, d_uv, d_obj);
	MeshInfo h_m;
	h_m.init(d_vert, d_normal, d_uv, d_obj);
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

}


__global__ void RayTracingKernel(unsigned int* d_pbo) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= w || y >= h) return;

	curandState_t localState = state[x + y * w];//use register for local efficiency
	d_pbo[x + y * w] = unsigned(255.f * curand_uniform(&localState));
	state[x + y * w] = localState;
}

void render(unsigned int* d_pbo, int _w, int _h)
{
	dim3 blockDim(16, 16, 1), gridDim(_w / blockDim.x + 1, _h / blockDim.y + 1, 1);
	RayTracingKernel << <gridDim, blockDim >> > (d_pbo);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
}
