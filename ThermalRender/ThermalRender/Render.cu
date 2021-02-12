#include "Render.cuh"
#include "Mesh.h"
#include "device_launch_parameters.h"

float* d_vert, * d_normal, * d_uv;
__constant__ Object* d_obj;
__constant__ int w, h;

void initRender() {
	std::vector<Object> h_obj;
	initMesh(d_vert, d_normal, d_uv, h_obj);
	gpuErrchk(cudaMemcpyToSymbol(*d_obj, h_obj.data(), h_obj.size() * sizeof(Object)));

}

__global__ void RayTracingKernel(unsigned int* d_pbo) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= w || y >= h) return;

	d_pbo[x + y * w] = 0x000000FF;
}

void render(unsigned int* d_pbo, int _w, int _h)
{
	gpuErrchk(cudaMemcpyToSymbol(w, &_w, sizeof(unsigned)));
	gpuErrchk(cudaMemcpyToSymbol(h, &_h, sizeof(unsigned)));

	dim3 blockDim(16, 16, 1), gridDim(_w / blockDim.x + 1, _h / blockDim.y + 1, 1);
	RayTracingKernel << <gridDim, blockDim >> > (d_pbo);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());


}
