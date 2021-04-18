#include <iostream>
#include "Render.cuh"
#include "Mesh.h"
#include <string>
#include <fstream>

const int WINDOW_WIDTH = 640, WINDOW_HEIGHT = 480;
cudaGraphicsResource_t pboCuda;
int Samples = 0;

int main() {
	
	initRender(WINDOW_WIDTH, WINDOW_HEIGHT);
	int TargetSample = 2048;
	//int SamplesArr[]{ 1, 256, 512, 768, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240 };
	int SamplesArr[]{ 1, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256};
	float* d_data, * h_data;
	size_t data_size = TOTAL_WAVE * WINDOW_WIDTH * WINDOW_HEIGHT;
	gpuErrchk(cudaMalloc((void**)&d_data, data_size * sizeof(float)));
	gpuErrchk(cudaMemset(d_data, 0, data_size * sizeof(float)));
	int Samples = 0;
	h_data = new float[data_size];
	for (int i = 0; i < sizeof(SamplesArr)/4; i++) {
		TargetSample = SamplesArr[i];

		while (Samples < TargetSample) {
			render(d_data, WINDOW_WIDTH, WINDOW_HEIGHT, Samples);
			printf("Sample: %i\n", Samples);
		}

		gpuErrchk(cudaMemcpy(h_data, d_data, data_size * sizeof(float), cudaMemcpyDeviceToHost));

		std::string name = "output/Human&Car_" + std::to_string(TargetSample) + "_right.txt";
		std::ofstream outFile(name);
		for (int y = 0; y < WINDOW_HEIGHT; y++) {
			for (int x = 0; x < WINDOW_WIDTH; x++) {
				float* ptr = h_data + TOTAL_WAVE * (y * WINDOW_WIDTH + x);
				for (int i = 0; i < TOTAL_WAVE; i++) {
					outFile << *(ptr + i);
					outFile << " ";
				}
				//outFile << "\t";

			}
			outFile << "\n";
		}
		outFile.close();
	}
	gpuErrchk(cudaFree(d_data));
	delete[] h_data;

	return 0;
}