#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "cuda_gl_interop.h"
#include "Render.cuh"

const int WINDOW_WIDTH = 640, WINDOW_HEIGHT = 480;
const int TargetSample = 2048;

int main() {
	initOpenGL();
	initRender(WINDOW_WIDTH, WINDOW_HEIGHT);

	h_data = new float[11 * WINDOW_WIDTH * WINDOW_HEIGHT];
	gpuErrchk(cudaMemcpy(&h_data[0], &d_data[0], data_size * sizeof(float), cudaMemcpyDeviceToHost));

	std::string name = "output/Human&Car_" + std::to_string(TargetSample) + ".txt";
	std::ofstream outFile(name);
	for (int y = 0; y < WINDOW_HEIGHT; y++) {
		for (int x = 0; x < WINDOW_WIDTH; x++) {
			float* ptr = h_data + 11 * (y * WINDOW_WIDTH + x);
			for (int i = 0; i < 11; i++) {
				outFile << *(ptr + i);
				outFile << " ";
			}
			//outFile << "\t";
		
		}
		outFile << "\n";
	}
	outFile.close();
	gpuErrchk(cudaFree(d_data));
	delete[] h_data;

	return 0;
}