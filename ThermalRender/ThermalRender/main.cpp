
#include <iostream>
#include "Render.cuh"
#include <string>
#include <fstream>

GLFWwindow* window;
const int WINDOW_WIDTH = 1280, WINDOW_HEIGHT = 720;
GLuint pbo, textureID;
cudaGraphicsResource_t pboCuda;
int Samples = 0;

int main() {
	
	initRender(WINDOW_WIDTH, WINDOW_HEIGHT);
	int Samples = 0;
	float* d_data, *h_data;
	size_t data_size = 11 * WINDOW_WIDTH * WINDOW_HEIGHT;
	gpuErrchk(cudaMalloc((void**)&d_data, data_size * sizeof(float)));
	while (Samples < TargetSample) {
		render(d_data, WINDOW_WIDTH, WINDOW_HEIGHT, Samples);
		printf("Sample: %i\n", Samples);
	}

void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

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