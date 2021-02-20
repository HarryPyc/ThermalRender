#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "cuda_gl_interop.h"
#include "Render.cuh"

GLFWwindow* window;
const int WINDOW_WIDTH = 1280, WINDOW_HEIGHT = 720;
GLuint pbo, textureID;
cudaGraphicsResource_t pboCuda;

void printGlInfo()
{
	std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
}

void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

void initOpenGL() {
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	if (!glfwInit()) {
		std::cout << "GLFW Init Failed" << std::endl;
	}
	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Thermal Render", NULL, NULL);
	if (!window) {
		std::cout << "Window Creation Failed" << std::endl;
	}
	glfwMakeContextCurrent(window);
	if (glewInit() != GLEW_OK)
	{
		std::cout << "GLEW initialization failed.\n";
	}
	glfwSwapInterval(1);
	glfwSetErrorCallback(error_callback);

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(unsigned int), 0, GL_DYNAMIC_COPY);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	gpuErrchk(cudaGraphicsGLRegisterBuffer(&pboCuda, pbo, cudaGraphicsRegisterFlagsNone));
}

void DrawTexture() {
	glUseProgram(0);
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glEnable(GL_TEXTURE_2D);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, 0.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, -1.0f, 0.0f);
	glEnd();

	glDisable(GL_TEXTURE_2D);
}

void DrawPBO() {
	gpuErrchk(cudaGraphicsMapResources(1, &pboCuda));

	unsigned int* d_pbo;
	size_t numBytes = WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(unsigned int);
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, pboCuda));

	render(d_pbo, WINDOW_WIDTH, WINDOW_HEIGHT);

	gpuErrchk(cudaGraphicsUnmapResources(1, &pboCuda));
}

int main() {
	initOpenGL();
	initRender(WINDOW_WIDTH, WINDOW_HEIGHT);

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
		//Draw
		DrawPBO();
		DrawTexture();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	return 0;
}