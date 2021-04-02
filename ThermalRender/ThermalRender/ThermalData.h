#pragma once
#include <GL/glew.h>
#include "Mesh.h"

const unsigned int TOTAL_WAVES = 11;
void FetchThermalData(int mat, float temp, Object &obj);
Wave GetSky();