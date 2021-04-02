#include "ThermalData.h"

const float Wave[] = {
   7.8576538e+02, 8.1770000e+02, 8.6250000e+02, 9.1025000e+02,
   9.4255000e+02, 9.7750000e+02, 1.0277500e+03, 1.0780000e+03,
   1.1255000e+03, 1.1860000e+03, 1.2766667e+03
};

const float Emissivity[11][11] = {
	//'human'0,       'marble'1,       'paint'2,       'glass'3,       'rubber'4,      'brass'5,        'road'6,        'al'7,         'al2o3'8,       'brick'9      'blackbody' 10    
	 9.5000000e-01,  9.5834758e-01,  8.7470001e-01,  5.0455443e-01,  9.2789246e-01,  1.2250251e-01,  9.6426578e-01,  5.5701898e-01,  4.1617280e-02,  9.7773773e-01, 1.0f,
	 9.5000000e-01,  9.5462609e-01,  8.8365367e-01,  2.8523451e-01,  9.2827028e-01,  1.1789014e-01,  9.7194589e-01,  5.4616836e-01,  4.1602933e-02,  9.7348785e-01, 1.0f,
	 9.5000000e-01,  9.5099592e-01,  9.6279529e-01,  3.8887318e-01,  9.2640468e-01,  1.2078545e-01,  9.6430868e-01,  5.2990503e-01,  4.0821044e-02,  9.6252597e-01, 1.0f,
	 9.5000000e-01,  9.5741246e-01,  8.6909910e-01,  4.2252257e-01,  9.2027605e-01,  1.2892990e-01,  9.4494491e-01,  5.1621436e-01,  4.8036999e-02,  9.4693874e-01, 1.0f,
	 9.5000000e-01,  9.6385735e-01,  8.5889954e-01,  4.4505789e-01,  9.2317386e-01,  1.3452107e-01,  9.5513005e-01,  5.0484414e-01,  1.4619579e-01,  9.3275042e-01, 1.0f,
	 9.5000000e-01,  9.6087765e-01,  9.3344199e-01,  4.7704424e-01,  8.9968776e-01,  1.4311263e-01,  9.5631467e-01,  4.9568769e-01,  2.6974721e-01,  9.1201603e-01, 1.0f,
	 9.5000000e-01,  9.5962251e-01,  9.4205163e-01,  5.6399482e-01,  8.6774658e-01,  1.4932587e-01,  9.5258259e-01,  4.7984848e-01,  4.2480553e-01,  8.7901868e-01, 1.0f,
	 9.5000000e-01,  9.5305901e-01,  9.4627694e-01,  3.2859562e-01,  8.8061124e-01,  1.4229701e-01,  9.1783893e-01,  4.6578646e-01,  4.7823023e-01,  8.5128884e-01, 1.0f,
	 9.5000000e-01,  9.5385122e-01,  9.5199753e-01,  4.2369253e-02,  8.9911606e-01,  1.3455656e-01,  9.1771733e-01,  4.5454008e-01,  5.1389488e-01,  9.0261137e-01, 1.0f,
	 9.5000000e-01,  9.5852822e-01,  9.5649050e-01,  2.7487807e-02,  9.1817783e-01,  1.2604779e-01,  9.1884949e-01,  4.3838823e-01,  5.4462383e-01,  9.3754130e-01, 1.0f,
	 9.5000000e-01,  9.5240096e-01,  9.5069231e-01,  8.9005827e-02,  9.3104627e-01,  1.1098321e-01,  9.5362853e-01,  4.1783501e-01,  5.6727138e-01,  9.7270040e-01, 1.0f,

};
const float Temprature[] = {
	//walls and floor
	20,20,20,20,20,20,
	//light, sphere, cone
	  100,   72.5f,  37
};
const double c = 299792458, k = 138064852e-31, PI = 3.141592653589793238463,
h = 2 * PI * 105457180e-42;

float BBP(float T, int index) {
	float v = Wave[index];
	return 2e8 * (h * c * c * v * v * v) / (exp(100 * h * c * v / k / T) - 1);
}

void uploadThermalData(GLuint program, int index) {
	glUseProgram(program);
	const int n = 9;//total objects
	const float wall_refl = 1 - Emissivity[index][9], floor_refl = 1 - Emissivity[index][6], light_refl = 1 - Emissivity[index][3],
		sphere_refl = 1 - Emissivity[index][7], cone_refl = 1-Emissivity[index][4];
	float Reflectivity[n] = {
		wall_refl,wall_refl,wall_refl,wall_refl,wall_refl,floor_refl,
		light_refl,sphere_refl,cone_refl
	};
	glUniform1fv(glGetUniformLocation(program, "Reflectivity"), n, Reflectivity);
	float intensity[n];
	for (int i = 0; i < n; i++) {
		intensity[i] = BBP(Temprature[i]+273.15, index)*(1-Reflectivity[i]);
	}
	glUniform1fv(glGetUniformLocation(program, "Intensity"), n, intensity);
}


Wave GetSky()
{
	Wave sky;
	for (int i = 0; i < 11; i++) {
		sky[i] = BBP(273.15f, i) * Emissivity[i][10];
	}
	return sky;
}

