#include <math.h>
#include "activation_functions.h"

float sigmoid(float s){
	return 1 / (1 + exp(-s));
}

//dériver de la sigmoide
float sigmoidPrime(float s){
	return s * (1 - s);
}


float ReLU(float x){
	return fmaxf(x, 0);
}


float ReLUPrime(float x){
	if (x > 0){
		return 1;
	}
	
	return 0;
}


float swish(float x, float b){
	return x * sigmoid(b * x);
}
