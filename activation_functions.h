#ifndef ANTI_LOOP_
#define ANTI_LOOP_


/////////////// FONCTIONS D'ACTIVATION AVEC LEURS DÉRIVÉ RESPECTIF //////////////
float sigmoid(float s);
//dériver de la sigmoide
float sigmoidPrime(float s);


float ReLU(float x);

float ReLUPrime(float x);


float swish(float x, float b);

#endif
