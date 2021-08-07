////////////////////////////////////////////////////////
//    \  /          /\            |\     |        /\  //
//    \ /          / \           | \    |        / \  //
//    \/          /__\          |  \   |        /__\  //
//    /\         /   \         |   \  |        /   \  //
//   / \        /    \        |    \ |        /    \  //
//  /  \   O   /     \   O   |     \|    O   /     \  //
////////////////////////////////////////////////////////


//     ### INTELIGENCE ARTIFICIEL MULTI AJENT ###     //


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "activation_functions.h"

#define MAX_HEIGHT 4 //largeur max du réseu de neurone (nombre max de neurone sur un layer)
#define PROFONDEUR_NETWORK 2
#define INPUT_HEIGHT 4


const int NETWORK_MAP[PROFONDEUR_NETWORK] = {4, 1};


///////Fontion de pert
float difference(float x, float y){
	return x - y;
}

float difference_prime(float x, float y){
	return x - y;
}


//donne des résulta toujours positif
//La différence est mise au carré afin que nous mesurions la valeur absolue de la différence.
float sum_of_sqaures_error(float x, float y){
	return powf(x - y, 2);
}


float sum_of_sqaures_error_prime(float x, float y){
	return 2 * (x - y);
}


//////////////////////// DEFINITION DES NEURONES ET DU RÉSEU DE NEURONE ////////////////////////////
typedef struct Neurone{
	float value_axone; //La valeur renvoyer par le neurone
	float poid_axone[MAX_HEIGHT]; //Le poid de la valeur de sorti du neurone
	float bias; //non utiliser
	float delta_error;
} Neurone;


//Notre réseu de neurones
Neurone network[PROFONDEUR_NETWORK][MAX_HEIGHT] = {};



///////////////DATASET//////////////

//Les inputs d'apprentissage
float inputs[8][INPUT_HEIGHT] = {
	{0, 1, 1, 0},
	{1, 0, 0, 1},
	{1, 0, 0, 0},
	{1, 1, 0, 0},
	{1, 1, 0, 1},
	{1, 1, 1, 0},
	{0, 1, 1, 0},
	{0, 0, 1, 1}
};

//Les reponses attendu
float reponses[8] = {1, 0, 0, 0, 0, 1, 1, 0};



/////////////INPUT TEST////////////

//Données d'input de test, une fois que le réseau a fini d'aprendre
float input_test[5][INPUT_HEIGHT] = {
	{0, 0, 1, 1},
	{1, 0, 0, 1},
	{1, 1, 0, 0},
	{0, 0, 0, 0},
	{1, 1, 1, 1}
};

/////  FIXÉÉÉ, CETTE FONCTION MARCHE!!! PAS TOUCHER!  //////
float *forward_propagate(float input_layer_values[]){
	//boucle qui parcour les layers de NETWORK_MAP
	unsigned int x;
	for (x = 0; x < sizeof(NETWORK_MAP) / sizeof(int); x++){
		//boucle qui parcour les neurones dans le layer
		unsigned int y;
		for (y = 0; y < NETWORK_MAP[x]; y++){
			//on refille les valeur dentré au neurone

			network[x][y].value_axone = 0;
			
			unsigned int l;
			if (!x){ //si c'est l'input layer (premier layer -> index 0)
				for(l = 0; l < INPUT_HEIGHT; l++){
					network[x][y].value_axone += input_layer_values[l] * network[x][y].poid_axone[l];
				}
			}else{
				for(l = 0; l < NETWORK_MAP[x - 1]; l++){
					network[x][y].value_axone += network[x - 1][l].value_axone * network[x][y].poid_axone[l];
				}
			}
			
			network[x][y].value_axone = sigmoid(network[x][y].value_axone + network[x][y].bias);
		}
	}
	
	//output layer values returned
	static float output_layer_value[1] = {};
	
	unsigned int i;
	for (i = 0; i < 1; i++){
		output_layer_value[i] = network[PROFONDEUR_NETWORK-1][i].value_axone;
	}
	
	return output_layer_value;
}


void backward_propagate(float reponce){	
	//On calcule le cou pour chaqun de nos neurones (a quel point la prédiction est éloignier de la bonne réponse pour trouver delta error
	//boucle qui parcour les layers de NETWORK_MAP de la fin au début de réseu de neurone (droit à gauche)
	unsigned int x;
	for(x = 0; x < sizeof(NETWORK_MAP) / sizeof(int); x++){
		//boucle qui parcour les neurones dans le layer
		unsigned int y;
		for(y = 0; y < NETWORK_MAP[PROFONDEUR_NETWORK - x - 1]; y++){
			float output_erreur = 0;
			float output = network[PROFONDEUR_NETWORK - x - 1][y].value_axone;
			
			if (PROFONDEUR_NETWORK - x == PROFONDEUR_NETWORK){ //si c'est l'output layer (premier layer -> index 0)
				output_erreur = reponce - output; //cost
			}else{
				unsigned int l;
				for(l = 0; l < NETWORK_MAP[PROFONDEUR_NETWORK - x]; l++){
					output_erreur += network[PROFONDEUR_NETWORK - x][l].poid_axone[y] * network[PROFONDEUR_NETWORK - x][l].delta_error;
				}
			}
			
			network[PROFONDEUR_NETWORK - x - 1][y].delta_error = output_erreur * sigmoidPrime(output);
		}
	}
}


///////////////////////////////////////////////  M A I N  //////////////////////////////////////////////////
int main(int argc, char *argv[]){
	printf("Entrez le nombre d'iterations d'apprentissage: ");
	int nombre_iteration_apprentissage;
	scanf("%d", &nombre_iteration_apprentissage);
	
	printf("En cours de calcule...\n\n");
	
	//On constitue notre resau de neurone
	//boucle qui parcour les layers de NETWORK_MAP
	srand(time(NULL));
	unsigned int x;
	for(x = 0; x < sizeof(NETWORK_MAP) / sizeof(int); x++){
		unsigned int y;
		
		//boucle qui parcour les neurones dans le layer
		for(y = 0; y < NETWORK_MAP[x]; y++){
			Neurone neurone;
			
			neurone.bias = 0;
			
			//on défini des valeurs de poid au assard;
			unsigned int p;
			for(p = 0; p < MAX_HEIGHT; p++){
				neurone.poid_axone[p] = (rand() / (float)RAND_MAX) * 2 - 1; //entre -1 et 1
			}
			//on met le neurone crée dans le reseu de neurone;
			network[x][y] = neurone;
		}
	}
	
	
	//Une foi le réseu inisialiser il faut lui faire passer la phase d'entrainement
	int i;
	for(i = 0; i < nombre_iteration_apprentissage; i++){
		float output_layer_values = forward_propagate(inputs[i % 6])[0];
		backward_propagate(reponses[i % 6]);
		
		/////correction des poid/////
		
		//boucle qui parcour les layers de NETWORK_MAP
		unsigned int x;
		for(x = 0; x < sizeof(NETWORK_MAP) / sizeof(int); x++){
			//boucle qui parcour les neurones dans le layer
			unsigned int y;
			for(y = 0; y < NETWORK_MAP[x]; y++){
				unsigned int l;
				if(!x){
					for(l = 0; l < INPUT_HEIGHT; l++){//intput layer
						network[x][y].poid_axone[l] += inputs[i % 5][l] * network[x][y].delta_error;
					}
				}else{
					for(l = 0; l < NETWORK_MAP[x - 1]; l++){
						network[x][y].poid_axone[l] += network[x - 1][l].value_axone * network[x][y].delta_error;
					}
				}
				
				network[x][y].bias += network[x][y].delta_error;
			}
		}
		
		//printf("%f\n", output_layer_values);
	}
	
	//rep
	for(i = 0; i < 5; i++){
		printf("%f\n", forward_propagate(input_test[i%5])[0]);
	}
	
	return 0;
}
