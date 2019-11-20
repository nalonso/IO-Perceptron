#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED
#include "Sigmoid.h"
#include<time.h>
#include<stdlib.h>
double GetRandomNumber(int idx)
{
    double num = rand();
    num = 1+rand()%(101-1+idx);
    num = num * (rand()%(20+idx));
    return num;
}

struct NeuronStruct
{
    double weights[10];
    int totalWeights;
    double lastActivation;
    double bias;
};

InitNeuron (struct NeuronStruct *neuron, int numberOfInputs)
{
    neuron->totalWeights = numberOfInputs;
    neuron->bias = 10 * GetRandomNumber(10) - 5;
        /*printf("bias %fl\n", neuron->bias);*/
    for(int idx = 0; idx < numberOfInputs; idx++)
    {
        neuron->weights[idx] = 10 * GetRandomNumber(idx * 3) - 5;
        /*printf("pesoNeuron %fl\n", neuron->weights[idx]);*/
    }
};

double ActivateNeuron(struct NeuronStruct *neuron, double inputs[10])
{
    double activation = neuron->bias;

    for (int idx = 0; idx < neuron->totalWeights; idx++)
    {
        activation += neuron->weights[idx] * inputs[idx];
    }

    neuron->lastActivation = activation;
    return Sigmoid(activation);
}

#endif // NEURON_H_INCLUDED
