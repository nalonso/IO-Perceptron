#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED
#include "Sigmoid.h"

double GetRandomNumber(int idx)
{
    return rand() % 100 + idx;
}

struct NeuronStruct
{
    double weights[10];
    int totalWeights;
    double lastActivation;
    double bias;
};

struct NeuronStruct *CreateNeuron (int numberOfInputs)
{
    struct NeuronStruct *toReturn = 0;
    toReturn = (struct NeuronStruct *)malloc(sizeof(struct NeuronStruct *));
    toReturn->totalWeights = 0;
    toReturn->bias = 1.0 * (GetRandomNumber(2) - GetRandomNumber(3));
    for(int idx = 0; idx < numberOfInputs; idx++)
    {
        toReturn->totalWeights++;
        toReturn->weights[idx] = 1.0 * (GetRandomNumber(idx) - GetRandomNumber(idx));
    }
    return toReturn;
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
