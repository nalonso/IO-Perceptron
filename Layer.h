#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED
#include "Neuron.h"

struct LayerStruct
{
    struct NeuronStruct *neurons[10];
    int totalNeurons;
    double output[10];
};

struct LayerStruct *CreateLayer(int numberOfNeurons, int numberOfInputs)
{
    struct LayerStruct *toReturn = 0;
    toReturn = (struct LayerStruct *)malloc(sizeof(struct LayerStruct *));
    for(int idx = 0; idx < numberOfNeurons; idx++)
    {
        toReturn->neurons[idx] = CreateNeuron(numberOfInputs);
    }
}

void ActivateLayer(struct LayerStruct *layer, double input[10])
{
    for(int idx = 0; idx < layer->totalNeurons; idx++)
    {
        layer->output[idx] = ActivateNeuron( layer->neurons[idx], input);
    }
}

#endif // LAYER_H_INCLUDED
