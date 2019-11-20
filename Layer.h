#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED
#include "Neuron.h"

struct LayerStruct
{
    struct NeuronStruct neurons[10];
    int totalNeurons;
    double output[10];
};

void InitLayer(struct LayerStruct *layer, int numberOfNeurons, int numberOfInputs)
{
    layer->totalNeurons = numberOfNeurons;
    for(int idx = 0; idx < numberOfNeurons; idx++)
    {
        InitNeuron(&layer->neurons[idx], numberOfInputs);
    }
}

void ActivateLayer(struct LayerStruct *layer, double input[10])
{
    for(int idx = 0; idx < layer->totalNeurons; idx++)
    {
        layer->output[idx] = ActivateNeuron( &layer->neurons[idx], input);
    }
}

#endif // LAYER_H_INCLUDED
