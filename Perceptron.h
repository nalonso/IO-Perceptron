#ifndef PERCEPTRON_H_INCLUDED
#define PERCEPTRON_H_INCLUDED
#include "Layer.h"
#include <math.h>

struct PerceptronStruct
{
    struct LayerStruct *layers[10];
    int numberOfLayer;
    int numberOfInput;
    double sigmas[10][10];
    double deltas[10][10][10];
};

struct PerceptronStruct *CreatePerceptron(int numberOfInput, int numberOfNeuronPerLayer[10], int numberOfLayers)
{
    struct PerceptronStruct *toReturn = 0;
    toReturn = (struct PerceptronStruct *)malloc(sizeof(struct PerceptronStruct *));
    toReturn->numberOfLayer = 1;
    toReturn->numberOfInput = numberOfInput;
    toReturn->layers[0] = CreateLayer(numberOfInput, numberOfNeuronPerLayer[0]);
    for(int idx = 0; idx < numberOfLayers; idx++)
    {
        toReturn->numberOfLayer++;
        toReturn->layers[idx] = CreateLayer(numberOfNeuronPerLayer[idx-1], numberOfNeuronPerLayer[idx]);
    }
}

double *ActivatePerceptron(struct PerceptronStruct *perceptron, double input[10])
{
    double inputAux[10] = {0};
    for(int idx = 0; idx < 10; idx++) inputAux[idx] = input[idx];
    for(int idx = 0; idx < perceptron->numberOfLayer; idx++)
    {
        ActivateLayer(perceptron->layers[idx], inputAux);
        for(int idxAux = 0; idxAux < 10; idxAux++) inputAux[idxAux] = perceptron->layers[idxAux]->output[idxAux];
    }
}

double IndividualError(double realOutput[10], double expectedOutput[10], int totalOutput)
{
    double currentError = 0;
    for(int idx = 0; idx < totalOutput; idx++)
    {
        currentError += pow(realOutput[idx] - expectedOutput[idx], 2);
    }
    return currentError;
}

double GeneralError(struct PerceptronStruct *perceptron, double input[10][10], double expectedOutput[10][10], int totalInput, int totalOutput)
{
    double generalError = 0;
    for(int idx = 0; idx < totalInput; idx++)
    {
        generalError += IndividualError(ActivatePerceptron(perceptron, input[idx]), expectedOutput[idx], totalOutput);
    }
    return generalError;
}

void SetSigmasPerceptron(struct PerceptronStruct *perceptron, double expectedOutput[10], int totalOutput)
{
    for(int idxLayer = perceptron->numberOfLayer - 1; idxLayer >= 0; idxLayer--)
    {
        for(int idxNeuron = 0; idxNeuron < perceptron->layers[idxLayer]->totalNeurons; idxNeuron++)
        {
            if(idxLayer == perceptron->numberOfLayer - 1)
            {
                double y = perceptron->layers[idxLayer]->neurons[idxNeuron]->lastActivation;
                perceptron->sigmas[idxLayer][idxNeuron] = (Sigmoid(y) - expectedOutput[idxNeuron]) * SigmoidDerivated(y);
            } else {
                double sum = 0;
                for(int idxWeight = 0; idxWeight < perceptron->layers[idxLayer + 1]->totalNeurons; idxWeight++)
                {
                    sum += perceptron->layers[idxLayer + 1]->neurons[idxWeight]->weights[idxNeuron] * perceptron->sigmas[idxLayer + 1][idxWeight];
                }
                perceptron->sigmas[idxLayer][idxNeuron] = SigmoidDerivated(perceptron->layers[idxLayer]->neurons[idxNeuron]->lastActivation) * sum;
            }
        }
    }
}

void AddDeltaPerceptron(struct PerceptronStruct *perceptron)
{
    for(int idxLayer = 1; idxLayer < perceptron->numberOfLayer; idxLayer++)
    {
        for(int idxNeuron = 0; idxNeuron < perceptron->layers[idxLayer]->totalNeurons; idxNeuron++)
        {
            for(int idxWeight = 0; idxWeight < perceptron->layers[idxLayer]->neurons[idxNeuron]->totalWeights; idxWeight++)
            {
                perceptron->deltas[idxLayer][idxNeuron][idxWeight] += perceptron->sigmas[idxLayer][idxNeuron] * Sigmoid(perceptron->layers[idxLayer - 1]->neurons[idxWeight]->lastActivation);
            }
        }
    }
}

void UpdateBiasPerceptron(struct PerceptronStruct *perceptron, double alpha)
{
    for(int idxLayer = 0; idxLayer < perceptron->numberOfLayer; idxLayer++)
    {
        for(int idxNeuron = 0; idxNeuron < perceptron->layers[idxLayer]->totalNeurons; idxNeuron++)
        {
            perceptron->layers[idxLayer]->neurons[idxNeuron]->bias -= alpha * perceptron->sigmas[idxLayer][idxNeuron];
        }
    }
}

void UpdateWeightsPerceptron(struct PerceptronStruct *perceptron, double alpha)
{
    for(int idxLayer = 0; idxLayer < perceptron->numberOfLayer; idxLayer++)
    {
        for(int idxNeuron = 0; idxNeuron < perceptron->layers[idxLayer]->totalNeurons; idxNeuron++)
        {
            for(int idxWeight = 0; idxWeight < perceptron->layers[idxLayer]->neurons[idxNeuron]->totalWeights; idxWeight++)
            {
                perceptron->layers[idxLayer]->neurons[idxNeuron]->weights[idxWeight] -= alpha * perceptron->deltas[idxLayer][idxNeuron][idxWeight];
            }
        }
    }
}

void ApplyBackPropagation(struct PerceptronStruct *perceptron, double input[10][10], double expectedOutput[10][10], double alpha, int totalInput, int totalOutput)
{
    for (int idx = 0; idx < totalInput; idx++)
    {
        ActivatePerceptron(perceptron, input[idx]);
        SetSigmasPerceptron(perceptron, expectedOutput[idx], totalOutput);
        UpdateBiasPerceptron(perceptron, alpha);
        AddDeltaPerceptron(perceptron);
    }
    UpdateWeightsPerceptron(perceptron, alpha);
}

void LearnPerceptron(struct PerceptronStruct *perceptron, double input[10][10], double expectedOutput[10][10], int totalInput, int totalOutput, int alpha, int errorMax)
{
    double currentError = 99999;
    while(currentError > errorMax)
    {
        ApplyBackPropagation(perceptron, input, expectedOutput, alpha, totalInput, totalOutput);
        currentError = GeneralError(perceptron, input, expectedOutput, totalInput, totalOutput);
        printf("Error Actual --> %fl\n", currentError);
    }
}

#endif // PERCEPTRON_H_INCLUDED
