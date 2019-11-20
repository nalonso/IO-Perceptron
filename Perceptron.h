#ifndef PERCEPTRON_H_INCLUDED
#define PERCEPTRON_H_INCLUDED
#include "Layer.h"
#include <math.h>

struct PerceptronStruct
{
    struct LayerStruct layers[10];
    int numberOfLayer;
    double sigmas[10][10];
    double deltas[10][10][10];
};

void InitPerceptron(struct PerceptronStruct * perceptron,int numberOfNeuronPerLayer[10])
{
    InitLayer(&perceptron->layers[0], numberOfNeuronPerLayer[0],numberOfNeuronPerLayer[0]);
    for(int idx = 1; idx < perceptron->numberOfLayer; idx++)
    {
        InitLayer(&perceptron->layers[idx], numberOfNeuronPerLayer[idx], numberOfNeuronPerLayer[idx-1]);
    }
}

double *ActivatePerceptron(struct PerceptronStruct *perceptron, double input[10])
{
    double inputAux[10] = {0};
    for(int idx = 0; idx < 10; idx++) inputAux[idx] = input[idx];
    for(int idx = 0; idx < perceptron->numberOfLayer; idx++)
    {
        ActivateLayer(&perceptron->layers[idx], inputAux);
        for(int idxAux = 0; idxAux < 10; idxAux++) inputAux[idxAux] = perceptron->layers[idx].output[idxAux];
    }
    return perceptron->layers[perceptron->numberOfLayer - 1].output;
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

void SetSigmasPerceptron(struct PerceptronStruct *perceptron, double expectedOutput[10])
{
    for(int idxLayer = perceptron->numberOfLayer - 1; idxLayer >= 0; idxLayer--)
    {
        for(int idxNeuron = 0; idxNeuron < perceptron->layers[idxLayer].totalNeurons; idxNeuron++)
        {
            perceptron->sigmas[idxLayer][idxNeuron] = 0;
        }
    }
    for(int idxLayer = perceptron->numberOfLayer - 1; idxLayer >= 0; idxLayer--)
    {
        for(int idxNeuron = 0; idxNeuron < perceptron->layers[idxLayer].totalNeurons; idxNeuron++)
        {
            if(idxLayer == perceptron->numberOfLayer - 1)
            {
                double y = perceptron->layers[idxLayer].neurons[idxNeuron].lastActivation;
                perceptron->sigmas[idxLayer][idxNeuron] = (Sigmoid(y) - expectedOutput[idxNeuron]) * SigmoidDerivated(y);
                //printf("Y --%fl\n", y);
                //printf("Sigmoid --%fl\n", Sigmoid(y));
                //printf("expectedOutput[%d] == %fl\n", idxNeuron,expectedOutput[idxNeuron]);

                //printf("SigmoidDerivated --%fl\n", SigmoidDerivated(y));
                //printf("First Sigma                            %fl\n", perceptron->sigmas[idxLayer][idxNeuron]);
                //printf("Asigno Sigma For Weight [%d][%d]\n", idxLayer,idxNeuron);
            } else {
                double sum = 0;
                for(int idxNeuronLayerPrevius = 0; idxNeuronLayerPrevius < perceptron->layers[idxLayer + 1].totalNeurons; idxNeuronLayerPrevius++)
                {
                    sum += perceptron->layers[idxLayer + 1].neurons[idxNeuronLayerPrevius].weights[idxNeuron] * perceptron->sigmas[idxLayer + 1][idxNeuronLayerPrevius];
                    //printf("REsult %fl\n", perceptron->sigmas[idxLayer + 1][idxNeuronLayerPrevius]);
                    //printf("Sigma For Weight [%d][%d]\n", idxLayer + 1,idxNeuronLayerPrevius);
                }
                perceptron->sigmas[idxLayer][idxNeuron] = SigmoidDerivated(perceptron->layers[idxLayer].neurons[idxNeuron].lastActivation) * sum;

                //printf("Asigno Sigma For Weight [%d][%d]\n", idxLayer,idxNeuron);
                //printf("SUM %fl\n", sum);
            }
            /*printf("SIgmaResult %fl\n", perceptron->layers[idxLayer].neurons[idxNeuron].lastActivation);*/
        }
    }
}

void SetDeltaPerceptron(struct PerceptronStruct *perceptron)
{
    for(int idxLayer = 0; idxLayer < perceptron->numberOfLayer; idxLayer++)
    {
        for(int idxNeuron = 0; idxNeuron < perceptron->layers[idxLayer].totalNeurons; idxNeuron++)
        {
            for(int idxWeight = 0; idxWeight < perceptron->layers[idxLayer].neurons[idxNeuron].totalWeights; idxWeight++)
            {
                perceptron->deltas[idxLayer][idxNeuron][idxWeight] = 0;
            }
        }
    }
}

void AddDeltaPerceptron(struct PerceptronStruct *perceptron)
{
    for(int idxLayer = 1; idxLayer < perceptron->numberOfLayer; idxLayer++)
    {
        for(int idxNeuron = 0; idxNeuron < perceptron->layers[idxLayer].totalNeurons; idxNeuron++)
        {
            for(int idxWeight = 0; idxWeight < perceptron->layers[idxLayer].neurons[idxNeuron].totalWeights; idxWeight++)
            {
                perceptron->deltas[idxLayer][idxNeuron][idxWeight] += perceptron->sigmas[idxLayer][idxNeuron] * Sigmoid(perceptron->layers[idxLayer - 1].neurons[idxWeight].lastActivation);
                /*printf("SeTEO DELTA %fl\n",perceptron->deltas[idxLayer][idxNeuron][idxWeight]);
                printf("Sigma para delta perceptron->sigmas[%d][%d] %fl\n",idxLayer, idxNeuron,perceptron->sigmas[idxLayer][idxNeuron]);
                printf("Sigmoid para delta %fl\n",Sigmoid(perceptron->layers[idxLayer - 1].neurons[idxWeight].lastActivation));*/
            }
        }
    }
}

void UpdateBiasPerceptron(struct PerceptronStruct *perceptron, double alpha)
{
    for(int idxLayer = 0; idxLayer < perceptron->numberOfLayer; idxLayer++)
    {
        for(int idxNeuron = 0; idxNeuron < perceptron->layers[idxLayer].totalNeurons; idxNeuron++)
        {
            perceptron->layers[idxLayer].neurons[idxNeuron].bias -= alpha * perceptron->sigmas[idxLayer][idxNeuron];
        }
    }
}

void UpdateWeightsPerceptron(struct PerceptronStruct *perceptron, double alpha)
{
    for(int idxLayer = 0; idxLayer < perceptron->numberOfLayer; idxLayer++)
    {
        for(int idxNeuron = 0; idxNeuron < perceptron->layers[idxLayer].totalNeurons; idxNeuron++)
        {
            for(int idxWeight = 0; idxWeight < perceptron->layers[idxLayer].neurons[idxNeuron].totalWeights; idxWeight++)
            {

                /*printf("weight Old %fl\n", perceptron->layers[idxLayer].neurons[idxNeuron].weights[idxWeight]);
                printf("alpha %fl\n", alpha);
                printf("delta %fl\n", perceptron->deltas[idxLayer][idxNeuron][idxWeight]);
                */
                perceptron->layers[idxLayer].neurons[idxNeuron].weights[idxWeight] -= alpha * perceptron->deltas[idxLayer][idxNeuron][idxWeight];

                //printf("Weight New %fl\n", perceptron->layers[idxLayer].neurons[idxNeuron].weights[idxWeight]);
            }
        }
    }
}

void ApplyBackPropagation(struct PerceptronStruct *perceptron, double input[10][10], double expectedOutput[10][10], double alpha, int totalInput)
{
    SetDeltaPerceptron(perceptron);
    for (int idx = 0; idx < totalInput; idx++)
    {
        ActivatePerceptron(perceptron, input[idx]);
        SetSigmasPerceptron(perceptron, expectedOutput[idx]);
        UpdateBiasPerceptron(perceptron, alpha);
        AddDeltaPerceptron(perceptron);
    }
    UpdateWeightsPerceptron(perceptron, alpha);
}

void LearnPerceptron(struct PerceptronStruct *perceptron, double input[10][10], double expectedOutput[10][10], int totalInput, int totalOutput, double alpha, double errorMax)
{
    double currentError = 99999;
    printf("Error Actual                   --> %fl\n", currentError);
    while(currentError > errorMax)
    {
        ApplyBackPropagation(perceptron, input, expectedOutput, alpha, totalInput);
        currentError = GeneralError(perceptron, input, expectedOutput, totalInput, totalOutput);
        printf("Error Actual                   --> %fl\n", currentError);
    }
}

#endif // PERCEPTRON_H_INCLUDED
