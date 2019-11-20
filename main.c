#include <stdio.h>
#include <stdlib.h>
#include "Perceptron.h"

int main()
{
    printf("Hello world!\n");
    double inputs[10][10] = {
    {0,0},
    {0,1},
    {1,0},
    {1,1}
    };
    double outputs[10][10] = {
    {0},
    {1},
    {1},
    {0}
    };
    int neuronPerLayer[10] = {2,3,1};

    struct PerceptronStruct * perceptron = CreatePerceptron(2, neuronPerLayer, 3);
    LearnPerceptron(perceptron, inputs, outputs, 2, 1, 0.7, 0.01);
    while (1 == 1)
    {
        printf("inserte valores, para verificar\n");
        double input1, input2;
        scanf("%fl", &input1);
        scanf("%fl", &input2);
        double toVerify [10] = {0};
        toVerify[0] = input1;
        toVerify[1] = input2;
        double *result = ActivatePerceptron(perceptron, toVerify);
        printf("Respuesta --> %fl\n", toVerify[0]);
    }

    return 0;
}
