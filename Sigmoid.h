#ifndef SIGMOID_H_INCLUDED
#define SIGMOID_H_INCLUDED
#include <math.h>

double Sigmoid(double input)
{
    return 1 / (1 + exp(-input));
}

double SigmoidDerivated(double input)
{
    double y = Sigmoid(input);
    return y * (1 - y);
}

#endif // SIGMOID_H_INCLUDED
