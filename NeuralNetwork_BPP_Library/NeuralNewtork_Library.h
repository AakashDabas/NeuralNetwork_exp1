#pragma once
#ifndef NeuralNetworkFlag
#define NeuralNetworkFlag

#include<cmath>

namespace Dabas
{
	namespace NeuralNetwork
	{
		enum TransferFuncType
		{
			NONE,
			SIGMOID,
			RATIONALSIGMOID,
			FASTSIGMOID,
			GAUSSIAN,
			TANH,
			LINEAR,
			RECTILINEAR
		};

		class Neuron
		{

		};

		class Connection
		{
		};

		static class TransferFunction
		{

			static double Evaluate(TransferFuncType tfuncType, double x)
			{
				double output = 0;

				switch (tfuncType)
				{
				case NONE:
					output = None(x);
					break;
				case SIGMOID:
					output = Sigmoid(x);
					break;
				case RATIONALSIGMOID:
					output = RationalSigmoid(x);
					break;
				case FASTSIGMOID:
					output = FastSigmoid(x);
					break;
				case GAUSSIAN:
					output = Gaussian(x);
					break;
				case TANH:
					output = Tanh(x);
					break;
				case LINEAR:
					output = Linear(x);
					break;
				case RECTILINEAR:
					output = RectiLinear(x);
					break;
				default:
					break;
				}

				return output;
			}

			static double EvaluateDerivate(TransferFuncType tfuncType, double x)
			{
				double output = 0;

				switch (tfuncType)
				{
				case NONE:
					output = None_Derivative(x);
					break;
				case SIGMOID:
					output = Sigmoid_Derivative(x);
					break;
				case RATIONALSIGMOID:
					output = RationalSigmoid_Derivative(x);
					break;
				case FASTSIGMOID:
					output = FastSigmoid_Derivative(x);
					break;
				case GAUSSIAN:
					output = Gaussian_Derivative(x);
					break;
				case TANH:
					output = Tanh_Derivative(x);
					break;
				case LINEAR:
					output = Linear_Derivative(x);
					break;
				case RECTILINEAR:
					output = RectiLinear_Derivative(x);
					break;
				default:
					break;
				}

				return output;
			}

			static double None(double x)
			{
				return x;
			}

			static double Sigmoid(double x)
			{
				return 1.0 / (1.0 + exp(-x));
			}

			static double RationalSigmoid(double x)
			{
				return x / sqrt(1 + x * x);
			}

			static double FastSigmoid(double x)
			{
				return x / (1 + abs(x));
			}

			static double Gaussian(double x)
			{
				return exp(-x * x);
			}

			static double Tanh(double x)
			{
				return tanh(x);
			}

			static double Linear(double x)
			{
				return x;
			}

			static double RectiLinear(double x)
			{
				return (x > 0 ? x : 0);
			}

			static double None_Derivative(double x)
			{
				return 1;
			}

			static double Sigmoid_Derivative(double x)
			{
				return Sigmoid(x) * (1 - Sigmoid(x));
			}

			static double RationalSigmoid_Derivative(double x)
			{
				double val = sqrt(1 + x * x);
				return 1.0 / (val * (1 + val));
			}

			static double FastSigmoid_Derivative(double x)
			{
				if (x <= 0)
					return 1.0 / pow(x - 1, 2);
				else
					return 1.0 / pow(x + 1, 2);
			}

			static double Gaussian_Derivative(double x)
			{
				return -2 * x * Gaussian(x);
			}

			static double Tanh_Derivative(double x)
			{
				return 1 - pow(tanh(x), 2);
			}

			static double Linear_Derivative(double x)
			{
				return 1;
			}

			static double RectiLinear_Derivative(double x)
			{
				return (x > 0 ? 1 : 0);
			}

		};
	}
}

#endif // !NeuralNetworkFlag