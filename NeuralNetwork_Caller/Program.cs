using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dabas.NeuralNewtork;

namespace NeuralNetwork_Caller
{
    class Program
    {
        static void Main(string[] args)
        {
            test2();
            Console.WriteLine("Execution Completed : )\nPress Any Key To Continue");
            Console.ReadKey();
        }

        static double test1()
        {
            NeuralNetwork nn = new NeuralNetwork(new int[] { 2, 2, 1 },
                                               new TransferFuncType[] { TransferFuncType.NONE,
                                                TransferFuncType.SIGMOID,
                                                TransferFuncType.LINEAR});

            double error = 0;

            for (int i = 0; i < 100000; i++)
            {
                error = 0;
                double learningRate = 0.15;
                double momentum = 0.0;
                bool displayOutput = false;
                if (i % 1000 == 0)
                {
                    Console.WriteLine("__________");
                    displayOutput = true;
                }
                error += nn.Train(new double[] { 0, 0 }, new double[] { 0 }, learningRate, momentum, displayOutput);
                error += nn.Train(new double[] { 0, 1 }, new double[] { 1 }, learningRate, momentum, displayOutput);
                error += nn.Train(new double[] { 1, 0 }, new double[] { 1 }, learningRate, momentum, displayOutput);
                error += nn.Train(new double[] { 1, 1 }, new double[] { 0 }, learningRate, momentum, displayOutput);
                if (displayOutput)
                    Console.WriteLine("Error : {0}", error);
            }
            return error;
        }

        static double test2()
        {
            NeuralNetwork nn = new NeuralNetwork(new int[] { 1, 1 },
                                               new TransferFuncType[] { TransferFuncType.NONE,
                                                TransferFuncType.LINEAR});

            double error = 0;

            for (int i = 0; i < 200; i++)
            {
                error = 0;
                double learningRate = 0.05;
                double momentum = 0.0;
                bool displayOutput = false;
                if (i % 1 == 0)
                {
                    Console.WriteLine("__________");
                    displayOutput = true;
                }
                Random gen = new Random();
                int input = (int)(gen.NextDouble() * 10);
                error += nn.Train(new double[] { input }, new double[] { input }, learningRate, momentum, displayOutput);
                if (displayOutput)
                    Console.WriteLine("Error : {0}", error);
            }
            return error;
        }
    }
}
