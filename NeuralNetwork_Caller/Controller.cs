﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dabas.NeuralNewtork;
using System.Threading;

namespace NeuralNetwork_Caller
{
    class Controller
    {
        static void Main(string[] args)
        {
            test1();
            Console.WriteLine("Execution Completed : )\nPress Any Key To Continue");
            Console.ReadKey();
        }

        static double test1()
        {
            NeuralNetwork nn = new NeuralNetwork(new int[] { 2, 20, 1 },
                                               new TransferFuncType[] { TransferFuncType.NONE,
                                                TransferFuncType.RECTILINEAR,
                                                TransferFuncType.SIGMOID});

            double error = 0;

            int limit = 10000;
            for (int i = 0; i < limit; i++)
            {
                error = 0;
                double learningRate = 0.25;
                double momentum = 0.3;
                bool displayOutput = false;
                if (i % (limit > 10 ? limit / 10 : 1) == 0)
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
            NeuralNetwork nn = new NeuralNetwork(new int[] { 1, 5, 1 },
                                               new TransferFuncType[] { TransferFuncType.NONE,
                                               TransferFuncType.SIGMOID,
                                               TransferFuncType.LINEAR});

            double error = 0;

            for (int i = 0; i < 20; i++)
            {
                error = 0;
                double learningRate =0.1;
                double momentum = 0.05;
                bool displayOutput = false;
                if (i % 1 == 0)
                {
                    Console.WriteLine("__________");
                    displayOutput = true;
                }
                for(int j = 0; j < 25; j++)
                {
                    Random gen = new Random();
                    double input = 2 * gen.NextDouble() - 1;
                    error += nn.Train(new double[] { input }, new double[] { Math.Cos(input) }, learningRate, momentum, false);
                }
                if (displayOutput)
                    Console.WriteLine("Error : {0}", error);
            }
            return error;
        }
    }
}
