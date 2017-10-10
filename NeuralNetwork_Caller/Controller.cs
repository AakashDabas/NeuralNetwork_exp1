using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dabas.NeuralNetwork;
using System.Threading;
using System.IO;

namespace NeuralNetwork_Caller
{
    class Controller
    {
        static void Main(string[] args)
        {
            test3();
            //Console.WriteLine("Execution Completed : )\nPress Any Key To Continue");
            //Console.ReadKey();
        }

        //static double test1()
        //{
        //    NeuralNetwork nn = new NeuralNetwork(new int[] { 2, 20, 1 },
        //                                       new TransferFuncType[] { TransferFuncType.NONE,
        //                                        TransferFuncType.RECTILINEAR,
        //                                        TransferFuncType.SIGMOID}, 10000);

        //    double error = 0;

        //    int limit = 10000;
        //    for (int i = 0; i < limit; i++)
        //    {
        //        error = 0;
        //        double learningRate = 0.25;
        //        double momentum = 0.3;
        //        bool displayOutput = false;
        //        if (i % (limit > 10 ? limit / 10 : 1) == 0)
        //        {
        //            Console.WriteLine("__________");
        //            displayOutput = true;
        //        }
        //        error += nn.Train(new double[] { 0, 0 }, new double[] { 0 }, learningRate, momentum, displayOutput);
        //        error += nn.Train(new double[] { 0, 1 }, new double[] { 1 }, learningRate, momentum, displayOutput);
        //        error += nn.Train(new double[] { 1, 0 }, new double[] { 1 }, learningRate, momentum, displayOutput);
        //        error += nn.Train(new double[] { 1, 1 }, new double[] { 0 }, learningRate, momentum, displayOutput);
        //        if (displayOutput)
        //            Console.WriteLine("Error : {0}", error);
        //    }
        //    return error;
        //}

        //static double test2()
        //{
        //    NeuralNetwork nn = new NeuralNetwork(new int[] { 1, 10, 10, 10, 1 },
        //                                       new TransferFuncType[] { TransferFuncType.NONE,
        //                                       TransferFuncType.RATIONALSIGMOID,
        //                                       TransferFuncType.RATIONALSIGMOID,
        //                                       TransferFuncType.RATIONALSIGMOID,
        //                                       TransferFuncType.LINEAR}, 2000 * 25);
        //    double error = 0;
        //    Random gen = new Random();

        //    for (int i = 0; i < 2000; i++)
        //    {
        //        error = 0;
        //        double learningRate = 0.01;
        //        double momentum = 0.01;
        //        bool displayOutput = false;
        //        if (i % 1 == 0)
        //        {
        //            nn.RegisterOutput("__________");
        //            displayOutput = true;
        //        }
        //        for (int j = 0; j < 25; j++)
        //        {
        //            double input = 4 * gen.NextDouble() - 2;
        //            error += nn.Train(new double[] { input }, new double[] { Math.Cos(input) }, learningRate, momentum, false);
        //        }
        //        if (displayOutput)
        //            nn.RegisterOutput(string.Format("Error : {0}", error));
        //    }
        //    nn.WaitTillDone();
        //    return error;
        //}

        public static int ReadInt(ref byte[] bytes, ref int idx)
        {
            int output = 0;
            for (int i = 0; i < 4; i++)
            {
                output <<= 8;
                output |= bytes[idx + i];
            }
            idx += 4;
            return output;
        }

        static double test3()
        {
            int limit = 0;
            int height, width;
            Dictionary<int, byte[,]> data = new Dictionary<int, byte[,]>();
            Dictionary<int, byte> label = new Dictionary<int, byte>();

            // Read Data
            byte[] imgBytes = File.ReadAllBytes(@"train-images.idx3-ubyte");
            byte[] labelBytes = File.ReadAllBytes(@"train-labels.idx1-ubyte");
            int idx1 = 0, idx2 = 8;
            int magicNumber = ReadInt(ref imgBytes, ref idx1);
            limit = ReadInt(ref imgBytes, ref idx1);
            height = ReadInt(ref imgBytes, ref idx1);
            width = ReadInt(ref imgBytes, ref idx1);

            for (int i = 0; i < limit; i++)
            {
                data[i] = new byte[height, width];
                label[i] = labelBytes[idx2++];
                for (int j = 0; j < height; j++)
                    for (int k = 0; k < width; k++)
                    {
                        data[i][j, k] = imgBytes[idx1++];
                    }
            }

            LayerData.FullyConnected inputLayer = new LayerData.FullyConnected()
            {
                cntNeurons = 784,
                tFuncType = TransferFuncType.NONE
            };
            LayerData.Convolutional conv1 = new LayerData.Convolutional()
            {
                filters = new int[]
                {
                    3, 3, 3
                },
                stride = 3
            };
            LayerData.FullyConnected outputLayer = new LayerData.FullyConnected()
            {
                cntNeurons = 10,
                tFuncType = TransferFuncType.SIGMOID
            };
            NeuralNetwork nn = new NeuralNetwork(limit, 1, true, inputLayer, conv1, outputLayer);
            //NeuralNetwork nn = NeuralNetwork.Load("Testing5.xml", true);
            //NeuralNetwork nn = new NeuralNetwork(new int[] { 784, 20, 10 },
            //                       new TransferFuncType[] { TransferFuncType.NONE, TransferFuncType.SIGMOID, TransferFuncType.SOFTMAX }, 60000, 100);
            nn.batchSize = 1;
            double error = 0;
            int cnt = 0;
            for (int i = nn.trainingCounter; i < limit; i++)
            {
                error = 0;
                double learningRate = 0.2;
                double momentum = 0.05;
                bool displayOutput = false;
                if (i % (limit > 10 ? limit / 1000 : 1) == 0)
                {
                    nn.RegisterOutput("_____________");
                    displayOutput = true;
                }

                double[] input = new double[height * width];

                for (int j = 0; j < height; j++)
                    for (int k = 0; k < width; k++)
                        input[j * width + k] = (data[i][j, k] >= 140 ? 1 : 0);

                double[] output = new double[10];
                for (int j = 0; j < 10; j++)
                    output[j] = 0;
                output[label[i]] = 1;

                if (displayOutput)
                    nn.RegisterOutput(string.Format("Label: {0}", label[i]));

                error = nn.Train(input, output, learningRate, momentum, displayOutput);
                if (error == double.NaN)
                    break;
                if (error < 0.00001)
                    cnt++;
                else
                    cnt = 0;
                if (cnt == 100)
                    break;
                if (displayOutput)
                    nn.RegisterOutput(string.Format("Error : {0}", error));

                //if (i % 1000 == 0 && i != 0)
                //    nn.Save("Testing.xml", "Testin01");
            }
             //nn.Save("Testing.xml", "DigitRecognizer");
            return error;
        }
    }
}
