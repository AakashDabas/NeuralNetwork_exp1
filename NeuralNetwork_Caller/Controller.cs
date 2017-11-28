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
            Console.WriteLine("Execution Completed : )\nPress Any Key To Continue");
            Console.ReadKey();
        }

        static double test1()
        {
            LayerData.FullyConnected inputLayer = new LayerData.FullyConnected()
            {
                cntNeurons = 2,
                tFuncType = TransferFuncType.NONE
            };
            LayerData.FullyConnected hidden = new LayerData.FullyConnected()
            {
                cntNeurons = 20,
                tFuncType = TransferFuncType.RECTILINEAR
            };
            LayerData.FullyConnected outputLayer = new LayerData.FullyConnected()
            {
                cntNeurons = 1,
                tFuncType = TransferFuncType.SIGMOID
            };
            int limit = 100;
            NeuralNetwork nn = new NeuralNetwork(limit * 4, 1, 0.25, 0.1, true, inputLayer, hidden, outputLayer);

            double error = 0;

            for (int i = 0; i < limit; i++)
            {
                error = 0;
                bool displayOutput = false;
                if (i % (limit > 10 ? limit / 10 : 1) == 0)
                {
                    Console.WriteLine("__________");
                    displayOutput = true;
                }
                error += nn.Train(new double[] { 0, 0 }, new double[] { 0 }, displayOutput);
                error += nn.Train(new double[] { 0, 1 }, new double[] { 1 }, displayOutput);
                error += nn.Train(new double[] { 1, 0 }, new double[] { 1 }, displayOutput);
                error += nn.Train(new double[] { 1, 1 }, new double[] { 0 }, displayOutput);
                if (displayOutput)
                    Console.WriteLine("Error : {0}", error);
                Thread.Sleep(10);
            }
            return error;
        }

        static double test2()
        {
            LayerData.FullyConnected inputLayer = new LayerData.FullyConnected()
            {
                cntNeurons = 3,
                tFuncType = TransferFuncType.NONE
            };
            LayerData.FullyConnected h1 = new LayerData.FullyConnected()
            {
                cntNeurons = 10,
                tFuncType = TransferFuncType.RATIONALSIGMOID
            };
            LayerData.FullyConnected outputLayer = new LayerData.FullyConnected()
            {
                cntNeurons = 1,
                tFuncType = TransferFuncType.TANH
            };
            int limit = 2000;
            NeuralNetwork nn = new NeuralNetwork(limit * 25, 1, 0.01, 0.1, true, inputLayer, outputLayer);
            double error = 0;
            Random gen = new Random();

            for (int i = 0; i < limit; i++)
            {
                error = 0;
                bool displayOutput = false;
                if (i % 1 == 0)
                {
                    nn.RegisterOutput("__________");
                    displayOutput = true;
                }
                for (int j = 0; j < 25; j++)
                {
                    double input = gen.NextDouble();
                    error += nn.Train(new double[] { input, 1, 1 }, new double[] { Math.Tanh(input) }, false);
                }
                if (displayOutput)
                    nn.RegisterOutput(string.Format("Error : {0}", error));
                Thread.Sleep(100);
            }
            nn.WaitTillDone();
            return error;
        }

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
            LayerData.FullyConnected hidden = new LayerData.FullyConnected()
            {
                cntNeurons = 20,
                tFuncType = TransferFuncType.SIGMOID
            };
            LayerData.Convolutional conv1 = new LayerData.Convolutional()
            {
                filters = new int[]
                {
                    14, 14, 14, 14, 14, 14
                },
                stride = 2,
                padding = false
            };
            LayerData.MaxPool maxPool1 = new LayerData.MaxPool()
            {
                size = 2,
                stride = 2
            };
            LayerData.Convolutional conv2 = new LayerData.Convolutional()
            {
                filters = new int[]
                {
                    7, 7, 7, 7, 7, 7
                },
                stride = 1,
                padding = false
            };
            LayerData.FullyConnected fully1 = new LayerData.FullyConnected()
            {
                cntNeurons = 30,
                tFuncType = TransferFuncType.RECTILINEAR
            };
            LayerData.FullyConnected outputLayer = new LayerData.FullyConnected()
            {
                cntNeurons = 10,
                tFuncType = TransferFuncType.SOFTMAX
            };
            NeuralNetwork nn = new NeuralNetwork(limit, 1000, 0.0001, 0.0, true, inputLayer, conv1, conv2, fully1, outputLayer);
            //nn.Save("temp.xml", "temp");
            //NeuralNetwork nn = NeuralNetwork.Load("Temp\\400.xml", true);
            //nn.Save("TestingSaveModule.xml", "CNNTEST");
            //NeuralNetwork nn = NeuralNetwork.Load("temp.xml", true);
            //NeuralNetwork nn = new NeuralNetwork(new int[] { 784, 20, 10 },
            //                       new TransferFuncType[] { TransferFuncType.NONE, TransferFuncType.SIGMOID, TransferFuncType.SOFTMAX }, 60000, 100);
            nn.batchSize = 1;
            double error = 0;
            Random randGen = new Random();
            //int saveCnt = 400;
            //nn.Save("Temp\\" + saveCnt++ + ".xml", "ForGraph");
            //limit = 180000;
            int[] wrongDetectedCnt = new int[10];
            for (int i = 0; i < 10; i++)
                wrongDetectedCnt[i] = 0;
            for (int i = nn.trainingCounter; i < limit; i++)
            {
                //learningRate = 0.005 * (1.00001 - (double)i / limit);
                //momentum = 0.005 * (1.00001 - (double)i / limit);
                error = 0;
                bool displayOutput = false;
                //if (i % (limit > 10 ? limit / 1000 : 1) == 0)
                //{
                //    nn.RegisterOutput("_____________");
                //    displayOutput = true;
                //}

                double[] input = new double[height * width];

                int idx = (int)(randGen.NextDouble() * 59999);
                for (int j = 0; j < height; j++)
                    for (int k = 0; k < width; k++)
                        input[j * width + k] = (data[idx][j, k] / 255.0);// >= 140 ? 1 : 0);

                double[] output = new double[10];
                for (int j = 0; j < 10; j++)
                    output[j] = 0;
                output[label[idx]] = 1;

                if (displayOutput)
                    nn.RegisterOutput(string.Format("Label: {0}", label[i]));

                error = nn.Train(input, output, displayOutput);
                if (error > 0.4)
                {
                    wrongDetectedCnt[label[i]]++;
                }

                if (displayOutput)
                    nn.RegisterOutput(string.Format("Error : {0}", error));

                if (i % 100 == 0 && i != 0)
                    nn.Save("Temporary.xml", "Testing");
                //nn.Save("Temp\\" + saveCnt++ + ".xml", "ForGraph");
                if (i % 50 == 0 && i != 0)
                {
                    for (int j = 0; j < 10; j++)
                        Console.WriteLine("{0} : {1}", j, wrongDetectedCnt[j]);
                    Console.WriteLine("**********************");
                }
            }
            nn.Save("Testing.xml", "DigitRecognizer");
            return error;
        }
    }
}
