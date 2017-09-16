using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Dabas.NeuralNetwork_UI;

namespace Dabas.NeuralNewtork
{
    public class NeuralNetwork
    {
        public int trainingCounter = 0, cntTotalTraining = 0;
        Queue<double> prevErrors = new Queue<double>();
        double prevErrorSum = 0;
        Layer inputLayer, outputLayer;
        Thread UIThread;

        NeuralNetworkUI ui;
        // To store graph data
        Queue<double> xAxisData, yAxisData;

        public NeuralNetwork(int[] NeuronCnt, TransferFuncType[] tFuncType, int totalTrainingData, double weightRescaleFactor = 1, bool showNNUI = true)
        {
            xAxisData = new Queue<double>();
            yAxisData = new Queue<double>();
            ui = new NeuralNetworkUI();
            ui.AddToChart(ref xAxisData, ref yAxisData);
            cntTotalTraining = totalTrainingData;
            UIThread = new Thread(ui.StartUI);
            if (showNNUI)
                UIThread.Start();
            if (NeuronCnt.Length != tFuncType.Length)
                throw new Exception("Input size mismatch! Invalid input to NeuralNetwork Constructor");

            Layer previousLayer = null;

            for (int i = 0; i < NeuronCnt.Length; i++)
            {
                if (NeuronCnt[i] == 0)
                    throw new Exception("Empty layer requested! Invalid input to NeuralNetwork Constructor");
                LayerType lType = LayerType.HIDDEN;
                if (i == 0)
                    lType = LayerType.INPUT;
                else if (i == NeuronCnt.Length - 1)
                    lType = LayerType.OUTPUT;

                Layer currentLayer = new Layer(NeuronCnt[i], tFuncType[i], previousLayer,
                                               (i == NeuronCnt.Length - 1 ? 0 : NeuronCnt[i + 1]), lType);

                previousLayer = currentLayer;
                if (lType == LayerType.INPUT)
                    inputLayer = currentLayer;
                else if (lType == LayerType.OUTPUT)
                    outputLayer = currentLayer;
            }

            RegisterOutput("Neural Network Ready");
        }

        public void Run(double[] input, bool displayOutput)
        {
            // Check for valid input
            if (input.Length != inputLayer.cntNeurons)
                throw new Exception("Input layer size mismatch! Invalid input given to NeuralNetwork.Run()");

            Neuron.Reset();
            // Initialize input neurons
            Queue<int> buffer = new Queue<int>();
            for (int i = 0; i < inputLayer.cntNeurons; i++)
            {
                Neuron.neurons[i].input = input[i];
                buffer.Enqueue(i);
            }

            // Perform BFS
            while (buffer.Count != 0)
            {
                int idx = buffer.Dequeue();

                // Evaluate current neuron's output
                // Use this output to add weighted output to all connected neurons
                Neuron.neurons[idx].Evaluate();
                foreach (int cIdx in Neuron.neurons[idx].outgoingConnection)
                {
                    Connection connection = Connection.connections[cIdx];
                    Neuron.neurons[connection.dest].input += Neuron.neurons[idx].output * connection.weight;
                    if (buffer.Contains(connection.dest) == false)
                        buffer.Enqueue(connection.dest);
                }
            }

            // Output Layer
            if (displayOutput)
            {
                RegisterOutput("OUTPUT: ");
                for (int itr = 0; itr < outputLayer.cntNeurons; itr++)
                    RegisterOutput(Neuron.neurons[itr + outputLayer.startIdx].output.ToString());
            }
        }

        public double Train(double[] input, double[] output, double learningRate, double momentum, bool displayOutput)
        {
            trainingCounter++;
            // Check for valid input
            if (input.Length != inputLayer.cntNeurons || output.Length != outputLayer.cntNeurons)
                throw new Exception("Input and layer size mismatch! Invalid input given to NeuralNetwork.Train()");

            // Forward pass
            Run(input, displayOutput);

            // Compute error

            double totalError = 0;
            Queue<int> buffer = new Queue<int>();

            for (int itr = 0; itr < outputLayer.cntNeurons; itr++)
            {
                Neuron neuron = Neuron.neurons[outputLayer.startIdx + itr];
                double error = neuron.output - output[itr];
                totalError += error * error / 2.0;
                Neuron.neurons[neuron.neuronIdx].deltaBack = error;

                buffer.Enqueue(neuron.neuronIdx);
            }

            // BackPropagate Error/Deltas

            while (buffer.Count != 0)
            {
                int currIdx = buffer.Dequeue();
                Neuron.neurons[currIdx].deltaBack *= Neuron.neurons[currIdx].EvaluateDerivative();
                Neuron neuron = Neuron.neurons[currIdx];

                foreach (int idx in neuron.incommingConnection)
                {
                    Connection connection = Connection.connections[idx];
                    double deltaBackward = connection.weight * neuron.deltaBack;
                    int src = connection.src;
                    Neuron.neurons[src].deltaBack += deltaBackward;
                    if (buffer.Contains(src) == false)
                        buffer.Enqueue(src);
                }
            }

            // Update Connection Weights

            foreach (Connection connection in Connection.connections.Values)
            {
                Neuron src = Neuron.neurons[connection.src];
                Neuron dest = Neuron.neurons[connection.dest];

                double weightDelta = -learningRate * dest.deltaBack * src.output;

                if (weightDelta * connection.previousWeightDelta < 0)
                    connection.previousWeightDelta *= 0.5;

                Connection.connections[connection.connectionIdx].weight += weightDelta + momentum * connection.previousWeightDelta;
                Connection.connections[connection.connectionIdx].previousWeightDelta = connection.previousWeightDelta + weightDelta;
                Neuron.neurons[connection.src].bias += -learningRate * dest.deltaBack;
            }

            //To Update UI
            if (ui.nnUIForm.graphUpdateOngoing == false)
            {
                ui.SetProgressBar(trainingCounter, cntTotalTraining);
                if (prevErrors.Count == 100)
                {
                    prevErrorSum -= prevErrors.Dequeue();
                }
                prevErrors.Enqueue(totalError);
                prevErrorSum += totalError;
                xAxisData.Enqueue(trainingCounter);
                yAxisData.Enqueue(prevErrorSum / prevErrors.Count);
            }
            return totalError;
        }

        public void WaitTillDone()
        {
            while (UIThread.IsAlive)
                Thread.Sleep(1000);
        }

        public void RegisterOutput(string text)
        {
            ui.RegisterOutput(text);
        }

    }

    public enum TransferFuncType
    {
        NONE,
        SIGMOID,
        RATIONALSIGMOID,
        FASTSIGMOID,
        GAUSSIAN,
        TANH,
        LINEAR,
        RECTILINEAR,
        SOFTMAX
    }

    public enum LayerType
    {
        INPUT,
        HIDDEN,
        OUTPUT
    }

    class Layer
    {
        public int startIdx;
        public int cntNeurons;
        TransferFuncType tFuncType;
        LayerType lType;

        public Layer(int cntNeurons, TransferFuncType tFuncType, Layer previousLayer, int nextLayerNeuronCnt, LayerType lType)
        {
            startIdx = Neuron.neuronCounter;
            this.cntNeurons = cntNeurons;
            this.tFuncType = tFuncType;
            this.lType = lType;

            for (int i = 0; i < cntNeurons; i++)
                new Neuron(nextLayerNeuronCnt, tFuncType, previousLayer);
        }
    }

    class Neuron
    {
        public double input, output, deltaBack, bias;
        public int neuronIdx;
        public List<int> incommingConnection, outgoingConnection;
        TransferFuncType tFuncType;

        public static int neuronCounter = 0;
        public static Dictionary<int, Neuron> neurons = new Dictionary<int, Neuron>();

        public Neuron(int nextLayerNeurons, TransferFuncType tFuncType, Layer previousLayer, double weightRescaleFactor = 1)
        {
            input = 0;
            deltaBack = 0;
            output = 0;
            bias = Gaussian.GetRandomGaussian();
            this.tFuncType = tFuncType;

            incommingConnection = new List<int>();
            outgoingConnection = new List<int>();

            for (int i = 0; i < nextLayerNeurons; i++)
            {
                int tmpIdx = new Connection(weightRescaleFactor).connectionIdx;
                outgoingConnection.Add(tmpIdx);
                Connection.connections[tmpIdx].src = neuronCounter;
            }

            if (previousLayer != null)
            {
                for (int itr = 0; itr < previousLayer.cntNeurons; itr++)
                    foreach (int idx in Neuron.neurons[previousLayer.startIdx + itr].outgoingConnection)
                        if (Connection.connections[idx].dest == -1)
                        {
                            Connection.connections[idx].dest = neuronCounter;
                            incommingConnection.Add(idx);
                            break;
                        }
            }

            neuronIdx = neuronCounter;
            neurons[neuronCounter++] = this;
        }

        public double Evaluate()
        {
            if (tFuncType != TransferFuncType.NONE)
                input += bias;
            output = TransferFunction.Evaluate(tFuncType, input);
            return output;
        }

        public double EvaluateDerivative()
        {
            if (tFuncType == TransferFuncType.SIGMOID)
                return output * (1 - output);
            return TransferFunction.EvaluateDerivate(tFuncType, input);
        }

        public static void Reset()
        {
            foreach (Neuron neuron in neurons.Values)
            {
                neuron.input = 0;
                neuron.deltaBack = 0;
                neuron.output = 0;
            }
        }
    }

    class Connection
    {
        public int src, dest;
        public double weight;
        public double weightDelta;
        public double previousWeightDelta;
        public int connectionIdx;

        public static int connectionCounter = 0;
        public static Dictionary<int, Connection> connections = new Dictionary<int, Connection>();

        public Connection(double weightRescaleFactor = 1)
        {
            weight = Gaussian.GetRandomGaussian() / weightRescaleFactor;
            weightDelta = 0;
            previousWeightDelta = 0;
            dest = -1;
            connectionIdx = connectionCounter;
            connections[connectionCounter++] = this;
        }
    }

    static class TransferFunction
    {
        public static double Evaluate(TransferFuncType tfuncType, double x)
        {
            double output = 0;

            switch (tfuncType)
            {
                case TransferFuncType.NONE:
                    output = None(x);
                    break;
                case TransferFuncType.SIGMOID:
                    output = Sigmoid(x);
                    break;
                case TransferFuncType.RATIONALSIGMOID:
                    output = RationalSigmoid(x);
                    break;
                case TransferFuncType.FASTSIGMOID:
                    output = FastSigmoid(x);
                    break;
                case TransferFuncType.GAUSSIAN:
                    output = Gaussian(x);
                    break;
                case TransferFuncType.TANH:
                    output = TANH(x);
                    break;
                case TransferFuncType.LINEAR:
                    output = Linear(x);
                    break;
                case TransferFuncType.RECTILINEAR:
                    output = RectiLinear(x);
                    break;
                default:
                    throw new FormatException("Invalid Input Provided To TransferFunction.Evaluate()");
                    break;
            }

            return output;
        }

        public static double EvaluateDerivate(TransferFuncType tfuncType, double x)
        {
            double output = 0;

            switch (tfuncType)
            {
                case TransferFuncType.NONE:
                    output = None_Derivative(x);
                    break;
                case TransferFuncType.SIGMOID:
                    output = Sigmoid_Derivative(x);
                    break;
                case TransferFuncType.RATIONALSIGMOID:
                    output = RationalSigmoid_Derivative(x);
                    break;
                case TransferFuncType.FASTSIGMOID:
                    output = FastSigmoid_Derivative(x);
                    break;
                case TransferFuncType.GAUSSIAN:
                    output = Gaussian_Derivative(x);
                    break;
                case TransferFuncType.TANH:
                    output = TANH_Derivative(x);
                    break;
                case TransferFuncType.LINEAR:
                    output = Linear_Derivative(x);
                    break;
                case TransferFuncType.RECTILINEAR:
                    output = RectiLinear_Derivative(x);
                    break;
                default:
                    throw new FormatException("Invalid Input Provided To TransferFunction.EvaluateDerivative()");
                    break;
            }

            return output;
        }

        #region Transfer Functions

        private static double None(double x)
        {
            return x;
        }

        private static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private static double RationalSigmoid(double x)
        {
            return x / Math.Sqrt(1 + x * x);
        }

        private static double FastSigmoid(double x)
        {
            return x / (1 + Math.Abs(x));
        }

        private static double Gaussian(double x)
        {
            return Math.Exp(-x * x);
        }

        private static double TANH(double x)
        {
            return Math.Tanh(x);
        }

        private static double Linear(double x)
        {
            return x;
        }

        private static double RectiLinear(double x)
        {
            return (x > 0 ? x : 0);
        }

        #endregion

        #region Transfer Functions Derivatives

        private static double None_Derivative(double x)
        {
            return 1;
        }

        private static double Sigmoid_Derivative(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        private static double RationalSigmoid_Derivative(double x)
        {
            double val = Math.Sqrt(1 + x * x);
            return 1.0 / (val * (1 + val));
        }

        private static double FastSigmoid_Derivative(double x)
        {
            if (x <= 0)
                return 1.0 / Math.Pow(x - 1, 2);
            else
                return 1.0 / Math.Pow(x + 1, 2);
        }

        private static double Gaussian_Derivative(double x)
        {
            return -2 * x * Gaussian(x);
        }

        private static double TANH_Derivative(double x)
        {
            return 1 - Math.Pow(Math.Tanh(x), 2);
        }

        private static double Linear_Derivative(double x)
        {
            return 1;
        }

        private static double RectiLinear_Derivative(double x)
        {
            return (x > 0 ? 1 : 0);
        }

        #endregion
    }

    static class Gaussian
    {
        private static Random gen = new Random();

        public static double GetRandomGaussian()
        {
            return GetRandomGaussian(0, 1);
        }

        public static double GetRandomGaussian(double mean, double stddev)
        {
            double rVal1, rVal2;
            GetRandomGaussian(mean, stddev, out rVal1, out rVal2);
            return rVal1;
        }

        public static void GetRandomGaussian(double mean, double stddev, out double val1, out double val2)
        {
            double u, v, s, t;
            do
            {
                u = 2 * gen.NextDouble() - 1;
                v = 2 * gen.NextDouble() - 1;
            } while (u * u + v * v > 1 || (u == 0 && v == 0));

            s = u * u + v * v;
            t = Math.Sqrt((-2.0 * Math.Log(s)) / s);

            val1 = stddev * u * t + mean;
            val2 = stddev * v * t + mean;
        }
    }
}