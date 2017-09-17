using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Dabas.NeuralNetwork_UI;
using System.IO;
using System.Xml;

namespace Dabas.NeuralNewtork
{
    public class NeuralNetwork
    {
        public int trainingCounter = 0;
        Queue<double> prevErrors = new Queue<double>();
        double prevErrorSum = 0;

        List<Layer> layers;
        Dictionary<int, Neuron> neurons;
        Dictionary<int, Connection> connections;
        int neuronCounter = 0, connectionCounter = 0;

        Thread UIThread;
        NeuralNetworkUI ui;
        // To store graph data
        Queue<double> xAxisData, yAxisData;

        NeuralNetworkArgs args;

        class NeuralNetworkArgs
        {
            public int[] layersCnt;
            public TransferFuncType[] tFuncType;
            public int intendedTrainingCnt = 0;
        }

        public NeuralNetwork(int[] layersCnt, TransferFuncType[] tFuncType, int intendedTrainingCnt = 1, double weightRescaleFactor = 1, bool showNNUI = true)
        {
            args = new NeuralNetworkArgs();
            args.layersCnt = layersCnt;
            args.tFuncType = tFuncType;
            args.intendedTrainingCnt = intendedTrainingCnt;

            xAxisData = new Queue<double>();
            yAxisData = new Queue<double>();
            ui = new NeuralNetworkUI();
            ui.AddToChart(ref xAxisData, ref yAxisData);
            UIThread = new Thread(ui.StartUI);
            if (showNNUI)
                UIThread.Start();
            if (layersCnt.Length != tFuncType.Length)
                throw new Exception("Input size mismatch! Invalid input to NeuralNetwork Constructor");

            Layer previousLayer = null;
            layers = new List<Layer>();
            neurons = new Dictionary<int, Neuron>();
            connections = new Dictionary<int, Connection>();

            for (int i = 0; i < layersCnt.Length; i++)
            {
                if (layersCnt[i] == 0)
                    throw new Exception("Empty layer requested! Invalid input to NeuralNetwork Constructor");

                Layer currentLayer = new Layer(layersCnt[i], tFuncType[i], previousLayer, ref neurons, ref neuronCounter, ref connections, ref connectionCounter, weightRescaleFactor);

                previousLayer = currentLayer;
                layers.Add(currentLayer);
            }

            RegisterOutput("Neural Network Initialized");
        }

        public void Run(double[] input, bool displayOutput)
        {
            // Check for valid input
            if (input.Length != layers[0].cntNeurons)
                throw new Exception("Input layer size mismatch! Invalid input given to NeuralNetwork.Run()");

            Neuron.Reset(ref neurons);
            // Initialize input neurons
            Queue<int> buffer = new Queue<int>();
            foreach (int idx in layers[0].neuronIdxs)
            {
                neurons[idx].input = input[idx];
                buffer.Enqueue(idx);
            }

            // Perform BFS
            while (buffer.Count != 0)
            {
                int idx = buffer.Dequeue();

                // Evaluate current neuron's output
                // Use this output to add weighted output to all connected neurons
                neurons[idx].Evaluate();
                foreach (int cIdx in neurons[idx].outgoingConnection)
                {
                    Connection connection = connections[cIdx];
                    neurons[connection.dest].input += neurons[idx].output * connection.weight;
                    if (buffer.Contains(connection.dest) == false)
                        buffer.Enqueue(connection.dest);
                }
            }

            // Output Layer
            if (displayOutput)
            {
                RegisterOutput("OUTPUT: ");
                foreach (int idx in layers.Last().neuronIdxs)
                    RegisterOutput(neurons[idx].output.ToString());
            }
        }

        public double Train(double[] input, double[] output, double learningRate, double momentum, bool displayOutput)
        {
            trainingCounter++;
            // Check for valid input
            if (input.Length != layers[0].cntNeurons || output.Length != layers.Last().cntNeurons)
                throw new Exception("Input and layer size mismatch! Invalid input given to NeuralNetwork.Train()");

            // Forward pass
            Run(input, displayOutput);

            // Compute error

            double totalError = 0;
            Queue<int> buffer = new Queue<int>();

            for (int itr = 0; itr < layers.Last().cntNeurons; itr++)
            {
                Neuron neuron = neurons[layers.Last().neuronIdxs[itr]];
                double error = neuron.output - output[itr];
                totalError += error * error / 2.0;
                neurons[neuron.neuronIdx].deltaBack = error;
                buffer.Enqueue(neuron.neuronIdx);
            }

            // BackPropagate Error/Deltas

            while (buffer.Count != 0)
            {
                int currIdx = buffer.Dequeue();
                neurons[currIdx].deltaBack *= neurons[currIdx].EvaluateDerivative();
                Neuron neuron = neurons[currIdx];

                foreach (int idx in neuron.incommingConnection)
                {
                    Connection connection = connections[idx];
                    double deltaBackward = connection.weight * neuron.deltaBack;
                    int src = connection.src;
                    neurons[src].deltaBack += deltaBackward;
                    if (buffer.Contains(src) == false)
                        buffer.Enqueue(src);
                }
            }

            // Update Connection Weights

            foreach (Connection connection in connections.Values)
            {
                Neuron src = neurons[connection.src];
                Neuron dest = neurons[connection.dest];

                double weightDelta = -learningRate * dest.deltaBack * src.output;

                if (weightDelta * connection.previousWeightDelta < 0)
                    connection.previousWeightDelta *= 0.5;

                connections[connection.connectionIdx].weight += weightDelta + momentum * connection.previousWeightDelta;
                connections[connection.connectionIdx].previousWeightDelta = connection.previousWeightDelta + weightDelta;
                neurons[connection.src].bias += -learningRate * dest.deltaBack;
            }

            //To Update UI
            ui.nnUIForm.graphUpdateSemaphore.WaitOne();
            {
                ui.SetProgressBar(trainingCounter, args.intendedTrainingCnt);
                if (prevErrors.Count == 100)
                {
                    prevErrorSum -= prevErrors.Dequeue();
                }
                prevErrors.Enqueue(totalError);
                prevErrorSum += totalError;
                xAxisData.Enqueue(trainingCounter);
                yAxisData.Enqueue(prevErrorSum / prevErrors.Count);
            }
            ui.nnUIForm.graphUpdateSemaphore.Release();
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

        public void Save(string filePath, string networkName)
        {
            XmlWriter writer;
            XmlWriterSettings writerSettings = new XmlWriterSettings
            {
                Indent = true,
                IndentChars = "     ",
                NewLineOnAttributes = false,
                OmitXmlDeclaration = true
            };

            try
            {
                writer = XmlWriter.Create(filePath, writerSettings);
            }
            catch
            {
                throw new Exception("Invalid filepath given to NeuralNetwork.Save() !");
            }

            writer.WriteStartElement("NeuralNetwork");

            writer.WriteAttributeString("Time", DateTime.Now.ToString());
            writer.WriteAttributeString("IndendedTrainingCnt", args.intendedTrainingCnt.ToString());
            writer.WriteAttributeString("TrainingDone", trainingCounter.ToString());
            writer.WriteAttributeString("Name", networkName);

            writer.WriteStartElement("Layers");
            writer.WriteAttributeString("Count", args.layersCnt.Length.ToString());
            for (int i = 0; i < args.layersCnt.Length; i++)
            {
                writer.WriteStartElement("Layer");
                writer.WriteAttributeString("Type", args.tFuncType[i].ToString());
                writer.WriteAttributeString("Neurons", args.layersCnt[i].ToString());
                writer.WriteAttributeString("Index", i.ToString());
                writer.WriteEndElement();
            }
            writer.WriteEndElement(); // Layers

            writer.WriteStartElement("Neurons");
            writer.WriteAttributeString("Count", neurons.Values.Count.ToString());
            foreach(Neuron neuron in neurons.Values)
            {
                writer.WriteStartElement("Neuron");
                writer.WriteAttributeString("Bias", neuron.bias.ToString());
                writer.WriteAttributeString("Index", neuron.neuronIdx.ToString());
                writer.WriteEndElement();
            }
            writer.WriteEndElement(); // Neurons

            writer.WriteStartElement("Connections");
            writer.WriteAttributeString("Count", connections.Values.Count.ToString());
            foreach (Connection connection in connections.Values)
            {
                writer.WriteStartElement("Connection");
                writer.WriteAttributeString("PreviousWeightDelta", connection.previousWeightDelta.ToString());
                writer.WriteAttributeString("Destination", connection.dest.ToString());
                writer.WriteAttributeString("Source", connection.src.ToString());
                writer.WriteAttributeString("Weight", connection.weight.ToString());
                writer.WriteAttributeString("Index", connection.connectionIdx.ToString());
                writer.WriteEndElement();
            }
            writer.WriteEndElement(); // Connections

            writer.WriteEndElement();// Neural Network Args

            writer.Flush();
            writer.Close();

            this.RegisterOutput("Saved Neural Network : " + networkName);
        }

        public static NeuralNetwork Load(string filePath, bool showNNUI = false)
        {
            XmlDocument doc = new XmlDocument();
            try
            {
                doc.Load(filePath);
            }
            catch
            {
                throw new Exception("Invalid filepath given to NeuralNetwork.Load() !");
            }

            int trainingCounter;
            int layerCnt = 0;
            NeuralNetworkArgs args = new NeuralNetworkArgs();

            string basePath = "NeuralNetwork/";
            int.TryParse(XPathValue(basePath + "@TrainingDone", ref doc), out trainingCounter);
            int.TryParse(XPathValue(basePath + "@IndendedTrainingCnt", ref doc), out args.intendedTrainingCnt);
            basePath += "Layers/";

            int.TryParse(XPathValue(basePath + "@Count", ref doc), out layerCnt);
            args.layersCnt = new int[layerCnt];
            args.tFuncType = new TransferFuncType[layerCnt];

            basePath += "Layer[@Index='{0}']/@{1}";
            for(int i = 0; i < layerCnt; i++)
            {
                int.TryParse(XPathValue(string.Format(basePath, i.ToString(), "Neurons"), ref doc), out args.layersCnt[i]);
                Enum.TryParse<TransferFuncType>(XPathValue(string.Format(basePath, i.ToString(), "Type"), ref doc), out args.tFuncType[i]);
            }

            NeuralNetwork nn = new NeuralNetwork(args.layersCnt, args.tFuncType, args.intendedTrainingCnt, 1, showNNUI);
            int.TryParse(XPathValue("NeuralNetwork/@TrainingDone", ref doc), out nn.trainingCounter);

            nn.RegisterOutput("Loading Neurons");
            basePath = "NeuralNetwork/Neurons/Neuron[@Index='{0}']/@Bias";
            int neuronCnt;
            int.TryParse(XPathValue("NeuralNetwork/Neurons/@Count", ref doc), out neuronCnt);
            for(int i = 0; i < neuronCnt; i++)
            {
                double.TryParse(XPathValue(string.Format(basePath, i.ToString()), ref doc), out nn.neurons[i].bias);
            }
            nn.RegisterOutput("Loading Connections");
            basePath = "NeuralNetwork/Connections/Connection[@Index='{0}']/@{1}";
            int connectionCnt;
            int.TryParse(XPathValue("NeuralNetwork/Connections/@Count", ref doc), out connectionCnt);
            for (int i = 0; i < connectionCnt; i++)
            {
                double.TryParse(XPathValue(string.Format(basePath, i.ToString(), "Weight"), ref doc), out nn.connections[i].weight);
                double.TryParse(XPathValue(string.Format(basePath, i.ToString(), "PreviousWeightDelta"), ref doc), out nn.connections[i].previousWeightDelta);
                double completed = (i + 1) * 100.0 / connectionCnt;
                if (completed == 25 || completed == 50 || completed == 75 || completed == 100)
                {
                    nn.RegisterOutput("Connections Loaded : " + completed + "%");
                }
            }
            nn.RegisterOutput("Neural Network : " + XPathValue("NeuralNetwork/@Name", ref doc) + "Loaded Successfully : )");
            doc = null;
            return nn;
        }

        private static string XPathValue(string xPath, ref XmlDocument doc)
        {
            XmlNode node = doc.SelectSingleNode(xPath);
            if (node == null)
                throw new Exception("Invalid XPath given to NeuralNetwork.XPathValue() !");
            return node.InnerText;
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

    class Layer
    {
        public int cntNeurons;
        public int[] neuronIdxs;
        TransferFuncType tFuncType;

        public Layer(int cntNeurons, TransferFuncType tFuncType, Layer previousLayer,
            ref Dictionary<int, Neuron> neurons, ref int neuronCounter,
            ref Dictionary<int, Connection> connections, ref int connectionCounter,
            double weightRescaleFactor = 1)
        {
            this.cntNeurons = cntNeurons;
            this.tFuncType = tFuncType;
            this.neuronIdxs = new int[cntNeurons];

            for (int i = 0; i < cntNeurons; i++)
            {
                Neuron neuron = new Neuron(tFuncType, previousLayer, ref neurons, ref neuronCounter, ref connections, ref connectionCounter, weightRescaleFactor);
                neuronIdxs[i] = neuron.neuronIdx;
            }
        }
    }

    class Neuron
    {
        public double input, output, deltaBack, bias;
        public int neuronIdx;
        public List<int> incommingConnection, outgoingConnection;
        TransferFuncType tFuncType;

        public Neuron(TransferFuncType tFuncType, Layer previousLayer,
            ref Dictionary<int, Neuron> neurons, ref int neuronCounter,
            ref Dictionary<int, Connection> connections, ref int connectionCounter,
            double weightRescaleFactor = 1)
        {
            input = 0;
            deltaBack = 0;
            output = 0;
            bias = Gaussian.GetRandomGaussian();
            this.tFuncType = tFuncType;

            incommingConnection = new List<int>();
            outgoingConnection = new List<int>();

            if (previousLayer != null)
            {
                foreach (int neuronIdx in previousLayer.neuronIdxs)
                {
                    Connection connection = new Connection(ref neurons, ref neuronCounter, ref connections, ref connectionCounter, weightRescaleFactor);
                    connection.dest = neuronCounter;
                    connection.src = neuronIdx;
                    this.incommingConnection.Add(connection.connectionIdx);
                    neurons[neuronIdx].outgoingConnection.Add(connection.connectionIdx);
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

        public static void Reset(ref Dictionary<int, Neuron> neurons)
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

        public Connection(ref Dictionary<int, Neuron> neurons, ref int neuronCounter,
            ref Dictionary<int, Connection> connections, ref int connectionCounter,
            double weightRescaleFactor = 1)
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