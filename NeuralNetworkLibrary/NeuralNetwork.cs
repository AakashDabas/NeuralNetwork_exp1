using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Dabas.NeuralNetwork_UI;
using System.IO;
using System.Xml;
using System.Diagnostics;

namespace Dabas.NeuralNetwork
{
    public class NeuralNetwork
    {
        public int trainingCounter = 0;
        Queue<double> prevErrors = new Queue<double>();
        double prevErrorSum = 0;
        public int batchSize = 1, batchUsed = 0;
        public bool ShowUi = true;
        double learningRate, momentum, learningRateT, momentumT;

        List<Layer> layers;
        Dictionary<int, Neuron> neurons;
        Dictionary<int, Connection> connections;
        int neuronCounter = 0, connectionCounter = 0;

        Thread UIThread;
        NeuralNetworkUI ui;
        // To store graph data
        Queue<double> xAxisData, yAxisData;

        // Testing Vars
        Queue<bool> results = new Queue<bool>();

        NeuralNetworkArgs args;

        class NeuralNetworkArgs
        {
            public object[] layersData;
            public double learningRate = 0;
            public double momentum = 0;
            public int intendedTrainingCnt = 0;
        }

        public NeuralNetwork(int intendedTrainingCnt, double weightRescaleFactor, double learningRate, double momentum, bool showNNUI, params object[] layersData)
        {
            layers = new List<Layer>();
            neurons = new Dictionary<int, Neuron>();
            connections = new Dictionary<int, Connection>();

            args = new NeuralNetworkArgs();
            args.intendedTrainingCnt = intendedTrainingCnt;
            args.layersData = layersData;
            args.learningRate = learningRate;
            args.momentum = momentum;
            this.learningRate = learningRate;
            this.momentum = momentum;
            learningRateT = learningRate;
            momentumT = momentum;

            xAxisData = new Queue<double>();
            yAxisData = new Queue<double>();
            ui = new NeuralNetworkUI(CallbackHandler);
            ui.AddToChart(ref xAxisData, ref yAxisData);
            UIThread = new Thread(ui.StartUI);
            if (showNNUI)
                UIThread.Start();
            ui.SetLearningRate(learningRate.ToString());
            ui.SetMomentum(momentum.ToString());

            Layer.ConstructLayers(layersData, ref layers, ref neurons, ref neuronCounter, ref connections, ref connectionCounter, learningRate, weightRescaleFactor);
            foreach (Layer layer in layers)
            {
                RegisterOutput("Count Neurons : " + layer.cntNeurons);
            }

            RegisterOutput("Neural Network Initialized");
        }

        private void Reset()
        {
            Neuron.Reset(ref neurons);

            // Reset weights for MAXPOOL layer
            foreach (Neuron neuron in neurons.Values)
                if (neuron.tFuncType == TransferFuncType.MAXPOOL)
                {
                    foreach (int cIdx in neuron.incomingConnection)
                    {
                        connections[cIdx].weight = 1;
                    }
                }
        }

        public void Run(double[] input, out double[] output, bool displayOutput)
        {
            // Check for valid input
            if (input.Length != layers[0].cntNeurons)
                throw new Exception("Input layer size mismatch! Invalid input given to NeuralNetwork.Run()");
            Reset();
            // Initialize input neurons
            Queue<int> buffer = new Queue<int>();
            foreach (int idx in layers[0].neuronIdxs)
            {
                neurons[idx].input = input[idx];
                buffer.Enqueue(idx);
            }
            double softMaxDenominator = 0;
            // Perform BFS
            while (buffer.Count != 0)
            {
                int idx = buffer.Dequeue();

                // Evaluate current neuron's output
                // Use this output to add weighted output to all connected neurons
                neurons[idx].Evaluate(ref neurons, ref connections);
                if (neurons[idx].tFuncType == TransferFuncType.SOFTMAX)
                    softMaxDenominator += neurons[idx].output;
                foreach (int cIdx in neurons[idx].outgoingConnection)
                {
                    Connection connection = connections[cIdx];
                    int dest = connection.srcDest[idx];
                    neurons[dest].input += neurons[idx].output * connection.weight;
                    if (buffer.Contains(dest) == false)
                        buffer.Enqueue(dest);
                }
            }
            //Console.WriteLine("---------------------------------");
            if (layers.Last().tFuncType == TransferFuncType.SOFTMAX)
                foreach (int idx in layers.Last().neuronIdxs)
                {
                    //Console.Write(neurons[idx].output);
                    neurons[idx].output /= softMaxDenominator;
                    //Console.WriteLine("  :  " + neurons[idx].output);
                    if (double.IsNaN(neurons[idx].output))
                    {
                        neurons[idx].output = 0;
                    }
                }

            output = new double[layers.Last().neuronIdxs.Count()];
            for (int i = 0; i < layers.Last().neuronIdxs.Count(); i++)
            {
                output[i] = neurons[layers.Last().neuronIdxs[i]].output;
            }

            // Output Layer
            if (displayOutput)
            {
                RegisterOutput("OUTPUT: ");
                foreach (int idx in layers.Last().neuronIdxs)
                    RegisterOutput(neurons[idx].output.ToString());
            }
        }

        public double Train(double[] input, double[] output, bool displayOutput)
        {
            if (learningRate != learningRateT || momentum != momentumT)
            {
                learningRate = learningRateT;
                momentum = momentumT;
                RegisterOutput(String.Format("Updated Learning Rate : {0:0.00000000}", learningRate));
                RegisterOutput(String.Format("Updated Momentum      : {0:0.00000000}",momentum));
            }
            trainingCounter++;
            // Check for valid input
            if (input.Length != layers[0].cntNeurons || output.Length != layers.Last().cntNeurons)
                throw new Exception("Input and layer size mismatch! Invalid input given to NeuralNetwork.Train()");

            // Forward pass
            double[] nnOutput = new double[output.Count()];
            Run(input, out nnOutput, displayOutput);
            if (nnOutput == null)
            {
                Console.Write("Bug!");
                return 0;
            }

            // Compute error

            double totalError = 0;
            Queue<int> buffer = new Queue<int>();

            for (int itr = 0; itr < layers.Last().cntNeurons; itr++)
            {
                Neuron neuron = neurons[layers.Last().neuronIdxs[itr]];
                double error = -neuron.output + output[itr];
                totalError += error * error / 2.0;
                neurons[neuron.Idx].deltaBack = error;
                buffer.Enqueue(neuron.Idx);
            }

            // BackPropagate Error/Deltas

            while (buffer.Count != 0)
            {
                int currIdx = buffer.Dequeue();
                neurons[currIdx].deltaBack *= neurons[currIdx].EvaluateDerivative();
                Neuron neuron = neurons[currIdx];

                foreach (int idx in neuron.incomingConnection)
                {
                    Connection connection = connections[idx];
                    double deltaBackward = connection.weight * neuron.deltaBack;
                    int src = connection.srcDest.FirstOrDefault(x => x.Value == neuron.Idx).Key;
                    neurons[src].deltaBack += deltaBackward;
                    if (buffer.Contains(src) == false)
                        buffer.Enqueue(src);
                }
            }

            // Update Connection Weights
            batchUsed++;
            foreach (Connection connection in connections.Values)
            {
                //if (connection.Idx == 2670)
                //    Console.Write(connection.weight);
                if (connection.updateAllowed == false)
                    continue;
                foreach (int srcIdx in connection.srcDest.Keys)
                {
                    Neuron src = neurons[srcIdx];
                    Neuron dest = neurons[connection.srcDest[srcIdx]];

                    double weightDelta = dest.deltaBack * src.output;
                    connections[connection.Idx].weightDelta += weightDelta / connection.srcDest.Count;
                    neurons[srcIdx].deltaBias += dest.deltaBack;
                }
                if (batchUsed == batchSize)
                {
                    double weightDelta = connections[connection.Idx].weightDelta / (double)batchSize;
                    //if (connection.Idx == 2670)
                    //    Console.Write(" : " + weightDelta);
                    if (weightDelta * connection.previousWeightDelta < 0)
                    {
                        connection.learningRate *= 0.9;
                        connection.previousWeightDelta *= 0.5;
                    }
                    else
                        connection.learningRate *= 1.01;
                    connection.learningRate = Math.Min(connection.learningRate, learningRate);
                    connection.learningRate = Math.Max(connection.learningRate, 0.0000001);
                    connections[connection.Idx].weight += learningRate * (weightDelta + momentum * connection.previousWeightDelta);
                    connections[connection.Idx].previousWeightDelta = 0.9 * connections[connection.Idx].previousWeightDelta + 0.1 * weightDelta;
                    connections[connection.Idx].weightDelta = 0;
                    connections[connection.Idx].learningRate = connection.learningRate;
                    //if (connection.Idx == 2670)
                    //    Console.WriteLine(" : " + connection.weight);
                }
            }
            if (batchUsed == batchSize)
            {
                foreach (Neuron neuron in neurons.Values)
                {
                    neuron.bias += learningRate * neuron.deltaBias / (double)batchSize;
                    neuron.deltaBias = 0;
                }
            }

            if (batchUsed >= batchSize)
                batchUsed = 0;

            //To Update UI
            ui.nnUIForm.graphUpdateSemaphore.WaitOne();
            {
                ui.SetProgressBar(trainingCounter, args.intendedTrainingCnt);
                if (prevErrors.Count == 1000)
                {
                    prevErrorSum -= prevErrors.Dequeue();
                }
                prevErrors.Enqueue(totalError);
                prevErrorSum += totalError;
                xAxisData.Enqueue(trainingCounter);

                double maxValue = output.Max();
                int maxIndex1 = output.ToList().IndexOf(maxValue);

                maxValue = nnOutput.Max();
                int maxIndex2 = nnOutput.ToList().IndexOf(maxValue);

                if (maxIndex1 == maxIndex2)
                    results.Enqueue(true);
                else
                    results.Enqueue(false);
                if (results.Count == 1000)
                    results.Dequeue();
                double correct = 0, wrong = 0;
                foreach (bool res in results)
                {
                    if (res)
                        correct++;
                    else
                        wrong++;
                }
                yAxisData.Enqueue(wrong / (correct + wrong));

                //yAxisData.Enqueue(prevErrorSum / prevErrors.Count);
            }
            ui.nnUIForm.graphUpdateSemaphore.Release();
            return totalError;
        }

        void CallbackHandler(string paramType, string paramValue)
        {
            if (paramType == "LearningRate")
                double.TryParse(paramValue, out learningRateT);
            else if (paramType == "Momentum")
                double.TryParse(paramValue, out momentumT);
        }

        public void SetLearningRate(double learningRate)
        {
            learningRateT = learningRate;
        }

        public void Momentum(double momentum)
        {
            momentumT = momentum;
        }

        public void WaitTillDone()
        {
            while (UIThread.IsAlive)
                Thread.Sleep(1000);
        }

        public void RegisterOutput(string text)
        {
            if (ShowUi && UIThread.IsAlive)
                ui.RegisterOutput(text);
            else
                Console.WriteLine(text);
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
            writer.WriteAttributeString("LearningRate", args.learningRate.ToString());
            writer.WriteAttributeString("Momentum", args.momentum.ToString());

            writer.WriteStartElement("Layers");
            writer.WriteAttributeString("Count", args.layersData.Length.ToString());
            for (int i = 0; i < args.layersData.Length; i++)
            {
                object layerData = args.layersData[i];
                writer.WriteStartElement("Layer");
                writer.WriteAttributeString("Index", i.ToString());
                if (layerData is LayerData.RELU)
                {
                    writer.WriteAttributeString("Type", "RELU");
                }
                else if (layerData is LayerData.MaxPool)
                {
                    writer.WriteAttributeString("Type", "MAXPOOL");
                    writer.WriteAttributeString("Size", ((LayerData.MaxPool)layerData).size.ToString());
                    writer.WriteAttributeString("Stride", ((LayerData.MaxPool)layerData).stride.ToString());
                }
                else if (layerData is LayerData.FullyConnected)
                {
                    LayerData.FullyConnected currLayerData = (LayerData.FullyConnected)layerData;
                    writer.WriteAttributeString("Type", "FULLYCONNECTED");
                    writer.WriteAttributeString("Neurons", currLayerData.cntNeurons.ToString());
                    writer.WriteAttributeString("NeuronType", currLayerData.tFuncType.ToString());
                }
                else if (layerData is LayerData.Convolutional)
                {
                    LayerData.Convolutional currLayerData = (LayerData.Convolutional)layerData;
                    writer.WriteAttributeString("Type", "CONVOLUTIONAL");
                    writer.WriteAttributeString("Stride", currLayerData.stride.ToString());
                    writer.WriteAttributeString("ZeroPadding", currLayerData.padding.ToString());
                    writer.WriteStartElement("Filters");
                    writer.WriteAttributeString("Count", currLayerData.filters.Length.ToString());
                    foreach (int filter in currLayerData.filters)
                    {
                        writer.WriteStartElement("Filter");
                        writer.WriteAttributeString("Size", filter.ToString());
                        writer.WriteEndElement();
                    }
                    writer.WriteEndElement();
                }
                writer.WriteEndElement();
            }
            writer.WriteEndElement(); // Layers

            writer.WriteStartElement("Neurons");
            writer.WriteAttributeString("Count", neurons.Values.Count.ToString());
            foreach (Neuron neuron in neurons.Values)
            {
                writer.WriteStartElement("Neuron");
                writer.WriteAttributeString("Index", neuron.Idx.ToString());
                writer.WriteAttributeString("Bias", neuron.bias.ToString());
                writer.WriteEndElement();
            }
            writer.WriteEndElement(); // Neurons

            writer.WriteStartElement("Connections");
            writer.WriteAttributeString("Count", connections.Values.Count.ToString());
            foreach (Connection connection in connections.Values)
            {
                foreach (int src in connection.srcDest.Keys)
                {
                    writer.WriteStartElement("Connection");
                    writer.WriteAttributeString("Index", connection.Idx.ToString());
                    writer.WriteAttributeString("Source", src.ToString());
                    writer.WriteAttributeString("Destination", connection.srcDest[src].ToString());
                    writer.WriteAttributeString("Weight", connection.weight.ToString());
                    writer.WriteAttributeString("LearningRate", connection.learningRate.ToString());
                    writer.WriteAttributeString("PreviousWeightDelta", connection.previousWeightDelta.ToString());
                    writer.WriteEndElement();
                }
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
            double learningRate = 0;
            NeuralNetworkArgs args = new NeuralNetworkArgs();

            string basePath = "NeuralNetwork/";
            int.TryParse(XPathValue(basePath + "@TrainingDone", ref doc), out trainingCounter);
            int.TryParse(XPathValue(basePath + "@IndendedTrainingCnt", ref doc), out args.intendedTrainingCnt);
            double.TryParse(XPathValue(basePath + "@LearningRate", ref doc), out args.learningRate);
            double.TryParse(XPathValue(basePath + "@Momentum", ref doc), out args.momentum);
            basePath += "Layers/";

            int.TryParse(XPathValue(basePath + "@Count", ref doc), out layerCnt);
            args.layersData = new object[layerCnt];

            basePath += "Layer[@Index='{0}']/@{1}";
            XmlNodeList layerList = doc.SelectNodes("NeuralNetwork/Layers/Layer");
            for (int i = 0; i < layerCnt; i++)
            {
                XmlNode layerNode = layerList[i];
                switch (layerNode.Attributes["Type"].Value)
                {
                    case "RELU":
                        LayerData.RELU reluLayer = new LayerData.RELU();
                        args.layersData[i] = reluLayer;
                        break;
                    case "MAXPOOL":
                        LayerData.MaxPool maxpoolLayer = new LayerData.MaxPool();
                        int.TryParse(layerNode.Attributes["Size"].Value, out maxpoolLayer.size);
                        int.TryParse(layerNode.Attributes["Stride"].Value, out maxpoolLayer.stride);
                        args.layersData[i] = maxpoolLayer;
                        break;
                    case "FULLYCONNECTED":
                        LayerData.FullyConnected fullyLayer = new LayerData.FullyConnected();
                        int.TryParse(layerNode.Attributes["Neurons"].Value, out fullyLayer.cntNeurons);
                        Enum.TryParse<TransferFuncType>(layerNode.Attributes["NeuronType"].Value, out fullyLayer.tFuncType);
                        args.layersData[i] = fullyLayer;
                        break;
                    case "CONVOLUTIONAL":
                        LayerData.Convolutional convolutionalLayer = new LayerData.Convolutional();
                        int.TryParse(layerNode.Attributes["Stride"].Value, out convolutionalLayer.stride);
                        bool.TryParse(layerNode.Attributes["ZeroPadding"].Value, out convolutionalLayer.padding);
                        XmlDocument filterXml = new XmlDocument();
                        filterXml.LoadXml(layerNode.OuterXml);
                        XmlNodeList filterList = filterXml.SelectNodes("Layer/Filters/Filter");
                        convolutionalLayer.filters = new int[filterList.Count];
                        int j = 0;
                        foreach (XmlNode filterNode in filterList)
                        {
                            int.TryParse(filterNode.Attributes["Size"].Value, out convolutionalLayer.filters[j++]);
                        }
                        args.layersData[i] = convolutionalLayer;
                        break;
                    default:
                        throw new Exception("Invalid Layer Type Entry Found!!!");
                }
            }

            NeuralNetwork nn = new NeuralNetwork(args.intendedTrainingCnt, 1, args.learningRate, args.momentum, showNNUI, args.layersData);
            int.TryParse(XPathValue("NeuralNetwork/@TrainingDone", ref doc), out nn.trainingCounter);

            nn.RegisterOutput("Loading Neurons");
            basePath = "NeuralNetwork/Neurons/Neuron[@Index='{0}']/@Bias";
            int neuronCnt;
            int.TryParse(XPathValue("NeuralNetwork/Neurons/@Count", ref doc), out neuronCnt);
            for (int i = 0; i < neuronCnt; i++)
            {
                double.TryParse(XPathValue(string.Format(basePath, i.ToString()), ref doc), out nn.neurons[i].bias);
            }
            nn.RegisterOutput("Loading Connections");
            basePath = "NeuralNetwork/Connections/Connection[@Index='{0}']/@{1}";

            XmlNodeList connectionList = doc.SelectNodes("NeuralNetwork/Connections/Connection");
            foreach (XmlNode connection in connectionList)
            {
                int idx, src, dest;
                int.TryParse(connection.Attributes["Index"].Value, out idx);
                int.TryParse(connection.Attributes["Source"].Value, out src);
                int.TryParse(connection.Attributes["Destination"].Value, out dest);
                double.TryParse(connection.Attributes["Weight"].Value, out nn.connections[idx].weight);
                double.TryParse(connection.Attributes["PreviousWeightDelta"].Value, out nn.connections[idx].previousWeightDelta);
                double.TryParse(connection.Attributes["LearningRate"].Value, out nn.connections[idx].learningRate);
                nn.connections[idx].srcDest[src] = dest;
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
        SOFTMAX,
        MAXPOOL
    }

    public class LayerData
    {
        public class FullyConnected
        {
            public int cntNeurons;
            public TransferFuncType tFuncType;
        }

        public class Convolutional
        {
            public int[] filters;
            public int stride;
            public bool padding;
        }

        public class RELU
        {

        }

        public class MaxPool
        {
            public int size;
            public int stride;
        }
    }

    class Layer
    {
        public int cntNeurons;
        public List<int> neuronIdxs;
        public TransferFuncType tFuncType;

        public Layer(TransferFuncType tFuncType)
        {
            cntNeurons = 0;
            neuronIdxs = new List<int>();
            this.tFuncType = tFuncType;
        }

        public Layer(Layer previousLayer, object layerData,
            ref Dictionary<int, Neuron> neurons, ref int neuronCounter,
            ref Dictionary<int, Connection> connections, ref int connectionCounter,
            double learningRate,
            double weightRescaleFactor)
        {
            // Form connections based upon layer type
            if (layerData is LayerData.RELU)
            {
                // One-To-One Connections
                neuronIdxs = new List<int>();
                tFuncType = TransferFuncType.RECTILINEAR;
                foreach (int prevNeuronIdx in previousLayer.neuronIdxs)
                {
                    Neuron neuron = new Neuron(tFuncType);
                    neuron.biasAllowed = false;
                    Connection connection = new Connection(ref connections, ref connectionCounter, learningRate, weightRescaleFactor, false);
                    connection.weight = 1;
                    neuron.Idx = neuronCounter;
                    connection.srcDest[prevNeuronIdx] = neuron.Idx;
                    neuron.incomingConnection.Add(connection.Idx);
                    neurons[prevNeuronIdx].outgoingConnection.Add(connection.Idx);
                    neurons[neuronCounter++] = neuron;
                    neuronIdxs.Add(neuron.Idx);
                }
            }
            else if (layerData is LayerData.FullyConnected)
            {
                // Cross Connections
                LayerData.FullyConnected currLayerData = (LayerData.FullyConnected)layerData;
                tFuncType = currLayerData.tFuncType;
                neuronIdxs = new List<int>();
                for (int i = 0; i < currLayerData.cntNeurons; i++)
                {
                    Neuron neuron = new Neuron(tFuncType, previousLayer, ref neurons, ref neuronCounter, ref connections,
                        ref connectionCounter, learningRate, weightRescaleFactor);
                    neuronIdxs.Add(neuron.Idx);
                }
            }
            else if (layerData is LayerData.Convolutional)
            {
                LayerData.Convolutional currLayerData = (LayerData.Convolutional)layerData;
                tFuncType = TransferFuncType.LINEAR;
                neuronIdxs = new List<int>();
                int dimIn = (int)Math.Sqrt(previousLayer.cntNeurons);

                // Form connections for each filter
                foreach (int filter in currLayerData.filters)
                {
                    int filterStartIdx, filterEndIdx;
                    if (currLayerData.padding)
                    {
                        filterStartIdx = -filter / 2;
                        filterEndIdx = filter / 2;
                    }
                    else
                    {
                        filterStartIdx = 0;
                        filterEndIdx = filter - 1;
                    }
                    Dictionary<int, int> filterConnections = new Dictionary<int, int>();
                    for (int k1 = filterStartIdx; k1 <= filterEndIdx; k1++)
                        for (int k2 = filterStartIdx; k2 <= filterEndIdx; k2++)
                        {
                            Connection connection = new Connection(ref connections, ref connectionCounter, learningRate, weightRescaleFactor);
                            int hashIdx = (k1 + filter / 2) * filter + (k2 + filter / 2);
                            connection.weight = 1;
                            connection.updateAllowed = false;
                            filterConnections[hashIdx] = connection.Idx;
                        }
                    // Zero padding is introduced for area which is not completly overlapped by filter
                    for (int i = 0; i + (currLayerData.padding ? 0 : filter - 1) < dimIn; i += currLayerData.stride)
                        for (int j = 0; j + (currLayerData.padding ? 0 : filter - 1) < dimIn; j += currLayerData.stride)
                        {
                            Neuron neuron = new Neuron(tFuncType);
                            neuron.Idx = neuronCounter;
                            neuron.biasAllowed = false;
                            neurons[neuronCounter++] = neuron;
                            neuronIdxs.Add(neuron.Idx);
                            for (int k1 = filterStartIdx; k1 <= filterEndIdx; k1++)
                                for (int k2 = filterStartIdx; k2 <= filterEndIdx; k2++)
                                {
                                    int idx = GetIndex(i + k1, j + k2, dimIn, previousLayer);
                                    if (idx == -1)
                                        continue;
                                    int hashIdx = (k1 + filter / 2) * filter + (k2 + filter / 2);
                                    int cIdx = filterConnections[hashIdx];
                                    connections[cIdx].srcDest[idx] = neuron.Idx;
                                    neurons[idx].outgoingConnection.Add(cIdx);
                                    neurons[neuron.Idx].incomingConnection.Add(cIdx);
                                }
                        }
                }
            }
            else if (layerData is LayerData.MaxPool)
            {
                LayerData.MaxPool currLayerData = (LayerData.MaxPool)layerData;
                this.tFuncType = TransferFuncType.MAXPOOL;
                neuronIdxs = new List<int>();
                int dimIn = (int)Math.Sqrt(previousLayer.cntNeurons);

                // Zero padding is introduced for area which is not completly overlapped by filter
                for (int i = 0; i < dimIn; i += currLayerData.stride)
                    for (int j = 0; j < dimIn; j += currLayerData.stride)
                    {
                        Neuron neuron = new Neuron(tFuncType);
                        neuron.biasAllowed = false;
                        neuron.Idx = neuronCounter;
                        neurons[neuronCounter++] = neuron;
                        neuronIdxs.Add(neuron.Idx);

                        Dictionary<int, int> filterConnections = new Dictionary<int, int>();
                        for (int k1 = 0; k1 < currLayerData.size; k1++)
                            for (int k2 = 0; k2 < currLayerData.size; k2++)
                            {
                                Connection connection = new Connection(ref connections, ref connectionCounter, learningRate, weightRescaleFactor, false);
                                connection.weight = 1;
                                int hashIdx = k1 * currLayerData.size + k2;
                                filterConnections[hashIdx] = connection.Idx;
                            }

                        // Form new connections
                        for (int k1 = 0; k1 < currLayerData.size; k1++)
                            for (int k2 = 0; k2 < currLayerData.size; k2++)
                            {
                                int idx = GetIndex(i + k1, j + k2, dimIn, previousLayer);
                                if (idx == -1)
                                    continue;
                                int hashIdx = k1 * currLayerData.size + k2;
                                int cIdx = filterConnections[hashIdx];
                                connections[cIdx].srcDest[idx] = neuron.Idx;
                                neurons[idx].outgoingConnection.Add(cIdx);
                                neurons[neuron.Idx].incomingConnection.Add(cIdx);
                            }
                    }
            }
            else
                throw new Exception("Invalid LayerConnectionStyle given to Layer.FormConnections  !!!!");
            cntNeurons = neuronIdxs.Count();
        }

        public void MergeLayer(Layer layer)
        {
            if (layer.tFuncType != this.tFuncType)
                throw new Exception("Transfer function type miss matched, in Layer.MergeLayer!!!");
            foreach (int neuronIdx in layer.neuronIdxs)
            {
                if (neuronIdxs.Contains(neuronIdx) == false)
                {
                    neuronIdxs.Add(neuronIdx);
                    cntNeurons++;
                }
            }
        }

        public static void ConstructLayers(object[] layersData, ref List<Layer> layers,
                                            ref Dictionary<int, Neuron> neurons, ref int neuronCounter,
                                            ref Dictionary<int, Connection> connections, ref int connectionCounter,
                                            double learningRate,
                                            double weightRescaleFactor)
        {
            List<Layer> lastSubLayers = new List<Layer>();
            Layer lastLayer = null;
            foreach (object layerData in layersData.ToArray())
            {
                if (layerData is LayerData.FullyConnected)
                {
                    lastLayer = new Layer(lastLayer, layerData, ref neurons, ref neuronCounter, ref connections, ref connectionCounter, learningRate, weightRescaleFactor);
                    lastSubLayers.Clear();
                    lastSubLayers.Add(lastLayer);
                }
                else if (layerData is LayerData.Convolutional)
                {
                    List<Layer> currSubLayers = new List<Layer>();
                    LayerData.Convolutional currLayerData = (LayerData.Convolutional)layerData;
                    lastLayer = new Layer(TransferFuncType.LINEAR);
                    foreach (Layer subLayer in lastSubLayers)
                    {
                        foreach (int filter in currLayerData.filters)
                        {
                            LayerData.Convolutional layerDataTmp = new LayerData.Convolutional();
                            layerDataTmp.stride = currLayerData.stride;
                            layerDataTmp.filters = new int[] { filter };

                            Layer layerTmp = new Layer(subLayer, layerDataTmp, ref neurons, ref neuronCounter, ref connections, ref connectionCounter, learningRate, weightRescaleFactor);
                            currSubLayers.Add(layerTmp);
                            lastLayer.MergeLayer(layerTmp);
                        }
                    }
                    lastSubLayers = currSubLayers;
                }
                else if (layerData is LayerData.RELU)
                {
                    List<Layer> currSubLayers = new List<Layer>();
                    lastLayer = new Layer(TransferFuncType.RECTILINEAR);
                    foreach (Layer subLayer in lastSubLayers)
                    {
                        Layer layerTmp = new Layer(subLayer, layerData, ref neurons, ref neuronCounter, ref connections, ref connectionCounter, learningRate, weightRescaleFactor);
                        currSubLayers.Add(layerTmp);
                        lastLayer.MergeLayer(layerTmp);
                    }
                    lastSubLayers = currSubLayers;
                }
                else if (layerData is LayerData.MaxPool)
                {
                    List<Layer> currSubLayers = new List<Layer>();
                    lastLayer = new Layer(TransferFuncType.MAXPOOL);
                    foreach (Layer subLayer in lastSubLayers)
                    {
                        Layer layerTmp = new Layer(subLayer, layerData, ref neurons, ref neuronCounter, ref connections, ref connectionCounter, learningRate, weightRescaleFactor);
                        currSubLayers.Add(layerTmp);
                        lastLayer.MergeLayer(layerTmp);
                    }
                    lastSubLayers = currSubLayers;
                }
                else
                {
                    throw new Exception("Invalid input given to Layer.ConstructLayers!!!!");
                }
                layers.Add(lastLayer);
            }
        }

        private int GetIndex(int i, int j, int dim, Layer layer)
        {
            if (i < 0 || i >= dim || j < 0 || j >= dim)
                return -1;
            else
                return layer.neuronIdxs[i * dim + j];
        }
    }

    class Neuron
    {
        public double input, output, deltaBack, bias, deltaBias;
        public int Idx;
        public List<int> incomingConnection, outgoingConnection;
        public TransferFuncType tFuncType;
        public bool biasAllowed;

        private void INIT(TransferFuncType tFuncType)
        {
            input = 0;
            deltaBack = 0;
            output = 0;
            deltaBias = 0;
            bias = Gaussian.GetRandomGaussian();
            this.tFuncType = tFuncType;
            biasAllowed = true;

            incomingConnection = new List<int>();
            outgoingConnection = new List<int>();
        }

        public Neuron(TransferFuncType tFuncType)
        {
            INIT(tFuncType);
        }

        public Neuron(TransferFuncType tFuncType, Layer previousLayer,
            ref Dictionary<int, Neuron> neurons, ref int neuronCounter,
            ref Dictionary<int, Connection> connections, ref int connectionCounter,
            double learningRate,
            double weightRescaleFactor)
        {
            INIT(tFuncType);

            if (previousLayer != null)
            {
                foreach (int neuronIdx in previousLayer.neuronIdxs)
                {
                    Connection connection = new Connection(ref connections, ref connectionCounter, learningRate, weightRescaleFactor);
                    connection.srcDest[neuronIdx] = neuronCounter;
                    this.incomingConnection.Add(connection.Idx);
                    neurons[neuronIdx].outgoingConnection.Add(connection.Idx);
                }
            }

            Idx = neuronCounter;
            neurons[neuronCounter++] = this;
        }

        public double Evaluate(ref Dictionary<int, Neuron> neurons, ref Dictionary<int, Connection> connections)
        {
            if (!biasAllowed)
                bias = 0;
            if (tFuncType == TransferFuncType.MAXPOOL)
            {
                int cFinalIdx = -1;
                double finalOut = int.MinValue;
                foreach (int cIdx in incomingConnection)
                {
                    connections[cIdx].weight = 0;
                    foreach (int key in connections[cIdx].srcDest.Keys)
                    {
                        int srcIdx = key;
                        if (neurons[srcIdx].output > finalOut)
                        {
                            finalOut = neurons[srcIdx].output;
                            cFinalIdx = cIdx;
                        }
                    }
                }
                connections[cFinalIdx].weight = 1;
                output = finalOut;
                return output;
            }
            if (tFuncType != TransferFuncType.NONE)
                input += bias;
            output = TransferFunction.Evaluate(tFuncType, input);
            return output;
        }

        public double EvaluateDerivative()
        {
            if (tFuncType == TransferFuncType.MAXPOOL)
                return 1;
            if (tFuncType == TransferFuncType.SIGMOID || tFuncType == TransferFuncType.SOFTMAX)
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
        public Dictionary<int, int> srcDest;
        public int Idx;
        public double weight;
        public double weightDelta;
        public double previousWeightDelta;
        public double learningRate;
        public bool updateAllowed;

        public Connection(ref Dictionary<int, Connection> connections, ref int connectionCounter, double learningRate,
                          double weightRescaleFactor, bool updateAllowed = true)
        {
            this.updateAllowed = updateAllowed;
            this.learningRate = learningRate;
            weight = Gaussian.GetRandomGaussian() / weightRescaleFactor;
            weightDelta = 0;
            previousWeightDelta = 0;
            srcDest = new Dictionary<int, int>();
            Idx = connectionCounter;
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
                case TransferFuncType.SOFTMAX:
                    output = Math.Exp(x);
                    break;
                default:
                    throw new FormatException("Invalid Input Provided To TransferFunction.Evaluate()");
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