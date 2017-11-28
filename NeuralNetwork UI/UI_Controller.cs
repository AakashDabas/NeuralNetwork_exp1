using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Dabas.NeuralNetwork_UI
{
    public delegate void Callback(string paramType, string paramValue);

    public class NeuralNetworkUI
    {
        public NNUIFORM nnUIForm;
        delegate void InvokeHelperStr1(string text);
        delegate void InvokeHelperInt2(int a, int b);
        static Semaphore semaphore = new Semaphore(1, 1);

        public NeuralNetworkUI(Callback _callback)
        {
            nnUIForm = new NNUIFORM(_callback);
        }

        [STAThread]
        public void StartUI()
        {
            Application.EnableVisualStyles();
            Application.Run(nnUIForm);
        }

        public void AddToChart(ref Queue<double> xAxisData, ref Queue<double> yAxisData)
        {
            nnUIForm.AddToChart(ref xAxisData, ref yAxisData);
        }

        public void RegisterOutput(string text)
        {
            if (nnUIForm.outputBox.InvokeRequired)
            {
                semaphore.WaitOne();
                nnUIForm.outputBox.BeginInvoke((MethodInvoker)(() => RegisterOutput(text)));
                semaphore.Release();
            }
            else
            {
                if (nnUIForm.outputBox.SelectedText == "")
                    nnUIForm.outputBox.Text += System.DateTime.Now.ToString() + " | " + text + "\n";
            }
        }

        public void SetProgressBar(int cntTrainingDone, int cntTotalTraining)
        {
            if (nnUIForm.trainingProgressBar.InvokeRequired)
                nnUIForm.trainingProgressBar.BeginInvoke(new InvokeHelperInt2(SetProgressBar), cntTrainingDone, cntTotalTraining);
            else
            {
                nnUIForm.trainingProgressBar.Maximum = cntTotalTraining;
                nnUIForm.trainingProgressBar.Value = cntTrainingDone;
                SetTrainingPercentange(cntTrainingDone, cntTotalTraining);
            }
        }

        public void SetLearningRate(string rate)
        {
            if (nnUIForm.rateInput.InvokeRequired)
                nnUIForm.rateInput.BeginInvoke(new InvokeHelperStr1(SetLearningRate), rate);
            else
            {
                nnUIForm.rateInput.Text = rate;
            }
        }

        public void SetMomentum(string momentum)
        {
            if (nnUIForm.momentumInput.InvokeRequired)
                nnUIForm.momentumInput.BeginInvoke(new InvokeHelperStr1(SetMomentum), momentum);
            else
            {
                nnUIForm.momentumInput.Text = momentum;
            }
        }

        public void SetTrainingPercentange(int cntTrainingDone, int cntTotalTraining)
        {
            if (nnUIForm.trainingPer.InvokeRequired)
                nnUIForm.trainingPer.BeginInvoke(new InvokeHelperInt2(SetProgressBar), cntTrainingDone, cntTotalTraining);
            else
            {
                double percentage = (double)cntTrainingDone / (double)cntTotalTraining * 100;
                nnUIForm.trainingPer.Text = string.Format("{0:0.0000} %", percentage);
            }
        }
    }
}
