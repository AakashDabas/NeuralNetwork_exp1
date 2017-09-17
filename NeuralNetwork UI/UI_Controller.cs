using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Dabas.NeuralNetwork_UI
{
    public class NeuralNetworkUI
    {
        public NNUIFORM nnUIForm;
        delegate void InvokeHelperStr1(string text);
        delegate void InvokeHelperInt2(int a, int b);
        static Semaphore semaphore = new Semaphore(3, 3);

        public NeuralNetworkUI()
        {
            nnUIForm = new NNUIFORM();
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
            semaphore.WaitOne();
            if (nnUIForm.outputBox.InvokeRequired)
            {

                nnUIForm.outputBox.BeginInvoke((MethodInvoker)(() => RegisterOutput(text)));
            }
            else
            {
                lock (nnUIForm.outputBox.Text)
                {
                    if (nnUIForm.outputBox.SelectedText == "")
                        nnUIForm.outputBox.Text += System.DateTime.Now.ToString() + " | " + text + "\n";
                }

            }
            semaphore.Release();
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
