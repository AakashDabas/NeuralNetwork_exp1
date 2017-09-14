using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Dabas.NeuralNetwork_UI
{
    public class NeuralNetworkUI
    {
        public NNUIFORM nnUIForm;
        delegate void InvokeHelperStr1(string text);
        delegate void InvokeHelperInt2(int a, int b);

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
            if (nnUIForm.outputBox.InvokeRequired)
            {
                nnUIForm.outputBox.Invoke(new InvokeHelperStr1(RegisterOutput), text);
            }
            else
            {
                nnUIForm.outputBox.Text += System.DateTime.Now.ToString() + " | " + text + "\n";
            }
        }

        public void SetProgressBar(int cntTrainingDone, int cntTotalTraining)
        {
            if (nnUIForm.trainingProgressBar.InvokeRequired)
                nnUIForm.trainingProgressBar.Invoke(new InvokeHelperInt2(SetProgressBar), cntTrainingDone, cntTotalTraining);
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
                nnUIForm.trainingPer.Invoke(new InvokeHelperInt2(SetProgressBar), cntTrainingDone, cntTotalTraining);
            else
            {
                double percentage = (double)cntTrainingDone / (double)cntTotalTraining * 100;
                nnUIForm.trainingPer.Text = string.Format("{0:0.0000} %", percentage);
            }

        }
    }
}
