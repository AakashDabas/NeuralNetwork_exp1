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
    }
}
