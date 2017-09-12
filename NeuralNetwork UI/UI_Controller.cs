using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Dabas.NeuralNetwork_UI
{
    public class NeuralNetworkUI
    {
        NNUIFORM nnUIForm;

        [STAThread]
        public void StartUI()
        {
            nnUIForm = new NNUIFORM();
            Application.EnableVisualStyles();
            Application.Run(nnUIForm);
        }
    }
}
