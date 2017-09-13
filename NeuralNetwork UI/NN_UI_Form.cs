using System;
using System.Timers;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace Dabas.NeuralNetwork_UI
{
    public partial class NNUIFORM : Form
    {
        private System.Windows.Forms.DataVisualization.Charting.Chart errorGraph;
        private Label label1;

        ChartArea chartArea;
        Series series;
        Queue<double> xAxisData, yAxisData;
        System.Timers.Timer graphUpdateTimer;

        delegate void UpdateCall();

        public bool graphUpdateOngoing = false;

        public NNUIFORM()
        {
            InitializeComponent();

            graphUpdateTimer = new System.Timers.Timer();
            graphUpdateTimer.Enabled = true;
            graphUpdateTimer.Interval = 1000;
            graphUpdateTimer.Elapsed += new ElapsedEventHandler(graphUpdateTimer_Tick);
            graphUpdateTimer.Start();

            errorGraph.Titles.Clear();
            errorGraph.ChartAreas.Clear();
            errorGraph.Series.Clear();

            Axis axisX = new Axis();
            Axis axisY = new Axis();
            axisX.MajorGrid.Enabled = false;

            chartArea = new ChartArea
            {
                AxisX = axisX,
                AxisY = axisY
            };

            series = new Series
            {
                Name = "Error",
                Color = Color.Red,
                BorderWidth = 1,
                ChartType = SeriesChartType.Line,
                IsVisibleInLegend = true
            };

            errorGraph.ChartAreas.Add(chartArea);
            errorGraph.Series.Add(series);
        }

        private void InitializeComponent()
        {
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.errorGraph = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.label1 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.errorGraph)).BeginInit();
            this.SuspendLayout();
            // 
            // errorGraph
            // 
            this.errorGraph.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
            | System.Windows.Forms.AnchorStyles.Left)
            | System.Windows.Forms.AnchorStyles.Right)));
            chartArea1.Name = "ChartArea1";
            this.errorGraph.ChartAreas.Add(chartArea1);
            legend1.Name = "Legend1";
            this.errorGraph.Legends.Add(legend1);
            this.errorGraph.Location = new System.Drawing.Point(12, 30);
            this.errorGraph.Name = "errorGraph";
            series1.ChartArea = "ChartArea1";
            series1.Legend = "Legend1";
            series1.Name = "Series1";
            this.errorGraph.Series.Add(series1);
            this.errorGraph.Size = new System.Drawing.Size(584, 349);
            this.errorGraph.TabIndex = 0;
            this.errorGraph.Text = "errorChart";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.Color.Maroon;
            this.label1.Location = new System.Drawing.Point(12, 11);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(124, 16);
            this.label1.TabIndex = 1;
            this.label1.Text = "Error Vs Iteration";
            // 
            // NNUIFORM
            // 
            this.ClientSize = new System.Drawing.Size(1044, 676);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.errorGraph);
            this.Name = "NNUIFORM";
            this.Text = "Neural Network UI";
            ((System.ComponentModel.ISupportInitialize)(this.errorGraph)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        private void UpdateErrorGraph()
        {
            if (errorGraph.InvokeRequired)
            {
                errorGraph.Invoke(new UpdateCall(UpdateErrorGraph));
            }
            else
            {
                if (xAxisData == null || yAxisData == null)
                    return;
                if (xAxisData.Count != yAxisData.Count)
                    return;

                while (xAxisData.Count != 0)
                {
                    double x, y;
                    x = xAxisData.Dequeue();
                    y = yAxisData.Dequeue();
                    series.Points.AddXY(x, y);
                }
            }
        }

        private void graphUpdateTimer_Tick(object sender, EventArgs e)
        {
            
            if (graphUpdateOngoing == false)
            {
                graphUpdateOngoing = true;
                UpdateErrorGraph();
                graphUpdateOngoing = false;
            }
        }

        public void AddToChart(ref Queue<double> xAxisData, ref Queue<double> yAxisData)
        {
            this.xAxisData = xAxisData;
            this.yAxisData = yAxisData;
        }
    }
}
