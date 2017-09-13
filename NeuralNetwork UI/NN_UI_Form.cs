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
using System.Threading;

namespace Dabas.NeuralNetwork_UI
{
    public partial class NNUIFORM : Form
    {

        ChartArea chartArea;
        Series series;
        Queue<double> xAxisData, yAxisData;
        System.Timers.Timer graphUpdateTimer;
        double zoomScale = 1;

        private Chart errorGraph;
        private PictureBox upArrow;
        private PictureBox downArrow;
        private RichTextBox outputBox;
        private Label label1;
        private Label label2;

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
            axisX.ArrowStyle = AxisArrowStyle.Triangle;
            axisY.ArrowStyle = AxisArrowStyle.Triangle;
            axisY.LineColor = Color.Orange;
            axisX.LineColor = Color.Orange;
            axisY.MajorGrid.LineColor = Color.Orange;
            axisX.LabelStyle.ForeColor = Color.Orange;
            axisY.LabelStyle.ForeColor = Color.Orange;

            chartArea = new ChartArea
            {
                AxisX = axisX,
                AxisY = axisY
            };

            series = new Series
            {
                Name = "Error",
                Color = Color.Plum,
                BorderWidth = 2,
                ChartType = SeriesChartType.Line,
                IsVisibleInLegend = true
            };

            chartArea.CursorX.IsUserSelectionEnabled = true;
            //chartArea.CursorY.IsUserSelectionEnabled = true;
            // Set automatic zooming
            chartArea.AxisX.ScaleView.Zoomable = true;
            chartArea.AxisY.ScaleView.Zoomable = true;

            // Set automatic scrolling 
            chartArea.CursorX.AutoScroll = true;
            //chartArea.CursorY.AutoScroll = true;

            // Allow user selection for Zoom
            chartArea.CursorX.IsUserSelectionEnabled = true;
            chartArea.BackColor = Color.Transparent;

            errorGraph.ChartAreas.Add(chartArea);
            errorGraph.Series.Add(series);

            for (int i = 0; i < 100; i++)
            {
                outputBox.Text += "\nTangoCharle";
                outputBox.Text += "\nAakash Dabas";
            }

        }

        private void InitializeComponent()
        {
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea3 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend3 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series3 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.errorGraph = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.downArrow = new System.Windows.Forms.PictureBox();
            this.upArrow = new System.Windows.Forms.PictureBox();
            this.outputBox = new System.Windows.Forms.RichTextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.errorGraph)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.downArrow)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.upArrow)).BeginInit();
            this.SuspendLayout();
            // 
            // errorGraph
            // 
            this.errorGraph.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.errorGraph.BackColor = System.Drawing.Color.SteelBlue;
            this.errorGraph.BackGradientStyle = System.Windows.Forms.DataVisualization.Charting.GradientStyle.TopBottom;
            this.errorGraph.BackSecondaryColor = System.Drawing.Color.Black;
            this.errorGraph.BorderlineColor = System.Drawing.Color.DimGray;
            this.errorGraph.BorderlineDashStyle = System.Windows.Forms.DataVisualization.Charting.ChartDashStyle.Solid;
            this.errorGraph.BorderlineWidth = 2;
            chartArea3.Name = "ChartArea1";
            this.errorGraph.ChartAreas.Add(chartArea3);
            legend3.Name = "Legend1";
            this.errorGraph.Legends.Add(legend3);
            this.errorGraph.Location = new System.Drawing.Point(17, 204);
            this.errorGraph.Name = "errorGraph";
            series3.ChartArea = "ChartArea1";
            series3.Legend = "Legend1";
            series3.Name = "Series1";
            this.errorGraph.Series.Add(series3);
            this.errorGraph.Size = new System.Drawing.Size(951, 526);
            this.errorGraph.TabIndex = 5;
            this.errorGraph.Text = "errorGraph";
            // 
            // downArrow
            // 
            this.downArrow.BackColor = System.Drawing.Color.SteelBlue;
            this.downArrow.BackgroundImage = global::NeuralNetwork_UI.Properties.Resources.Down;
            this.downArrow.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            this.downArrow.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.downArrow.Cursor = System.Windows.Forms.Cursors.Hand;
            this.downArrow.Location = new System.Drawing.Point(26, 252);
            this.downArrow.Name = "downArrow";
            this.downArrow.Size = new System.Drawing.Size(30, 30);
            this.downArrow.TabIndex = 7;
            this.downArrow.TabStop = false;
            this.downArrow.Click += new System.EventHandler(this.downArrow_Click);
            // 
            // upArrow
            // 
            this.upArrow.BackColor = System.Drawing.Color.SteelBlue;
            this.upArrow.BackgroundImage = global::NeuralNetwork_UI.Properties.Resources.Up;
            this.upArrow.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            this.upArrow.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.upArrow.Cursor = System.Windows.Forms.Cursors.Hand;
            this.upArrow.Location = new System.Drawing.Point(26, 216);
            this.upArrow.Name = "upArrow";
            this.upArrow.Size = new System.Drawing.Size(30, 30);
            this.upArrow.TabIndex = 6;
            this.upArrow.TabStop = false;
            this.upArrow.Click += new System.EventHandler(this.upArrow_Click);
            // 
            // outputBox
            // 
            this.outputBox.BackColor = System.Drawing.Color.GhostWhite;
            this.outputBox.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.outputBox.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.outputBox.ForeColor = System.Drawing.Color.Coral;
            this.outputBox.Location = new System.Drawing.Point(12, 28);
            this.outputBox.Name = "outputBox";
            this.outputBox.Size = new System.Drawing.Size(951, 145);
            this.outputBox.TabIndex = 8;
            this.outputBox.Text = "";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.ForeColor = System.Drawing.Color.OrangeRed;
            this.label1.Location = new System.Drawing.Point(14, 185);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(155, 16);
            this.label1.TabIndex = 9;
            this.label1.Text = "Avg Error Vs Iteration";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.ForeColor = System.Drawing.Color.OrangeRed;
            this.label2.Location = new System.Drawing.Point(14, 7);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(71, 16);
            this.label2.TabIndex = 10;
            this.label2.Text = "OUTPUT";
            // 
            // NNUIFORM
            // 
            this.BackColor = System.Drawing.Color.Black;
            this.ClientSize = new System.Drawing.Size(984, 742);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.outputBox);
            this.Controls.Add(this.downArrow);
            this.Controls.Add(this.upArrow);
            this.Controls.Add(this.errorGraph);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.ForeColor = System.Drawing.Color.SeaShell;
            this.MinimumSize = new System.Drawing.Size(1000, 700);
            this.Name = "NNUIFORM";
            this.Text = "Neural Network UI";
            ((System.ComponentModel.ISupportInitialize)(this.errorGraph)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.downArrow)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.upArrow)).EndInit();
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

        private void upArrow_Click(object sender, EventArgs e)
        {
            chartArea.AxisY.ScaleView.Zoomable = true;
            zoomScale *= 2;
            double lowerLimit = chartArea.AxisY.Minimum;
            double upperLimit = chartArea.AxisX.Maximum;
            chartArea.AxisY.ScaleView.Zoom(lowerLimit * zoomScale, upperLimit * zoomScale);
        }

        private void downArrow_Click(object sender, EventArgs e)
        {
            chartArea.AxisY.ScaleView.Zoomable = true;
            zoomScale /= 2;
            double lowerLimit = chartArea.AxisY.Minimum;
            double upperLimit = chartArea.AxisX.Maximum;
            chartArea.AxisY.ScaleView.Zoom(lowerLimit * zoomScale, upperLimit * zoomScale);
        }

        public void AddToChart(ref Queue<double> xAxisData, ref Queue<double> yAxisData)
        {
            this.xAxisData = xAxisData;
            this.yAxisData = yAxisData;
        }
    }
}
