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
        System.Timers.Timer updateTimer;
        double zoomScale = 1;
        public string outputText = "";
        Semaphore graphDataSemaphore = new Semaphore(1, 1);
        public Semaphore graphUpdateSemaphore = new Semaphore(1, 1);

        delegate void UpdateCall();

        private Chart errorGraph;
        private PictureBox upArrow;
        private PictureBox downArrow;
        public RichTextBox outputBox;
        public ProgressBar trainingProgressBar;
        private Label trainingLabel;
        public Label trainingPer;


        public NNUIFORM()
        {
            InitializeComponent();
            updateTimer = new System.Timers.Timer();
            updateTimer.Enabled = true;
            updateTimer.Interval = 1000;
            updateTimer.Elapsed += new ElapsedEventHandler(updateTimer_Tick);
            updateTimer.Start();
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
                BorderWidth = 1,
                ChartType = SeriesChartType.Line
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

            series.IsVisibleInLegend = false;

            errorGraph.ChartAreas.Add(chartArea);
            errorGraph.Series.Add(series);

        }

        private void InitializeComponent()
        {
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.errorGraph = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.downArrow = new System.Windows.Forms.PictureBox();
            this.upArrow = new System.Windows.Forms.PictureBox();
            this.outputBox = new System.Windows.Forms.RichTextBox();
            this.trainingProgressBar = new System.Windows.Forms.ProgressBar();
            this.trainingLabel = new System.Windows.Forms.Label();
            this.trainingPer = new System.Windows.Forms.Label();
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
            chartArea1.Name = "ChartArea1";
            this.errorGraph.ChartAreas.Add(chartArea1);
            legend1.Name = "Legend1";
            this.errorGraph.Legends.Add(legend1);
            this.errorGraph.Location = new System.Drawing.Point(17, 242);
            this.errorGraph.Name = "errorGraph";
            series1.ChartArea = "ChartArea1";
            series1.Legend = "Legend1";
            series1.Name = "Series1";
            this.errorGraph.Series.Add(series1);
            this.errorGraph.Size = new System.Drawing.Size(743, 249);
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
            this.downArrow.Location = new System.Drawing.Point(26, 295);
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
            this.upArrow.Location = new System.Drawing.Point(26, 259);
            this.upArrow.Name = "upArrow";
            this.upArrow.Size = new System.Drawing.Size(30, 30);
            this.upArrow.TabIndex = 6;
            this.upArrow.TabStop = false;
            this.upArrow.Click += new System.EventHandler(this.upArrow_Click);
            // 
            // outputBox
            // 
            this.outputBox.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.outputBox.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(0)))), ((int)(((byte)(64)))));
            this.outputBox.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.outputBox.Font = new System.Drawing.Font("Corbel", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.outputBox.ForeColor = System.Drawing.Color.White;
            this.outputBox.Location = new System.Drawing.Point(12, 12);
            this.outputBox.MinimumSize = new System.Drawing.Size(550, 140);
            this.outputBox.Name = "outputBox";
            this.outputBox.ReadOnly = true;
            this.outputBox.Size = new System.Drawing.Size(748, 204);
            this.outputBox.TabIndex = 8;
            this.outputBox.Text = "";
            // 
            // trainingProgressBar
            // 
            this.trainingProgressBar.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.trainingProgressBar.Location = new System.Drawing.Point(115, 223);
            this.trainingProgressBar.Name = "trainingProgressBar";
            this.trainingProgressBar.Size = new System.Drawing.Size(645, 12);
            this.trainingProgressBar.Step = 1;
            this.trainingProgressBar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.trainingProgressBar.TabIndex = 11;
            // 
            // trainingLabel
            // 
            this.trainingLabel.AutoSize = true;
            this.trainingLabel.BackColor = System.Drawing.Color.Transparent;
            this.trainingLabel.ForeColor = System.Drawing.Color.Black;
            this.trainingLabel.Location = new System.Drawing.Point(17, 220);
            this.trainingLabel.Name = "trainingLabel";
            this.trainingLabel.Size = new System.Drawing.Size(95, 20);
            this.trainingLabel.TabIndex = 12;
            this.trainingLabel.Text = "TRAINING";
            // 
            // trainingPer
            // 
            this.trainingPer.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.trainingPer.AutoSize = true;
            this.trainingPer.BackColor = System.Drawing.Color.Transparent;
            this.trainingPer.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.trainingPer.ForeColor = System.Drawing.Color.Black;
            this.trainingPer.Location = new System.Drawing.Point(508, 219);
            this.trainingPer.Name = "trainingPer";
            this.trainingPer.Size = new System.Drawing.Size(33, 20);
            this.trainingPer.TabIndex = 13;
            this.trainingPer.Text = "0%";
            // 
            // NNUIFORM
            // 
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(782, 503);
            this.Controls.Add(this.trainingPer);
            this.Controls.Add(this.trainingLabel);
            this.Controls.Add(this.trainingProgressBar);
            this.Controls.Add(this.outputBox);
            this.Controls.Add(this.downArrow);
            this.Controls.Add(this.upArrow);
            this.Controls.Add(this.errorGraph);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.ForeColor = System.Drawing.Color.SeaShell;
            this.MinimumSize = new System.Drawing.Size(800, 550);
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
                graphDataSemaphore.WaitOne();
                errorGraph.BeginInvoke(new UpdateCall(UpdateErrorGraph));
                graphDataSemaphore.Release();
            }
            else
            {
                if (xAxisData == null || yAxisData == null)
                    return;
                if (xAxisData.Count != yAxisData.Count)
                    return;

                while (xAxisData.Count != 0 && yAxisData.Count != 0)
                {
                    double x, y;
                    x = xAxisData.Dequeue();
                    y = yAxisData.Dequeue();
                    series.Points.AddXY(x, y);
                }
            }
        }

        private void updateTimer_Tick(object sender, EventArgs e)
        {
            graphUpdateSemaphore.WaitOne();
            UpdateErrorGraph();
            graphUpdateSemaphore.Release();
        }

        private void upArrow_Click(object sender, EventArgs e)
        {
            chartArea.AxisY.ScaleView.Zoomable = true;
            zoomScale *= 2;
            double lowerLimit = chartArea.AxisY.Minimum * zoomScale;
            double upperLimit = chartArea.AxisX.Maximum * zoomScale;
            chartArea.AxisY.ScaleView.Zoom(lowerLimit * zoomScale, upperLimit * zoomScale);
        }

        private void downArrow_Click(object sender, EventArgs e)
        {
            chartArea.AxisY.ScaleView.Zoomable = true;
            zoomScale /= 2;
            double lowerLimit = chartArea.AxisY.Minimum * zoomScale;
            double upperLimit = chartArea.AxisY.Maximum * zoomScale;
            chartArea.AxisY.ScaleView.Zoom(lowerLimit * zoomScale, upperLimit * zoomScale);
        }

        public void AddToChart(ref Queue<double> xAxisData, ref Queue<double> yAxisData)
        {
            this.xAxisData = xAxisData;
            this.yAxisData = yAxisData;
        }

    }
}
