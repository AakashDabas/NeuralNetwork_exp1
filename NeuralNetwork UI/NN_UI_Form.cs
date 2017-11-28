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
        private Label label1;
        private Label label2;
        public TextBox rateInput;
        public TextBox momentumInput;
        public Label trainingPer;
        private Button button1;
        Callback _callback;

        public NNUIFORM(Callback _callback)
        {
            this._callback = _callback;
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
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea3 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend3 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series3 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.errorGraph = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.downArrow = new System.Windows.Forms.PictureBox();
            this.upArrow = new System.Windows.Forms.PictureBox();
            this.outputBox = new System.Windows.Forms.RichTextBox();
            this.trainingProgressBar = new System.Windows.Forms.ProgressBar();
            this.trainingLabel = new System.Windows.Forms.Label();
            this.trainingPer = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.rateInput = new System.Windows.Forms.TextBox();
            this.momentumInput = new System.Windows.Forms.TextBox();
            this.button1 = new System.Windows.Forms.Button();
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
            this.errorGraph.Location = new System.Drawing.Point(17, 242);
            this.errorGraph.Name = "errorGraph";
            series3.ChartArea = "ChartArea1";
            series3.Legend = "Legend1";
            series3.Name = "Series1";
            this.errorGraph.Series.Add(series3);
            this.errorGraph.Size = new System.Drawing.Size(897, 363);
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
            this.outputBox.Size = new System.Drawing.Size(902, 204);
            this.outputBox.TabIndex = 8;
            this.outputBox.Text = "";
            // 
            // trainingProgressBar
            // 
            this.trainingProgressBar.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.trainingProgressBar.Location = new System.Drawing.Point(115, 223);
            this.trainingProgressBar.Name = "trainingProgressBar";
            this.trainingProgressBar.Size = new System.Drawing.Size(799, 12);
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
            this.trainingLabel.Size = new System.Drawing.Size(80, 16);
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
            this.trainingPer.Size = new System.Drawing.Size(27, 16);
            this.trainingPer.TabIndex = 13;
            this.trainingPer.Text = "0%";
            // 
            // label1
            // 
            this.label1.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 11.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.Color.Black;
            this.label1.Location = new System.Drawing.Point(16, 626);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(99, 18);
            this.label1.TabIndex = 14;
            this.label1.Text = "Learning Rate";
            // 
            // label2
            // 
            this.label2.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 11.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.ForeColor = System.Drawing.Color.Black;
            this.label2.Location = new System.Drawing.Point(298, 626);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(84, 18);
            this.label2.TabIndex = 15;
            this.label2.Text = "Momentum";
            // 
            // rateInput
            // 
            this.rateInput.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.rateInput.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.rateInput.Font = new System.Drawing.Font("Bahnschrift", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.rateInput.ForeColor = System.Drawing.Color.DarkCyan;
            this.rateInput.Location = new System.Drawing.Point(121, 626);
            this.rateInput.Name = "rateInput";
            this.rateInput.Size = new System.Drawing.Size(100, 23);
            this.rateInput.TabIndex = 16;
            // 
            // momentumInput
            // 
            this.momentumInput.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.momentumInput.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.momentumInput.Font = new System.Drawing.Font("Bahnschrift", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.momentumInput.ForeColor = System.Drawing.Color.DarkCyan;
            this.momentumInput.Location = new System.Drawing.Point(388, 626);
            this.momentumInput.Name = "momentumInput";
            this.momentumInput.Size = new System.Drawing.Size(100, 23);
            this.momentumInput.TabIndex = 17;
            // 
            // button1
            // 
            this.button1.BackColor = System.Drawing.Color.DodgerBlue;
            this.button1.ForeColor = System.Drawing.Color.White;
            this.button1.Location = new System.Drawing.Point(566, 626);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(89, 23);
            this.button1.TabIndex = 18;
            this.button1.Text = "UPDATE";
            this.button1.UseVisualStyleBackColor = false;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // NNUIFORM
            // 
            this.BackColor = System.Drawing.Color.White;
            this.ClientSize = new System.Drawing.Size(938, 660);
            this.ControlBox = false;
            this.Controls.Add(this.button1);
            this.Controls.Add(this.momentumInput);
            this.Controls.Add(this.rateInput);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.trainingPer);
            this.Controls.Add(this.trainingLabel);
            this.Controls.Add(this.trainingProgressBar);
            this.Controls.Add(this.outputBox);
            this.Controls.Add(this.downArrow);
            this.Controls.Add(this.upArrow);
            this.Controls.Add(this.errorGraph);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.ForeColor = System.Drawing.Color.SeaShell;
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
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
            double upperLimit = chartArea.AxisY.Maximum * zoomScale;
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

        private void button1_Click(object sender, EventArgs e)
        {
            _callback("Momentum", momentumInput.Text);
            _callback("LearningRate", rateInput.Text);
        }

        public void AddToChart(ref Queue<double> xAxisData, ref Queue<double> yAxisData)
        {
            this.xAxisData = xAxisData;
            this.yAxisData = yAxisData;
        }

    }
}
