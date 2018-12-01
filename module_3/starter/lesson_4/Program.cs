using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using Pensar;

namespace ml_csharp_lesson4
{
    /// <summary>
    /// The Plot class encapsulates a plotting window. 
    /// </summary>
    class Plot : System.Windows.Window
    {
        /// <summary>
        /// Construct a new instance of the class.
        /// </summary>
        /// <param name="title">The plot title.</param>
        /// <param name="results">The data to plot.</param>
        public Plot(string title, List<List<double>> results)
        {
            // set up plot model
            var plotModel = new OxyPlot.PlotModel();
            plotModel.Title = title;

            // set up axes and colors
            plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() { Position = OxyPlot.Axes.AxisPosition.Left, Title = "Error" });
            plotModel.Axes.Add(new OxyPlot.Axes.LinearAxis() { Position = OxyPlot.Axes.AxisPosition.Bottom, Title = "Epochs" });
            var colors = new OxyPlot.OxyColor[] { OxyPlot.OxyColors.Blue, OxyPlot.OxyColors.Green, OxyPlot.OxyColors.Red, OxyPlot.OxyColors.Black };

            // set up lines
            for (int i = 0; i < results.Count; i++)
            {
                var lineSeries = new OxyPlot.Series.LineSeries();
                lineSeries.ItemsSource = results[i].Select((value, index) => new OxyPlot.DataPoint(index, value));
                lineSeries.Title = string.Format("KFold {0}/{1}", i + 1, results.Count);
                //lineSeries.Color = colors[i];
                plotModel.Series.Add(lineSeries);
            }

            var plotView = new OxyPlot.Wpf.PlotView();
            plotView.Model = plotModel;

            Title = title;
            Content = plotView;
        }
    }

    /// <summary>
    /// The main application class.
    /// </summary>
    class Program
    {
        // local members
        private static CNTK.Variable features;
        private static CNTK.Variable labels;
        private static CNTK.Trainer trainer;
        private static CNTK.Evaluator evaluator;

        /// <summary>
        /// Create the neural network for this app.
        /// </summary>
        /// <returns>The neural network to use</returns>
        public static CNTK.Function CreateNetwork()
        {
            // build features and labels
            features = NetUtil.Var(new int[] { 13 }, DataType.Float);
            labels = NetUtil.Var(new int[] { 1 }, DataType.Float);

            // build the network
            var network = features
                .Dense(64, CNTKLib.ReLU)
                .Dense(64, CNTKLib.ReLU)
                .Dense(1)
                .ToNetwork();

            // set up the loss function and the classification error function
            var lossFunc = NetUtil.MeanSquaredError(network.Output, labels);
            var errorFunc = NetUtil.MeanAbsoluteError(network.Output, labels);

            // use the Adam learning algorithm
            var learner = network.GetAdamLearner(
                learningRateSchedule: (0.001, 1),
                momentumSchedule: (0.9, 1),
                unitGain: true);

            // set up a trainer and an evaluator
            trainer = network.GetTrainer(learner, lossFunc, errorFunc);
            evaluator = network.GetEvaluator(errorFunc);

            // return the completed network
            return network;
        }

        /// <summary>
        /// The main entry point of the application.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        [STAThread]
        public static void Main(string[] args)
        {
            // unzip archive
            if (!System.IO.File.Exists("x_train.bin"))
            {
                DataUtil.Unzip(@"..\..\..\..\..\boston_housing.zip", ".");
            }

            // load training and test data
            var training_data = DataUtil.LoadBinary<float>("x_train.bin", 404, 13);
            var test_data = DataUtil.LoadBinary<float>("x_test.bin", 102, 13);
            var training_labels = DataUtil.LoadBinary<float>("y_train.bin", 404);
            var test_labels = DataUtil.LoadBinary<float>("y_test.bin", 102);



            Console.ReadLine();
        }
    }
}
