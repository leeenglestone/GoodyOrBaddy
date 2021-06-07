using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;

namespace ImageClassification.CoreConsoleApplication
{
    class Program
    {
        // https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification-api-transfer-learning

        static MLContext mlContext;

        static void Main(string[] args)
        {
            //CleaningHelper.RenameImages();
            //CleaningHelper.RemoveCopiedFiles();

            Process();

            Console.ReadKey();
        }

        private static void Process()
        {
            //var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var workspaceRelativePath = @"C:\Development\GoodyOrBaddy\workspace";// Path.Combine(projectDirectory, "workspace");
            //var assetsRelativePath = Path.Combine(projectDirectory, "images\\training");

            var assetsRelativePath = @"C:\Development\GoodyOrBaddy\images\training";

            mlContext = new MLContext();

            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);
            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);
            var imageDataPreview = imageData.Preview();

            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);
            var shuffledDataPreview = shuffledData.Preview();

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "Label",
                outputColumnName: "LabelAsKey")
            .Append(mlContext.Transforms.LoadRawImageBytes(
                outputColumnName: "Image",
                imageFolder: assetsRelativePath,
                inputColumnName: "ImagePath"));

            IDataView preProcessedData = preprocessingPipeline
                    .Fit(shuffledData)
                    .Transform(shuffledData);

            var preProcessedDataPreview = preProcessedData.Preview(); // Is this empty?

            // https://rubikscode.net/2021/03/22/transfer-learning-and-image-classification-with-ml-net/

            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            IDataView trainSet = trainSplit.TrainSet;
            //var trainSetPreview = trainSet.Preview();

            IDataView validationSet = validationTestSplit.TrainSet;
            //var validationSetPreview = validationSet.Preview();

            IDataView testSet = validationTestSplit.TestSet;
            //var testSetPreview = testSet.Preview(); // Empty??

            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                //Epoch = 500,
                //WorkspacePath = workspaceRelativePath,
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true
            };

            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            ITransformer trainedModel = trainingPipeline.Fit(trainSet);

            //mlContext.Model.Save(trainedModel, trainSet.Schema, @"C:\Temp\model.zip");

            //ClassifyImageUsingModel(@"C:\Development\GoodyOrBaddy\images\test\good\315be4a2-b2dc-4b32-be4b-ec3d24b0d396.jpg", @"C:\Temp\model.zip");

            ClassifySingleImage(mlContext, testSet, trainedModel);

            //ClassifyImages(mlContext, testSet, trainedModel);
        }

        public static void ClassifyImageUsingModel(string pathToImage, string pathToModel)
        {
            var loadModel = mlContext.Model.Load(pathToModel, out var modelInputSchema);
            var PredictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(loadModel);

            ModelInput image = new ModelInput();
            image.ImagePath = pathToImage;
            image.Image = File.ReadAllBytes(pathToImage);

            //image.Label 
            //;// LoadImage(pathToImage);

            var prediction = PredictionEngine.Predict(image);

            Console.WriteLine($"Classifying image using existing model {pathToModel}");
            OutputPrediction(prediction);
        }

        public static void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            var dataPreview = data.Preview();

            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();

            ModelOutput prediction = predictionEngine.Predict(image);

            Console.WriteLine("Classifying single image");
            OutputPrediction(prediction);
        }

        public static void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            IDataView predictionData = trainedModel.Transform(data);

            IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);

            Console.WriteLine("Classifying multiple images");
            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction);
            }
        }

        private static void OutputPrediction(ModelOutput prediction)
        {
            string imageName = Path.GetFileName(prediction.ImagePath);
            Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
        }

        public static ImageData LoadImage(string file, bool useFolderNameAsLabel = true)
        {
            var label = Path.GetFileName(file);

            if (useFolderNameAsLabel)
                label = Directory.GetParent(file).Name;
            else
            {
                for (int index = 0; index < label.Length; index++)
                {
                    if (!char.IsLetter(label[index]))
                    {
                        label = label.Substring(0, index);
                        break;
                    }
                }
            }

            var imageData = new ImageData()
            {
                Label = label,
                ImagePath = file
            };

            return imageData;
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var images = new List<ImageData>();

            var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);

                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                images.Add(new ImageData() { Label = label, ImagePath = file });

                //yield return new ImageData()
                //{
                //    ImagePath = file,
                //    Label = label
                //};
            }

            return images.AsEnumerable();

        }
    }

    public class ImageData
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    public class ModelInput
    {
        public byte[] Image { get; set; }
        public UInt32 LabelAsKey { get; set; }
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    public class ModelOutput
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public string PredictedLabel { get; set; }
    }
}
