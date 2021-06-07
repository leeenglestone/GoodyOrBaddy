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
        static readonly string trainedModelPath = @"C:\Temp\model.zip";

        static void Main(string[] args)
        {
            // These are helper methods, you probably wont need these
            // CleaningHelper.RenameImages(); // This makes the filenames unique
            // CleaningHelper.RemoveCopiedFiles(); // This removes the copied files if I want

            // 1. You will need to create the model first before step 2
            //CreateModel();

            // 2. Testing images from the model
            //ClassifyImageUsingModel(@"C:\Development\GoodyOrBaddy\images\test\baddy1.jpg", trainedModelPath, "bad");
            //ClassifyImageUsingModel(@"C:\Development\GoodyOrBaddy\images\test\baddy2.jpg", trainedModelPath, "bad");
            //ClassifyImageUsingModel(@"C:\Development\GoodyOrBaddy\images\test\baddy3.jpg", trainedModelPath, "bad");
            
            //ClassifyImageUsingModel(@"C:\Development\GoodyOrBaddy\images\test\goody1.jpg", trainedModelPath, "good");
            //ClassifyImageUsingModel(@"C:\Development\GoodyOrBaddy\images\test\goody2.jpg", trainedModelPath, "good");
            //ClassifyImageUsingModel(@"C:\Development\GoodyOrBaddy\images\test\goody3.jpg", trainedModelPath, "good");

            Console.ReadKey();
        }

        private static void CreateModel()
        {
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

            var preProcessedDataPreview = preProcessedData.Preview();

            // https://rubikscode.net/2021/03/22/transfer-learning-and-image-classification-with-ml-net/

            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            IDataView trainSet = trainSplit.TrainSet;
            //var trainSetPreview = trainSet.Preview();

            IDataView validationSet = validationTestSplit.TrainSet;
            //var validationSetPreview = validationSet.Preview();

            IDataView testSet = validationTestSplit.TestSet;
            //var testSetPreview = testSet.Preview(); // Empty when there aren't sufficient images!

            var classifierOptions = new ImageClassificationTrainer.Options()
            {
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

            // You can comment out this code once you have produced the model zip file
            mlContext.Model.Save(trainedModel, trainSet.Schema, trainedModelPath);

            // These methods are from the code sample I found
            // They use the in memory model rather than the physical model file
            //ClassifySingleImage(mlContext, testSet, trainedModel);
            //ClassifyImages(mlContext, testSet, trainedModel);
        }

        public static void ClassifyImageUsingModel(string pathToImage, string pathToModel, string expectedLabel)
        {
            mlContext = new MLContext();

            var loadModel = mlContext.Model.Load(pathToModel, out var modelInputSchema);
            var PredictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(loadModel);

            ModelInput image = new ModelInput();
            image.Label = expectedLabel;
            image.ImagePath = pathToImage;
            image.Image = File.ReadAllBytes(pathToImage);

            var prediction = PredictionEngine.Predict(image);

            Console.WriteLine($"Classifying image using existing model {pathToModel}");
            OutputPrediction(prediction);
        }

        public static void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
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
