using Microsoft.ML;
using System;

namespace ImageClassification.Library
{
    public class ImageClassifier
    {
        MLContext mlContext;
        PredictionEngine<ModelInput, ModelOutput> predictionEngine;

        public ImageClassifier(string pathToModel)
        {
            mlContext = new MLContext();

            var loadModel = mlContext.Model.Load(pathToModel, out var modelInputSchema);
            predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(loadModel);
        }

        public string ClassifyImage(byte[] imageData)
        {
            if (imageData == null)
                throw new ArgumentNullException(nameof(imageData));

            ModelInput image = new ModelInput();
            image.Label = null;
            image.ImagePath = null;
            image.Image = imageData;

            var prediction = predictionEngine.Predict(image);
            var predictedLabel = prediction.PredictedLabel;

            return predictedLabel;
        }
    }
}
