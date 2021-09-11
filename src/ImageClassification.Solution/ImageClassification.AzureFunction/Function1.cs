using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using ImageClassification.Library;

namespace ImageClassification.AzureFunction
{
    public static class Function1
    {
        [FunctionName("Function1")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log, ExecutionContext ctx)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            try
            {
                var formdata = await req.ReadFormAsync();
                var file = req.Form.Files["file"];
                var pathToModel = Path.Combine(ctx.FunctionAppDirectory, "Model", "model-v1.zip"); ; // Needed
                var fileName = file.FileName;
                byte[] fileBytes = null;

                using (var memoryStream = new MemoryStream())
                {
                    file.CopyTo(memoryStream);
                    fileBytes = memoryStream.ToArray();
                }

                MLContext mlContext = new MLContext();
                var model = mlContext.Model.Load(pathToModel, out var modelInputSchema);

                var PredictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

                ModelInput image = new ModelInput();
                image.Label = "<Expected Label>"; // Not really using this
                image.ImagePath = fileName;
                image.Image = fileBytes;

                var prediction = PredictionEngine.Predict(image);
                var predictedLabel = prediction.PredictedLabel;

                return new OkObjectResult(predictedLabel);
            }
            catch (Exception ex)
            {
                return new BadRequestObjectResult(ex);
            }
        }
    }
}
