using ImageClassification.Library;
using System.IO;
using Xunit;

namespace ImageClassification.UnitTests
{
    public class UnitTest1
    {
        readonly ImageClassifier imageClassifier;
        private readonly string currentDirectory;

        public UnitTest1()
        {
            currentDirectory = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
            string pathToModel = Path.Combine(currentDirectory, "Debug\\netcoreapp3.1\\Model\\model-v1.zip");
            imageClassifier = new ImageClassifier(pathToModel);
        }

        [Fact]
        public void Test_Bananaman_Good()
        {
            string pathToImage = Path.Combine(currentDirectory, "Debug\\netcoreapp3.1\\TestImages\\goody1.jpg");
            byte[] imageBytes = File.ReadAllBytes(pathToImage);

            var expectedPrediction = "good";
            var actualPrediction = imageClassifier.ClassifyImage(imageBytes);

            Assert.Equal(expectedPrediction, actualPrediction);
        }

        [Fact]
        public void Test_Cruela_Bad()
        {
            string pathToImage = Path.Combine(currentDirectory, "Debug\\netcoreapp3.1\\TestImages\\baddy1.jpg");
            byte[] imageBytes = File.ReadAllBytes(pathToImage);

            var expectedPrediction = "bad";
            var actualPrediction = imageClassifier.ClassifyImage(imageBytes);

            Assert.Equal(expectedPrediction, actualPrediction);
        }
    }
}
