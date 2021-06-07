using System;
using System.IO;

namespace ImageClassification.CoreConsoleApplication
{
    public class CleaningHelper
    {
        public static void RenameImages()
        {
            RenameImages(@"C:\Development\GoodyOrBaddy\images\training\good");
            RenameImages(@"C:\Development\GoodyOrBaddy\images\training\bad");
        }

        private static void RenameImages(string folderPath)
        {
            foreach (var filePath in Directory.GetFiles(folderPath))
            {
                var guid = Guid.NewGuid();
                var file = new FileInfo(filePath);
                var extension = file.Extension;

                var newFilePath = Path.Combine(file.DirectoryName, guid.ToString() + extension);

                Console.WriteLine(filePath);
                Console.WriteLine(newFilePath);

                File.Move(filePath, newFilePath);
            }
        }

        public static void RemoveCopiedFiles()
        {
            RemoveCoppiedFiles(@"C:\Development\GoodyOrBaddy\images\training\good");
            RemoveCoppiedFiles(@"C:\Development\GoodyOrBaddy\images\training\bad");
        }

        private static void RemoveCoppiedFiles(string folderPath)
        {
            foreach (var filePath in Directory.GetFiles(folderPath))
            {
                var file = new FileInfo(filePath);

                if (file.Name.ToLower().Contains("copy"))
                {
                    File.Delete(filePath);
                    Console.WriteLine($"Deleting: {filePath}");
                }
            }
        }
    }
}
