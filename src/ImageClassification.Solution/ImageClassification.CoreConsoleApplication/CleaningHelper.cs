using System;
using System.IO;

namespace ImageClassification.CoreConsoleApplication
{
    public class CleaningHelper
    {
        public static void RenameImages()
        {
            string folderPath = @"C:\Development\GoodyOrBaddy\images\training\good";

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
    }
}
