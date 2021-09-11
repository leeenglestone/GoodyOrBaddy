using System;

namespace ImageClassification.Library
{
    public class ModelInput
    {
        public byte[] Image { get; set; }
        public UInt32 LabelAsKey { get; set; }
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }
}
