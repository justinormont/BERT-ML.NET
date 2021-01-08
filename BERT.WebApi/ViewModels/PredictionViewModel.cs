using System.Collections.Generic;

namespace BERT.WebApi.ViewModels
{
    /*public class PredictionViewModel
    {
        public IEnumerable<string> Tokens { get; set; }

        public float Probability { get; set; }
    }*/

    public class PredictionViewModel
    {
        public float[] BertEmbeddings1 { get; set; }
        public float[] BertEmbeddings2 { get; set; }
    }

}
