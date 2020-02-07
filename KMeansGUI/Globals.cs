using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeans.GUI
{
    public static class Globals
    {
        public static string WindowName = "Cuda Kmeans";
        public static string ImagesFolderPath = "..\\..\\Images\\";
        public static string DefaultImageSource = "..\\..\\Images\\image1.jpg";
        public static CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");

        //frame
        public static int BorderWidth = 20;
        public static int ImageWidth = 800;
        public static int ImageHeight = 600;
        public static int Radius = 250;

        //kmeans
        public static int k_means = 10;
        public static int max_iter = 20;
        public static double eps = 1e-4;

    }
}
