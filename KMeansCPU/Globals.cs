using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeansCPU
{
    public static class Globals
    {
        public static string WindowName = "GK Project 3";
        public static string ImagesFolderPath = "..\\..\\Images\\";
        public static string DefaultImageSource = "..\\..\\Images\\image1.jpg";
        public static CultureInfo culture = CultureInfo.CreateSpecificCulture("en-US");

        //frame
        public static int BorderWidth = 20;
        public static int ImageWidth = 800;
        public static int ImageHeight = 600;
        public static int Radius = 250;

    }
}
