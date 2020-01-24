using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeansCPU
{
    public struct DoubleColor
    {
        public double R { get; set; }
        public double G { get; set; }
        public double B { get; set; }
        public DoubleColor(double r, double g, double b)
        {
            R = r;
            G = g;
            B = b;
        }
        public DoubleColor(int r, int g, int b)
        {
            R = (double)r / 255d;
            G = (double)g / 255d;
            B = (double)b / 255d;
        }

        public DoubleColor(SimpleColor color)
        {
            R = (double)color.R / 255d;
            G = (double)color.G / 255d;
            B = (double)color.B / 255d;
        }

        public void InverseGammaCorrection(double gamma)
        {
            R = Math.Pow(R, gamma);
            G = Math.Pow(G, gamma);
            B = Math.Pow(B, gamma);
        }
        public void GammaCorrection(double gamma)
        {
            double inv = 1 / gamma;
            R = Math.Pow(R, inv);
            G = Math.Pow(G, inv);
            B = Math.Pow(B, inv);
        }

        public SimpleColor ToSimpleColor()
        {
            return new SimpleColor(Clamp(R * 255), Clamp(G * 255), Clamp(B * 255));
        }

        public void ToXYZ()
        {
            if (Math.Abs(G) < 1e-5)
            {
                R = 0;
                G = 0;
                B = 0;
            }
            else
            {
                R /= G;
                B /= G;
                G = 1;
            }
        }

        private byte Clamp(double d)
        {
            if (d > 255d)
                d = 255d;
            if (d < 0d)
                d = 0d;
            d = Math.Round(d);
            return (byte)d;
        }

        public override string ToString()
        {
            return $"R: {Math.Round(R, 4),5}, G: {Math.Round(G, 4),5}, B: {Math.Round(B, 4),5}";
        }
    }
}
