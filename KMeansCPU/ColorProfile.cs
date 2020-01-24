using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeansCPU
{
    public class ColorProfile
    {
        public MyMatrix RGBtoXYZ { get; set; }
        public MyMatrix XYZtoRGB { get; set; }
        public double Gamma { get; set; }
        public double White_X { get; set; }
        public double White_Y { get; set; }
        public double White_Z
        {
            get
            {
                return 1d - White_X - White_Y;
            }
        }
        public double Red_X { get; set; }
        public double Red_Y { get; set; }
        public double Red_Z
        {
            get
            {
                return 1d - Red_X - Red_Y;
            }
        }
        public double Green_X { get; set; }
        public double Green_Y { get; set; }
        public double Green_Z
        {
            get
            {
                return 1d - Green_X - Green_Y;
            }
        }
        public double Blue_X { get; set; }
        public double Blue_Y { get; set; }
        public double Blue_Z
        {
            get
            {
                return 1d - Blue_X - Blue_Y;
            }
        }

        public void SetColorMatrix()
        {
            double X_W;
            double Y_W;
            double Z_W;
            if (White_Y < 1e-5)
            {
                X_W = 0;
                Y_W = 0;
                Z_W = 0;
            }
            else
            {
                X_W = White_X / White_Y;
                Y_W = 1d;
                Z_W = White_Z / White_Y;
            }
            MyMatrix matrix = new MyMatrix(this);

            MyMatrix inv = matrix.Invert();

            if (inv == null)
                throw new Exception();

            (double S_R, double S_G, double S_B) = inv.MultiplyByVectorRight(X_W, Y_W, Z_W);

            matrix.MultiplyBySRSGSB(S_R, S_G, S_B);
            RGBtoXYZ = matrix;
            XYZtoRGB = matrix.Invert();
        }
    }
}
