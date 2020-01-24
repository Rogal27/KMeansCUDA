using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeansCPU
{
    public class MyMatrix
    {
        public double [,] array { get; set; }
        private static int size = 3;
        public MyMatrix()
        {
            array = new double[3, 3];
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    array[i, j] = 0d;
                }
            }
        }
        public MyMatrix(ColorProfile cp)
        {
            array = new double[3, 3];
            array[0, 0] = cp.Red_X;
            array[1, 0] = cp.Red_Y;
            array[2, 0] = cp.Red_Z;
            array[0, 1] = cp.Green_X;
            array[1, 1] = cp.Green_Y;
            array[2, 1] = cp.Green_Z;
            array[0, 2] = cp.Blue_X;
            array[1, 2] = cp.Blue_Y;
            array[2, 2] = cp.Blue_Z;
        }

        public MyMatrix Invert()
        {
            MyMatrix inv = new MyMatrix();
            double A = array[1, 1] * array[2, 2] - array[1, 2] * array[2, 1];
            double B = array[1, 2] * array[2, 0] - array[1, 0] * array[2, 2];
            double C = array[1, 0] * array[2, 1] - array[1, 1] * array[2, 0];
            double det = array[0, 0] * A + array[0, 1] * B + array[0, 2] * C;
            if (Math.Abs(det) < 1e-5)
                return null;
            inv[0, 0] = A / det;
            inv[1, 0] = B / det;
            inv[2, 0] = C / det;
            inv[0, 1] = (array[0, 2] * array[2, 1] - array[0, 1] * array[2, 2]) / det;
            inv[1, 1] = (array[0, 0] * array[2, 2] - array[0, 2] * array[2, 0]) / det;
            inv[2, 1] = (array[0, 1] * array[2, 0] - array[0, 0] * array[2, 1]) / det;
            inv[0, 2] = (array[0, 1] * array[1, 2] - array[0, 2] * array[1, 1]) / det;
            inv[1, 2] = (array[0, 2] * array[1, 0] - array[0, 0] * array[1, 2]) / det;
            inv[2, 2] = (array[0, 0] * array[1, 1] - array[0, 1] * array[1, 0]) / det;
            return inv;
        }

        public void Transpose()
        {
            for (int i = 0; i < array.GetLength(0) / 2; i++)
            {
                for (int j = 0; j < array.GetLength(1) / 2; j++)
                {
                    double tmp = array[i, j];
                    array[i, j] = array[j,i];
                    array[j, i] = tmp;
                }
            }
        }

        public (double X, double Y, double Z) MultiplyByVectorRight(double x, double y, double z)
        {
            double X = array[0, 0] * x + array[0, 1] * y + array[0, 2] * z;
            double Y = array[1, 0] * x + array[1, 1] * y + array[1, 2] * z;
            double Z = array[2, 0] * x + array[2, 1] * y + array[2, 2] * z;
            return (X, Y, Z);
        }

        public void MultiplyBySRSGSB(double S_R, double S_G, double S_B)
        {
            for (int i = 0; i < array.GetLength(0); i++)
            {
                array[i, 0] *= S_R;
            }
            for (int i = 0; i < array.GetLength(0); i++)
            {
                array[i, 1] *= S_G;
            }
            for (int i = 0; i < array.GetLength(0); i++)
            {
                array[i, 2] *= S_B;
            }
        }

        public double this[int x,int y]
        {
            get
            {
                return array[x, y];
            }
            set
            {
                array[x, y] = value;
            }
        }

        public static MyMatrix GenerateBradfordMatrix(ColorProfileEnum from, ColorProfileEnum to)
        {
            //from is D65
            if(from == ColorProfileEnum.sRGB || from == ColorProfileEnum.adobeRGB || from == ColorProfileEnum.appleRGB || from == ColorProfileEnum.PAL)
            {
                //to is D65
                if (to == ColorProfileEnum.sRGB || to == ColorProfileEnum.adobeRGB || to == ColorProfileEnum.appleRGB || to == ColorProfileEnum.PAL)
                {
                    return null;
                }
                //to is D50
                else if (to == ColorProfileEnum.WideGamut)
                {
                    var bm = new MyMatrix();
                    bm[0, 0] = 1.0478;
                    bm[0, 1] = 0.0229;
                    bm[0, 2] = -0.0501;
                    bm[1, 0] = 0.0295;
                    bm[1, 1] = 0.9905;
                    bm[1, 2] = -0.0171;
                    bm[2, 0] = -0.0092;
                    bm[2, 1] = 0.0150;
                    bm[2, 2] = 0.7521;
                    return bm;
                }
                //to is E
                else if (to == ColorProfileEnum.CIE_RGB)
                {
                    var bm = new MyMatrix();
                    bm[0, 0] = 1.0503;
                    bm[0, 1] = 0.0271;
                    bm[0, 2] = -0.0233;
                    bm[1, 0] = 0.0391;
                    bm[1, 1] = 0.9730;
                    bm[1, 2] = -0.0093;
                    bm[2, 0] = -0.0024;
                    bm[2, 1] = 0.0026;
                    bm[2, 2] = 0.9181;
                    return bm;
                }
            }
            //from is D50
            else if(from == ColorProfileEnum.WideGamut)
            {
                //to is D65
                if (to == ColorProfileEnum.sRGB || to == ColorProfileEnum.adobeRGB || to == ColorProfileEnum.appleRGB || to == ColorProfileEnum.PAL)
                {
                    var bm = new MyMatrix();
                    bm[0, 0] = 0.9556;
                    bm[0, 1] = -0.0230;
                    bm[0, 2] = 0.0632;
                    bm[1, 0] = -0.0283;
                    bm[1, 1] = 1.0099;
                    bm[1, 2] = 0.0210;
                    bm[2, 0] = 0.0123;
                    bm[2, 1] = -0.0205;
                    bm[2, 2] = 1.3299;
                    return bm;
                }
                //to is D50
                else if (to == ColorProfileEnum.WideGamut)
                {
                    return null;
                }
                //to is E
                else if (to == ColorProfileEnum.CIE_RGB)
                {
                    var bm = new MyMatrix();
                    bm[0, 0] = 1.0026;
                    bm[0, 1] = 0.0036;
                    bm[0, 2] = 0.0360;
                    bm[1, 0] = 0.0097;
                    bm[1, 1] = 0.9819;
                    bm[1, 2] = 0.0106;
                    bm[2, 0] = 0.0089;
                    bm[2, 1] = -0.0161;
                    bm[2, 2] = 1.2209;
                    return bm;
                }
            }
            //from is E
            else if(from == ColorProfileEnum.CIE_RGB)
            {
                //to is D65
                if (to == ColorProfileEnum.sRGB || to == ColorProfileEnum.adobeRGB || to == ColorProfileEnum.appleRGB || to == ColorProfileEnum.PAL)
                {
                    var bm = new MyMatrix();
                    bm[0, 0] = 0.9532;
                    bm[0, 1] = -0.0266;
                    bm[0, 2] = 0.0239;
                    bm[1, 0] = -0.0382;
                    bm[1, 1] = 1.0288;
                    bm[1, 2] = 0.0094;
                    bm[2, 0] = 0.0026;
                    bm[2, 1] = -0.0030;
                    bm[2, 2] = 1.0893;
                    return bm;
                }
                //to is D50
                else if (to == ColorProfileEnum.WideGamut)
                {
                    var bm = new MyMatrix();
                    bm[0, 0] = 0.9978;
                    bm[0, 1] = -0.0042;
                    bm[0, 2] = -0.0294;
                    bm[1, 0] = -0.0098;
                    bm[1, 1] = 1.0183;
                    bm[1, 2] = -0.0085;
                    bm[2, 0] = -0.0074;
                    bm[2, 1] = 0.0134;
                    bm[2, 2] = 0.8192;
                    return bm;
                }
                //to is E
                else if (to == ColorProfileEnum.CIE_RGB)
                {
                    return null;
                }
            }
            return null;
        }

        public static MyMatrix GenerateBradfordMatrix(ColorProfile from, ColorProfile to)
        {
            var ConeResponse = GetConeResponseMatrix();
            var InverseConeResponse = GetInverseConeResponseMatrix();
            double fromX;
            double fromY;
            double fromZ;
            double toX;
            double toY;
            double toZ;
            double m;
            if (from.White_Y < 1e-5)
            {
                fromX = 0;
                fromY = 0;
                fromZ = 0;
            }
            else
            {
                fromY = 100;
                m = fromY / from.White_Y;                
                fromX = from.White_X * m;
                fromZ = from.White_Z * m;
            }
            if(to.White_Y < 1e-5)
            {
                toX = 0;
                toY = 0;
                toZ = 0;
            }
            else
            {
                toY = 100;
                m = toY / to.White_Y;
                toX = to.White_X * m;
                toZ = to.White_Z * m;
            }


            var v1 = ConeResponse.MultiplyByVectorRight(fromX, fromY, fromZ);
            var v2 = ConeResponse.MultiplyByVectorRight(toX, toY, toZ);
            var middleMatrix = new MyMatrix();
            middleMatrix[0, 0] = v2.X / v1.X;
            middleMatrix[1, 1] = v2.Y / v1.Y;
            middleMatrix[2, 2] = v2.Z / v1.Z;
            var result = InverseConeResponse * middleMatrix * ConeResponse;
            return result;
        }

        private static MyMatrix GetConeResponseMatrix()
        {
            var bm = new MyMatrix();
            bm[0, 0] = 0.8951;
            bm[0, 1] = 0.2664;
            bm[0, 2] = -0.1614;
            bm[1, 0] = -0.7502;
            bm[1, 1] = 1.7135;
            bm[1, 2] = 0.0367;
            bm[2, 0] = 0.0389;
            bm[2, 1] = -0.0685;
            bm[2, 2] = 1.0296;
            return bm;
        }

        private static MyMatrix GetInverseConeResponseMatrix()
        {
            var bm = new MyMatrix();
            bm[0, 0] = 0.9870;
            bm[0, 1] = -0.1471;
            bm[0, 2] = 0.1600;
            bm[1, 0] = 0.4323;
            bm[1, 1] = 0.5184;
            bm[1, 2] = 0.0493;
            bm[2, 0] = -0.0085;
            bm[2, 1] = 0.0400;
            bm[2, 2] = 0.9685;
            return bm;
        }

        public static MyMatrix operator * (MyMatrix m1, MyMatrix m2)
        {
            MyMatrix result = new MyMatrix();
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    result[i, j] = 0;
                    for (int k = 0; k < size; k++)
                    {
                        result[i, j] += m1[i, k] * m2[k, j];
                    }
                }
            }
            return result;
        }

        public void Print()
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    sb.Append($"{Math.Round(array[i, j],4),5}\t");
                }
                sb.AppendLine();
            }
            Debug.WriteLine(sb.ToString());
        }
    }
}
