using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeansCPU
{
    public static class ColorProfileConverter
    {
        public static DoubleColor ConvertColorToXYZ(DoubleColor color, ColorProfileEnum from)
        {
            ColorProfile cp = null;
            switch (from)
            {
                case ColorProfileEnum.sRGB:
                    {
                        cp = ColorProfileFactory.GetFactory().sRBGcolorProfile;
                    }
                    break;
                case ColorProfileEnum.adobeRGB:
                    {
                        cp = ColorProfileFactory.GetFactory().adobeRBGcolorProfile;
                    }
                    break;
                case ColorProfileEnum.appleRGB:
                    {
                        cp = ColorProfileFactory.GetFactory().appleRBGcolorProfile;
                    }
                    break;
                case ColorProfileEnum.CIE_RGB:
                    {
                        cp = ColorProfileFactory.GetFactory().CIE_RBGcolorProfile;
                    }
                    break;
                case ColorProfileEnum.WideGamut:
                    {
                        cp = ColorProfileFactory.GetFactory().WideGamutcolorProfile;
                    }
                    break;
                case ColorProfileEnum.PAL:
                    {
                        cp = ColorProfileFactory.GetFactory().PALcolorProfile;
                    }
                    break;
                default:
                    break;
            }

            color.InverseGammaCorrection(cp.Gamma);
            var v = cp.RGBtoXYZ.MultiplyByVectorRight(color.R, color.G, color.B);
            DoubleColor XYZ = new DoubleColor(v.X, v.Y, v.Z);
            //XYZ.ToXYZ();
            return XYZ;




            return new DoubleColor(-1, -1, -1);
        }

        public static DoubleColor ConvertColorToXYZ(DoubleColor color, ColorProfile src)
        {
            color.InverseGammaCorrection(src.Gamma);
            var v = src.RGBtoXYZ.MultiplyByVectorRight(color.R, color.G, color.B);
            DoubleColor XYZ = new DoubleColor(v.X, v.Y, v.Z);
            return XYZ;
        }

        public static DoubleColor ConvertColorFromXYZ(DoubleColor color, ColorProfileEnum from, ColorProfileEnum to)
        {
            ColorProfile cp = null;
            switch (to)
            {
                case ColorProfileEnum.sRGB:
                    {
                        cp = ColorProfileFactory.GetFactory().sRBGcolorProfile;
                    }
                    break;
                case ColorProfileEnum.adobeRGB:
                    {
                        cp = ColorProfileFactory.GetFactory().adobeRBGcolorProfile;
                    }
                    break;
                case ColorProfileEnum.appleRGB:
                    {
                        cp = ColorProfileFactory.GetFactory().appleRBGcolorProfile;
                    }
                    break;
                case ColorProfileEnum.CIE_RGB:
                    {
                        cp = ColorProfileFactory.GetFactory().CIE_RBGcolorProfile;
                    }
                    break;
                case ColorProfileEnum.WideGamut:
                    {
                        cp = ColorProfileFactory.GetFactory().WideGamutcolorProfile;
                    }
                    break;
                case ColorProfileEnum.PAL:
                    {
                        cp = ColorProfileFactory.GetFactory().PALcolorProfile;
                    }
                    break;
                default:
                    break;
            }


            //color.InverseGammaCorrection(cp.Gamma);
            MyMatrix bradford = MyMatrix.GenerateBradfordMatrix(from, to);
            if (bradford != null)
            {
                (color.R, color.G, color.B) = bradford.MultiplyByVectorRight(color.R, color.G, color.B);
            }
            var v = cp.XYZtoRGB.MultiplyByVectorRight(color.R, color.G, color.B);
            DoubleColor XYZ = new DoubleColor(v.X, v.Y, v.Z);
            XYZ.GammaCorrection(cp.Gamma);
            return XYZ;


            return new DoubleColor(-1, -1, -1);
        }

        public static DoubleColor ConvertColorFromXYZ(DoubleColor color, ColorProfile dest, MyMatrix bradfordMatrix = null)
        {
            if (bradfordMatrix != null)
            {
                (color.R, color.G, color.B) = bradfordMatrix.MultiplyByVectorRight(color.R, color.G, color.B);
            }
            var v = dest.XYZtoRGB.MultiplyByVectorRight(color.R, color.G, color.B);
            DoubleColor XYZ = new DoubleColor(v.X, v.Y, v.Z);
            XYZ.GammaCorrection(dest.Gamma);
            return XYZ;
        }

        public static void ConvertImage(SimpleColor[,] imageSource, SimpleColor[,] imageDest, ColorProfileEnum from, ColorProfileEnum to)
        {
            Parallel.For(0, imageSource.GetLength(0), i =>
            {
                for (int j = 0; j < imageSource.GetLength(1); j++)
                {
                    DoubleColor color = new DoubleColor(imageSource[i, j]);
                    var c1 = ConvertColorToXYZ(color, from);
                    var c2 = ConvertColorFromXYZ(c1, from, to);
                    imageDest[i, j] = c2.ToSimpleColor();
                }
            });
        }

        public static void ConvertImage(SimpleColor[,] imageSource, SimpleColor[,] imageDest, ColorProfile from, ColorProfile to)
        {
            var bradfordMatrix = MyMatrix.GenerateBradfordMatrix(from, to);
            Parallel.For(0, imageSource.GetLength(0), i =>
            {
                for (int j = 0; j < imageSource.GetLength(1); j++)
                {
                    DoubleColor color = new DoubleColor(imageSource[i, j]);
                    var c1 = ConvertColorToXYZ(color, from);
                    var c2 = ConvertColorFromXYZ(c1, to, bradfordMatrix);
                    imageDest[i, j] = c2.ToSimpleColor();
                }
            });
        }
    }
}
