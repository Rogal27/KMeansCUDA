using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeans.GUI
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




            //return new DoubleColor(-1, -1, -1);
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


            //return new DoubleColor(-1, -1, -1);
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

        public static void ConvertImageToLAB(SimpleColor[,] imageSource, DoubleColor[,] imageDest, ColorProfileEnum from)
        {
            Parallel.For(0, imageSource.GetLength(0), i =>
            {
                for (int j = 0; j < imageSource.GetLength(1); j++)
                {
                    DoubleColor color = new DoubleColor(imageSource[i, j]);
                    var c1 = ConvertColorToXYZ(color, from);
                    var c2 = ConvertColorFromXYZToLAB(c1, from);
                    imageDest[i, j] = c2;
                }
            });
        }

        public static void ConvertImageToLAB(SimpleColor[,] imageSource, DoubleColor[,] imageDest, ColorProfile from)
        {
            Parallel.For(0, imageSource.GetLength(0), i =>
            {
                for (int j = 0; j < imageSource.GetLength(1); j++)
                {
                    DoubleColor color = new DoubleColor(imageSource[i, j]);
                    var c1 = ConvertColorToXYZ(color, from);
                    var c2 = ConvertColorFromXYZToLAB(c1, from);
                    imageDest[i, j] = c2;
                }
            });
        }

        public static DoubleColor ConvertColorFromXYZToLAB(DoubleColor color, ColorProfileEnum from)
        {
            var cp = ColorProfileFactory.GetFactory().GetColorProfile(from);
            double YR = 100d;
            double XR = cp.White_X * YR / cp.White_Y;
            double ZR = cp.White_Z * YR / cp.White_Y;

            double xr = color.R / XR;
            double yr = color.G / YR;
            double zr = color.B / ZR;

            double k_modifier = 903.3;
            double eps_modifier = 0.008856;

            double fx;
            double fy;
            double fz;

            if (xr > eps_modifier)
            {
                fx = Math.Pow(xr, 1d / 3d);
            }
            else
            {
                fx = (k_modifier * xr + 16d) / 116d;
            }

            if (yr > eps_modifier)
            {
                fy = Math.Pow(yr, 1d / 3d);
            }
            else
            {
                fy = (k_modifier * yr + 16d) / 116d;
            }

            if (zr > eps_modifier)
            {
                fz = Math.Pow(zr, 1d / 3d);
            }
            else
            {
                fz = (k_modifier * zr + 16d) / 116d;
            }

            double L = 116 * fy - 16;
            double a = 500 * (fx - fy);
            double b = 200 * (fy - fz);

            return new DoubleColor(L, a, b);

        }

        public static DoubleColor ConvertColorFromXYZToLAB(DoubleColor color, ColorProfile cp)
        {
            //var cp = ColorProfileFactory.GetFactory().GetColorProfile(from);
            double YR = 100d;
            double XR = cp.White_X * YR / cp.White_Y;
            double ZR = cp.White_Z * YR / cp.White_Y;

            double xr = color.R / XR;
            double yr = color.G / YR;
            double zr = color.B / ZR;

            double k_modifier = 903.3;
            double eps_modifier = 0.008856;

            double fx;
            double fy;
            double fz;

            if (xr > eps_modifier)
            {
                fx = Math.Pow(xr, 1d / 3d);
            }
            else
            {
                fx = (k_modifier * xr + 16d) / 116d;
            }

            if (yr > eps_modifier)
            {
                fy = Math.Pow(yr, 1d / 3d);
            }
            else
            {
                fy = (k_modifier * yr + 16d) / 116d;
            }

            if (zr > eps_modifier)
            {
                fz = Math.Pow(zr, 1d / 3d);
            }
            else
            {
                fz = (k_modifier * zr + 16d) / 116d;
            }

            double L = 116d * fy - 16d;
            double a = 500d * (fx - fy);
            double b = 200d * (fy - fz);

            return new DoubleColor(L, a, b);

        }

        public static void ConvertImageFromLAB(DoubleColor[,] imageSource, SimpleColor[,] imageDest, ColorProfileEnum from)
        {
            Parallel.For(0, imageSource.GetLength(0), i =>
            {
                for (int j = 0; j < imageSource.GetLength(1); j++)
                {
                    //DoubleColor color = new DoubleColor(imageSource[i, j]);
                    var c1 = ConvertColorFromLABToXYZ(imageSource[i, j], from);
                    var c2 = ConvertColorFromXYZ(c1, from, from);
                    imageDest[i, j] = c2.ToSimpleColor();
                }
            });
        }

        public static void ConvertImageFromLAB(DoubleColor[,] imageSource, SimpleColor[,] imageDest, ColorProfile from)
        {
            Parallel.For(0, imageSource.GetLength(0), i =>
            {
                for (int j = 0; j < imageSource.GetLength(1); j++)
                {
                    //DoubleColor color = new DoubleColor(imageSource[i, j]);
                    var c1 = ConvertColorFromLABToXYZ(imageSource[i, j], from);
                    var c2 = ConvertColorFromXYZ(c1, from);
                    imageDest[i, j] = c2.ToSimpleColor();
                }
            });
        }

        public static DoubleColor ConvertColorFromLABToXYZ(DoubleColor color, ColorProfileEnum from)
        {
            var cp = ColorProfileFactory.GetFactory().GetColorProfile(from);
            double YR = 100d;
            double XR = cp.White_X * YR / cp.White_Y;
            double ZR = cp.White_Z * YR / cp.White_Y;

            double xr;
            double yr;
            double zr;

            double k_modifier = 903.3;
            double eps_modifier = 0.008856;

            
            double fy = (color.R + 16d) / 116d;
            double fx = color.G / 500d + fy;
            double fz = fy - color.B / 200d;

            xr = Math.Pow(fx, 3);
            if (xr <= eps_modifier)
            {
                xr = (116d * fx - 16d) / k_modifier;
            }

            if (color.R > k_modifier * eps_modifier)
            {
                yr = Math.Pow((color.R + 16d) / 116d, 3);
            }
            else
            {
                yr = color.R / k_modifier;
            }

            zr = Math.Pow(fz, 3);
            if (zr <= eps_modifier)
            {
                zr = (116d * fz - 16d) / k_modifier;
            }


            double X = xr * XR;
            double Y = yr * YR;
            double Z = zr * ZR;

            return new DoubleColor(X, Y, Z);

        }

        public static DoubleColor ConvertColorFromLABToXYZ(DoubleColor color, ColorProfile cp)
        {
            //var cp = ColorProfileFactory.GetFactory().GetColorProfile(from);
            double YR = 100d;
            double XR = cp.White_X * YR / cp.White_Y;
            double ZR = cp.White_Z * YR / cp.White_Y;

            double xr;
            double yr;
            double zr;

            double k_modifier = 903.3;
            double eps_modifier = 0.008856;


            double fy = (color.R + 16d) / 116d;
            double fx = color.G / 500d + fy;
            double fz = fy - color.B / 200d;

            xr = Math.Pow(fx, 3);
            if (xr <= eps_modifier)
            {
                xr = (116d * fx - 16d) / k_modifier;
            }

            if (color.R > k_modifier * eps_modifier)
            {
                yr = Math.Pow((color.R + 16d) / 116d, 3);
            }
            else
            {
                yr = color.R / k_modifier;
            }

            zr = Math.Pow(fz, 3);
            if (zr <= eps_modifier)
            {
                zr = (116d * fz - 16d) / k_modifier;
            }


            double X = xr * XR;
            double Y = yr * YR;
            double Z = zr * ZR;

            return new DoubleColor(X, Y, Z);

        }

        public static void ConvertImageToDoubleColor(SimpleColor[,] imageSource, DoubleColor[,] imageDest)
        {
            Parallel.For(0, imageSource.GetLength(0), i =>
            {
                for (int j = 0; j < imageSource.GetLength(1); j++)
                {
                    imageDest[i, j] = new DoubleColor(imageSource[i, j]);
                }
            });
        }

        public static void ConvertImageFromDoubleColor(DoubleColor[,] imageSource, SimpleColor[,] imageDest)
        {
            Parallel.For(0, imageSource.GetLength(0), i =>
            {
                for (int j = 0; j < imageSource.GetLength(1); j++)
                {
                    imageDest[i, j] = imageSource[i, j].ToSimpleColor();
                }
            });
        }
    }
}
