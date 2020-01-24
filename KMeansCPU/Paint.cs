using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;

namespace KMeansCPU
{
    public static class Paint
    {
        public static void CopyToWriteableBitmap(WriteableBitmap writeableBitmap, SimpleColor[,] pixels)
        {
            writeableBitmap.Lock();
            unsafe
            {
                int writeablebpp = writeableBitmap.Format.BitsPerPixel / 8;
                int writeableBuffer = (int)writeableBitmap.BackBuffer;
                int bufferstride = writeableBitmap.BackBufferStride;
                Parallel.For(0, pixels.GetLength(0), y =>
                {
                    int place = writeableBuffer + y * bufferstride;
                    for (int x = 0; x < pixels.GetLength(1); x++)
                    {
                        *((int*)place) = pixels[y, x].ToInt();
                        place += writeablebpp;
                    }
                });
            }
            writeableBitmap.AddDirtyRect(new Int32Rect(0, 0, writeableBitmap.PixelWidth, writeableBitmap.PixelHeight));
            writeableBitmap.Unlock();
        }
        public static void ReadColorsFromBitmap(WriteableBitmap writeableBitmap, SimpleColor[,] pixels)
        {
            writeableBitmap.Lock();
            unsafe
            {
                int writeablebpp = writeableBitmap.Format.BitsPerPixel / 8;
                int writeableBuffer = (int)writeableBitmap.BackBuffer;
                int bufferstride = writeableBitmap.BackBufferStride;
                for (int y = 0; y < pixels.GetLength(0); y++)
                {
                    for (int x = 0; x < pixels.GetLength(1); x++)
                    {
                        int col = *((int*)writeableBuffer);
                        //SimpleColor sc = new SimpleColor(col);
                        pixels[y, x].ConvertFromInt(col);
                        writeableBuffer += writeablebpp;
                    }
                }
            }
            writeableBitmap.AddDirtyRect(new Int32Rect(0, 0, writeableBitmap.PixelWidth, writeableBitmap.PixelHeight));
            writeableBitmap.Unlock();
        }

        public static void SetGrayscale(SimpleColor[,] pixels)
        {
            Parallel.For(0, pixels.GetLength(0), y =>
            {
                for (int j = 0; j < pixels.GetLength(1); j++)
                {
                    pixels[y, j].ToGrayscale();
                }
            });
        }

        public static void CreateHSVBitmap(SimpleColor[,] pixels)
        {
            int middle_x = pixels.GetLength(1) / 2;
            int middle_y = pixels.GetLength(0) / 2;
            for (int y = 0; y < pixels.GetLength(0); y++)
            {
                for (int x = 0; x < pixels.GetLength(1); x++)
                {
                    if (y < Globals.BorderWidth || y > pixels.GetLength(0) - Globals.BorderWidth ||
                        x < Globals.BorderWidth || x > pixels.GetLength(1) - Globals.BorderWidth)
                    {
                        pixels[y, x] = new SimpleColor(0, 0, 0);
                    }
                    else
                    {
                        pixels[y, x] = new SimpleColor(255, 255, 255);
                        double dist = Distance(middle_x, middle_y, x, y);
                        if (dist <= Globals.Radius)
                        {
                            //calculate HSV
                            double V = 1;
                            double C = dist / (double)Globals.Radius;
                            double m = 1 - C;
                            double cos = Cosinus(middle_x, middle_y, x, y, dist);
                            if (y <= middle_y)// angle between 0 and pi
                            {
                                double H = Math.Acos(cos);
                                double angleH = H * 180d / Math.PI;
                                double int_H = (angleH / 60d);
                                //C to dist
                                double X = C * (1 - Math.Abs((int_H % 2) - 1));
                                if (int_H <= 1)
                                {
                                    var color = new DoubleColor(C + m, X + m, m);
                                    var simpleColor = color.ToSimpleColor();
                                    //double R = 0.5 + C * Math.Cos(angleH) / Math.Cos(60 - H);
                                    //double B = 0.5 - C;
                                    //double G = 1.5 - B - R;
                                    //color = new DoubleColor(R, G, B);
                                    //simpleColor = color.ToSimpleColor();

                                    pixels[y, x] = simpleColor;
                                }
                                else if (int_H <= 2)
                                {
                                    var color = new DoubleColor(X + m, C + m, m);
                                    pixels[y, x] = color.ToSimpleColor();
                                }
                                else
                                {
                                    var color = new DoubleColor(m, C + m, X + m);
                                    pixels[y, x] = color.ToSimpleColor();
                                }
                            }
                            else // angle between pi and 2 pi
                            {
                                double H = Math.Acos(cos);
                                double angleH = 360d - H * 180 / Math.PI;
                                double int_H = (angleH / 60);
                                //C to dist
                                double X = C * (1 - Math.Abs((int_H % 2) - 1));
                                if (int_H <= 4)
                                {
                                    var color = new DoubleColor(m, X + m, C + m);
                                    pixels[y, x] = color.ToSimpleColor();
                                }
                                else if (int_H <= 5)
                                {
                                    var color = new DoubleColor(X + m, m, C + m);
                                    pixels[y, x] = color.ToSimpleColor();
                                }
                                else
                                {
                                    var color = new DoubleColor(C + m, m, X + m);
                                    pixels[y, x] = color.ToSimpleColor();
                                }
                            }
                        }



                    }
                }
            }
        }



        private static double Distance(int x_1, int y_1, int x_2, int y_2)
        {
            int a = x_2 - x_1;
            int b = y_2 - y_1;
            return Math.Sqrt(a * a + b * b);
        }

        private static double Cosinus(int middle_x, int middle_y, int x, int y, double length) // cosinus to vector [1,0]
        {
            int v_x = x - middle_x;
            //int v_y = y - middle_y;
            //double dot = (double)v_x;
            return (double)v_x / length;
        }





    }
}
