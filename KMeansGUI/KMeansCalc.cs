using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeans.GUI
{
    public static class KMeansCalc
    {
        public static DoubleColor[,] CalculateKMeans(DoubleColor[,] array, int k_means, int max_iter, double eps)
        {
            double[,] distances = new double[array.GetLength(0), array.GetLength(1)];
            int[,] clusters = new int[array.GetLength(0), array.GetLength(1)];
            DoubleColor[] centroids = new DoubleColor[k_means];
            List<(int, int)> centroids_cords = new List<(int, int)>();
            Random rand = new Random();
            for (int i = 0; i < centroids.Length; i++)
            {
                int x = rand.Next(0, array.GetLength(0));
                int y = rand.Next(0, array.GetLength(1));

                int iters = 0;
                while (centroids_cords.Exists(a => DoubleColor.Distance(array[a.Item1, a.Item2], array[x, y]) < eps) == true || iters > max_iter)
                {
                    x = rand.Next(0, array.GetLength(0));
                    y = rand.Next(0, array.GetLength(1));
                    iters++;
                }

                centroids[i] = array[x, y];
            }

            for (int iter_number = 0; iter_number < max_iter; iter_number++)
            {
                //Phase Distance Calculation
                for (int x = 0; x < array.GetLength(0); x++)
                {
                    for (int y = 0; y < array.GetLength(1); y++)
                    {
                        var minDist = double.MaxValue;
                        int index = -1;
                        for (int k = 0; k < centroids.Length; k++)
                        {
                            //if(x==0&&y==0)
                            //{
                            //    ;
                            //}
                            var cur_dist = DoubleColor.Distance(centroids[k], array[x, y]);
                            if (cur_dist < minDist)
                            {
                                minDist = cur_dist;
                                index = k;
                            }
                        }
                        //if (index == -1)
                        //    ;
                        clusters[x, y] = index;
                    }
                }

                //Phase Centroid Recalculation
                bool hasCentroidChanged = false;

                for (int k = 0; k < centroids.Length; k++)
                {
                    int count = 0;
                    DoubleColor color_sum = new DoubleColor(0d, 0d, 0d);
                    for (int x = 0; x < array.GetLength(0); x++)
                    {
                        for (int y = 0; y < array.GetLength(1); y++)
                        {
                            if (clusters[x, y] == k)
                            {
                                count++;
                                color_sum.R += array[x, y].R;
                                color_sum.G += array[x, y].G;
                                color_sum.B += array[x, y].B;
                            }
                        }
                    }
                    if (count != 0)
                    {
                        color_sum.R /= (double)count;
                        color_sum.G /= (double)count;
                        color_sum.B /= (double)count;
                    }

                    var centroid_dist = DoubleColor.Distance(centroids[k], color_sum);
                    if (centroid_dist < eps)
                    {
                        continue;
                    }
                    centroids[k] = color_sum;
                    hasCentroidChanged = true;
                }

                if (hasCentroidChanged == false)
                {
                    break;
                }
            }

            DoubleColor[,] resultArray = new DoubleColor[array.GetLength(0), array.GetLength(1)];
            for (int x = 0; x < array.GetLength(0); x++)
            {
                for (int y = 0; y < array.GetLength(1); y++)
                {
                    resultArray[x, y] = centroids[clusters[x, y]];
                }
            }

            return resultArray;
        }

        public static DoubleColor[,] CalculateKMeansAsync(DoubleColor[,] array, int k_means, int max_iter, double eps)
        {
            double[,] distances = new double[array.GetLength(0), array.GetLength(1)];
            int[,] clusters = new int[array.GetLength(0), array.GetLength(1)];
            DoubleColor[] centroids = new DoubleColor[k_means];
            List<(int, int)> centroids_cords = new List<(int, int)>();
            Random rand = new Random();
            for (int i = 0; i < centroids.Length; i++)
            {
                int x = rand.Next(0, array.GetLength(0));
                int y = rand.Next(0, array.GetLength(1));

                int iters = 0;
                while (centroids_cords.Exists(a => DoubleColor.Distance(array[a.Item1, a.Item2], array[x, y]) < eps) == true || iters > max_iter)
                {
                    x = rand.Next(0, array.GetLength(0));
                    y = rand.Next(0, array.GetLength(1));
                    iters++;
                }

                centroids[i] = array[x, y];
            }

            for (int iter_number = 0; iter_number < max_iter; iter_number++)
            {
                //Phase Distance Calculation
                Parallel.For(0, array.GetLength(0), x =>
                {
                    for (int y = 0; y < array.GetLength(1); y++)
                    {
                        var minDist = double.MaxValue;
                        int index = -1;
                        for (int k = 0; k < centroids.Length; k++)
                        {
                            //if(x==0&&y==0)
                            //{
                            //    ;
                            //}
                            var cur_dist = DoubleColor.Distance(centroids[k], array[x, y]);
                            if (cur_dist < minDist)
                            {
                                minDist = cur_dist;
                                index = k;
                            }
                        }
                        //if (index == -1)
                        //    ;
                        clusters[x, y] = index;
                    }
                });

                //Phase Centroid Recalculation
                bool hasCentroidChanged = false;


                Parallel.For(0, centroids.Length, k =>
                 {
                     int count = 0;
                     DoubleColor color_sum = new DoubleColor(0d, 0d, 0d);
                     for (int x = 0; x < array.GetLength(0); x++)
                     {
                         for (int y = 0; y < array.GetLength(1); y++)
                         {
                             if (clusters[x, y] == k)
                             {
                                 count++;
                                 color_sum.R += array[x, y].R;
                                 color_sum.G += array[x, y].G;
                                 color_sum.B += array[x, y].B;
                             }
                         }
                     }
                     if (count != 0)
                     {
                         color_sum.R /= (double)count;
                         color_sum.G /= (double)count;
                         color_sum.B /= (double)count;
                     }

                     var centroid_dist = DoubleColor.Distance(centroids[k], color_sum);
                     if (centroid_dist > eps)
                     {
                         centroids[k] = color_sum;
                         hasCentroidChanged = true;
                     }
                 });

                if (hasCentroidChanged == false)
                {
                    break;
                }
            }

            DoubleColor[,] resultArray = new DoubleColor[array.GetLength(0), array.GetLength(1)];
            Parallel.For(0, array.GetLength(0), x =>
            {
                for (int y = 0; y < array.GetLength(1); y++)
                {
                    resultArray[x, y] = centroids[clusters[x, y]];
                }
            });

            return resultArray;
        }

    }
}
