using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeans.GUI
{
    public struct SimpleColor
    {
        public byte R { get; set; }
        public byte G { get; set; }
        public byte B { get; set; }
        public SimpleColor(byte r, byte g, byte b)
        {
            R = r;
            G = g;
            B = b;
        }
        public SimpleColor(int col)
        {
            int mask = 255;
            R = (byte)((col & (mask << 16)) >> 16);
            G = (byte)((col & (mask << 8)) >> 8);
            B = (byte)(col & mask);
        }
        public void ConvertFromInt(int col)
        {
            int mask = 255;
            R = (byte)((col & (mask << 16)) >> 16);
            G = (byte)((col & (mask << 8)) >> 8);
            B = (byte)(col & mask);
        }

        public void ConvertFromInt(uint col)
        {
            int mask = 255;
            R = (byte)((col & (mask << 16)) >> 16);
            G = (byte)((col & (mask << 8)) >> 8);
            B = (byte)(col & mask);
        }
        public int ToInt()
        {
            int colorData = 0;
            colorData |= 255 << 24;
            colorData |= R << 16; // R
            colorData |= G << 8; // G
            colorData |= B; //B
            return colorData;
        }
        public void ToGrayscale()
        {
            double y = 0.299 * (double)R + 0.587 * (double)G + 0.114 * (double)B;
            if (y < 0d)
                y = 0d;
            if (y > 255d)
                y = 255d;
            byte Y = (byte)Math.Round(y);
            R = Y;
            G = Y;
            B = Y;
        }
        public override string ToString()
        {
            return $"R: {R,3}, G: {G,3}, B: {B,3}";
        }
    }
}
