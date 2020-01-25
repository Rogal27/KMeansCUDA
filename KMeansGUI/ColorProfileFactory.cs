using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeans.GUI
{
    public class ColorProfileFactory
    {
        private static ColorProfileFactory Factory;

        public static ColorProfileFactory GetFactory()
        {
            if(Factory == null)
            {
                Factory = new ColorProfileFactory();
            }
            return Factory;
        }

        public ColorProfile sRBGcolorProfile { get; set; }
        public ColorProfile adobeRBGcolorProfile { get; set; }
        public ColorProfile appleRBGcolorProfile { get; set; }
        public ColorProfile CIE_RBGcolorProfile { get; set; }
        public ColorProfile WideGamutcolorProfile { get; set; }
        public ColorProfile PALcolorProfile { get; set; }

        public ColorProfile GetColorProfile(ColorProfileEnum cp)
        {
            switch (cp)
            {
                case ColorProfileEnum.sRGB:
                    return sRBGcolorProfile;
                case ColorProfileEnum.adobeRGB:
                    return adobeRBGcolorProfile;
                case ColorProfileEnum.appleRGB:
                    return appleRBGcolorProfile;
                case ColorProfileEnum.CIE_RGB:
                    return CIE_RBGcolorProfile;
                case ColorProfileEnum.WideGamut:
                    return WideGamutcolorProfile;
                case ColorProfileEnum.PAL:
                    return PALcolorProfile;
                default:
                    return null;
            }
        }

        private ColorProfileFactory()
        {
            sRBGcolorProfile = SetsRBGcolorProfile();
            adobeRBGcolorProfile = SetadobeRBGcolorProfile();
            appleRBGcolorProfile = SetappleRBGcolorProfile();
            CIE_RBGcolorProfile = SetCIE_RBGcolorProfile();
            WideGamutcolorProfile = SetWideGamutcolorProfile();
            PALcolorProfile = SetPALcolorProfile();
        }

        private ColorProfile SetsRBGcolorProfile()
        {
            var cp = new ColorProfile();
            cp.Gamma = 2.20;
            cp.White_X = 0.3127;
            cp.White_Y = 0.3290;
            cp.Red_X = 0.6400;
            cp.Red_Y = 0.3300;
            cp.Green_X = 0.3000;
            cp.Green_Y = 0.6000;
            cp.Blue_X = 0.1500;
            cp.Blue_Y = 0.0600;
            cp.SetColorMatrix();
            return cp;
        }
        private ColorProfile SetadobeRBGcolorProfile()
        {
            var cp = new ColorProfile();
            cp.Gamma = 2.20;
            cp.White_X = 0.3127;
            cp.White_Y = 0.3290;
            cp.Red_X = 0.6400;
            cp.Red_Y = 0.3300;
            cp.Green_X = 0.2100;
            cp.Green_Y = 0.7100;
            cp.Blue_X = 0.1500;
            cp.Blue_Y = 0.0600;
            cp.SetColorMatrix();
            return cp;
        }
        private ColorProfile SetappleRBGcolorProfile()
        {
            var cp = new ColorProfile();
            cp.Gamma = 1.80;
            cp.White_X = 0.3127;
            cp.White_Y = 0.3290;
            cp.Red_X = 0.6250;
            cp.Red_Y = 0.3400;
            cp.Green_X = 0.2800;
            cp.Green_Y = 0.5950;
            cp.Blue_X = 0.1550;
            cp.Blue_Y = 0.0700;
            cp.SetColorMatrix();
            return cp;
        }
        private ColorProfile SetCIE_RBGcolorProfile()
        {
            var cp = new ColorProfile();
            cp.Gamma = 2.20;
            cp.White_X = 0.3333;
            cp.White_Y = 0.3333;
            cp.Red_X = 0.7350;
            cp.Red_Y = 0.2650;
            cp.Green_X = 0.2740;
            cp.Green_Y = 0.7170;
            cp.Blue_X = 0.1670;
            cp.Blue_Y = 0.0090;
            cp.SetColorMatrix();
            return cp;
        }
        private ColorProfile SetWideGamutcolorProfile()
        {
            var cp = new ColorProfile();
            cp.Gamma = 2.20;
            cp.White_X = 0.3457;
            cp.White_Y = 0.3585;
            cp.Red_X = 0.7347;
            cp.Red_Y = 0.2653;
            cp.Green_X = 0.1152;
            cp.Green_Y = 0.8264;
            cp.Blue_X = 0.1566;
            cp.Blue_Y = 0.0177;
            cp.SetColorMatrix();
            return cp;
        }
        private ColorProfile SetPALcolorProfile()
        {
            var cp = new ColorProfile();
            cp.Gamma = 1.95;
            cp.White_X = 0.3127;
            cp.White_Y = 0.3290;
            cp.Red_X = 0.6400;
            cp.Red_Y = 0.3300;
            cp.Green_X = 0.2900;
            cp.Green_Y = 0.6000;
            cp.Blue_X = 0.1500;
            cp.Blue_Y = 0.0600;
            cp.SetColorMatrix();
            return cp;
        }
    }
}
