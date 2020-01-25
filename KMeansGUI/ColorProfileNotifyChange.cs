using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace KMeans.GUI
{
    public class ColorProfileNotifyChange : ColorProfile, INotifyPropertyChanged
    {
        public bool HasChanged { get; set; }
        public new double Gamma
        {
            get
            {
                return base.Gamma;
            }
            set
            {
                base.Gamma = value;
                gamma_String = value.ToString();
                NotifyPropertyChanged("Gamma_String");
            }
        }
        public new double White_X
        {
            get
            {
                return base.White_X;
            }
            set
            {
                base.White_X = value;
                white_X_String = value.ToString();
                NotifyPropertyChanged("White_X_String");
            }
        }
        public new double White_Y
        {
            get
            {
                return base.White_Y;
            }
            set
            {
                base.White_Y = value;
                white_Y_String = value.ToString();
                NotifyPropertyChanged("White_Y_String");
            }
        }
        public new double Red_X
        {
            get
            {
                return base.Red_X;
            }
            set
            {
                base.Red_X = value;
                red_X_String = value.ToString();
                NotifyPropertyChanged("Red_X_String");
            }
        }
        public new double Red_Y
        {
            get
            {
                return base.Red_Y;
            }
            set
            {
                base.Red_Y = value;
                red_Y_String = value.ToString();
                NotifyPropertyChanged("Red_Y_String");
            }
        }
        public new double Green_X
        {
            get
            {
                return base.Green_X;
            }
            set
            {
                base.Green_X = value;
                green_X_String = value.ToString();
                NotifyPropertyChanged("Green_X_String");
            }
        }
        public new double Green_Y
        {
            get
            {
                return base.Green_Y;
            }
            set
            {
                base.Green_Y = value;
                green_Y_String = value.ToString();
                NotifyPropertyChanged("Green_Y_String");
            }
        }
        public new double Blue_X
        {
            get
            {
                return base.Blue_X;
            }
            set
            {
                base.Blue_X = value;
                blue_X_String = value.ToString();
                NotifyPropertyChanged("Blue_X_String");
            }
        }
        public new double Blue_Y
        {
            get
            {
                return base.Blue_Y;
            }
            set
            {
                base.Blue_Y = value;
                blue_Y_String = value.ToString();
                NotifyPropertyChanged("Blue_Y_String");
            }
        }

        private string gamma_String;
        public string Gamma_String
        {
            get
            {
                return gamma_String;
            }
            set
            {
                gamma_String = value;
                base.Gamma = double.Parse(value, NumberStyles.Float, Globals.culture);
                NotifyPropertyChanged();
            }
        }
        private string white_X_String;
        public string White_X_String
        {
            get
            {
                return white_X_String;
            }
            set
            {
                white_X_String = value;
                base.White_X = double.Parse(value, NumberStyles.Float, Globals.culture);
                NotifyPropertyChanged();
            }
        }
        private string white_Y_String;
        public string White_Y_String
        {
            get
            {
                return white_Y_String;
            }
            set
            {
                white_Y_String = value;
                base.White_Y = double.Parse(value, NumberStyles.Float, Globals.culture);
                NotifyPropertyChanged();
            }
        }
        private string red_X_String;
        public string Red_X_String
        {
            get
            {
                return red_X_String;
            }
            set
            {
                red_X_String = value;
                base.Red_X = double.Parse(value, NumberStyles.Float, Globals.culture);
                NotifyPropertyChanged();
            }
        }
        private string red_Y_String;
        public string Red_Y_String
        {
            get
            {
                return red_Y_String;
            }
            set
            {
                red_Y_String = value;
                base.Red_Y = double.Parse(value, NumberStyles.Float, Globals.culture);
                NotifyPropertyChanged();
            }
        }
        private string green_X_String;
        public string Green_X_String
        {
            get
            {
                return green_X_String;
            }
            set
            {
                green_X_String = value;
                base.Green_X = double.Parse(value, NumberStyles.Float, Globals.culture);
                NotifyPropertyChanged();
            }
        }
        private string green_Y_String;
        public string Green_Y_String
        {
            get
            {
                return green_Y_String;
            }
            set
            {
                green_Y_String = value;
                base.Green_Y = double.Parse(value, NumberStyles.Float, Globals.culture);
                NotifyPropertyChanged();
            }
        }
        private string blue_X_String;
        public string Blue_X_String
        {
            get
            {
                return blue_X_String;
            }
            set
            {
                blue_X_String = value;
                base.Blue_X = double.Parse(value, NumberStyles.Float, Globals.culture);
                NotifyPropertyChanged();
            }
        }
        private string blue_Y_String;
        public string Blue_Y_String
        {
            get
            {
                return blue_Y_String;
            }
            set
            {
                blue_Y_String = value;
                base.Blue_Y = double.Parse(value, NumberStyles.Float, Globals.culture);
                NotifyPropertyChanged();
            }
        }


        public ColorProfileNotifyChange(ColorProfile cp) : base()
        {
            base.Gamma = cp.Gamma;
            base.White_X = cp.White_X;
            base.White_Y = cp.White_Y;
            base.Red_X = cp.Red_X;
            base.Red_Y = cp.Red_Y;
            base.Green_X = cp.Green_X;
            base.Green_Y = cp.Green_Y;
            base.Blue_X = cp.Blue_X;
            base.Blue_Y = cp.Blue_Y;
            gamma_String = Gamma.ToString(Globals.culture);
            white_X_String = White_X.ToString(Globals.culture);
            white_Y_String = White_Y.ToString(Globals.culture);
            red_X_String = Red_X.ToString(Globals.culture);
            red_Y_String = Red_Y.ToString(Globals.culture);
            green_X_String = Green_X.ToString(Globals.culture);
            green_Y_String = Green_Y.ToString(Globals.culture);
            blue_X_String = Blue_X.ToString(Globals.culture);
            blue_Y_String = Blue_Y.ToString(Globals.culture);
            HasChanged = false;
            SetColorMatrix();
        }


        public event PropertyChangedEventHandler PropertyChanged;

        private void NotifyPropertyChanged([CallerMemberName] String propertyName = "")
        {
            HasChanged = true;
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public (bool isValid, string info) Validate()
        {
            if (double.TryParse(Gamma_String, NumberStyles.Float, Globals.culture, out _) == false)
                return (false, "Gamma should be decimal");
            if (Gamma < 0)
                return (false, "Gamma should be greater than 0");
            if (double.TryParse(White_X_String, NumberStyles.Float, Globals.culture, out _) == false)
                return (false, "White X should be decimal");
            if (White_X < 0 || White_X > 1)
                return (false, "White X should be between 0 and 1");
            if (double.TryParse(White_Y_String, NumberStyles.Float, Globals.culture, out _) == false)
                return (false, "White Y should be decimal");
            if (White_Y < 0 || White_Y > 1)
                return (false, "White Y should be between 0 and 1");
            if (White_Z < 0)
                return (false, "White X and White Y sum should be less than 1");
            if (double.TryParse(Red_X_String, NumberStyles.Float, Globals.culture, out _) == false)
                return (false, "Red X should be decimal");
            if (Red_X < 0 || Red_X > 1)
                return (false, "Red X should be between 0 and 1");
            if (double.TryParse(Red_Y_String, NumberStyles.Float, Globals.culture, out _) == false)
                return (false, "Red Y should be decimal");
            if (Red_Y < 0 || Red_Y > 1)
                return (false, "Red Y should be between 0 and 1");
            if (Red_Z < 0)
                return (false, "Red X and Red Y sum should be less than 1");
            if (double.TryParse(Green_X_String, NumberStyles.Float, Globals.culture, out _) == false)
                return (false, "Green X should be decimal");
            if (Green_X < 0 || Green_X > 1)
                return (false, "Green X should be between 0 and 1");
            if (double.TryParse(Green_Y_String, NumberStyles.Float, Globals.culture, out _) == false)
                return (false, "Green Y should be decimal");
            if (Green_Y < 0 || Green_Y > 1)
                return (false, "Green Y should be between 0 and 1");
            if (Green_Z < 0)
                return (false, "Green X and Green Y sum should be less than 1");
            if (double.TryParse(Blue_X_String, NumberStyles.Float, Globals.culture, out _) == false)
                return (false, "Blue X should be decimal");
            if (Blue_X < 0 || Blue_X > 1)
                return (false, "Blue X should be between 0 and 1");
            if (double.TryParse(Blue_Y_String, NumberStyles.Float, Globals.culture, out _) == false)
                return (false, "Blue Y should be decimal");
            if (Blue_Y < 0 || Blue_Y > 1)
                return (false, "Blue Y should be between 0 and 1");
            if (Blue_Z < 0)
                return (false, "Blue X and Blue Y sum should be less than 1");
            SetColorMatrix();
            return (true, "");
        }
    }
}
