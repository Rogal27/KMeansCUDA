using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;

namespace KMeans.GUI
{
    class TextBoxValidationRule : ValidationRule
    {
        public override ValidationResult Validate(object value, CultureInfo cultureInfo)
        {
            if(value is string s)
            {
                if (double.TryParse(s, NumberStyles.Float, Globals.culture, out double res) == true)
                {
                    if (res >= 0 && res <= 1)
                    {
                        return ValidationResult.ValidResult;
                    }
                    return new ValidationResult(false, "Value must be between 0 and 1!");
                }
            }
            return new ValidationResult(false, "Value must be decimal!");
        }
    }
    class GammaTextBoxValidationRule : ValidationRule
    {
        public override ValidationResult Validate(object value, CultureInfo cultureInfo)
        {
            if (value is string s)
            {
                if (double.TryParse(s, NumberStyles.Float, Globals.culture, out double res) == true)
                {
                    if (res >= 0)
                    {
                        return ValidationResult.ValidResult;
                    }
                    return new ValidationResult(false, "Value must be greater than 0 !");
                }
            }
            return new ValidationResult(false, "Value must be decimal!");
        }
    }
}
