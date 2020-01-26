using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Runtime.InteropServices.WindowsRuntime;
using System.IO;

namespace KMeans.GUI
{
    using Cpp.CLI;
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        private bool IsWindowInitialized = false;

        private WriteableBitmap SourceImageWB;
        private WriteableBitmap DestImageWB;


        private SimpleColor[,] OriginalSourceImageColorArray;
        private SimpleColor[,] SourceImageColorArray;
        private SimpleColor[,] DestImageColorArray;

        public List<string> ColorSpaceList { get; set; }

        private ColorProfileNotifyChange SourceImageCP;
        private ColorProfileNotifyChange DestImageCP;
        public ColorProfileNotifyChange SourceImageColorProfile
        {
            get
            {
                return SourceImageCP;
            }
            set
            {
                SourceImageCP = value;
                NotifyPropertyChanged();
            }
        }
        public ColorProfileNotifyChange DestImageColorProfile
        {
            get
            {
                return DestImageCP;
            }
            set
            {
                DestImageCP = value;
                NotifyPropertyChanged();
            }
        }

        private int _KMeansParam;
        public int KMeansParam
        {
            get
            {
                return _KMeansParam;
            }
            set
            {
                _KMeansParam = value;
                NotifyPropertyChanged();
            }
        }

        private bool _useLAB;
        public bool useLAB
        {
            get
            {
                return _useLAB;
            }
            set
            {
                _useLAB = value;
                NotifyPropertyChanged();
            }
        }

        private bool GrayscaleButtonClicked;


        public MainWindow()
        {
            SetVariables();
            InitializeComponent();
        }

        public event PropertyChangedEventHandler PropertyChanged;

        private void NotifyPropertyChanged([CallerMemberName] String propertyName = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private void SetVariables()
        {
            ColorSpaceList = new List<string>
            {
                "sRGB",
                "Adobe RGB",
                "Apple RGB",
                "CIE RGB",
                "Wide Gamut",
                "PAL/SECAM"
            };
            SourceImageCP = new ColorProfileNotifyChange(ColorProfileFactory.GetFactory().sRBGcolorProfile);
            DestImageCP = new ColorProfileNotifyChange(ColorProfileFactory.GetFactory().WideGamutcolorProfile);
            GrayscaleButtonClicked = false;
            KMeansParam = Globals.k_means;
            useLAB = true;
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            SourceColorSpaceComboBox.ItemsSource = ColorSpaceList;
            SourceColorSpaceComboBox.SelectedIndex = 0;//srgb
            DestColorSpaceComboBox.ItemsSource = ColorSpaceList;
            DestColorSpaceComboBox.SelectedIndex = 4;//widegamut

            var bmp = KMeans.GUI.Properties.Resources.image1;
            var SourceImageBitmap = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(bmp.GetHbitmap(), IntPtr.Zero, System.Windows.Int32Rect.Empty, BitmapSizeOptions.FromWidthAndHeight(bmp.Width, bmp.Height));

            LoadWritableBitamp(SourceImageBitmap, IsWindowInitialized);

            IsWindowInitialized = true;
        }

        private void LoadWritableBitamp(BitmapSource bitmap, bool IsWindowInit)
        {
            SourceImageWB = new WriteableBitmap(bitmap);
            DestImageWB = new WriteableBitmap(bitmap);

            int width = SourceImageWB.PixelWidth;
            int height = SourceImageWB.PixelHeight;

            SourceImageControl.Source = SourceImageWB;
            SourceImageControl.Width = width;
            SourceImageControl.Height = height;

            DestImageControl.Source = DestImageWB;
            DestImageControl.Width = width;
            DestImageControl.Height = height;

            SourceImageColorArray = new SimpleColor[height, width];
            DestImageColorArray = new SimpleColor[height, width];


            Paint.ReadColorsFromBitmap(SourceImageWB, SourceImageColorArray);
            OriginalSourceImageColorArray = (SimpleColor[,])SourceImageColorArray.Clone();

            if (IsWindowInit == false)
            {
                ColorProfileConverter.ConvertImage(SourceImageColorArray, DestImageColorArray, (ColorProfileEnum)SourceColorSpaceComboBox.SelectedIndex, (ColorProfileEnum)DestColorSpaceComboBox.SelectedIndex);
                Paint.CopyToWriteableBitmap(DestImageWB, DestImageColorArray);
            }
            else
            {
                TryGenerate();
            }
        }

        private void SourceColorSpaceComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (IsWindowInitialized == true)
            {
                var cb = sender as ComboBox;
                int selected = cb.SelectedIndex;
                int to = DestColorSpaceComboBox.SelectedIndex;

                SourceImageColorProfile = new ColorProfileNotifyChange(ColorProfileFactory.GetFactory().GetColorProfile((ColorProfileEnum)selected));

                ColorProfileConverter.ConvertImage(SourceImageColorArray, DestImageColorArray, (ColorProfileEnum)selected, (ColorProfileEnum)to);
                Paint.CopyToWriteableBitmap(DestImageWB, DestImageColorArray);
            }
        }

        private void DestColorSpaceComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (IsWindowInitialized == true)
            {
                var cb = sender as ComboBox;
                int selected = cb.SelectedIndex;
                int from = SourceColorSpaceComboBox.SelectedIndex;

                DestImageColorProfile = new ColorProfileNotifyChange(ColorProfileFactory.GetFactory().GetColorProfile((ColorProfileEnum)selected));

                ColorProfileConverter.ConvertImage(SourceImageColorArray, DestImageColorArray, (ColorProfileEnum)from, (ColorProfileEnum)selected);
                Paint.CopyToWriteableBitmap(DestImageWB, DestImageColorArray);
            }
        }

        private void GenerateRadioButton_Click(object sender, RoutedEventArgs e)
        {
            TryGenerate();
        }

        private bool TryGenerate(bool paint = true)
        {
            var isValid = ValidateTextBoxes();
            if (isValid.IsSourceValid == false && isValid.IsDestValid == false)
            {
                var info = new StringBuilder();
                info.AppendLine("Error in source and destination color space!");
                MessageBox.Show(info.ToString(), Globals.WindowName, MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }
            else if (isValid.IsSourceValid == false)
            {
                var info = new StringBuilder();
                info.AppendLine("Error in source color space!");
                MessageBox.Show(info.ToString(), Globals.WindowName, MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }
            else if (isValid.IsDestValid == false)
            {
                var info = new StringBuilder();
                info.AppendLine("Error in destination color space!");
                MessageBox.Show(info.ToString(), Globals.WindowName, MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }
            if (SourceImageColorProfile.HasChanged == false && DestImageColorProfile.HasChanged == false)
            {
                if (paint == true)
                {
                    int from = SourceColorSpaceComboBox.SelectedIndex;
                    int to = DestColorSpaceComboBox.SelectedIndex;
                    ColorProfileConverter.ConvertImage(SourceImageColorArray, DestImageColorArray, (ColorProfileEnum)from, (ColorProfileEnum)to);
                    Paint.CopyToWriteableBitmap(DestImageWB, DestImageColorArray);
                }
            }
            else
            {
                var sourceValidation = SourceImageColorProfile.Validate();
                var destValidation = DestImageColorProfile.Validate();
                if (sourceValidation.isValid == false && destValidation.isValid == false)
                {
                    //error
                    var info = new StringBuilder();
                    info.AppendLine("Error in source and destination color space:");
                    info.AppendLine("Source: " + sourceValidation.info);
                    info.Append("Destination: " + destValidation.info);

                    MessageBox.Show(info.ToString(), Globals.WindowName, MessageBoxButton.OK, MessageBoxImage.Error);
                    return false;
                }
                else if (sourceValidation.isValid == false)
                {
                    //error
                    var info = new StringBuilder();
                    info.AppendLine("Error in source color space:");
                    info.Append(sourceValidation.info);

                    MessageBox.Show(info.ToString(), Globals.WindowName, MessageBoxButton.OK, MessageBoxImage.Error);
                    return false;
                }
                else if (destValidation.isValid == false)
                {
                    //error
                    var info = new StringBuilder();
                    info.AppendLine("Error in destination color space:");
                    info.Append(destValidation.info);

                    MessageBox.Show(info.ToString(), Globals.WindowName, MessageBoxButton.OK, MessageBoxImage.Error);
                    return false;
                }
                else
                {
                    if (paint == true)
                    {
                        ColorProfileConverter.ConvertImage(SourceImageColorArray, DestImageColorArray, SourceImageColorProfile, DestImageColorProfile);
                        Paint.CopyToWriteableBitmap(DestImageWB, DestImageColorArray);
                    }
                }
            }
            return true;
        }
        private (bool IsSourceValid, bool IsDestValid) ValidateTextBoxes()
        {
            var validationRule = new TextBoxValidationRule();
            var gammaValidationRule = new GammaTextBoxValidationRule();

            var sourceTextBoxes1 = SourceGrid.Children.OfType<TextBox>();
            var sourceTextBoxes2 = SourceGrid2.Children.OfType<TextBox>();
            var destTextBoxes1 = DestGrid.Children.OfType<TextBox>();
            var destTextBoxes2 = DestGrid2.Children.OfType<TextBox>();
            var sourceTextBoxes = sourceTextBoxes1.Concat(sourceTextBoxes2);
            var destTextBoxes = destTextBoxes1.Concat(destTextBoxes2);
            bool IsSourceValid = true;
            bool IsDestValid = true;
            foreach (var box in sourceTextBoxes)
            {
                if (validationRule.Validate(box.Text, null).IsValid == false)
                {
                    IsSourceValid = false;
                    break;
                }
            }
            if (gammaValidationRule.Validate(SourceGammaTextBox.Text, null).IsValid == false)
            {
                IsSourceValid = false;
            }
            foreach (var box in destTextBoxes)
            {
                if (box.Name == "KMeansTextBox")
                {
                    if(int.TryParse(box.Text,out int res)==true)
                    {
                        if (res <= 0)
                        {
                            IsDestValid = false;
                            break;
                        }
                    }
                    else
                    {
                        IsDestValid = false;
                        break;
                    }
                }
                else if (validationRule.Validate(box.Text, null).IsValid == false)
                {
                    IsDestValid = false;
                    break;
                }
            }
            if (gammaValidationRule.Validate(DestGammaTextBox.Text, null).IsValid == false)
            {
                IsDestValid = false;
            }
            return (IsSourceValid, IsDestValid);
        }

        private void GrayscaleRadioButton_Click(object sender, RoutedEventArgs e)
        {
            if (GrayscaleButtonClicked == false)
            {
                Paint.SetGrayscale(SourceImageColorArray);
                if (TryGenerate() == true)
                {
                    GrayscaleButtonClicked = true;
                    Paint.CopyToWriteableBitmap(SourceImageWB, SourceImageColorArray);
                }
                else
                {
                    SourceImageColorArray = (SimpleColor[,])OriginalSourceImageColorArray.Clone();
                }
            }
            else
            {
                SourceImageColorArray = (SimpleColor[,])OriginalSourceImageColorArray.Clone();
                if (TryGenerate() == true)
                {
                    GrayscaleButtonClicked = false;
                    Paint.CopyToWriteableBitmap(SourceImageWB, SourceImageColorArray);
                }
                else
                {
                    Paint.SetGrayscale(SourceImageColorArray);
                }
            }
        }
        private void KMeansRadioButton_Click(object sender, RoutedEventArgs e)
        {
            DoubleColor[,] LABImageArray = new DoubleColor[SourceImageColorArray.GetLength(0), SourceImageColorArray.GetLength(1)];
            int from = SourceColorSpaceComboBox.SelectedIndex;

            
            if(useLAB==true)
            {
                ColorProfileConverter.ConvertImageToLAB(SourceImageColorArray, LABImageArray, (ColorProfileEnum)from);
                var result = KMeansCalc.CalculateKMeans(LABImageArray, KMeansParam, Globals.max_iter, Globals.eps);
                ColorProfileConverter.ConvertImageFromLAB(result, DestImageColorArray, (ColorProfileEnum)from);
            }
            else
            {
                ColorProfileConverter.ConvertImageToDoubleColor(SourceImageColorArray, LABImageArray);
                var result = KMeansCalc.CalculateKMeans(LABImageArray, KMeansParam, Globals.max_iter, Globals.eps);
                ColorProfileConverter.ConvertImageFromDoubleColor(result, DestImageColorArray);
            }

            
            
            Paint.CopyToWriteableBitmap(DestImageWB, DestImageColorArray);
        }


        private void LoadImageRadioButton_Click(object sender, RoutedEventArgs e)
        {
            if (TryGenerate(false) == false)
                return;

            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Title = "Open an Image File";
            openFileDialog.Filter = "Image Files (*.gif;*.jpg;*.jpe*;*.png;*.bmp)|*.gif;*.jpg;*.jpe*;*.png;*.bmp|All files (*.*)|*.*";

            openFileDialog.InitialDirectory = System.IO.Path.GetFullPath(Globals.ImagesFolderPath);
            openFileDialog.Multiselect = false;

            try
            {
                if (openFileDialog.ShowDialog() == true)
                {
                    BitmapImage OpenedBitmap;
                    string filename;
                    try
                    {
                        filename = openFileDialog.FileName;
                        OpenedBitmap = new BitmapImage(new Uri(filename, UriKind.Absolute));
                    }
                    catch (Exception)
                    {
                        MessageBox.Show("Could not open file!", Globals.WindowName, MessageBoxButton.OK, MessageBoxImage.Exclamation);
                        return;
                    }

                    LoadWritableBitamp(OpenedBitmap, IsWindowInitialized);
                }
            }
            catch (Exception)
            {
                openFileDialog.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
                if (openFileDialog.ShowDialog() == true)
                {
                    BitmapImage OpenedBitmap;
                    string filename;
                    try
                    {
                        filename = openFileDialog.FileName;
                        OpenedBitmap = new BitmapImage(new Uri(filename, UriKind.Absolute));
                    }
                    catch (Exception)
                    {
                        MessageBox.Show("Could not open file!", Globals.WindowName, MessageBoxButton.OK, MessageBoxImage.Exclamation);
                        return;
                    }

                    LoadWritableBitamp(OpenedBitmap, IsWindowInitialized);
                }
            }
        }

        private void SaveImageRadioButton_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog saveFileDialog = new SaveFileDialog();
            saveFileDialog.Filter = "Jpeg Image|*.jpeg|Bitmap Image|*.bmp|PNG Image|*.png|Gif Image|*.gif";
            saveFileDialog.Title = "Save an Image File";
            saveFileDialog.FileName = "Untitled";
            saveFileDialog.InitialDirectory = System.IO.Path.GetFullPath(Globals.ImagesFolderPath);
            try
            {
                if (saveFileDialog.ShowDialog() == true)
                {
                    if (saveFileDialog.FileName != "")
                    {
                        switch (saveFileDialog.FilterIndex)
                        {
                            case 1:
                                SaveWriteableBitmap(DestImageWB, saveFileDialog.FileName, FileFormat.Jpeg);
                                break;
                            case 2:
                                SaveWriteableBitmap(DestImageWB, saveFileDialog.FileName, FileFormat.Bmp);
                                break;
                            case 3:
                                SaveWriteableBitmap(DestImageWB, saveFileDialog.FileName, FileFormat.Png);
                                break;
                            case 4:
                                SaveWriteableBitmap(DestImageWB, saveFileDialog.FileName, FileFormat.Gif);
                                break;
                        }
                    }
                }
            }
            catch (Exception)
            {
                saveFileDialog.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
                if (saveFileDialog.ShowDialog() == true)
                {
                    if (saveFileDialog.FileName != "")
                    {
                        switch (saveFileDialog.FilterIndex)
                        {
                            case 1:
                                SaveWriteableBitmap(DestImageWB, saveFileDialog.FileName, FileFormat.Jpeg);
                                break;
                            case 2:
                                SaveWriteableBitmap(DestImageWB, saveFileDialog.FileName, FileFormat.Bmp);
                                break;
                            case 3:
                                SaveWriteableBitmap(DestImageWB, saveFileDialog.FileName, FileFormat.Png);
                                break;
                            case 4:
                                SaveWriteableBitmap(DestImageWB, saveFileDialog.FileName, FileFormat.Gif);
                                break;
                        }
                    }
                }
            }

        }


        private void SaveWriteableBitmap(WriteableBitmap bitmap, string filename, FileFormat format)
        {
            using (FileStream stream = new FileStream(filename, FileMode.Create))
            {
                switch (format)
                {
                    case FileFormat.Jpeg:
                        JpegBitmapEncoder encoder1 = new JpegBitmapEncoder();
                        encoder1.Frames.Add(BitmapFrame.Create(bitmap));
                        encoder1.Save(stream);
                        break;
                    case FileFormat.Bmp:
                        BmpBitmapEncoder encoder2 = new BmpBitmapEncoder();
                        encoder2.Frames.Add(BitmapFrame.Create(bitmap));
                        encoder2.Save(stream);
                        break;
                    case FileFormat.Png:
                        PngBitmapEncoder encoder3 = new PngBitmapEncoder();
                        encoder3.Frames.Add(BitmapFrame.Create(bitmap));
                        encoder3.Save(stream);
                        break;
                    case FileFormat.Gif:
                        GifBitmapEncoder encoder4 = new GifBitmapEncoder();
                        encoder4.Frames.Add(BitmapFrame.Create(bitmap));
                        encoder4.Save(stream);
                        break;
                }
            }
        }

        private enum FileFormat
        {
            Jpeg,
            Bmp,
            Png,
            Gif
        }

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                TryGenerate();
            }
        }


        private void CreateImageRadioButton_Click(object sender, RoutedEventArgs e)
        {
            if (TryGenerate(false) == false)
                return;

            var bitmap = new WriteableBitmap(Globals.ImageWidth, Globals.ImageHeight, 96, 96, PixelFormats.Bgr24, null);

            SourceImageWB = new WriteableBitmap(bitmap);
            DestImageWB = new WriteableBitmap(bitmap);

            int width = SourceImageWB.PixelWidth;
            int height = SourceImageWB.PixelHeight;

            SourceImageControl.Source = SourceImageWB;
            SourceImageControl.Width = width;
            SourceImageControl.Height = height;

            DestImageControl.Source = DestImageWB;
            DestImageControl.Width = width;
            DestImageControl.Height = height;

            SourceImageColorArray = new SimpleColor[height, width];
            DestImageColorArray = new SimpleColor[height, width];

            Paint.CreateHSVBitmap(SourceImageColorArray);
            Paint.CopyToWriteableBitmap(SourceImageWB, SourceImageColorArray);

            OriginalSourceImageColorArray = (SimpleColor[,])SourceImageColorArray.Clone();

            TryGenerate();
        }

        private void TestButtonRadioButton_Click(object sender, RoutedEventArgs e)
        {
            int[] tab1 = new int[10];
            int[] tab2 = new int[10];
            for (int i = 0; i < tab1.Length; i++)
            {
                tab1[i] = 3;
                tab2[i] = 7;
            }

            

            using (var wrapper = new Logic())
            { 
                var result = wrapper.addParallelVectors(tab1, tab2, tab1.Length);

                for (int i = 0; i < result.Length; i++)
                {
                    Debug.Write($"{result[i]}, ");
                }
                Debug.WriteLine("");

            }
            
        }
    }
}
