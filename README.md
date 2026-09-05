# YOLO Detection and Training GUI

A modern PyQt5-based desktop application for YOLO object detection, training, and model export. Built with a clean, flat design interface.

## 📋 Features

### Detection Tab
- **Single Model Detection**: Run inference with one YOLO model
- **Dual Model Comparison**: Compare detection results between two models
- **Single Model + Labels**: Visualize ground truth labels on images
- **Batch/Single Mode**: Process entire directories or single images
- **Navigation**: Previous/Next buttons with keyboard shortcuts (←/→)
- **Loop Navigation**: Auto-loop from last to first image

### Training Tab
- **Model Configuration**: Select pre-trained models (.pt files)
- **Dataset Settings**: Configure YAML dataset files
- **Project Settings**: Experiment name, epochs, batch size, image size
- **Data Augmentation**: Flip probabilities, rotation, mosaic, mixup
- **Color Space Enhancement**: HSV hue/saturation/value adjustments
- **Loss Weights**: Configure box, class, and DFL loss weights
- **Quick Presets**: Default, Aggressive, Light training presets
- **Config Import/Export**: Save and load training configurations

### Format Conversion Tab
- **Model Export**: Convert PyTorch models to ONNX or TensorRT Engine
- **Export Options**: Half precision, device selection, image size
- **Progress Tracking**: Real-time export progress

### UI Features
- **Modern Flat Design**: Clean, contemporary interface
- **Responsive Layout**: Resizable panels with splitter controls
- **Configuration Persistence**: Auto-save/load settings to JSON
- **Keyboard Navigation**: Arrow keys for image browsing
- **Progress Indicators**: Visual progress bars for all operations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Windows 10/11 (recommended)
- NVIDIA GPU (optional, for CUDA acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/khjg2025/YOLO-pt-testing-GUI.git
   cd YOLO-pt-testing-GUI
   ```

2. **Create virtual environment (recommended)**
   ```bash
   # Using conda (recommended)
   conda create -n yolo-gui python=3.9
   conda activate yolo-gui

   # Or using venv
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
# Method 1: Direct execution
python app_v4.py

# Method 2: Using batch file (Windows)
双击 启动_UI版本.bat

# Method 3: Using specific Python environment
C:\ProgramData\miniconda3\envs\pyqt-yolo\python.exe app_v4.py
```

## 📁 Project Structure

```
YOLO-pt-testing-GUI/
├── app_v4.py              # Main application (modern UI version)
├── app_config.json         # Application configuration (auto-generated)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

The application automatically saves configuration to `app_config.json` including:
- Model paths (model1, model2)
- Image/label directory paths
- Detection parameters (confidence, IOU, image size, etc.)
- Training settings (epochs, batch size, augmentation, etc.)
- Export format settings

## 🎨 UI Customization

The application uses a modern flat design style defined in `APP_STYLE` stylesheet. You can customize colors, fonts, and spacing by editing the style sheet in `app_v4.py`.

### Key Style Elements
- Primary color: `#1976d2` (blue)
- Background: `#f5f5f5` (light gray)
- Card background: `#ffffff` (white)
- Font: Microsoft YaHei / PingFang SC

## 📝 Usage Guide

### Detection Workflow
1. Select model file(s) using "选择模型" buttons
2. Choose image directory using "选择图片目录"
3. For label comparison, select label directory
4. Choose detection mode (Single/Dual/Single+Label)
5. Adjust parameters (confidence, IOU, etc.)
6. Click "开始检测" to start
7. Use "上一页"/"下一页" or arrow keys to navigate

### Training Workflow
1. Select pre-trained model (.pt file)
2. Choose dataset YAML configuration
3. Set project directory and experiment name
4. Configure training parameters (epochs, batch size, etc.)
5. Adjust augmentation settings as needed
6. Click "开始训练" to start
7. Monitor training progress in log area

### Export Workflow
1. Select model file to export
2. Choose export format (ONNX/Engine)
3. Configure export parameters
4. Click "转换开始" to export

## ⚠️ Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'ultralytics'**
   ```bash
   pip install ultralytics
   ```

2. **ModuleNotFoundError: No module named 'torch'**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **CUDA not available**
   - Install CPU-only PyTorch: `pip install torch torchvision`
   - Or install CUDA version matching your NVIDIA driver

4. **UI display issues**
   - Ensure PyQt5 is installed: `pip install PyQt5`
   - Try different Python versions (3.8-3.10 recommended)

5. **Image loading fails**
   - Ensure OpenCV is installed: `pip install opencv-python`
   - Check image file permissions

### Performance Tips

- Use GPU (CUDA) for faster inference and training
- Reduce `imgsz` for faster processing
- Lower `max_det` if processing many objects
- Use `quantize` for model optimization

## 📦 Dependencies

See `requirements.txt` for complete dependency list.

### Core Dependencies
- PyQt5 >= 5.15
- ultralytics >= 8.0
- opencv-python >= 4.5
- torch >= 2.0
- numpy >= 1.20
- Pillow >= 9.0

### Optional Dependencies
- onnxruntime-gpu (for TensorRT export)
- tensorrt (for Engine export)

## 🔄 Version History

### v4.0 (Current)
- Modern flat UI design
- UI separation architecture
- Enhanced configuration management
- Improved error handling
- Better keyboard navigation

### v3.0
- Added format conversion tab
- Improved training interface
- Bug fixes and stability improvements

### v2.0
- Added dual model comparison
- Enhanced detection parameters
- Progress tracking improvements

### v1.0
- Initial release
- Basic detection and training
- Single model support

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - Modern GUI framework
- [OpenCV](https://opencv.org/) - Computer vision library

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the source code comments

---

**Developed with ❤️ for the YOLO community**
