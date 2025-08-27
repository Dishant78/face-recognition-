# AI-Based Missing Person Face Detection System

A real-time face detection system that can identify missing persons using advanced computer vision techniques.

## Features

- **High Accuracy Face Detection**: Uses multiple cascade classifiers and advanced image processing
- **Real-time Recognition**: Processes live camera feed with improved accuracy
- **Web Interface**: Easy-to-use Flask web application
- **Database Management**: Store and manage missing person records
- **Multiple Detection Methods**: Supports both face_recognition library and OpenCV fallback

## Quick Start

### Option 1: Easy Start (Recommended)
1. Double-click `start_app.bat` (Windows)
2. Open your browser and go to: http://localhost:5000
3. Click "Camera Feed" to start face detection

### Option 2: Command Line
```bash
python run_app.py
```

### Option 3: Direct Flask App
```bash
python app.py
```

## System Requirements

- Python 3.7+
- Webcam
- Windows/Linux/macOS

## Installation

The system will automatically install required packages on first run:
- Flask
- OpenCV (opencv-python)
- NumPy
- Pillow

## How to Use

### 1. Web Interface
- **Home**: Main dashboard
- **Add Person**: Add new missing person with photo
- **Camera Feed**: Live face detection
- **View Persons**: Manage existing records

### 2. Face Detection
- **Green Box**: Detected missing person (high confidence)
- **Red Box**: Unknown person
- **Real-time**: Continuous monitoring with improved accuracy

### 3. Adding Your Face
1. Go to "Add Person" in the web interface
2. Upload your photo (dishant.jpg)
3. Fill in your details
4. The system will automatically detect your face

## Accuracy Improvements

The system now includes:

### Enhanced Face Processing
- **9 Different Variations**: Original, histogram equalization, blur, sharpening, edge enhancement, gamma correction
- **Advanced Matching**: Template matching, histogram comparison, MSE, SSIM-like scoring
- **Stricter Thresholds**: 75% confidence required for positive identification

### Optimized Detection
- **Multiple Cascade Classifiers**: Different sensitivity levels for various scenarios
- **Duplicate Removal**: 60% overlap threshold to prevent false positives
- **Better Face Extraction**: Improved region of interest detection

### Improved Matching Algorithm
- **Weighted Scoring**: 60% template matching + 40% similarity calculation
- **SSIM-like Analysis**: Structural similarity for better face comparison
- **Multiple Methods**: CCOEFF, CCORR, SQDIFF template matching

## File Structure

```
face_fiii/
├── app.py                 # Main Flask web application
├── main.py               # Standalone face detection
├── run_app.py            # Easy startup script
├── start_app.bat         # Windows batch file
├── db.py                 # Database management
├── camera.py             # Camera handling
├── logger.py             # Detection logging
├── notifier.py           # Email notifications
├── requirements.txt      # Python dependencies
├── data/                 # Database and images
│   ├── dishant.jpg      # Your face image
│   └── missing_persons.db
└── templates/            # Web interface templates
```

## Troubleshooting

### Camera Not Working
- Check if webcam is connected and not in use by another application
- Try restarting the application
- Check camera permissions

### Low Detection Accuracy
- Ensure good lighting conditions
- Face should be clearly visible
- Try adjusting distance from camera
- Check if face image in database is clear

### Application Won't Start
- Make sure Python is installed
- Check if all dependencies are installed
- Try running `python run_app.py` for detailed error messages

## Technical Details

### Detection Thresholds
- **OpenCV Method**: 75% confidence required
- **face_recognition**: 65% confidence required (tolerance 0.35)
- **Multiple Processing**: 9 different image variations per face

### Performance
- **Real-time Processing**: ~30 FPS on standard hardware
- **Memory Efficient**: Optimized image processing
- **Scalable**: Can handle multiple faces simultaneously

## Support

For issues or questions:
1. Check the troubleshooting section
2. Ensure all dependencies are installed
3. Verify camera permissions
4. Check database connectivity

## License

This project is for educational and research purposes. 