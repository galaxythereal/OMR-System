# Automated OMR System
## Installation Guide

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Kaggle account

### Installation Steps

1. Create a new Kaggle notebook
```bash
# Clone repository if using locally
git clone https://github.com/your-username/omr-system.git
cd omr-system
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Upload the following folder structure to Kaggle:
```
omr_data/
├── filled/
│   └── box_*.png
├── empty/
│   └── box_*.png
└── test_sheets/
    └── *.jpg
```

4. Mount Google Drive in Kaggle notebook
```python
from google.colab import drive
drive.mount('/content/drive')
```

5. Run the training script
```python
python train_model.py
```

6. Test the system
```python
python test_omr.py --image_path /path/to/test/sheet.jpg
```

### Requirements.txt
```
tensorflow==2.14.0
opencv-python==4.8.1
numpy==1.24.3
Pillow==10.0.0
matplotlib==3.7.2
scikit-learn==1.3.0
pandas==2.0.3
```

### Usage
1. Prepare the answer sheet according to the template
2. Scan or photograph the sheet (ensure good lighting)
3. Run the processing script
4. Review results in the generated output

### Troubleshooting
- Ensure proper lighting conditions
- Check image is properly aligned
- Verify bubble filling is clear and complete

### Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

### License
This project is licensed under the MIT License - see the LICENSE.md file for details.
