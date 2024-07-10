# Continuous Sign Language Recognition System

## Project Overview
This project utilizes a Seq2Seq deep learning model to recognize continuous sign language, supporting a CNN-LSTM architecture. It aims to provide a more accurate and fluent communication method for the deaf and mute.
## Requirements
- Python >= 3.7
- PyTorch >= 1.7.0
- Flask >= 1.1.2
- OpenCV-python
- PIL
- Additional dependencies as needed

## Installation Guide

### Create a Python Virtual Environment
```bash
conda create -n handread python=3.8
```

### Activate the Virtual Environment
```bash
conda activate handread
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
Note: If you encounter dependency conflicts, specify the correct version based on the error message.

## Running the Project

### Start the Backend Service
```bash
python app.py
```
This command will start the Flask server, which handles video uploads and sign language recognition requests.

### Access the Frontend
Open `http://127.0.0.1:5001` in your browser to upload sign language videos for recognition.

## Model Training
To train your own model, modify the dataset paths and model configurations in the `train.py` file, then run:
```bash
python train.py
```
Once the basic model is trained, adjust the processing methods in `dataset.py` according to your dataset, then run:
```bash
python CSL_Continuous_Seq2Seq.py
```

## Project Structure
```plaintext
.
├── app.py                        # Main Flask server file
├── CSL_Continuous_Seq2Seq.py     # Main model training file
├── dataset.py                    # Dataset processing
├── logits.py                     # Model prediction logic
├── main.py                       # Pre-run environment and configuration check
├── models                        # Model architecture code
├── runs                          # Experiment records folder
├── SLR_Dataset                   # Sign language dataset directory
├── static                        # Static files
├── templates                     # HTML template files
│   └── index.html                # Frontend page
├── test.py                       # Model testing script
├── tools.py                      # Utility code
├── train.py                      # Model training script
├── validation.py                 # Model validation script
├── tmps                          # Temporary files folder
├── log                           # Log files folder
├── checkpoint                    # Model checkpoint folder
│   └── seq2seq_models            # Seq2Seq model checkpoints
│       └── slr_seq2seq_epoch097.pth # Retained weight file
└── font                          # Font files folder
    └── simsun.ttc                # Font file

```

## 注意事项
- The project uses GPU by default. To run on CPU, modify the device configuration in `logits.py`.
- For new sign language datasets, adjust the data loading and preprocessing methods in `dataset.py`.
- When training models, ensure the dataset paths are correct and adjust model parameters as needed.
