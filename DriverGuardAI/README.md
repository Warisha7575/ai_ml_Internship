# DriverGuard AI
AI-Based Driver Drowsiness Detection and Monitoring System

## Features
- Real-time driver fatigue detection using computer vision and machine learning
- Eye Aspect Ratio (EAR) calculation for both eyes
- Blink rate, eye closure duration, and PERCLOS (Percentage of Eye Closure) monitoring
- Visual dashboard with real-time metrics
- Audio alarm when drowsiness is detected
- Synthetic dataset generation for model training

## Project Structure
```
DriverGuardAI/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── README.md
├── app.py (main Streamlit application)
├── assets/
│   └── alarm.wav
├── dataset/
│   └── fatigue_dataset.csv
├── generate_dataset.py (generates synthetic training data)
├── models/
│   └── fatigue_model.pkl (trained Random Forest model)
├── requirements.txt
├── src/
│   ├── alarm.py (alarm handling using simpleaudio)
│   ├── live_predictor.py (standalone OpenCV-based predictor)
│   └── utils.py (utility functions for EAR, fatigue score, etc.)
└── train_model.py (trains the fatigue detection model)
```

## Installation
1. Clone or download the project
2. Create and activate a virtual environment (recommended)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generate Dataset (Optional)
To generate a new synthetic dataset:
```bash
python generate_dataset.py
```

### Train Model (Optional)
To train the model on the dataset:
```bash
python train_model.py
```

### Run Streamlit App
To start the main application:
```bash
streamlit run app.py
```

### Run Standalone Predictor
To run the OpenCV-based live predictor:
```bash
python src/live_predictor.py
```

## Model Details
- Algorithm: Random Forest Classifier
- Features used:
  - EAR (Eye Aspect Ratio)
  - Blink Rate (blinks per minute)
  - Closure Duration (seconds)
  - PERCLOS (Percentage of Eye Closure)

## Technologies Used
- Python 3.x
- Streamlit (for UI)
- OpenCV (for computer vision)
- MediaPipe (for facial landmark detection)
- Scikit-learn (for machine learning model)
- SimpleAudio (for alarm sounds)
