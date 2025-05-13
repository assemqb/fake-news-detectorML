# Real-Time Fake News Detector

## Overview
This project provides a web-based fake news detection service. It trains an XGBoost-based classifier on a dataset of real and fake news, serializes the model pipeline, and serves predictions via a FastAPI web server with a simple front-end.

## Features
- **Text Cleaning**: Custom transformer for text preprocessing (lowercasing, URL/HTML removal, non-letter filtering).
- **TF-IDF Vectorization**: Uses bigrams and unigrams for feature extraction.
- **XGBoost Classifier**: High-performance gradient boosting model.
- **FastAPI Backend**: Exposes a `/predict` endpoint for real-time predictions.
- **Modern UI**: Responsive front-end with classification results.

## Folder Structure
```
fake-news-detector/
├── data/                     # Raw datasets (Fake.csv, True.csv)
├── models/
│   ├── fake_news_pipeline.joblib  # Trained model pipeline
│   ├── model_utils.py             # Prediction utility
│   └── text_cleaner.py            # Text cleaning transformer
├── scripts/
│   └── train_model.py         # Model training script
├── static/
│   └── index.html             # Front-end HTML/CSS/JS
├── app.py                     # FastAPI application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Setup & Installation

1. **Clone the repository**  
   ```bash
   git clone <your-repo-url>
   cd fake-news-detector
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Training the Model

1. **Place your data** in `data/`:
   - `Fake.csv` (fake news)
   - `True.csv` (real news)

2. **Run the training script**  
   ```bash
   python -m scripts.train_model
   ```
   The script will:
   - Load and preprocess data
   - Train a TF-IDF + XGBoost pipeline
   - Evaluate on validation and test splits
   - Save the pipeline to `models/fake_news_pipeline.joblib`

## Running the Server

Start the FastAPI server with Uvicorn:
```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Then open your browser at: `http://localhost:8000`

## Usage

- Paste news article text into the input box.
- Click **Check News**.
- The detector will display **REAL**, **FAKE**, or **UNKNOWN** with confidence scores.

### cURL Example
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":"Your news text here"}'
```

## Contributing

Feel free to submit issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.
