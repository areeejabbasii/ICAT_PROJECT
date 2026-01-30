# EZ PZ Image to Text Extractor

A simple and powerful image-to-text extraction tool built with Streamlit that supports 7+ languages.

## Features

- 📤 Easy image upload from desktop
- 🌍 Multi-language OCR support (13 languages)
- 📝 Real-time text extraction
- 💾 Download extracted text as TXT file
- 📊 Text statistics (word count, character count)
- 🔧 Image preprocessing for better accuracy

## Supported Languages

- English
- Spanish
- French
- German
- Italian
- Portuguese
- Russian
- Chinese (Simplified & Traditional)
- Japanese
- Korean
- Arabic
- Hindi

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
   - **Windows**: 
     - Download from https://github.com/UB-Mannheim/tesseract/wiki
     - During installation, make sure to select additional language packs
     - Default installation path: `C:\Program Files\Tesseract-OCR\`
   - **Mac**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

3. For additional languages after installation:
   - **Windows**: Download .traineddata files from https://github.com/tesseract-ocr/tessdata and copy to `C:\Program Files\Tesseract-OCR\tessdata\`
   - **Mac/Linux**: Install language packs:
   ```bash
   # Example for additional languages
   sudo apt-get install tesseract-ocr-spa tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-hin
   ```

4. Set environment variable (if needed):
   - **Windows**: Add `TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata` to system environment variables
   - **Mac/Linux**: Add `export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata` to your shell profile

## Usage

1. Run the Streamlit app:
```bash
streamlit run new.py
```

2. Open your browser and go to the displayed URL (usually http://localhost:8501)

3. Select the languages present in your image

4. Upload an image file (PNG, JPG, JPEG, TIFF, BMP, GIF)

5. View and download the extracted text

## Tips for Better Results

- Use high-resolution images with clear text
- Ensure good contrast between text and background
- Select the correct languages for your image
- Avoid blurry or skewed images
- For best results, use images with horizontal text

## Requirements

- Python 3.7+
- Tesseract OCR installed on your system
- All packages listed in requirements.txt