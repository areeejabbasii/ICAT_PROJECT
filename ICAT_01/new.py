
import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import os

# Configure Tesseract path - try common installation locations
def find_tesseract_path():
    """Find Tesseract installation path"""
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
        r'C:\Users\GEO\Downloads\tesseract.exe',  # Your Downloads folder
        r'C:\Users\GEO\Downloads\Tesseract-OCR\tesseract.exe',  # If extracted in Downloads
        r'tesseract.exe'  # If in PATH
    ]
    
    for path in common_paths:
        if os.path.exists(path) or path == 'tesseract.exe':
            return path
    return None

# Try to find Tesseract automatically
tesseract_path = find_tesseract_path()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # Default fallback
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_available_languages():
    """Get list of available Tesseract languages on the system"""
    try:
        available_langs = pytesseract.get_languages(config='')
        return available_langs
    except Exception as e:
        return None  # Return None to indicate Tesseract is not working

def check_tesseract_installation():
    """Check if Tesseract is properly installed"""
    try:
        # Try to get version to test if Tesseract works
        version = pytesseract.get_tesseract_version()
        langs = pytesseract.get_languages(config='')
        return True, version, langs
    except Exception as e:
        return False, str(e), []

# Get available languages from Tesseract
tesseract_working, version_or_error, available_tesseract_langs = check_tesseract_installation()

# Language options with their Tesseract codes (only show available ones)
ALL_LANGUAGES = {
    'English': 'eng',
    'Spanish': 'spa',
    'French': 'fra',
    'German': 'deu',
    'Italian': 'ita',
    'Portuguese': 'por',
    'Russian': 'rus',
    'Chinese (Simplified)': 'chi_sim',
    'Chinese (Traditional)': 'chi_tra',
    'Japanese': 'jpn',
    'Korean': 'kor',
    'Arabic': 'ara',
    'Hindi': 'hin'
}

# Filter to only show available languages if Tesseract is working
if tesseract_working:
    LANGUAGES = {name: code for name, code in ALL_LANGUAGES.items() 
                 if code in available_tesseract_langs}
else:
    LANGUAGES = {'English': 'eng'}  # Fallback

def preprocess_image(image):
    """Apply various preprocessing techniques to improve OCR accuracy"""
    # Convert PIL image to numpy array for OpenCV
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Apply different preprocessing techniques
    processed_images = []
    
    # 1. Original grayscale
    processed_images.append(("Original", gray))
    
    # 2. Gaussian blur + OTSU threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(("OTSU Threshold", thresh1))
    
    # 3. Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_images.append(("Adaptive Threshold", adaptive_thresh))
    
    # 4. Morphological operations to clean up
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    processed_images.append(("Morphological", morph))
    
    # 5. Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, enhanced_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(("Enhanced Contrast", enhanced_thresh))
    
    # 6. Noise removal
    denoised = cv2.medianBlur(gray, 3)
    _, denoised_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(("Denoised", denoised_thresh))
    
    return processed_images

def extract_text_from_image(image, languages):
    """Extract text from image using multiple preprocessing techniques and OCR configurations"""
    try:
        # Validate languages are available
        available_langs = get_available_languages()
        valid_languages = [lang for lang in languages if lang in available_langs]
        
        if not valid_languages:
            valid_languages = ['eng']  # Fallback to English
        
        # Join selected languages with '+'
        lang_string = '+'.join(valid_languages)
        
        # Get preprocessed images
        processed_images = preprocess_image(image)
        
        # Try different OCR configurations
        ocr_configs = [
            '--psm 3',  # Fully automatic page segmentation (default)
            '--psm 6',  # Uniform block of text
            '--psm 4',  # Single column of text
            '--psm 1',  # Automatic page segmentation with OSD
            '--psm 11', # Sparse text
            '--psm 12', # Sparse text with OSD
        ]
        
        all_results = []
        
        # Try each preprocessing method with different PSM modes
        for method_name, processed_img in processed_images:
            for config in ocr_configs:
                try:
                    # Convert numpy array back to PIL Image for pytesseract
                    pil_img = Image.fromarray(processed_img)
                    
                    # Extract text with current configuration
                    text = pytesseract.image_to_string(pil_img, lang=lang_string, config=config)
                    
                    if text.strip() and len(text.strip()) > 5:  # Only add meaningful results
                        all_results.append({
                            'method': method_name,
                            'config': config,
                            'text': text.strip(),
                            'length': len(text.strip()),
                            'word_count': len(text.strip().split())
                        })
                except:
                    continue
        
        if not all_results:
            return "No text could be extracted from the image."
        
        # Sort results by word count and length (more comprehensive text is better)
        all_results.sort(key=lambda x: (x['word_count'], x['length']), reverse=True)
        
        # Return the best result (most comprehensive)
        best_result = all_results[0]['text']
        
        # Optional: Show which method worked best in debug
        # st.info(f"Best result from: {all_results[0]['method']} with {all_results[0]['config']}")
        
        return best_result
    
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def main():
    st.set_page_config(
        page_title="EZ PZ Image to Text Extractor",
        page_icon="📄",
        layout="wide"
    )
    
    # Header
    st.title("📄 EZ PZ Image to Text Extractor")
    st.markdown("---")
    
    # Check Tesseract installation status
    if not tesseract_working:
        st.error("🚨 Tesseract OCR is not properly installed or configured!")
        st.markdown(f"**Error:** {version_or_error}")
        
        # Manual path configuration
        st.markdown("### 🔧 Manual Path Configuration")
        manual_path = st.text_input(
            "Enter the full path to tesseract.exe:",
            value=r"C:\Users\GEO\Downloads\tesseract.exe",
            help="Example: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        )
        
        if st.button("Test Path"):
            if os.path.exists(manual_path):
                pytesseract.pytesseract.tesseract_cmd = manual_path
                try:
                    version = pytesseract.get_tesseract_version()
                    st.success(f"✅ Tesseract found! Version: {version}")
                    st.info("Please refresh the page to use the application.")
                except Exception as e:
                    st.error(f"❌ Path exists but Tesseract not working: {str(e)}")
            else:
                st.error("❌ File not found at this path")
        
        st.markdown("""
        ### 📥 Installation Steps:
        
        **Option 1: Install Tesseract (Recommended)**
        1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
        2. Run installer as Administrator
        3. Select "Additional language data (download)"
        4. Install to default location: `C:\\Program Files\\Tesseract-OCR\\`
        
        **Option 2: If you have tesseract.exe in Downloads**
        1. Look for `tesseract.exe` in your Downloads folder
        2. Enter the full path above (like `C:\\Users\\GEO\\Downloads\\tesseract.exe`)
        3. Click "Test Path"
        
        ### 🔍 Current Search Paths:
        The app is looking for tesseract.exe in these locations:
        - `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
        - `C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe`
        - `C:\\Users\\GEO\\Downloads\\tesseract.exe`
        - `C:\\Users\\GEO\\Downloads\\Tesseract-OCR\\tesseract.exe`
        """)
        
        return
    
    st.success(f"✅ Tesseract OCR is working! Version: {version_or_error}")
    st.markdown("Upload an image and extract text in multiple languages using OCR technology!")
    
    # Sidebar for language selection
    st.sidebar.header("🌍 Language Settings")
    st.sidebar.markdown("Select the languages present in your image:")
    
    # Show available languages info
    if len(LANGUAGES) < len(ALL_LANGUAGES):
        missing_langs = [name for name, code in ALL_LANGUAGES.items() 
                        if code not in available_tesseract_langs]
        st.sidebar.info(f"📋 Available languages: {len(LANGUAGES)}")
        if missing_langs:
            with st.sidebar.expander("ℹ️ Missing Languages"):
                st.write("To add more languages, install additional Tesseract language packs:")
                for lang in missing_langs:
                    st.write(f"• {lang}")
    
    selected_languages = []
    for lang_name, lang_code in LANGUAGES.items():
        if st.sidebar.checkbox(lang_name, value=(lang_name == 'English')):
            selected_languages.append(lang_code)
    
    if not selected_languages:
        st.sidebar.warning("Please select at least one language!")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'],
            help="Supported formats: PNG, JPG, JPEG, TIFF, BMP, GIF"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"📊 Image Info: {image.size[0]}x{image.size[1]} pixels, Format: {image.format}")
    
    with col2:
        st.header("📝 Extracted Text")
        
        if uploaded_file is not None:
            with st.spinner("🔍 Extracting text from image..."):
                # Extract text
                extracted_text = extract_text_from_image(image, selected_languages)
                
                if extracted_text and extracted_text.strip():
                    # Display extracted text
                    st.success("✅ Text extraction completed!")
                    
                    # Text area with extracted content
                    st.text_area(
                        "Extracted Text:",
                        value=extracted_text,
                        height=300,
                        help="You can copy this text"
                    )
                    
                    # Download button for extracted text
                    st.download_button(
                        label="💾 Download Text as TXT",
                        data=extracted_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                    
                    # Statistics
                    word_count = len(extracted_text.split())
                    char_count = len(extracted_text)
                    st.metric("Word Count", word_count)
                    st.metric("Character Count", char_count)
                    
                else:
                    st.warning("⚠️ No text found in the image. Try:")
                    st.markdown("""
                    - Selecting different languages
                    - Using a higher quality image
                    - Ensuring the text is clearly visible
                    """)
        else:
            st.info("👆 Please upload an image to extract text")
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("### 💡 Tips for Better OCR Results:")
    st.markdown("""
    - **Image Quality**: Use high-resolution images (300+ DPI) with clear, sharp text
    - **Contrast**: Ensure good contrast between text and background (black text on white background works best)
    - **Language Selection**: Select the correct languages present in your image
    - **Text Orientation**: Horizontal text works best - rotate skewed images before uploading
    - **Image Format**: PNG and TIFF formats often work better than JPG
    - **Text Size**: Very small text (< 12pt) may not be recognized well
    - **Fonts**: Simple, clean fonts work better than decorative or handwritten text
    - **Lighting**: Avoid shadows, glare, or uneven lighting in photos
    """)
    
    # Advanced tips
    with st.expander("🔧 Advanced OCR Tips"):
        st.markdown("""
        **The app automatically tries multiple preprocessing techniques:**
        - Original image processing
        - OTSU thresholding for better contrast
        - Adaptive thresholding for varying lighting
        - Morphological operations to clean up noise
        - Contrast enhancement (CLAHE)
        - Noise removal with median filtering
        
        **Multiple OCR modes are tested:**
        - PSM 3: Full automatic page segmentation (default)
        - PSM 6: Uniform block of text
        - PSM 4: Single column of text
        - PSM 1: Automatic with orientation detection
        - PSM 11 & 12: Sparse text detection
        
        **If OCR is still not working well:**
        - Try cropping to focus on specific text areas
        - Increase image resolution before uploading
        - Ensure text is not rotated or skewed
        - Check if the language pack is installed
        """)
    
    # Technical information
    with st.expander("ℹ️ Technical Information"):
        st.markdown(f"""
        **OCR Engine:** Tesseract {version_or_error if tesseract_working else 'Not Available'}
        
        **Available Languages:** {len(LANGUAGES)} out of {len(ALL_LANGUAGES)} supported
        
        **Image Processing:** OpenCV with multiple preprocessing techniques
        
        **Supported Formats:** PNG, JPG, JPEG, TIFF, BMP, GIF
        
        **Processing Pipeline:**
        1. Image format conversion and color space adjustment
        2. Multiple preprocessing techniques applied
        3. Various OCR page segmentation modes tested
        4. Best result selected based on text length and word count
        """)
        
        if tesseract_working:
            available_lang_names = [name for name, code in ALL_LANGUAGES.items() 
                                   if code in available_tesseract_langs]
            st.write("**Available Languages:**")
            st.write(", ".join(available_lang_names))
            
            missing_lang_names = [name for name, code in ALL_LANGUAGES.items() 
                                 if code not in available_tesseract_langs]
            if missing_lang_names:
                st.write("**Missing Languages:**")
                st.write(", ".join(missing_lang_names))
    
    # Technical info
    with st.expander("🔧 Technical Information"):
        st.markdown(f"""
        **Selected Languages:** {', '.join([k for k, v in LANGUAGES.items() if v in selected_languages])}
        
        **OCR Engine:** Tesseract OCR
        
        **Available Languages:** {len(LANGUAGES)} out of {len(ALL_LANGUAGES)} supported languages
        
        **Image Processing:** The application applies Gaussian blur and OTSU thresholding to improve OCR accuracy.
        """)
        
        # Show available vs missing languages
        available_lang_names = [name for name, code in ALL_LANGUAGES.items() 
                               if code in available_tesseract_langs]
        missing_lang_names = [name for name, code in ALL_LANGUAGES.items() 
                             if code not in available_tesseract_langs]
        
        st.write("**Available Languages:**")
        st.write(", ".join(available_lang_names))
        
        if missing_lang_names:
            st.write("**Missing Languages:**")
            st.write(", ".join(missing_lang_names))
            st.markdown("""
            **To install missing languages on Windows:**
            1. Download additional language packs from: https://github.com/tesseract-ocr/tessdata
            2. Copy .traineddata files to: `C:\\Program Files\\Tesseract-OCR\\tessdata\\`
            3. Restart the application
            """)
    
    # Troubleshooting section
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        
        1. **Language not found error:**
           - Install the required language pack
           - Check if Tesseract is properly installed
           - Verify TESSDATA_PREFIX environment variable
        
        2. **No text extracted:**
           - Try different languages
           - Use higher quality images
           - Ensure good contrast
        
        3. **Tesseract not found:**
           - Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
           - Update the tesseract_cmd path in the code if needed
        """)

if __name__ == "__main__":
    main()
