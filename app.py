import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import hashlib
import time
import json
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Try to import OpenCV and additional libraries
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("OpenCV not found. Some image processing features will be limited. Install with `pip install opencv-python-headless`")


try:
    from skimage import filters, measure, morphology, segmentation
    from skimage.feature import local_binary_pattern
    from scipy import ndimage
    ADVANCED_PROCESSING = True
except ImportError:
    ADVANCED_PROCESSING = False
    st.warning("Scikit-image and Scipy not found. Advanced image processing and feature extraction will be limited. Install with `pip install scikit-image scipy`")

# Page configuration
st.set_page_config(
    page_title="Pro-Level Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Pro-Level CSS Styling ---
def inject_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Montserrat:wght=400;600;800&display=swap" rel="stylesheet">
    <style>
        /* General Body and Main Container Styling */
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #e0f2f7; /* Light blue background */
            color: #333333;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-right: 3rem;
            padding-left: 3rem;
            padding-bottom: 2rem;
        }

        /* Headers */
        .main-header {
            font-family: 'Montserrat', sans-serif;
            font-size: 3.2rem; /* Slightly larger */
            font-weight: 800; /* Extra bold */
            background: linear-gradient(90deg, #1f77b4, #ff7f0e); /* Existing vibrant gradient */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem; /* More space */
            text-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Subtle shadow */
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Montserrat', sans-serif;
            color: #2c3e50; /* Darker blue for headers */
        }
        h2 { font-size: 2rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 1rem; }
        h3 { font-size: 1.7rem; font-weight: 600; margin-top: 1.2rem; margin-bottom: 0.8rem; }

        /* Custom Cards for Metrics and Features */
        .metric-card, .feature-box, .alert-critical, .alert-warning, .alert-success {
            background: #ffffff; /* White background for cards */
            border-radius: 12px; /* Softer corners */
            padding: 1.5rem; /* More padding */
            margin: 1rem 0;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08); /* More pronounced shadow */
            transition: transform 0.3s ease-in-out; /* Smooth hover effect */
        }
        .metric-card:hover, .feature-box:hover {
            transform: translateY(-5px); /* Lift effect on hover */
        }
        .metric-card h4, .feature-box h4 {
            color: #1f77b4; /* Blue accent for card titles */
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        .metric-card p, .feature-box p {
            color: #555555;
            line-height: 1.6;
        }

        /* Alert Styling */
        .alert-critical {
            background: linear-gradient(135deg, #e74c3c, #c0392b); /* Reddish gradients */
            color: white;
            box-shadow: 0 5px 15px rgba(231, 76, 60, 0.3);
        }
        .alert-warning {
            background: linear-gradient(135deg, #f39c12, #e67e22); /* Orangeish gradients */
            color: white;
            box-shadow: 0 5px 15px rgba(243, 156, 18, 0.3);
        }
        .alert-success {
            background: linear-gradient(135deg, #27ae60, #2ecc71); /* Greenish gradients */
            color: white;
            box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
        }
        .alert-critical h3, .alert-warning h3, .alert-success h3 {
            color: white;
            margin-top: 0;
            margin-bottom: 0.5rem;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px; /* More space between tabs */
            justify-content: center; /* Center the tabs */
            margin-bottom: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px; /* Taller tabs */
            width: 200px; /* Wider tabs */
            background: #ecf0f1; /* Light grey for inactive tabs */
            border-radius: 30px; /* Pill shape */
            color: #34495e; /* Darker text */
            font-weight: bold;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease-in-out;
            border: 2px solid transparent; /* Border for active state */
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: #bdc3c7; /* Darker grey on hover */
            color: #2c3e50;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(90deg, #1f77b4, #ff7f0e); /* Active tab color */
            color: white;
            border-color: #2980b9; /* Border accent for active */
            box-shadow: 0 6px 15px rgba(31, 119, 180, 0.4);
            transform: translateY(-2px); /* Slight lift for active tab */
        }

        /* Buttons */
        .stButton button {
            background: linear-gradient(45deg, #3498db, #2980b9); /* Blue gradient */
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .stButton button:hover {
            background: linear-gradient(45deg, #2980b9, #3498db); /* Darker blue on hover */
            box-shadow: 0 6px 15px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }
        .stButton button:active {
            transform: translateY(0);
        }
        /* Primary button specific styling */
        .stButton button[data-testid="baseButton-secondary"] { /* Targets default Streamlit primary */
             background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        }
        .stButton button[data-testid="baseButton-secondary"]:hover {
             background: linear-gradient(45deg, #ff7f0e, #1f77b4);
        }

        /* Sidebar Styling */
        .stSidebar {
            background-color: #2c3e50; /* Dark blue sidebar */
            color: white;
            padding-top: 2rem;
            font-family: 'Roboto', sans-serif;
        }
        .stSidebar .stSelectbox label, .stSidebar .stCheckbox label, .stSidebar .stFileUploader label {
            color: #ecf0f1; /* Light text for labels */
        }
        .stSidebar h2, .stSidebar h3 {
            color: #ecf0f1;
            margin-bottom: 1rem;
        }
        .stSidebar .stExpander {
            background-color: #34495e; /* Slightly lighter blue for expander */
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        .stSidebar .stExpander div[data-testid="stExpanderTitleIcon"] {
            color: #ecf0f1;
        }
        .stSidebar .stExpander div[role="button"] {
            color: #ecf0f1;
            padding: 0.8rem 1rem;
        }
        .stSidebar .stExpander div[data-testid="stExpanderContent"] {
            padding: 1rem;
            background-color: #2c3e50; /* Maintain sidebar background inside content */
        }

        /* Scrollbar Styling (Webkit - Chrome, Safari) */
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }

        /* Slider Styling */
        .stSlider .st-cq { /* Track */
            background-color: #bbdefb; /* Light blue track */
        }
        .stSlider .st-by { /* Thumb */
            background-color: #2196f3; /* Darker blue thumb */
        }

        /* Input Fields */
        .stTextInput div[data-baseweb="input"] input, .stFileUploader div[data-baseweb="file-uploader-file-container"] {
            border-radius: 8px;
            border: 1px solid #cccccc;
            padding: 0.5rem;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.06);
        }
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 8px;
            border: 1px solid #cccccc;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.06);
        }

        /* Markdown elements */
        a {
            color: #1f77b4;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        code {
            background-color: #e0e0e0;
            border-radius: 4px;
            padding: 0.2em 0.4em;
            font-family: 'Cascadia Code', 'Fira Code', monospace;
        }

        /* Additional specific adjustments for layout */
        .streamlit-expanderHeader { /* Targeting expander header for better spacing */
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Call the CSS injection function
inject_css()

# Initialize session state with enhanced features
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'analysis_history' not in st.session_state: # This might be redundant with processed_images, but kept for clarity
    st.session_state.analysis_history = []
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}
if 'user_annotations' not in st.session_state:
    st.session_state.user_annotations = {}

class AdvancedBrainTumorDetector:
    def __init__(self):
        # These would ideally be loaded pre-trained deep learning models
        self.model_names = ["ResNet-50", "DenseNet-121", "EfficientNet-B0", "Vision Transformer", "ConvNeXt"]
        self.tumor_types = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
        self.tumor_characteristics = {
            "Glioma": {
                "description": "Most common brain tumor, arises from glial cells. Often irregularly shaped and can be infiltrative.",
                "typical_location": "Cerebral hemispheres, brainstem, cerebellum.",
                "prognosis": "Variable, depends on grade (low to high). Aggressive forms have poor prognosis.",
                "treatment": "Surgery (maximal safe resection), radiation therapy, chemotherapy (e.g., Temozolomide)."
            },
            "Meningioma": {
                "description": "Usually benign, arises from the meninges (membranes surrounding the brain and spinal cord). Characterized by a well-defined border.",
                "typical_location": "Surface of the brain, falx cerebri, tentorium, sphenoid wing.",
                "prognosis": "Generally good if completely removed. Recurrence possible if incomplete resection or atypical/malignant features.",
                "treatment": "Surgery (primary), radiation if incomplete resection or atypical/malignant features."
            },
            "Pituitary": {
                "description": "Tumor of the pituitary gland, a small gland at the base of the brain controlling hormones. Can lead to hormonal imbalances or vision issues.",
                "typical_location": "Sella turcica (the bony pocket at the base of the skull).",
                "prognosis": "Often good with treatment, but requires careful management of hormonal effects.",
                "treatment": "Surgery (transsphenoidal approach), medication (for hormone-secreting tumors), radiation."
            },
            "No Tumor": {
                "description": "No abnormal tissue or lesions indicative of a tumor were detected in the scan. Represents normal brain tissue findings.",
                "typical_location": "N/A (Normal brain anatomy)",
                "prognosis": "Normal brain tissue, healthy scan findings.",
                "treatment": "None required."
            }
        }
        
    def advanced_preprocessing(self, image, target_size=(224, 224)):
        """Advanced image preprocessing with multiple enhancement techniques.
        This simulates steps that would precede deep learning inference."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        results = {}
        
        # Basic preprocessing
        img_resized = cv2.resize(img_array, target_size) if OPENCV_AVAILABLE else np.array(image.resize(target_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        results['resized'] = img_resized
        results['normalized'] = img_normalized
        
        if OPENCV_AVAILABLE:
            # Noise reduction
            img_denoised = cv2.bilateralFilter(img_resized, 9, 75, 75)
            results['denoised'] = img_denoised
            
            # Contrast enhancement (CLAHE for adaptive contrast)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_gray)
            results['enhanced'] = img_enhanced
            
            # Edge detection
            edges = cv2.Canny(img_gray, 50, 150)
            results['edges'] = edges
            
            # Morphological operations (Opening and Closing to remove small objects and fill small holes)
            kernel = np.ones((3, 3), np.uint8)
            img_opened = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
            img_closed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel)
            results['morphological'] = img_closed
        
        if ADVANCED_PROCESSING:
            # Advanced filtering (Gaussian and Median for smoothing)
            img_gaussian = filters.gaussian(img_normalized, sigma=1.0)
            img_median = filters.median(img_normalized)
            results['gaussian'] = img_gaussian
            results['median'] = img_median
            
            # Texture analysis (Local Binary Pattern for texture description)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY) if OPENCV_AVAILABLE else np.mean(img_resized, axis=2).astype(np.uint8)
            lbp = local_binary_pattern(gray, 24, 3, method='uniform')
            results['texture'] = lbp
        
        return results
    
    def extract_features(self, image):
        """Extract comprehensive features from the image.
        These features would ideally be fed into a classical ML model or used as input for a deep learning model."""
        features = {}
        
        # Ensure image is in a suitable format for feature extraction
        if len(image.shape) == 3 and image.shape[2] == 3: # If RGB, convert to grayscale for some ops
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if OPENCV_AVAILABLE else np.mean(image, axis=2).astype(np.uint8)
        else:
            gray_image = image.astype(np.uint8) # Assume it's already grayscale or single channel
            
        # Basic statistics
        features['mean_intensity'] = np.mean(gray_image)
        features['std_intensity'] = np.std(gray_image)
        features['min_intensity'] = np.min(gray_image)
        features['max_intensity'] = np.max(gray_image)
        
        # Histogram features
        hist, _ = np.histogram(gray_image.flatten(), bins=256, range=(0, 256))
        # Normalize histogram for entropy calculation
        hist_norm = hist / (hist.sum() + 1e-10) # Add small epsilon to prevent division by zero
        features['histogram_entropy'] = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        features['histogram_variance'] = np.var(hist_norm)
        
        if ADVANCED_PROCESSING:
            # Texture features using LBP
            lbp = local_binary_pattern(gray_image, 24, 3, method='uniform')
            # Histogram of LBP values as texture feature
            lbp_hist, _ = np.histogram(lbp.flatten(), bins=np.arange(0, 27), range=(0, 26))
            features['lbp_texture_uniformity'] = len(np.unique(lbp))
            
            # Regional properties (segmentation based on Otsu's thresholding)
            try:
                # Use a smoothed image for thresholding to reduce noise in segmentation
                img_smooth_for_seg = filters.gaussian(gray_image, sigma=2)
                binary = img_smooth_for_seg > filters.threshold_otsu(img_smooth_for_seg)
                labeled = measure.label(binary)
                regions = measure.regionprops(labeled)
                
                if regions:
                    # Select the largest region that is not the entire image border
                    img_area = gray_image.shape[0] * gray_image.shape[1]
                    filtered_regions = [r for r in regions if r.area < img_area * 0.9] # Exclude too large regions (background)
                    
                    if filtered_regions:
                        largest_region = max(filtered_regions, key=lambda r: r.area)
                        features['largest_region_area'] = largest_region.area
                        features['largest_region_perimeter'] = largest_region.perimeter
                        features['largest_region_eccentricity'] = largest_region.eccentricity # How elongated the region is
                        features['largest_region_solidity'] = largest_region.solidity       # Area / Convex Hull Area
                        features['largest_region_extent'] = largest_region.extent           # Area / Bounding Box Area
                    else: # If only background-like large regions or no regions after filtering
                        features['largest_region_area'] = 0
                        features['largest_region_perimeter'] = 0
                        features['largest_region_eccentricity'] = 0
                        features['largest_region_solidity'] = 0
                        features['largest_region_extent'] = 0
                else: # No regions found at all
                    features['largest_region_area'] = 0
                    features['largest_region_perimeter'] = 0
                    features['largest_region_eccentricity'] = 0
                    features['largest_region_solidity'] = 0
                    features['largest_region_extent'] = 0
            except ValueError: # Handles cases where thresholding might fail (e.g., uniform image)
                st.warning("Segmentation failed for region properties. Image might be too uniform.")
                features['largest_region_area'] = 0
                features['largest_region_perimeter'] = 0
                features['largest_region_eccentricity'] = 0
                features['largest_region_solidity'] = 0
                features['largest_region_extent'] = 0
        
        return features
    
    def simulate_advanced_prediction(self, image, features):
        """Advanced prediction simulation with feature-based logic.
        This simulates the output of a deep learning model for multiple architectures."""
        predictions = {}
        
        # Base prediction logic on extracted features
        # Create a deterministic seed based on image content for consistent simulation
        img_hash = hashlib.md5(image.tobytes()).hexdigest()
        base_seed = int(img_hash[:8], 16) % 10000 # Use a larger range for seed
        
        for i, model_name in enumerate(self.model_names):
            np.random.seed(base_seed + i) # Vary seed slightly for each model
            
            # Feature-influenced prediction weights
            feature_influence = np.zeros(len(self.tumor_types))
            
            # Get tumor type indices
            glioma_idx = self.tumor_types.index("Glioma")
            meningioma_idx = self.tumor_types.index("Meningioma")
            pituitary_idx = self.tumor_types.index("Pituitary")
            no_tumor_idx = self.tumor_types.index("No Tumor")

            # Apply influence based on extracted features
            # High intensity variation, high entropy, irregular shape -> Glioma
            if features.get('std_intensity', 0) > 60:
                feature_influence[glioma_idx] += 0.1
            if features.get('histogram_entropy', 0) > 7.5:
                feature_influence[glioma_idx] += 0.08
            if features.get('largest_region_eccentricity', 1) > 0.8 and features.get('largest_region_solidity', 0) < 0.9: # Elongated/irregular
                feature_influence[glioma_idx] += 0.15

            # Large, well-defined (high solidity/extent) region -> Meningioma
            if features.get('largest_region_area', 0) > 2500 and features.get('largest_region_solidity', 0) > 0.9:
                feature_influence[meningioma_idx] += 0.2
            if features.get('largest_region_extent', 0) > 0.7: # Compact, fills bounding box
                feature_influence[meningioma_idx] += 0.1

            # Moderate area, low eccentricity, high mean intensity (dense mass) -> Pituitary
            if 600 < features.get('largest_region_area', 0) < 1800 and features.get('largest_region_eccentricity', 1) < 0.6:
                feature_influence[pituitary_idx] += 0.18
            if features.get('mean_intensity', 0) > 150: # Brighter regions
                feature_influence[pituitary_idx] += 0.07

            # If features indicate minimal abnormalities -> No Tumor
            if features.get('largest_region_area', 0) < 100 and \
               features.get('std_intensity', 0) < 30 and \
               features.get('histogram_entropy', 0) < 6.0:
                feature_influence[no_tumor_idx] += 0.6 # Strong boost for "No Tumor"

            # Simulate raw scores, then apply feature influence
            raw_scores = np.random.rand(len(self.tumor_types)) * 0.5 # Initial random scores, dampen for influence to matter
            raw_scores = raw_scores + feature_influence
            
            # Ensure "No Tumor" doesn't become a primary prediction if strong tumor features are present
            if np.max(feature_influence[:-1]) > 0.2 and feature_influence[no_tumor_idx] > 0.5:
                # If there's a strong tumor signal, reduce "No Tumor" boost
                feature_influence[no_tumor_idx] -= np.max(feature_influence[:-1]) * 0.5
                raw_scores = np.random.rand(len(self.tumor_types)) * 0.5 + feature_influence # Re-calculate with adjusted influence
            
            probabilities = np.exp(raw_scores) / np.sum(np.exp(raw_scores)) # Softmax-like normalization
            
            # Add model-specific noise to make them slightly different
            model_noise = np.random.normal(0, 0.03, len(probabilities)) # Small noise
            probabilities = probabilities + model_noise
            probabilities = np.clip(probabilities, 0.001, 0.999) # Clip to avoid 0 or 1
            probabilities = probabilities / np.sum(probabilities) # Re-normalize
            
            predictions[model_name] = {
                tumor_type: prob for tumor_type, prob in zip(self.tumor_types, probabilities)
            }
        
        return predictions
    
    def ensemble_prediction(self, predictions):
        """Combines predictions from multiple models using a simple averaging ensemble."""
        ensemble_scores = {tumor_type: 0 for tumor_type in self.tumor_types}
        
        for model_name in self.model_names:
            for tumor_type, prob in predictions[model_name].items():
                ensemble_scores[tumor_type] += prob
        
        # Average the scores
        num_models = len(self.model_names)
        ensemble_scores = {k: v / num_models for k, v in ensemble_scores.items()}
        
        # Re-normalize to ensure they sum to 1, accounting for floating point inaccuracies
        sum_scores = sum(ensemble_scores.values())
        if sum_scores > 0:
            ensemble_scores = {k: v / sum_scores for k, v in ensemble_scores.items()}
        else: # Handle case where all scores are zero
            ensemble_scores = {k: 1/num_models for k in ensemble_scores} # Assign equal probability
            
        return ensemble_scores

    def uncertainty_quantification(self, predictions):
        """Calculate prediction uncertainty metrics across models."""
        all_predictions = []
        for model_preds in predictions.values():
            all_predictions.append(list(model_preds.values()))
        
        predictions_array = np.array(all_predictions) # Shape: (num_models, num_tumor_types)
        
        # Mean probabilities for entropy calculation
        mean_probs = np.mean(predictions_array, axis=0)
        
        # Entropy (measures predictive uncertainty for the ensemble mean)
        # Higher entropy means more uncertainty/less clear prediction
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        
        # Variance (measures disagreement among models for each class)
        # Higher variance means more disagreement
        variance_per_class = np.var(predictions_array, axis=0)
        mean_variance = np.mean(variance_per_class) # Average variance across all classes
        
        # Confidence interval for each class (2.5th and 97.5th percentiles)
        # This gives a range for the predicted probability of each class
        confidence_interval = np.percentile(predictions_array, [2.5, 97.5], axis=0)
        
        return {
            'entropy': entropy,
            'variance_per_class': variance_per_class, # Keep per-class variance
            'mean_variance': mean_variance,
            'confidence_interval': confidence_interval,
            'model_disagreement_range': np.max(predictions_array, axis=0) - np.min(predictions_array, axis=0)
        }
    
    def generate_detailed_attention_map(self, image, ensemble_prediction_scores):
        """Generate more sophisticated attention maps.
        This is a *simulated* attention map as actual CAM generation requires a DL model."""
        height, width = image.shape[:2]
        
        attention_maps = {}
        
        # Main attention based on the highest predicted class
        predicted_class = max(ensemble_prediction_scores, key=ensemble_prediction_scores.get)
        confidence = ensemble_prediction_scores[predicted_class]
        
        # Initialize a base attention map (e.g., random noise or uniform)
        attention_map = np.zeros((height, width))
        
        # Simulate different attention patterns for different tumor types
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        
        if predicted_class == "Glioma":
            # Gliomas often have irregular shapes and diffuse margins.
            # Simulate a more spread-out, less defined attention with multiple foci.
            num_foci = np.random.randint(2, 5)
            for _ in range(num_foci):
                focus_x = np.random.randint(width * 0.2, width * 0.8)
                focus_y = np.random.randint(height * 0.2, height * 0.8)
                sigma_x = width / (4 + np.random.rand() * 4) # Vary sigma for irregularity
                sigma_y = height / (4 + np.random.rand() * 4)
                attention_map += np.exp(-(((x - focus_x)**2 / (2 * sigma_x**2)) + ((y - focus_y)**2 / (2 * sigma_y**2)))) * (confidence * 0.5 + 0.1)
            attention_map *= (1 + np.random.rand() * 0.2) # Add some overall intensity variation

        elif predicted_class == "Meningioma":
            # Meningiomas are often well-defined, pushing on brain tissue.
            # Simulate a more localized, distinct attention blob.
            radius = min(width, height) / (3 + np.random.rand())
            main_region = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * radius**2))
            attention_map = main_region * (confidence * 1.5 + 0.2)
            # Add a subtle, brighter border to simulate encapsulation
            border_radius = radius * 0.9
            border_map = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * border_radius**2))
            attention_map += (main_region - border_map) * (confidence * 0.5 + 0.1) # Simulate rim enhancement

        elif predicted_class == "Pituitary":
            # Pituitary tumors are centrally located at the base of the brain.
            # Simulate a very central, compact attention.
            radius = min(width, height) / (5 + np.random.rand() * 2)
            main_region = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * radius**2))
            attention_map = main_region * (confidence * 2.0 + 0.3) # Very strong central activation
            # Add slight vertical emphasis due to location at skull base
            attention_map *= (1 + (y / height - 0.5) * 0.5)

        else: # "No Tumor"
            # If no tumor, attention should be more diffuse or focus on general brain structures.
            # Simulate general low-level activity across the image with slight emphasis on anatomical landmarks.
            attention_map = np.random.random((height, width)) * 0.1 # Base background
            # Simulate focus on ventricles/midline
            midline_attention = np.exp(-(np.abs(x - center_x)**2) / (2 * (width / 10)**2)) * 0.2
            attention_map += midline_attention
            # Simulate diffuse brain tissue activity
            diffuse_attention = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(width, height) / 1.5)**2)) * 0.3
            attention_map += diffuse_attention

        # Add some realistic, subtle background noise
        noise = np.random.normal(0, 0.02, (height, width)) # Gaussian noise, smaller
        attention_map += noise
        
        # Normalize the attention map to [0, 1]
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-10)
        
        # Store different simulated attention types
        attention_maps['Grad-CAM (Simulated)'] = attention_map
        
        # Example for another type, slightly different pattern
        # Could be more localized or highlight different features
        attention_maps['LIME (Simulated)'] = (np.random.random((height, width)) * 0.3 + attention_map * 0.7) # Mix of random and main
        attention_maps['LIME (Simulated)'] = (attention_maps['LIME (Simulated)'] - attention_maps['LIME (Simulated)'].min()) / (attention_maps['LIME (Simulated)'].max() - attention_maps['LIME (Simulated)'].min() + 1e-10)

        attention_maps['Integrated Gradients (Simulated)'] = (np.random.random((height, width)) * 0.2 + np.sqrt(attention_map) * 0.8) # Non-linear transformation
        attention_maps['Integrated Gradients (Simulated)'] = (attention_maps['Integrated Gradients (Simulated)'] - attention_maps['Integrated Gradients (Simulated)'].min()) / (attention_maps['Integrated Gradients (Simulated)'].max() - attention_maps['Integrated Gradients (Simulated)'].min() + 1e-10)
        
        return attention_maps

def create_performance_dashboard():
    """Create a comprehensive performance dashboard using synthetic data."""
    st.subheader("📊 Model Performance Dashboard")
    st.markdown("<p style='text-align: center; color: #777;'>This dashboard presents simulated performance metrics for various deep learning architectures.</p>", unsafe_allow_html=True)

    # Generate synthetic performance data
    models = ["ResNet-50", "DenseNet-121", "EfficientNet-B0", "Vision Transformer", "ConvNeXt"]
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    tumor_types = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    
    np.random.seed(42) # For reproducibility of synthetic data

    # Main performance data
    performance_data_overall = {}
    for i, model in enumerate(models):
        perf_values = []
        for j, metric in enumerate(metrics):
            score = np.random.uniform(0.78, 0.93)
            if "ResNet" in model and metric == "Accuracy": score = np.random.uniform(0.88, 0.95)
            elif "DenseNet" in model and metric == "Recall": score = np.random.uniform(0.85, 0.92)
            elif "EfficientNet" in model and metric == "F1-Score": score = np.random.uniform(0.87, 0.94)
            elif "Vision Transformer" in model and metric == "AUC-ROC": score = np.random.uniform(0.90, 0.96)
            elif "ConvNeXt" in model and metric == "Precision": score = np.random.uniform(0.86, 0.93)
            perf_values.append(score)
        performance_data_overall[model] = perf_values
    performance_df_overall = pd.DataFrame(performance_data_overall, index=metrics).T
    
    # Per-class performance (simulated)
    performance_data_per_class = {}
    for model in models:
        for tumor_type in tumor_types:
            for metric in ["Precision", "Recall", "F1-Score"]: # Common per-class metrics
                score = np.random.uniform(0.70, 0.95)
                # Introduce some variations for realism
                if tumor_type == "No Tumor" and "Recall" in metric: score = np.random.uniform(0.90, 0.99) # High recall for normal
                if tumor_type == "Glioma" and "Precision" in metric: score = np.random.uniform(0.70, 0.85) # Glioma harder to pinpoint
                performance_data_per_class[f"{model}_{tumor_type}_{metric}"] = score
    
    per_class_df = pd.DataFrame.from_dict(performance_data_per_class, orient='index', columns=['Score'])
    per_class_df[['Model', 'Tumor Type', 'Metric']] = per_class_df.index.to_series().str.split('_', n=2, expand=True)


    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
    st.write("#### 🎯 Overall Model Performance Metrics")
    # Create heatmap using matplotlib and seaborn
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(performance_df_overall, 
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                linewidths=.5,
                linecolor='lightgray',
                cbar_kws={'label': 'Score'},
                ax=ax)
    ax.set_title('Simulated Overall Model Performance Metrics Heatmap', fontsize=16, pad=20)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('Model Architecture', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', rotation=0, labelsize=12)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
    st.write("#### 📊 Model Performance Comparison (Bar Chart)")
    # Performance comparison chart using Plotly Express
    fig_comparison = px.bar(
        performance_df_overall.reset_index().melt(id_vars='index'),
        x='index', 
        y='value',
        color='variable',
        barmode='group', # Group bars by model
        title="Simulated Model Performance Comparison by Metric",
        labels={'index': 'Model', 'value': 'Score', 'variable': 'Metric'},
        hover_data={'value': ':.3f'},
        height=550
    )
    fig_comparison.update_layout(xaxis_title="Model Architecture", yaxis_title="Score", legend_title="Metric")
    fig_comparison.update_traces(marker_line_width=1, marker_line_color='black')
    st.plotly_chart(fig_comparison, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
    st.write("#### 📈 Metric Trends Across Models")
    # Line chart showing how each metric performs across models
    fig_line = px.line(
        performance_df_overall.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score'),
        x='index',
        y='Score',
        color='Metric',
        markers=True,
        title="Simulated Metric Performance Across Models",
        labels={'index': 'Model', 'Score': 'Score'},
        height=450
    )
    fig_line.update_layout(xaxis_title="Model Architecture", yaxis_title="Score", legend_title="Metric")
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
    st.write("#### 🔬 Per-Class Performance Breakdown (Example: F1-Score)")
    # Filter for F1-Score for per-class comparison
    f1_df = per_class_df[per_class_df['Metric'] == 'F1-Score']
    fig_per_class = px.bar(
        f1_df,
        x='Tumor Type',
        y='Score',
        color='Model',
        barmode='group',
        title="Simulated F1-Score Per Tumor Type by Model",
        labels={'Score': 'F1-Score'},
        height=500
    )
    fig_per_class.update_layout(xaxis_title="Tumor Type", yaxis_title="F1-Score", legend_title="Model")
    st.plotly_chart(fig_per_class, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def generate_report(analysis_data):
    """Generates a detailed markdown report from analysis data."""
    report_content = io.StringIO()
    
    report_content.write(f"# Medical Imaging Analysis Report\n")
    report_content.write(f"### Generated by Advanced Brain Tumor Detection System\n")
    report_content.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
    report_content.write(f"**Time:** {datetime.now().strftime('%H:%M:%S')}\n\n")
    report_content.write("---")
    report_content.write("## 📄 Patient & Scan Information\n")
    report_content.write(f"- **Scan Filename:** `{analysis_data['filename']}`\n")
    report_content.write(f"- **Analysis Timestamp:** {analysis_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_content.write(f"- **Original Image Dimensions:** {analysis_data['original'].size[0]} x {analysis_data['original'].size[1]} pixels\n")
    report_content.write(f"- **Original Image Mode:** {analysis_data['original'].mode}\n")
    
    report_content.write("\n---\n")
    report_content.write("## 🧠 AI Prediction Summary\n")
    
    ensemble_scores = analysis_data['ensemble']
    predicted_class = max(ensemble_scores, key=ensemble_scores.get)
    confidence = ensemble_scores[predicted_class]
    uncertainty_metrics = analysis_data['uncertainty']
    
    if predicted_class == "No Tumor":
        report_content.write(f"### ✅ **Prediction: {predicted_class}**\n")
        report_content.write(f"No significant abnormal findings indicative of a tumor were detected.\n")
    else:
        report_content.write(f"### ⚠️ **Prediction: {predicted_class}**\n")
        report_content.write(f"Potential tumor of type **{predicted_class}** detected. Medical consultation highly recommended.\n")

    report_content.write(f"- **Ensemble Confidence:** `{confidence:.2%}`\n")
    report_content.write(f"- **Prediction Uncertainty (Entropy):** `{uncertainty_metrics['entropy']:.3f}`\n")
    report_content.write(f"- **Mean Model Disagreement (Variance):** `{uncertainty_metrics['mean_variance']:.5f}`\n")
    
    if predicted_class != "No Tumor":
        characteristics = AdvancedBrainTumorDetector().tumor_characteristics[predicted_class]
        report_content.write(f"\n#### {predicted_class} Characteristics:\n")
        report_content.write(f"- **Description:** {characteristics['description']}\n")
        report_content.write(f"- **Typical Location:** {characteristics['typical_location']}\n")
        report_content.write(f"- **Prognosis:** {characteristics['prognosis']}\n")
        report_content.write(f"- **Treatment Considerations:** {characteristics['treatment']}\n")

    report_content.write("\n---\n")
    report_content.write("## 📈 Detailed Prediction Scores (Ensemble)\n")
    report_content.write("| Tumor Type    | Confidence |\n")
    report_content.write("|:--------------|:-----------|\n")
    for tumor_type, score in ensemble_scores.items():
        report_content.write(f"| {tumor_type} | {score:.2%} |\n")

    report_content.write("\n---\n")
    report_content.write("## 🔬 Image Feature Analysis\n")
    features = analysis_data['features']
    report_content.write("| Feature                 | Value      |\n")
    report_content.write("|:------------------------|:-----------|\n")
    for feature_name, value in features.items():
        if isinstance(value, (int, float)):
            report_content.write(f"| {feature_name.replace('_', ' ').title()} | `{value:.3f}` |\n")
        else:
            report_content.write(f"| {feature_name.replace('_', ' ').title()} | `{value}` |\n")

    report_content.write("\n---\n")
    report_content.write("## 📊 Model-Specific Predictions\n")
    for model_name, preds in analysis_data['predictions'].items():
        report_content.write(f"### {model_name}\n")
        report_content.write("| Tumor Type    | Confidence |\n")
        report_content.write("|:--------------|:-----------|\n")
        for tumor_type, prob in preds.items():
            report_content.write(f"| {tumor_type} | {prob:.2%} |\n")
        report_content.write("\n") # Add a newline after each model's predictions

    report_content.write("\n---\n")
    report_content.write("## ❓ Uncertainty Quantification Details\n")
    report_content.write("- **Ensemble Prediction Entropy:** `{:.3f}` (Higher value indicates more uncertainty)\n".format(uncertainty_metrics['entropy']))
    report_content.write("- **Mean Inter-Model Variance:** `{:.5f}` (Average disagreement across models)\n".format(uncertainty_metrics['mean_variance']))
    report_content.write("\n#### Confidence Interval (95%) per Class:\n")
    report_content.write("| Tumor Type    | Lower Bound | Upper Bound |\n")
    report_content.write("|:--------------|:------------|:------------|\n")
    for i, tumor_type in enumerate(AdvancedBrainTumorDetector().tumor_types):
        lower = uncertainty_metrics['confidence_interval'][0, i]
        upper = uncertainty_metrics['confidence_interval'][1, i]
        report_content.write(f"| {tumor_type} | `{lower:.2%}` | `{upper:.2%}` |\n")
    
    report_content.write("\n---")
    report_content.write("\n## ⚠️ **Disclaimer**\n")
    report_content.write("This report is generated by an AI system for **research and informational purposes only**. It is not intended for clinical diagnosis or as a substitute for professional medical advice. Always consult a qualified healthcare professional for any medical concerns. AI predictions may contain errors (false positives/negatives) and should not be used for making critical medical decisions.\n")
    report_content.write("\n---")
    report_content.write("\n*End of Report*")


    return report_content.getvalue()

def main():
    st.markdown('<h1 class="main-header">🧠 Pro-Level Brain Tumor Detection System</h1>', unsafe_allow_html=True)
    
    # Enhanced Medical Disclaimer
    st.markdown("""
    <div class="alert-critical">
        <h3>⚠️ CRITICAL MEDICAL DISCLAIMER</h3>
        <p><strong>This application is for educational and research purposes ONLY.</strong> It is built for demonstrating advanced image analysis techniques and simulating AI prediction workflows.</p>
        <ul>
            <li>🚫 <strong>NOT for clinical diagnosis or medical decision-making.</strong></li>
            <li>👨‍⚕️ Always consult qualified healthcare professionals for any health concerns.</li>
            <li>🔬 AI predictions may contain errors (false positives/negatives are possible) and are based on simulated models.</li>
            <li>🔒 All image processing is done locally within your browser/Streamlit session; no images are uploaded to external servers for privacy.</li>
            <li>📚 Use this tool solely for learning, research, and understanding AI in medical imaging.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize detector
    detector = AdvancedBrainTumorDetector()
    
    # Create tabs for different functionalities
    tabs = st.tabs(["🔍 Analysis", "📊 Performance Dashboard", "📈 Batch Processing", "🎯 Model Comparison", "📋 Reports & History", "ℹ️ About"])
    
    with tabs[0]:  # Analysis Tab
        st.header("Single Image Advanced Analysis")
        st.markdown("<p style='text-align: center; color: #777;'>Upload a brain scan to perform comprehensive AI-powered analysis.</p>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📁 Upload Brain Scan")
            st.markdown("<p style='color: #555;'>Upload MRI or CT scan images. Supported formats: JPG, PNG, JPEG, BMP, TIFF.</p>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a brain scan image...",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                accept_multiple_files=False, # Ensure only one file can be uploaded here
                key="single_image_uploader"
            )
            
            current_image = None
            if uploaded_file is not None:
                current_image = Image.open(uploaded_file)
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.image(current_image, caption="Uploaded Image", use_column_width=True)
                
                # Enhanced image information
                st.markdown(f"""
                    <h4>📋 Image Information</h4>
                    <p><strong>Dimensions:</strong> {current_image.size[0]} × {current_image.size[1]} pixels</p>
                    <p><strong>Color Mode:</strong> {current_image.mode}</p>
                    <p><strong>Format:</strong> {current_image.format}</p>
                    <p><strong>File Size:</strong> {len(uploaded_file.getvalue()) / 1024:.2f} KB</p>
                    <p><strong>Aspect Ratio:</strong> {current_image.size[0]/current_image.size[1]:.2f}</p>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.subheader("🔧 Processing Options (Automatic)")
                st.info("Key preprocessing steps like noise reduction, contrast enhancement, and feature extraction are automatically applied for optimal analysis. The options below are for demonstration and are 'locked' to show included steps.")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.checkbox("Enhanced Contrast (CLAHE)", value=True, disabled=True)
                    st.checkbox("Noise Reduction (Bilateral Filter)", value=True, disabled=True)
                    st.checkbox("Edge Detection (Canny)", value=OPENCV_AVAILABLE, disabled=True)
                
                with col_b:
                    st.checkbox("Morphological Operations", value=OPENCV_AVAILABLE, disabled=True)
                    st.checkbox("Texture Analysis (LBP)", value=ADVANCED_PROCESSING, disabled=True)
                    st.checkbox("Uncertainty Quantification", value=True, disabled=True)
                
                # Process button
                if st.button("🚀 Run Advanced Analysis", type="primary", key="run_analysis_button"):
                    with st.spinner("Running advanced AI analysis... This might take a moment."):
                        progress_bar = st.progress(0, text="Initializing analysis...")
                        
                        # Advanced preprocessing
                        progress_bar.progress(20, text="Step 1/5: Preprocessing image (resizing, denoising, enhancing)...")
                        processed_results = detector.advanced_preprocessing(current_image)
                        
                        # Feature extraction
                        progress_bar.progress(40, text="Step 2/5: Extracting comprehensive image features...")
                        features = detector.extract_features(processed_results['resized'])
                        
                        # Predictions
                        progress_bar.progress(60, text="Step 3/5: Simulating deep learning predictions across models...")
                        predictions = detector.simulate_advanced_prediction(processed_results['resized'], features)
                        ensemble_scores = detector.ensemble_prediction(predictions)
                        
                        # Uncertainty quantification
                        progress_bar.progress(80, text="Step 4/5: Quantifying prediction uncertainty and model agreement...")
                        uncertainty_metrics = detector.uncertainty_quantification(predictions)
                        
                        # Attention maps
                        progress_bar.progress(90, text="Step 5/5: Generating explainability maps...")
                        attention_maps = detector.generate_detailed_attention_map(processed_results['resized'], ensemble_scores)
                        
                        progress_bar.progress(100, text="Analysis complete!")
                        
                        # Store results
                        analysis_id = hashlib.md5(uploaded_file.getvalue()).hexdigest() + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
                        st.session_state.processed_images.append({
                            'id': analysis_id,
                            'original': current_image,
                            'processed_results': processed_results,
                            'features': features,
                            'predictions': predictions, # Individual model predictions
                            'ensemble': ensemble_scores, # Ensemble prediction
                            'uncertainty': uncertainty_metrics,
                            'attention_maps': attention_maps,
                            'timestamp': datetime.now(),
                            'filename': uploaded_file.name
                        })
                        
                        st.success("✅ Analysis completed successfully!")
                        st.balloons() # Add a little celebration
                        st.rerun() # Rerun to display results in col2

            else:
                st.info("Please upload an image to begin the advanced analysis.")

        with col2:
            st.subheader("📊 Advanced Analysis Results")
            st.markdown("<p style='color: #555;'>View the AI's predictions, feature analysis, and explainability maps.</p>", unsafe_allow_html=True)
            
            if st.session_state.processed_images:
                latest_result = st.session_state.processed_images[-1]
                ensemble_scores = latest_result['ensemble']
                uncertainty_metrics = latest_result['uncertainty']
                features = latest_result['features']
                attention_maps = latest_result['attention_maps']
                
                # Main prediction with uncertainty
                predicted_class = max(ensemble_scores, key=ensemble_scores.get)
                confidence = ensemble_scores[predicted_class]
                
                # Enhanced result display
                if predicted_class == "No Tumor":
                    st.markdown(f"""
                    <div class="alert-success">
                        <h3>✅ Prediction: {predicted_class}</h3>
                        <p><strong>Ensemble Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Uncertainty (Entropy):</strong> {uncertainty_metrics['entropy']:.3f}</p>
                        <p>Based on the analysis, no tumor was detected in this scan.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-warning">
                        <h3>⚠️ Prediction: {predicted_class}</h3>
                        <p><strong>Ensemble Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Uncertainty (Entropy):</strong> {uncertainty_metrics['entropy']:.3f}</p>
                        <p>A potential <strong>{predicted_class.lower()}</strong> has been indicated. Immediate medical consultation is highly advised.</p>
                        <p><strong>Recommendation:</strong> Do not rely on this software for diagnosis. Seek professional medical evaluation.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Tumor characteristics
                if predicted_class != "No Tumor":
                    characteristics = detector.tumor_characteristics[predicted_class]
                    st.markdown(f"""
                    <div class="feature-box">
                        <h4>🏥 {predicted_class} - Key Characteristics</h4>
                        <p><strong>Description:</strong> {characteristics['description']}</p>
                        <p><strong>Typical Location:</strong> {characteristics['typical_location']}</p>
                        <p><strong>Prognosis Overview:</strong> {characteristics['prognosis']}</p>
                        <p><strong>General Treatment:</strong> {characteristics['treatment']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
                st.subheader("📈 Ensemble Prediction Confidence")
                st.info("The bar chart below shows the confidence of the ensemble model for each tumor type. Error bars indicate the standard deviation of predictions across individual models, reflecting inter-model disagreement.")
                
                # Create confidence plot with uncertainty bars
                tumor_types = list(ensemble_scores.keys())
                confidences = [ensemble_scores[tt] for tt in tumor_types]
                uncertainties_variance_per_class = uncertainty_metrics['variance_per_class']
                
                fig_confidence = go.Figure()
                
                fig_confidence.add_trace(go.Bar(
                    x=tumor_types,
                    y=confidences,
                    # Error bars using standard deviation (sqrt of variance)
                    error_y=dict(type='data', array=np.sqrt(uncertainties_variance_per_class), visible=True, color='darkgray', thickness=2, width=3),
                    marker_color=['#ff7f0e' if t == predicted_class else '#1f77b4' for t in tumor_types],
                    text=[f'{c:.2%}' for c in confidences],
                    textposition='outside', # Position text outside bars
                    name='Confidence',
                    hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2%}<br>Std Dev: %{error_y.array[0]:.3f}<extra></extra>' # Custom hover text
                ))
                
                fig_confidence.update_layout(
                    title="Ensemble Predictions with Model Disagreement",
                    xaxis_title="Tumor Type",
                    yaxis_title="Confidence Score",
                    yaxis_range=[0, 1], # Ensure y-axis is from 0 to 1
                    showlegend=False,
                    height=480,
                    hovermode="x unified",
                    plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_confidence, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


                st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
                st.subheader("🔍 Key Image Feature Analysis")
                st.info("These numerical features are extracted from the image and can provide insights into tissue characteristics.")
                
                feature_cols = st.columns(3)
                
                with feature_cols[0]:
                    st.metric("Mean Intensity", f"{features.get('mean_intensity', 0):.2f}", help="Average pixel intensity (brightness).")
                    st.metric("Std Intensity", f"{features.get('std_intensity', 0):.2f}", help="Variation in pixel intensity (contrast/heterogeneity).")
                
                with feature_cols[1]:
                    st.metric("Hist. Entropy", f"{features.get('histogram_entropy', 0):.3f}", help="Measure of randomness or complexity in the image's intensity distribution.")
                    st.metric("Hist. Variance", f"{features.get('histogram_variance', 0):.2f}", help="Spread of values in the image's intensity histogram.")
                
                with feature_cols[2]:
                    if 'largest_region_area' in features:
                        st.metric("Largest Region Area", f"{features.get('largest_region_area', 0):.0f} px²", help="Area of the largest segmented non-background region.")
                        st.metric("Region Eccentricity", f"{features.get('largest_region_eccentricity', 0):.3f}", help="How elongated the largest region is (0 for circle, 1 for line).")
                        st.metric("Region Solidity", f"{features.get('largest_region_solidity', 0):.3f}", help="Area of the region divided by the area of its convex hull (compactness).")
                
                if ADVANCED_PROCESSING and 'texture' in latest_result['processed_results']:
                    st.write("#### Texture Analysis Preview (LBP)")
                    st.image(latest_result['processed_results']['texture'], caption="Local Binary Pattern Texture Map", use_column_width=True, clamp=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
                st.subheader("👁️‍🗨️ AI Explainability: Attention Maps")
                st.warning("These attention maps are **simulated** to illustrate potential areas of focus for an AI model. They are not derived from a real deep learning model's internal activations.")
                
                if attention_maps:
                    attention_map_types = list(attention_maps.keys())
                    selected_attention_map_type = st.selectbox("Select Attention Map Type", attention_map_types, key="attention_map_selector")
                    
                    if selected_attention_map_type:
                        att_map = attention_maps[selected_attention_map_type]
                        original_image_array_for_overlay = np.array(latest_result['original'].resize(att_map.shape[::-1])) # Resize original to match map
                        
                        # Create an interactive slider for opacity
                        opacity_level = st.slider("Attention Map Opacity", 0.0, 1.0, 0.4, 0.05, key="attention_opacity_slider")

                        # Plotly figure for overlay
                        fig_att = go.Figure()
                        fig_att.add_trace(px.imshow(original_image_array_for_overlay).data[0]) # Add original image
                        
                        # Add heatmap layer (normalized attention map)
                        fig_att.add_trace(go.Heatmap(
                            z=att_map,
                            colorscale='Hot', #'Jet', 'Viridis', 'Plasma'
                            showscale=True,
                            opacity=opacity_level,
                            colorbar=dict(title='Attention Intensity', titleside='right')
                        ))
                        fig_att.update_layout(
                            title=f"{selected_attention_map_type} Overlay on Original Image",
                            hovermode="closest",
                            width=original_image_array_for_overlay.shape[1],
                            height=original_image_array_for_overlay.shape[0],
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        fig_att.update_xaxes(showticklabels=False, zeroline=False, showgrid=False).update_yaxes(showticklabels=False, zeroline=False, showgrid=False)
                        st.plotly_chart(fig_att, use_container_width=True)
                        
                        # Show the raw attention map for closer inspection
                        st.image(att_map, caption=f"Raw {selected_attention_map_type} (Normalized)", use_column_width=True, clamp=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No analysis results to display yet. Upload an image and click 'Run Advanced Analysis'.")

    with tabs[1]:  # Performance Dashboard Tab
        create_performance_dashboard()
    
    with tabs[2]:  # Batch Processing Tab
        st.subheader("📈 Batch Processing for Multiple Scans")
        st.markdown("<p style='color: #555;'>Upload multiple brain scans to process them in a batch and get a summary report.</p>", unsafe_allow_html=True)

        uploaded_files_batch = st.file_uploader(
            "Upload multiple brain scan images...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            key="batch_image_uploader"
        )
        
        if uploaded_files_batch:
            st.info(f"📁 **{len(uploaded_files_batch)}** files selected for batch processing.")
            
            if st.button("🚀 Process All Images in Batch", type="primary", key="run_batch_analysis_button"):
                progress_bar_batch = st.progress(0, text="Starting batch processing...")
                batch_results = []
                
                for i, uploaded_file in enumerate(uploaded_files_batch):
                    file_name = uploaded_file.name
                    progress_text = f"Processing {file_name} ({i+1}/{len(uploaded_files_batch)})..."
                    progress_bar_batch.progress((i + 1) / len(uploaded_files_batch), text=progress_text)
                    
                    try:
                        image = Image.open(uploaded_file)
                        processed_results = detector.advanced_preprocessing(image)
                        features = detector.extract_features(processed_results['resized'])
                        predictions = detector.simulate_advanced_prediction(processed_results['resized'], features)
                        ensemble_scores = detector.ensemble_prediction(predictions)
                        uncertainty_metrics = detector.uncertainty_quantification(predictions)
                        
                        predicted_class = max(ensemble_scores, key=ensemble_scores.get)
                        confidence = ensemble_scores[predicted_class]
                        
                        batch_results.append({
                            'filename': file_name,
                            'prediction': predicted_class,
                            'confidence': confidence,
                            'uncertainty_entropy': uncertainty_metrics['entropy'],
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        # st.success(f"Processed: {file_name} -> Predicted: **{predicted_class}**") # Too verbose for many files
                    except Exception as e:
                        st.error(f"Error processing {file_name}: {e}")
                        batch_results.append({
                            'filename': file_name,
                            'prediction': "Error",
                            'confidence': 0.0,
                            'uncertainty_entropy': float('inf'),
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                
                progress_bar_batch.empty()
                st.success("✅ Batch processing completed!")
                st.session_state.last_batch_results = batch_results # Store for later display/download

                # Display batch results
                st.subheader("📊 Batch Results Summary")
                
                if st.session_state.last_batch_results:
                    batch_df = pd.DataFrame(st.session_state.last_batch_results)
                    
                    # Summary statistics
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    cols_sum = st.columns(4)
                    with cols_sum[0]: st.metric("Total Images", len(batch_df))
                    with cols_sum[1]: st.metric("Potential Tumors", len(batch_df[batch_df['prediction'] != 'No Tumor']))
                    with cols_sum[2]: st.metric("Avg. Confidence", f"{batch_df['confidence'].mean():.2%}")
                    with cols_sum[3]: st.metric("High Conf. Tumors (>70%)", len(batch_df[(batch_df['prediction'] != 'No Tumor') & (batch_df['confidence'] > 0.7)]))
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
                    st.write("#### Detailed Batch Results Table")
                    st.dataframe(batch_df.set_index('filename'), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
                    st.write("#### Prediction Distribution")
                    fig_batch_dist = px.histogram(
                        batch_df,
                        x='prediction',
                        color='prediction',
                        title="Distribution of Predictions in Batch",
                        labels={'prediction': 'Predicted Tumor Type', 'count': 'Number of Images'},
                        height=400
                    )
                    st.plotly_chart(fig_batch_dist, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
                    st.write("#### Confidence vs. Uncertainty")
                    fig_scatter_conf = px.scatter(
                        batch_df,
                        x='confidence',
                        y='uncertainty_entropy',
                        color='prediction',
                        size='confidence', # Size points by confidence
                        hover_name='filename',
                        title="Confidence vs. Uncertainty for Batch Predictions",
                        labels={'confidence': 'Confidence Score', 'uncertainty_entropy': 'Uncertainty (Entropy)'},
                        height=500
                    )
                    st.plotly_chart(fig_scatter_conf, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    csv_export = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Batch Results as CSV",
                        data=csv_export,
                        file_name="batch_analysis_results.csv",
                        mime="text/csv",
                        key="download_batch_csv"
                    )
                else:
                    st.info("No results to display for batch processing yet.")
        else:
            st.info("Upload images above to enable batch processing.")

    with tabs[3]:  # Model Comparison Tab
        st.subheader("🎯 Model Comparison for Last Analyzed Image")
        st.markdown("<p style='color: #777;'>This section compares the simulated predictions from different model architectures for the <strong>most recently analyzed single image</strong>, highlighting model agreement and divergence.</p>", unsafe_allow_html=True)
        
        if st.session_state.processed_images:
            latest_result = st.session_state.processed_images[-1]
            predictions = latest_result['predictions']
            
            # Create comparison visualization
            comparison_data = []
            for model_name, preds in predictions.items():
                for tumor_type, confidence in preds.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Tumor Type': tumor_type,
                        'Confidence': confidence
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
            st.write("#### Individual Model Confidence Scores Table")
            st.dataframe(comparison_df.pivot_table(index='Model', columns='Tumor Type', values='Confidence').style.format("{:.3f}"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Model agreement analysis
            st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
            st.write("#### 📊 Model Prediction Heatmap")
            st.info("This heatmap shows each model's confidence for each tumor type. Darker shades indicate lower confidence, brighter shades indicate higher confidence.")
            
            # Create heatmap of model predictions
            pivot_df = comparison_df.pivot(index='Model', columns='Tumor Type', values='Confidence')
            
            fig_heatmap_compare = px.imshow(
                pivot_df.values,
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale='Plasma', # Good for highlighting high values
                text_auto=True, # Show values on heatmap
                aspect="auto",
                title="Model Prediction Confidence Heatmap"
            )
            fig_heatmap_compare.update_layout(xaxis_title="Tumor Type", yaxis_title="Model Architecture")
            st.plotly_chart(fig_heatmap_compare, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Model consensus
            st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
            st.write("#### 🤝 Model Consensus and Disagreement")
            st.info("This table quantifies the agreement among models for each tumor type. A higher Standard Deviation indicates greater disagreement.")
            
            ensemble_scores = latest_result['ensemble']
            consensus_data = []
            
            for tumor_type, ensemble_score in ensemble_scores.items():
                individual_scores = [predictions[model][tumor_type] for model in predictions.keys()]
                std_dev = np.std(individual_scores)
                
                agreement_level = 'High' if std_dev < 0.05 else 'Medium' if std_dev < 0.15 else 'Low'
                
                consensus_data.append({
                    'Tumor Type': tumor_type,
                    'Ensemble Score': ensemble_score,
                    'Standard Deviation': std_dev,
                    'Agreement Level': agreement_level
                })
            
            consensus_df = pd.DataFrame(consensus_data)
            st.dataframe(consensus_df.set_index('Tumor Type').style.format({
                'Ensemble Score': "{:.3f}",
                'Standard Deviation': "{:.4f}"
            }), use_container_width=True)

            # Box plot for model agreement
            st.write("##### Model Prediction Distribution per Class (Box Plot)")
            fig_box = px.box(
                comparison_df,
                x="Tumor Type",
                y="Confidence",
                color="Tumor Type",
                points="all", # Show individual model points
                title="Distribution of Model Predictions per Tumor Class",
                labels={"Confidence": "Predicted Confidence"},
                height=450
            )
            fig_box.update_layout(yaxis_range=[0,1])
            st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info("Please run a single image analysis in the 'Analysis' tab first to see model comparisons.")

    with tabs[4]: # Reports & History Tab
        st.subheader("📋 Generate Analysis Report & History")
        st.markdown("<p style='color: #777;'>Access detailed reports for previous analyses or browse the analysis history.</p>", unsafe_allow_html=True)

        if st.session_state.processed_images:
            # Sort by timestamp, newest first
            sorted_history = sorted(st.session_state.processed_images, key=lambda x: x['timestamp'], reverse=True)
            
            analysis_options_display = [
                f"[{res['timestamp'].strftime('%Y-%m-%d %H:%M')}] {res['filename']} (Pred: {max(res['ensemble'], key=res['ensemble'].get)}, Conf: {max(res['ensemble'].values()):.2%})"
                for res in sorted_history
            ]
            
            selected_analysis_str = st.selectbox(
                "Choose an analysis from history to view/generate report:",
                options=['-- Select an analysis --'] + analysis_options_display,
                key="history_selector"
            )

            selected_analysis_data = None
            if selected_analysis_str and selected_analysis_str != '-- Select an analysis --':
                # Find the actual data based on the selected string
                selected_analysis_data = next(
                    (res for res in sorted_history if f"[{res['timestamp'].strftime('%Y-%m-%d %H:%M')}] {res['filename']} (Pred: {max(res['ensemble'], key=res['ensemble'].get)}, Conf: {max(res['ensemble'].values()):.2%})" == selected_analysis_str),
                    None
                )

            if selected_analysis_data:
                st.markdown("---")
                st.write("#### Selected Analysis Details:")
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.write(f"**Filename:** `{selected_analysis_data['filename']}`")
                st.write(f"**Analysis Time:** `{selected_analysis_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}`")
                st.write(f"**Primary Prediction:** <span style='font-weight:bold; color:#1f77b4;'>{max(selected_analysis_data['ensemble'], key=selected_analysis_data['ensemble'].get)}</span>", unsafe_allow_html=True)
                st.write(f"**Confidence:** `{max(selected_analysis_data['ensemble'].values()):.2%}`")
                st.markdown("</div>", unsafe_allow_html=True)

                if st.button("📄 Generate Full Report (Markdown)", type="primary", key="generate_full_report_button"):
                    report_text = generate_report(selected_analysis_data)
                    
                    st.download_button(
                        label="Download Report as Markdown (.md) File",
                        data=report_text.encode('utf-8'), # Encode for download
                        file_name=f"Brain_Tumor_Report_{selected_analysis_data['filename'].split('.')[0]}_{selected_analysis_data['timestamp'].strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        help="Downloads a comprehensive markdown report of the analysis. Can be viewed with any markdown editor."
                    )
                    st.markdown("---")
                    st.write("#### Preview of Generated Report:")
                    st.markdown(report_text, unsafe_allow_html=True) # Display as markdown
            else:
                st.info("Please select an analysis from the dropdown above to view its details or generate a report.")
        else:
            st.info("No analyses have been performed yet. Go to the 'Analysis' tab and upload an image first to build history.")

    with tabs[5]: # About Tab
        st.subheader("ℹ️ About This Application")
        st.markdown("""
        <div class="feature-box">
            <h4>Application Overview</h4>
            <p>The <strong>Advanced Brain Tumor Detection System</strong> is an AI-powered prototype developed for educational and research purposes in the field of medical image analysis. It demonstrates a sophisticated workflow for processing brain MRI/CT scans and simulating tumor detection using advanced computer vision techniques and deep learning concepts.</p>
            <p>This system integrates several modules:</p>
            <ul>
                <li><strong>Advanced Preprocessing:</strong> Applies techniques like noise reduction, contrast enhancement (CLAHE), edge detection, and morphological operations to prepare images for analysis.</li>
                <li><strong>Feature Extraction:</strong> Derives quantitative features (e.g., intensity statistics, histogram characteristics, regional properties, texture via LBP) that are crucial for traditional machine learning and can inform deep learning.</li>
                <li><strong>Simulated Deep Learning Prediction:</strong> Mimics the behavior of multiple state-of-the-art deep learning architectures (ResNet, DenseNet, EfficientNet, Vision Transformer, ConvNeXt) to provide ensemble predictions for different tumor types.</li>
                <li><strong>Uncertainty Quantification:</strong> Provides metrics (entropy, variance) to assess the reliability and agreement of the AI's predictions, a critical aspect of responsible AI in medicine.</li>
                <li><strong>Simulated Explainability (Attention Maps):</strong> Generates visual "attention maps" to illustrate which parts of the image the simulated AI models "focused" on, aiding interpretability.</li>
            </ul>
            <p>The application is built using the <a href="https://streamlit.io/" target="_blank">Streamlit</a> framework, enabling rapid development of interactive web applications in Python.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
            <h4>Technologies Used</h4>
            <ul>
                <li><strong>Python:</strong> The core programming language.</li>
                <li><strong>Streamlit:</strong> For building the interactive web interface.</li>
                <li><strong>NumPy:</strong> Fundamental package for numerical computation.</li>
                <li><strong>PIL (Pillow):</strong> For image loading and basic manipulations.</li>
                <li><strong>OpenCV (cv2):</strong> For advanced image processing operations like resizing, denoising, and CLAHE.</li>
                <li><strong>Scikit-image (skimage):</strong> For advanced filtering, segmentation, and feature extraction (e.g., LBP).</li>
                <li><strong>SciPy:</strong> For scientific computing, used by scikit-image.</li>
                <li><strong>Pandas:</strong> For data manipulation and tabular display.</li>
                <li><strong>Plotly & Plotly Express:</strong> For interactive and visually rich data visualizations.</li>
                <li><strong>Matplotlib & Seaborn:</strong> For static data visualizations (e.g., heatmaps).</li>
                <li><strong>Scikit-learn:</strong> Used for general machine learning utilities (though not for model training in this demo).</li>
            </ul>
            <p><strong>Note on Deep Learning Models:</strong> This application simulates the predictions of deep learning models. It does not load or run actual pre-trained deep learning models due to the computational and model hosting requirements not being feasible within a standard Streamlit Cloud free tier setup. The aim is to demonstrate the *workflow* and *outputs* of such a system.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
            <h4>Acknowledgements</h4>
            <p>This project is inspired by the critical need for advanced tools in medical imaging and aims to contribute to the understanding and responsible development of AI in healthcare.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #777;'>Developed with ❤️ for educational and research purposes.</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
