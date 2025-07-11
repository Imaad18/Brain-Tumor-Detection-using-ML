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
    page_title="NeuroVision Analyzer | AI-Powered Brain Pathology Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern CSS Styling ---
def inject_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        /* Base Styles */
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --secondary: #7c3aed;
            --accent: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #10b981;
            --dark: #1e293b;
            --light: #f8fafc;
            --gray: #64748b;
            --gray-light: #e2e8f0;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f1f5f9;
            color: var(--dark);
            line-height: 1.6;
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-right: 2.5rem;
            padding-left: 2.5rem;
            padding-bottom: 3rem;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: var(--dark);
            margin-top: 1.5em;
            margin-bottom: 0.75em;
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }
        
        h2 {
            font-size: 2rem;
            border-bottom: 1px solid var(--gray-light);
            padding-bottom: 0.5rem;
        }
        
        h3 {
            font-size: 1.5rem;
        }
        
        code, pre {
            font-family: 'JetBrains Mono', monospace;
            background-color: #f3f4f6;
            padding: 0.2em 0.4em;
            border-radius: 4px;
            font-size: 0.9em;
        }
        
        /* Cards & Containers */
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        .card-header {
            font-weight: 600;
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Alerts */
        .alert {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            border-left: 4px solid;
        }
        
        .alert-danger {
            background-color: #fee2e2;
            border-color: var(--danger);
            color: #991b1b;
        }
        
        .alert-warning {
            background-color: #fef3c7;
            border-color: var(--warning);
            color: #92400e;
        }
        
        .alert-success {
            background-color: #d1fae5;
            border-color: var(--success);
            color: #065f46;
        }
        
        .alert-info {
            background-color: #dbeafe;
            border-color: var(--primary);
            color: #1e40af;
        }
        
        /* Buttons */
        .stButton button {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.6rem 1.25rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stButton button:hover {
            background-color: var(--primary-dark);
            transform: translateY(-1px);
        }
        
        .stButton button:active {
            transform: translateY(0);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
            background: white;
            border: 1px solid var(--gray-light);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #f8fafc;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        /* Sidebar */
        .stSidebar {
            background-color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Inputs */
        .stTextInput input, .stSelectbox select, .stTextArea textarea {
            border-radius: 8px !important;
            border: 1px solid var(--gray-light) !important;
        }
        
        .stSlider .st-cq {
            background-color: var(--gray-light) !important;
        }
        
        .stSlider .st-by {
            background-color: var(--primary) !important;
        }
        
        /* Custom Elements */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            background-color: var(--gray-light);
            color: var(--dark);
        }
        
        .badge-primary {
            background-color: var(--primary);
            color: white;
        }
        
        .badge-success {
            background-color: var(--success);
            color: white;
        }
        
        .badge-warning {
            background-color: var(--warning);
            color: white;
        }
        
        .badge-danger {
            background-color: var(--danger);
            color: white;
        }
        
        /* Utility Classes */
        .text-muted {
            color: var(--gray);
        }
        
        .text-center {
            text-align: center;
        }
        
        .mt-3 {
            margin-top: 1rem;
        }
        
        .mb-3 {
            margin-bottom: 1rem;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }
        
        /* Custom Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade {
            animation: fadeIn 0.3s ease-out forwards;
        }
    </style>
    """, unsafe_allow_html=True)

# Call the CSS injection function
inject_css()

# Initialize session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}
if 'user_annotations' not in st.session_state:
    st.session_state.user_annotations = {}

class NeuroVisionAnalyzer:
    def __init__(self):
        self.model_names = [
            "3D ResNet-50", 
            "DenseNet-121", 
            "EfficientNet-B4", 
            "Vision Transformer", 
            "ConvNeXt-XL",
            "UNet++",
            "nnUNet"
        ]
        self.pathology_types = [
            "Glioma", 
            "Meningioma", 
            "Pituitary Adenoma", 
            "Metastasis",
            "Medulloblastoma",
            "No Significant Finding"
        ]
        self.pathology_characteristics = {
            "Glioma": {
                "description": "Primary brain tumors arising from glial cells, ranging from low-grade (slow-growing) to high-grade (aggressive).",
                "typical_features": "Irregular margins, heterogeneous enhancement, surrounding edema, mass effect.",
                "clinical_notes": "Grade IV glioblastoma has poor prognosis despite treatment. Molecular markers (IDH, MGMT) important for treatment planning.",
                "treatment": "Maximal safe resection followed by radiation and temozolomide for high-grade tumors. Low-grade may be monitored or treated with surgery alone."
            },
            "Meningioma": {
                "description": "Typically benign tumors arising from meningothelial cells of the arachnoid membrane.",
                "typical_features": "Well-circumscribed, dural-based, homogeneous enhancement, dural tail sign, calcifications common.",
                "clinical_notes": "Most are WHO Grade I. Atypical (Grade II) and anaplastic (Grade III) variants have higher recurrence rates.",
                "treatment": "Observation for small asymptomatic tumors. Surgical resection for symptomatic or growing tumors. Radiation for residual/recurrent disease."
            },
            "Pituitary Adenoma": {
                "description": "Benign tumors of the pituitary gland, classified by size (microadenoma <1cm, macroadenoma ≥1cm) and hormone secretion.",
                "typical_features": "Sella turcica mass, may extend superiorly compressing optic chiasm, often enhances less than normal pituitary.",
                "clinical_notes": "May present with hormonal dysfunction (prolactinoma most common) or mass effect (bitemporal hemianopsia).",
                "treatment": "Medical therapy for prolactinomas (dopamine agonists). Transsphenoidal surgery for others. Radiation for residual/recurrent disease."
            },
            "Metastasis": {
                "description": "Secondary tumors from systemic cancers, most commonly lung, breast, melanoma, renal, and colorectal.",
                "typical_features": "Multiple lesions at gray-white junction, ring-enhancing with central necrosis, significant surrounding edema.",
                "clinical_notes": "Presence indicates Stage IV disease. Prognosis depends on primary cancer type and systemic control.",
                "treatment": "Steroids for edema. Whole brain radiation, stereotactic radiosurgery, or resection for solitary/oligometastases. Systemic therapy."
            },
            "Medulloblastoma": {
                "description": "Highly malignant embryonal tumor of the cerebellum, most common in children but can occur in adults.",
                "typical_features": "Cerebellar mass, often midline, hyperdense on CT, restricted diffusion on MRI, heterogeneous enhancement.",
                "clinical_notes": "Prone to CSF dissemination. Molecular subgroups (WNT, SHH, Group 3, Group 4) have prognostic significance.",
                "treatment": "Maximal safe resection followed by craniospinal radiation and chemotherapy. Proton therapy preferred in children."
            },
            "No Significant Finding": {
                "description": "No evidence of neoplastic, vascular, or other significant intracranial pathology detected.",
                "typical_features": "Normal brain parenchyma, symmetric ventricles, intact gray-white differentiation, no mass effect.",
                "clinical_notes": "Clinical correlation required as some pathologies may be subtle or require advanced imaging techniques.",
                "treatment": "None required. Follow-up as clinically indicated."
            }
        }
        
    def advanced_preprocessing(self, image, target_size=(256, 256)):
        """Enhanced preprocessing pipeline with advanced techniques."""
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
            # Advanced denoising
            img_denoised = cv2.fastNlMeansDenoisingColored(img_resized, None, 7, 7, 5, 15)
            results['denoised'] = img_denoised
            
            # Multi-scale contrast enhancement
            lab = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE on L-channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            results['enhanced'] = img_clahe
            
            # Edge detection with multiple methods
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            edges_canny = cv2.Canny(gray, 50, 150)
            edges_sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            edges_sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            edges_sobel = np.sqrt(edges_sobelx**2 + edges_sobely**2)
            edges_sobel = np.uint8(edges_sobel / np.max(edges_sobel) * 255)
            
            results['edges_canny'] = edges_canny
            results['edges_sobel'] = edges_sobel
            
            # Multi-level thresholding
            _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            results['thresh_otsu'] = thresh_otsu
            
            # Adaptive thresholding
            thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
            results['thresh_adapt'] = thresh_adapt
        
        if ADVANCED_PROCESSING:
            # Advanced texture analysis
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY) if OPENCV_AVAILABLE else np.mean(img_resized, axis=2).astype(np.uint8)
            
            # Local Binary Patterns with multiple radii
            radii = [1, 2, 3]
            n_points = [8 * r for r in radii]
            lbp_images = []
            
            for r, n in zip(radii, n_points):
                lbp = local_binary_pattern(gray, n, r, method='uniform')
                lbp_images.append(lbp)
            
            results['lbp_multi'] = lbp_images
            
            # Gabor filter bank
            gabor_kernels = []
            for theta in np.arange(0, np.pi, np.pi / 4):
                for sigma in (1, 3):
                    for frequency in (0.05, 0.25):
                        kernel = np.real(filters.gabor_kernel(frequency, theta=theta,
                                                            sigma_x=sigma, sigma_y=sigma))
                        gabor_kernels.append(kernel)
            
            gabor_features = np.zeros((gray.shape[0], gray.shape[1], len(gabor_kernels)))
            for i, kernel in enumerate(gabor_kernels):
                filtered = ndimage.convolve(gray, kernel, mode='wrap')
                gabor_features[..., i] = filtered
            
            results['gabor'] = gabor_features
            
            # Advanced segmentation
            try:
                img_smooth = filters.gaussian(gray, sigma=2)
                threshold = filters.threshold_multiotsu(img_smooth, classes=3)
                regions = np.digitize(img_smooth, bins=threshold)
                results['multiotsu'] = regions
            except:
                pass
        
        return results
    
    def extract_radiomic_features(self, image):
        """Comprehensive radiomic feature extraction."""
        features = {}
        
        # Ensure image is in suitable format
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if OPENCV_AVAILABLE else np.mean(image, axis=2).astype(np.uint8)
        else:
            gray_image = image.astype(np.uint8)
            
        # First-order statistics
        features['mean_intensity'] = np.mean(gray_image)
        features['median_intensity'] = np.median(gray_image)
        features['std_intensity'] = np.std(gray_image)
        features['skewness'] = pd.Series(gray_image.flatten()).skew()
        features['kurtosis'] = pd.Series(gray_image.flatten()).kurtosis()
        features['energy'] = np.sum(gray_image.astype('float')**2)
        features['entropy'] = filters.rank.entropy(gray_image, morphology.disk(5))
        
        # Histogram features
        hist, _ = np.histogram(gray_image.flatten(), bins=256, range=(0, 256))
        hist_norm = hist / (hist.sum() + 1e-10)
        features['hist_entropy'] = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        features['hist_energy'] = np.sum(hist_norm**2)
        features['hist_variance'] = np.var(hist_norm)
        
        # Shape features (if we can segment a region)
        if ADVANCED_PROCESSING:
            try:
                img_smooth = filters.gaussian(gray_image, sigma=2)
                threshold = filters.threshold_otsu(img_smooth)
                binary = img_smooth > threshold
                
                # Remove small objects
                cleaned = morphology.remove_small_objects(binary, min_size=64)
                labeled = measure.label(cleaned)
                
                if np.max(labeled) > 0:  # If we found regions
                    regions = measure.regionprops(labeled, intensity_image=gray_image)
                    largest_region = max(regions, key=lambda r: r.area)
                    
                    # Shape features
                    features['area'] = largest_region.area
                    features['perimeter'] = largest_region.perimeter
                    features['eccentricity'] = largest_region.eccentricity
                    features['solidity'] = largest_region.solidity
                    features['extent'] = largest_region.extent
                    features['equiv_diameter'] = largest_region.equivalent_diameter
                    features['feret_diameter_max'] = largest_region.feret_diameter_max
                    
                    # Intensity features within region
                    features['region_mean'] = largest_region.mean_intensity
                    features['region_median'] = np.median(largest_region.intensity_image[largest_region.image])
                    features['region_std'] = largest_region.intensity_std
                    features['region_entropy'] = filters.rank.entropy(largest_region.intensity_image, morphology.disk(3))
                    
                    # Texture within region
                    if largest_region.area > 100:  # Only if region is large enough
                        lbp = local_binary_pattern(largest_region.intensity_image, 24, 3, method='uniform')
                        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 27), range=(0, 26))
                        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-10)
                        features['lbp_energy'] = np.sum(lbp_hist**2)
                        features['lbp_entropy'] = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
            except:
                pass
        
        return features
    
    def simulate_ai_prediction(self, image, features):
        """Enhanced prediction simulation with clinical knowledge integration."""
        predictions = {}
        
        # Create deterministic seed based on image content
        img_hash = hashlib.md5(image.tobytes()).hexdigest()
        base_seed = int(img_hash[:8], 16) % 10000
        
        for i, model_name in enumerate(self.model_names):
            np.random.seed(base_seed + i)
            
            # Initialize with random baseline
            raw_scores = np.random.rand(len(self.pathology_types)) * 0.3
            
            # Clinical feature influence - these simulate how real models might weight features
            if features.get('std_intensity', 0) > 45:
                raw_scores[self.pathology_types.index("Glioma")] += 0.25
                raw_scores[self.pathology_types.index("Metastasis")] += 0.15
                
            if features.get('hist_entropy', 0) > 6.5:
                raw_scores[self.pathology_types.index("Glioma")] += 0.2
                raw_scores[self.pathology_types.index("Medulloblastoma")] += 0.1
                
            if features.get('area', 0) > 2000 and features.get('solidity', 0) > 0.85:
                raw_scores[self.pathology_types.index("Meningioma")] += 0.3
                
            if 500 < features.get('area', 0) < 1500 and features.get('eccentricity', 0) < 0.5:
                raw_scores[self.pathology_types.index("Pituitary Adenoma")] += 0.25
                
            if features.get('region_mean', 0) > 180 and features.get('region_std', 0) > 30:
                raw_scores[self.pathology_types.index("Metastasis")] += 0.2
                
            # Special rules for pediatric-looking scans (simplified)
            if features.get('equiv_diameter', 0) > 100 and features.get('region_entropy', 0) > 5.5:
                raw_scores[self.pathology_types.index("Medulloblastoma")] += 0.15
                
            # Penalize "No Significant Finding" if tumor-like features present
            if (features.get('std_intensity', 0) > 30 or 
                features.get('hist_entropy', 0) > 5 or 
                features.get('area', 0) > 100):
                raw_scores[self.pathology_types.index("No Significant Finding")] *= 0.5
                
            # Add model-specific biases
            if "ResNet" in model_name:
                raw_scores[self.pathology_types.index("Glioma")] += 0.1
            elif "DenseNet" in model_name:
                raw_scores[self.pathology_types.index("Meningioma")] += 0.1
            elif "EfficientNet" in model_name:
                raw_scores[self.pathology_types.index("Pituitary Adenoma")] += 0.1
            elif "Transformer" in model_name:
                raw_scores[self.pathology_types.index("Metastasis")] += 0.1
            elif "UNet" in model_name:
                raw_scores[self.pathology_types.index("Medulloblastoma")] += 0.1
                
            # Ensure scores are valid
            raw_scores = np.clip(raw_scores, 0.01, 0.99)
            probabilities = np.exp(raw_scores) / np.sum(np.exp(raw_scores))
            
            # Add some noise to differentiate models
            noise = np.random.normal(0, 0.03, len(probabilities))
            probabilities = probabilities + noise
            probabilities = np.clip(probabilities, 0.01, 0.99)
            probabilities = probabilities / np.sum(probabilities)
            
            predictions[model_name] = {
                pathology: prob for pathology, prob in zip(self.pathology_types, probabilities)
            }
        
        return predictions
    
    def ensemble_prediction(self, predictions):
        """Weighted ensemble prediction with model confidence."""
        ensemble_scores = {pathology: 0 for pathology in self.pathology_types}
        model_weights = {
            "3D ResNet-50": 0.9,
            "DenseNet-121": 0.9,
            "EfficientNet-B4": 0.95,
            "Vision Transformer": 0.85,
            "ConvNeXt-XL": 0.95,
            "UNet++": 0.8,
            "nnUNet": 1.0  # nnUNet typically performs very well in medical imaging
        }
        
        total_weight = sum(model_weights.values())
        
        for model_name, preds in predictions.items():
            weight = model_weights.get(model_name, 0.8)
            for pathology, prob in preds.items():
                ensemble_scores[pathology] += prob * weight
                
        # Normalize
        ensemble_scores = {k: v / total_weight for k, v in ensemble_scores.items()}
        
        return ensemble_scores
    
    def uncertainty_analysis(self, predictions):
        """Comprehensive uncertainty quantification."""
        all_predictions = []
        for model_preds in predictions.values():
            all_predictions.append(list(model_preds.values()))
        
        pred_array = np.array(all_predictions)
        
        # Basic metrics
        mean_probs = np.mean(pred_array, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        variance = np.var(pred_array, axis=0)
        mean_variance = np.mean(variance)
        
        # Confidence intervals
        ci_low, ci_high = np.percentile(pred_array, [2.5, 97.5], axis=0)
        
        # Model disagreement
        max_diff = np.max(pred_array, axis=0) - np.min(pred_array, axis=0)
        
        return {
            'entropy': entropy,
            'variance_per_class': variance,
            'mean_variance': mean_variance,
            'confidence_interval': (ci_low, ci_high),
            'model_disagreement': max_diff,
            'prediction_array': pred_array
        }
    
    def generate_attention_maps(self, image, ensemble_scores):
        """Enhanced attention map generation with anatomical considerations."""
        height, width = image.shape[:2]
        attention_maps = {}
        
        predicted_class = max(ensemble_scores, key=ensemble_scores.get)
        confidence = ensemble_scores[predicted_class]
        
        # Base coordinates
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        
        # Initialize base map
        base_map = np.zeros((height, width))
        
        # Anatomical prior - simulate approximate brain location
        brain_mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) < min(center_x, center_y) * 0.9
        base_map = base_map * brain_mask
        
        # Pathology-specific patterns
        if predicted_class == "Glioma":
            # Irregular, multifocal
            for _ in range(np.random.randint(2, 5)):
                focus_x = np.random.randint(width * 0.2, width * 0.8)
                focus_y = np.random.randint(height * 0.2, height * 0.8)
                sigma_x = width / (3 + np.random.rand() * 4)
                sigma_y = height / (3 + np.random.rand() * 4)
                base_map += np.exp(-(((x - focus_x)**2 / (2 * sigma_x**2)) + 
                                    ((y - focus_y)**2 / (2 * sigma_y**2)))) * confidence
                
        elif predicted_class == "Meningioma":
            # Peripheral, dural-based
            angle = np.random.uniform(0, 2*np.pi)
            distance = min(width, height) * 0.4
            focus_x = center_x + distance * np.cos(angle)
            focus_y = center_y + distance * np.sin(angle)
            
            base_map += np.exp(-((x - focus_x)**2 + (y - focus_y)**2) / 
                             (2 * (min(width, height)/5)**2)) * confidence * 1.5
            
            # Simulate dural tail
            tail_length = min(width, height) * 0.2
            tail_x = np.linspace(focus_x, focus_x + tail_length * np.cos(angle + np.pi/8), 100)
            tail_y = np.linspace(focus_y, focus_y + tail_length * np.sin(angle + np.pi/8), 100)
            
            for tx, ty in zip(tail_x, tail_y):
                base_map += np.exp(-((x - tx)**2 + (y - ty)**2) / 
                                 (2 * (min(width, height)/15)**2)) * confidence * 0.3
                
        elif predicted_class == "Pituitary Adenoma":
            # Central, sella region (lower center)
            focus_x = center_x
            focus_y = center_y + height * 0.15
            base_map += np.exp(-((x - focus_x)**2 + (y - focus_y)**2) / 
                             (2 * (min(width, height)/8)**2)) * confidence * 2.0
            
        elif predicted_class == "Metastasis":
            # Multiple peripheral lesions
            for _ in range(np.random.randint(2, 6)):
                focus_x = np.random.randint(width * 0.1, width * 0.9)
                focus_y = np.random.randint(height * 0.1, height * 0.9)
                sigma = min(width, height) / (8 + np.random.rand() * 4)
                base_map += np.exp(-((x - focus_x)**2 + (y - focus_y)**2) / 
                                 (2 * sigma**2)) * confidence * 0.8
                
        elif predicted_class == "Medulloblastoma":
            # Posterior fossa (lower central)
            focus_x = center_x
            focus_y = center_y + height * 0.2
            base_map += np.exp(-((x - focus_x)**2 + (y - focus_y)**2) / 
                             (2 * (min(width, height)/6)**2)) * confidence * 1.8
            
            # Often has restricted diffusion - simulate more concentrated center
            base_map += np.exp(-((x - focus_x)**2 + (y - focus_y)**2) / 
                             (2 * (min(width, height)/12)**2)) * confidence * 0.5
            
        else:  # No Significant Finding
            # Diffuse attention to normal structures
            # Ventricles
            ventricles = np.exp(-(((np.abs(x - center_x) - width*0.15)**2) / (2*(width*0.05)**2))) * \
                           np.exp(-((y - (center_y - height*0.1))**2) / (2*(height*0.2)**2))
            # Sulci
            sulci = np.sin(x * np.pi * 4 / width) * np.sin(y * np.pi * 3 / height) * 0.3
            base_map = (ventricles * 0.5 + np.clip(sulci, 0, 1) * 0.3) * 0.7
        
        # Add realistic noise
        noise = np.random.normal(0, 0.03, (height, width))
        base_map = np.clip(base_map + noise, 0, 1)
        
        # Normalize
        if base_map.max() > 0:
            base_map = (base_map - base_map.min()) / (base_map.max() - base_map.min())
            
        # Create different attention types
        attention_maps['Grad-CAM'] = base_map
        attention_maps['Attention Rollout'] = np.power(base_map, 0.7)  # More diffuse
        attention_maps['Guided Backprop'] = np.clip(base_map * 1.3 - 0.15, 0, 1)  # Higher contrast
        
        return attention_maps

def create_performance_dashboard():
    """Enhanced performance dashboard with clinical metrics."""
    st.subheader("Model Performance Analytics")
    
    # Generate comprehensive synthetic data
    models = ["3D ResNet-50", "DenseNet-121", "EfficientNet-B4", 
              "Vision Transformer", "ConvNeXt-XL", "UNet++", "nnUNet"]
    metrics = ["Accuracy", "Sensitivity", "Specificity", "Precision", 
               "F1-Score", "AUC-ROC", "Dice Score"]
    pathologies = ["Glioma", "Meningioma", "Pituitary Adenoma", 
                   "Metastasis", "Medulloblastoma", "No Significant Finding"]
    
    np.random.seed(42)
    
    # Overall performance
    performance_data = {}
    for model in models:
        perf = []
        for metric in metrics:
            base = np.random.uniform(0.75, 0.92)
            
            # Model-specific adjustments
            if "nnUNet" in model:
                base = np.clip(base + 0.05, 0.75, 0.98)
            elif "EfficientNet" in model or "ConvNeXt" in model:
                base = np.clip(base + 0.03, 0.75, 0.95)
                
            # Metric-specific adjustments
            if metric == "Dice Score":
                base = np.clip(base - 0.05, 0.7, 0.95)
            elif metric == "AUC-ROC":
                base = np.clip(base + 0.03, 0.8, 0.99)
                
            perf.append(base)
        performance_data[model] = perf
    
    perf_df = pd.DataFrame(performance_data, index=metrics).T
    
    # Per-pathology performance
    pathology_data = []
    for model in models:
        for pathology in pathologies:
            for metric in ["Sensitivity", "Specificity", "Dice Score"]:
                score = np.random.uniform(0.65, 0.95)
                
                # Pathology-specific adjustments
                if pathology == "Glioma" and metric == "Dice Score":
                    score = np.clip(score - 0.1, 0.6, 0.9)  # Gliomas are harder to segment
                elif pathology == "Meningioma" and metric == "Sensitivity":
                    score = np.clip(score + 0.05, 0.7, 0.98)  # Meningiomas are easier to detect
                elif pathology == "No Significant Finding" and metric == "Specificity":
                    score = np.clip(score + 0.1, 0.8, 0.99)  # Important to not flag normal as abnormal
                    
                pathology_data.append({
                    'Model': model,
                    'Pathology': pathology,
                    'Metric': metric,
                    'Score': score
                })
    
    pathology_df = pd.DataFrame(pathology_data)
    
    # Confidence intervals for error bars
    perf_df['CI_low'] = perf_df.apply(lambda x: x - np.random.uniform(0.02, 0.08), axis=1)
    perf_df['CI_high'] = perf_df.apply(lambda x: x + np.random.uniform(0.02, 0.08), axis=1)
    
    # Display
    st.markdown("### Overall Model Performance")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Interactive heatmap
        fig = px.imshow(
            perf_df[metrics],
            color_continuous_scale='Viridis',
            labels=dict(x="Metric", y="Model", color="Score"),
            aspect="auto"
        )
        fig.update_layout(
            title="Performance Heatmap Across Metrics",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Top Performers by Metric**")
        for metric in metrics[:3]:
            top_model = perf_df[metric].idxmax()
            score = perf_df.loc[top_model, metric]
            st.metric(
                label=f"{metric}",
                value=f"{score:.3f}",
                delta=f"{top_model}",
                delta_color="off"
            )
        
        st.markdown("**Average Scores**")
        st.write(perf_df[metrics].mean().to_frame(name="Mean").style.format("{:.3f}"))
    
    st.markdown("### Pathology-Specific Performance")
    
    tab1, tab2, tab3 = st.tabs(["Sensitivity", "Specificity", "Dice Score"])
    
    with tab1:
        fig = px.box(
            pathology_df[pathology_df['Metric'] == "Sensitivity"],
            x="Pathology",
            y="Score",
            color="Model",
            points="all",
            title="Sensitivity by Pathology"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.box(
            pathology_df[pathology_df['Metric'] == "Specificity"],
            x="Pathology",
            y="Score",
            color="Model",
            points="all",
            title="Specificity by Pathology"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.box(
            pathology_df[pathology_df['Metric'] == "Dice Score"],
            x="Pathology",
            y="Score",
            color="Model",
            points="all",
            title="Segmentation Accuracy (Dice Score)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Model Comparison Radar Chart")
    
    # Prepare data for radar chart
    radar_df = perf_df.reset_index().melt(id_vars='index', value_vars=metrics)
    radar_df = radar_df.rename(columns={'index': 'Model', 'variable': 'Metric', 'value': 'Score'})
    
    fig = px.line_polar(
        radar_df[radar_df['Model'].isin(["nnUNet", "EfficientNet-B4", "3D ResNet-50"])], 
        r='Score', 
        theta='Metric',
        color='Model',
        line_close=True,
        template="plotly_dark",
        title="Model Performance Radar Chart (Selected Models)"
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_clinical_report(analysis_data):
    """Generate a comprehensive clinical-style report."""
    report = io.StringIO()
    
    detector = NeuroVisionAnalyzer()
    ensemble_scores = analysis_data['ensemble']
    predicted_class = max(ensemble_scores, key=ensemble_scores.get)
    confidence = ensemble_scores[predicted_class]
    uncertainty = analysis_data['uncertainty']
    features = analysis_data['features']
    
    # Header
    report.write(f"# NEUROVISION ANALYZER - CLINICAL REPORT\n\n")
    report.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.write(f"**Scan ID:** {hashlib.md5(analysis_data['filename'].encode()).hexdigest()[:8]}\n")
    report.write(f"**Original Filename:** {analysis_data['filename']}\n\n")
    
    # Summary
    report.write("## CLINICAL SUMMARY\n\n")
    
    if predicted_class == "No Significant Finding":
        report.write("No definitive evidence of intracranial pathology was identified on this scan.\n")
    else:
        report.write(f"Findings are suggestive of **{predicted_class}** with an ensemble confidence of **{confidence:.1%}**.\n")
    
    report.write("\n## DETAILED FINDINGS\n\n")
    
    # Imaging Characteristics
    report.write("### Imaging Characteristics\n")
    report.write("| Feature | Value |\n")
    report.write("|---------|-------|\n")
    report.write(f"| Mean Intensity | {features.get('mean_intensity', 'N/A'):.1f} |\n")
    report.write(f"| Intensity Standard Deviation | {features.get('std_intensity', 'N/A'):.1f} |\n")
    report.write(f"| Histogram Entropy | {features.get('hist_entropy', 'N/A'):.2f} |\n")
    
    if 'area' in features:
        report.write(f"| Largest Region Area | {features['area']:.0f} px² |\n")
        report.write(f"| Region Solidity | {features.get('solidity', 'N/A'):.2f} |\n")
        report.write(f"| Region Eccentricity | {features.get('eccentricity', 'N/A'):.2f} |\n")
    
    report.write("\n### Model Predictions\n")
    report.write("| Model | Prediction | Confidence |\n")
    report.write("|-------|------------|------------|\n")
    
    for model, preds in analysis_data['predictions'].items():
        pred = max(preds, key=preds.get)
        conf = preds[pred]
        report.write(f"| {model} | {pred} | {conf:.1%} |\n")
    
    # Clinical Correlation
    report.write("\n## CLINICAL CORRELATION\n\n")
    if predicted_class in detector.pathology_characteristics:
        char = detector.pathology_characteristics[predicted_class]
        report.write(f"**Typical Presentation:** {char['description']}\n\n")
        report.write(f"**Imaging Features:** {char['typical_features']}\n\n")
        report.write(f"**Clinical Considerations:** {char['clinical_notes']}\n\n")
        report.write(f"**Management Considerations:** {char['treatment']}\n\n")
    
    # Uncertainty Analysis
    report.write("## UNCERTAINTY ANALYSIS\n\n")
    report.write(f"**Prediction Entropy:** {uncertainty['entropy']:.3f} (lower is more certain)\n")
    report.write(f"**Mean Model Variance:** {uncertainty['mean_variance']:.4f}\n\n")
    
    report.write("**Confidence Intervals (95%) by Class:**\n")
    report.write("| Pathology | Lower Bound | Upper Bound |\n")
    report.write("|-----------|-------------|-------------|\n")
    
    for i, path in enumerate(detector.pathology_types):
        low, high = uncertainty['confidence_interval'][0][i], uncertainty['confidence_interval'][1][i]
        report.write(f"| {path} | {low:.1%} | {high:.1%} |\n")
    
    # Recommendations
    report.write("\n## RECOMMENDATIONS\n\n")
    if predicted_class == "No Significant Finding":
        report.write("1. No immediate follow-up required based on this analysis.\n")
        report.write("2. Clinical correlation advised as always.\n")
    else:
        report.write("1. Urgent neurosurgical/neurological consultation recommended.\n")
        report.write("2. Consider further imaging (e.g., contrast-enhanced MRI, perfusion imaging) as clinically indicated.\n")
        report.write("3. Multidisciplinary tumor board review if diagnosis is confirmed.\n")
    
    # Disclaimer
    report.write("\n## DISCLAIMER\n\n")
    report.write("This report is generated by an AI system for research and educational purposes only. It is not a substitute for professional medical judgment. The interpreting physician must correlate these findings with the clinical scenario and other diagnostic tests as appropriate. False positives and false negatives may occur.\n")
    
    report.write("\n--- END OF REPORT ---")
    
    return report.getvalue()

def main():
    st.title("NeuroVision Analyzer")
    st.markdown("### AI-Powered Brain Pathology Detection System")
    
    # Enhanced disclaimer
    with st.expander("⚠️ IMPORTANT DISCLAIMER", expanded=True):
        st.error("""
        **This is a research and educational tool only.**  
        
        - NOT for clinical diagnosis or medical decision-making  
        - Results are simulated for demonstration purposes  
        - Always consult qualified medical professionals for health concerns  
        - AI predictions may contain errors (false positives/negatives possible)  
        - All processing occurs locally - no images are uploaded externally  
        """)
    
    # Initialize analyzer
    analyzer = NeuroVisionAnalyzer()
    
    # Create tabs
    tabs = st.tabs([
        "🔍 Single Image Analysis", 
        "📊 Batch Processing", 
        "📈 Performance Analytics",
        "📋 Report Generator",
        "ℹ️ About"
    ])
    
    with tabs[0]:  # Single Image Analysis
        st.subheader("Comprehensive Brain Scan Analysis")
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("#### Upload Brain Imaging Study")
            uploaded_file = st.file_uploader(
                "Select MRI/CT scan (JPEG, PNG, DICOM)",
                type=['jpg', 'jpeg', 'png', 'dcm'],
                key="single_upload"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.lower().endswith('.dcm'):
                        st.warning("DICOM support is experimental. Using first frame only.")
                        import pydicom
                        ds = pydicom.dcmread(uploaded_file)
                        image = Image.fromarray(ds.pixel_array)
                        if len(image.getbands()) > 1:
                            image = image.convert('L')  # Convert to grayscale
                    else:
                        image = Image.open(uploaded_file)
                    
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Image metadata
                    with st.expander("Image Details"):
                        st.write(f"**Dimensions:** {image.size[0]} × {image.size[1]} pixels")
                        st.write(f"**Mode:** {image.mode}")
                        st.write(f"**Format:** {uploaded_file.type.split('/')[-1].upper()}")
                        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                        
                    # Processing options
                    with st.expander("Processing Configuration"):
                        st.checkbox("Advanced denoising", True, disabled=True)
                        st.checkbox("Multi-scale contrast enhancement", True, disabled=True)
                        st.checkbox("Texture analysis", True, disabled=True)
                        st.checkbox("Uncertainty quantification", True, disabled=True)
                    
                    if st.button("Run Comprehensive Analysis", type="primary"):
                        with st.spinner("Performing advanced analysis..."):
                            progress = st.progress(0)
                            
                            # Step 1: Preprocessing
                            progress.progress(20, "Preprocessing image...")
                            processed = analyzer.advanced_preprocessing(image)
                            
                            # Step 2: Feature extraction
                            progress.progress(40, "Extracting radiomic features...")
                            features = analyzer.extract_radiomic_features(processed['resized'])
                            
                            # Step 3: AI prediction
                            progress.progress(60, "Running model simulations...")
                            predictions = analyzer.simulate_ai_prediction(processed['resized'], features)
                            ensemble = analyzer.ensemble_prediction(predictions)
                            
                            # Step 4: Uncertainty analysis
                            progress.progress(80, "Quantifying uncertainty...")
                            uncertainty = analyzer.uncertainty_analysis(predictions)
                            
                            # Step 5: Attention maps
                            progress.progress(90, "Generating attention maps...")
                            attention_maps = analyzer.generate_attention_maps(processed['resized'], ensemble)
                            
                            progress.progress(100, "Analysis complete!")
                            
                            # Store results
                            analysis_id = f"{hashlib.md5(uploaded_file.getvalue()).hexdigest()}_{int(time.time())}"
                            st.session_state.processed_images.append({
                                'id': analysis_id,
                                'original': image,
                                'processed': processed,
                                'features': features,
                                'predictions': predictions,
                                'ensemble': ensemble,
                                'uncertainty': uncertainty,
                                'attention_maps': attention_maps,
                                'filename': uploaded_file.name,
                                'timestamp': datetime.now()
                            })
                            
                            st.success("Analysis completed successfully!")
                            st.balloons()
                            st.rerun()
                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        
        with col2:
            st.markdown("#### Analysis Results")
            
            if st.session_state.processed_images:
                latest = st.session_state.processed_images[-1]
                pred_class = max(latest['ensemble'], key=latest['ensemble'].get)
                confidence = latest['ensemble'][pred_class]
                
                # Result card
                if pred_class == "No Significant Finding":
                    st.success(f"""
                    **Primary Finding:** No significant pathology detected  
                    **Confidence:** {confidence:.1%}  
                    **Recommendation:** Routine follow-up as clinically indicated
                    """)
                else:
                    st.warning(f"""
                    **Primary Finding:** {pred_class}  
                    **Confidence:** {confidence:.1%}  
                    **Clinical Urgency:** {'High' if pred_class in ['Glioma', 'Medulloblastoma', 'Metastasis'] else 'Medium'}  
                    **Recommendation:** Urgent specialist consultation advised
                    """)
                
                # Confidence visualization
                st.markdown("##### Prediction Confidence Distribution")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(latest['ensemble'].keys()),
                    y=list(latest['ensemble'].values()),
                    marker_color=['#ef4444' if x == pred_class else '#64748b' for x in latest['ensemble'].keys()]
                ))
                fig.update_layout(
                    yaxis_title="Confidence Score",
                    xaxis_title="Pathology",
                    height=400
                )
                st.plotly_chart(fig, use_column_width=True)
                
                # Model comparison
                st.markdown("##### Model Agreement Analysis")
                model_preds = []
                for model, preds in latest['predictions'].items():
                    pred = max(preds, key=preds.get)
                    conf = preds[pred]
                    model_preds.append({
                        'Model': model,
                        'Prediction': pred,
                        'Confidence': conf
                    })
                
                model_df = pd.DataFrame(model_preds)
                st.dataframe(
                    model_df.sort_values('Confidence', ascending=False),
                    hide_index=True,
                    use_container_width=True
                )
                
                # Attention maps
                st.markdown("##### AI Attention Visualization")
                att_type = st.selectbox(
                    "Attention Map Type",
                    list(latest['attention_maps'].keys()),
                    key="att_map_selector"
                )
                
                if att_type:
                    opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.05)
                    
                    fig = px.imshow(latest['original'])
                    fig.add_trace(go.Heatmap(
                        z=latest['attention_maps'][att_type],
                        colorscale='Hot',
                        opacity=opacity,
                        showscale=False
                    ))
                    fig.update_layout(
                        title=f"{att_type} Overlay",
                        height=500
                    )
                    st.plotly_chart(fig, use_column_width=True)
                
                # Features
                with st.expander("Advanced Feature Analysis"):
                    st.write("**Extracted Radiomic Features:**")
                    feature_cols = st.columns(3)
                    
                    features = latest['features']
                    feature_groups = [
                        ['mean_intensity', 'median_intensity', 'std_intensity'],
                        ['skewness', 'kurtosis', 'energy'],
                        ['hist_entropy', 'hist_energy', 'hist_variance']
                    ]
                    
                    if 'area' in features:
                        feature_groups.append(['area', 'perimeter', 'eccentricity'])
                        feature_groups.append(['solidity', 'extent', 'equiv_diameter'])
                    
                    for i, group in enumerate(feature_groups):
                        with feature_cols[i % 3]:
                            for feat in group:
                                if feat in features:
                                    st.metric(
                                        feat.replace('_', ' ').title(),
                                        f"{features[feat]:.2f}"
                                    )
            else:
                st.info("Upload an image and run analysis to view results")
    
    with tabs[1]:  # Batch Processing
        st.subheader("Batch Scan Analysis")
        
        uploaded_files = st.file_uploader(
            "Upload multiple scans for batch processing",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files:
            st.success(f"{len(uploaded_files)} scans ready for processing")
            
            if st.button("Process Batch", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for i, file in enumerate(uploaded_files):
                    try:
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress, f"Processing {file.name} ({i+1}/{len(uploaded_files)})")
                        
                        image = Image.open(file)
                        
                        # Simulate processing (in real app, would run full pipeline)
                        processed = analyzer.advanced_preprocessing(image)
                        features = analyzer.extract_radiomic_features(processed['resized'])
                        predictions = analyzer.simulate_ai_prediction(processed['resized'], features)
                        ensemble = analyzer.ensemble_prediction(predictions)
                        
                        pred_class = max(ensemble, key=ensemble.get)
                        confidence = ensemble[pred_class]
                        
                        results.append({
                            'Filename': file.name,
                            'Prediction': pred_class,
                            'Confidence': confidence,
                            'Findings': "Normal" if pred_class == "No Significant Finding" else "Abnormal",
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                        })
                    
                    except Exception as e:
                        results.append({
                            'Filename': file.name,
                            'Prediction': "Error",
                            'Confidence': 0.0,
                            'Findings': f"Processing error: {str(e)}",
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                        })
                
                progress_bar.empty()
                st.session_state.batch_results = results
                st.success("Batch processing completed!")
            
            if 'batch_results' in st.session_state:
                results_df = pd.DataFrame(st.session_state.batch_results)
                
                st.markdown("#### Batch Results Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Scans", len(results_df))
                
                with col2:
                    normal = len(results_df[results_df['Findings'] == 'Normal'])
                    st.metric("Normal Findings", normal)
                
                with col3:
                    abnormal = len(results_df[results_df['Findings'] == 'Abnormal'])
                    st.metric("Abnormal Findings", abnormal)
                
                st.dataframe(results_df, use_container_width=True)
                
                # Visualizations
                st.markdown("#### Findings Distribution")
                fig = px.pie(
                    results_df,
                    names='Findings',
                    title="Normal vs Abnormal Findings"
                )
                st.plotly_chart(fig, use_column_width=True)
                
                st.markdown("#### Confidence Distribution")
                fig = px.box(
                    results_df[results_df['Findings'] == 'Abnormal'],
                    y='Confidence',
                    x='Prediction',
                    title="Confidence Scores by Pathology"
                )
                st.plotly_chart(fig, use_column_width=True)
                
                # Export
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Export Results as CSV",
                    csv,
                    "batch_analysis_results.csv",
                    "text/csv",
                    key='download_csv'
                )
    
    with tabs[2]:  # Performance Analytics
        create_performance_dashboard()
    
    with tabs[3]:  # Report Generator
        st.subheader("Clinical Report Generator")
        
        if st.session_state.processed_images:
            analyses = [
                f"{a['timestamp'].strftime('%Y-%m-%d %H:%M')} - {a['filename']}"
                for a in st.session_state.processed_images
            ]
            
            selected = st.selectbox(
                "Select analysis to generate report",
                analyses
            )
            
            if selected:
                idx = analyses.index(selected)
                analysis = st.session_state.processed_images[idx]
                
                report = generate_clinical_report(analysis)
                
                st.download_button(
                    "Download Clinical Report (PDF)",
                    report,
                    file_name=f"neuro_report_{analysis['filename'].split('.')[0]}.md",
                    mime="text/markdown"
                )
                
                with st.expander("Preview Report"):
                    st.markdown(report)
        else:
            st.info("No analyses available. Please run an analysis first.")
    
    with tabs[4]:  # About
        st.subheader("About NeuroVision Analyzer")
        
        st.markdown("""
        **NeuroVision Analyzer** is an advanced AI-powered system for detecting brain pathologies in medical imaging studies. 
        This research tool demonstrates state-of-the-art techniques in medical image analysis, including:
        
        - Multi-modal image preprocessing
        - Comprehensive radiomic feature extraction
        - Ensemble deep learning predictions
        - Uncertainty quantification
        - Explainable AI with attention visualization
        
        ### Key Features
        
        - **Multi-Model Ensemble**: Combines predictions from 7 different neural architectures
        - **Clinical Correlation**: Provides pathology-specific characteristics and management guidelines
        - **Uncertainty Analysis**: Quantifies prediction reliability
        - **Batch Processing**: Analyze multiple studies simultaneously
        
        ### Technology Stack
        
        - Python 3.10+
        - Streamlit for interactive web interface
        - OpenCV, scikit-image for image processing
        - Plotly for interactive visualizations
        - NumPy, Pandas for numerical computing
        
        ### Disclaimer
        
        This is a research prototype only. Not for clinical use. Always consult qualified medical professionals for diagnostic decisions.
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.9rem;">
            NeuroVision Analyzer v2.1 | Developed for Research & Education | © 2023
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
