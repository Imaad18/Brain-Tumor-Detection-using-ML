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

try:
    from skimage import filters, measure, morphology, segmentation
    from skimage.feature import local_binary_pattern
    from scipy import ndimage
    ADVANCED_PROCESSING = True
except ImportError:
    ADVANCED_PROCESSING = False
    st.info("Install scikit-image and scipy for advanced image processing features for full functionality.")

# Page configuration
st.set_page_config(
    page_title="Advanced Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling - Using a function for clarity
def inject_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .alert-critical {
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .alert-warning {
            background: linear-gradient(135deg, #ffeaa7, #fab1a0);
            color: #2d3436;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .alert-success {
            background: linear-gradient(135deg, #00b894, #00a085);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .feature-box {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            color: white;
            font-weight: bold;
            padding: 10px 20px; /* Adjust padding for better look */
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); /* Hover effect */
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%); /* Active tab color */
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Call the CSS injection function
inject_css()

# Initialize session state with enhanced features
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'analysis_history' not in st.session_state:
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
                "description": "Usually benign, arises from the meninges (membranes surrounding the brain and spinal cord).",
                "typical_location": "Surface of the brain, falx cerebri, tentorium, sphenoid wing.",
                "prognosis": "Generally good if completely removed. Recurrence possible if incomplete resection.",
                "treatment": "Surgery (primary), radiation if incomplete resection or atypical/malignant features."
            },
            "Pituitary": {
                "description": "Tumor of the pituitary gland, a small gland at the base of the brain controlling hormones.",
                "typical_location": "Sella turcica (the bony pocket at the base of the skull).",
                "prognosis": "Often good with treatment, but can cause hormonal imbalances and vision problems.",
                "treatment": "Surgery (transsphenoidal approach), medication (for hormone-secreting tumors), radiation."
            },
            "No Tumor": {
                "description": "No abnormal tissue or lesions indicative of a tumor were detected in the scan.",
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
        hist_norm = hist / hist.sum()
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
                binary = gray_image > filters.threshold_otsu(gray_image)
                labeled = measure.label(binary)
                regions = measure.regionprops(labeled)
                
                if regions:
                    largest_region = max(regions, key=lambda r: r.area)
                    features['largest_region_area'] = largest_region.area
                    features['largest_region_perimeter'] = largest_region.perimeter
                    features['largest_region_eccentricity'] = largest_region.eccentricity
                    features['largest_region_solidity'] = largest_region.solidity
                    features['largest_region_extent'] = largest_region.extent # Ratio of area to bounding box area
                else:
                    # Default values if no regions are found
                    features['largest_region_area'] = 0
                    features['largest_region_perimeter'] = 0
                    features['largest_region_eccentricity'] = 0
                    features['largest_region_solidity'] = 0
                    features['largest_region_extent'] = 0
            except ValueError: # Handles cases where thresholding might fail (e.g., uniform image)
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
            # Adjust these weights based on what features might indicate certain tumors
            feature_influence = np.zeros(len(self.tumor_types))
            
            # Example: High mean intensity or entropy might suggest abnormalities
            if features.get('mean_intensity', 0) > 120:
                feature_influence[0] += 0.1 # Glioma
                feature_influence[1] += 0.05 # Meningioma
            if features.get('histogram_entropy', 0) > 7.0:
                feature_influence[0] += 0.08 # Glioma
            
            # Example: Large, well-defined region might suggest Meningioma
            if features.get('largest_region_area', 0) > 2000 and features.get('largest_region_solidity', 0) > 0.8:
                feature_influence[1] += 0.15 # Meningioma
            
            # Example: Centrally located, smaller region might suggest Pituitary
            # (Simplified check: if area is moderate and eccentricity is low)
            if 500 < features.get('largest_region_area', 0) < 1500 and features.get('largest_region_eccentricity', 1) < 0.7:
                feature_influence[2] += 0.12 # Pituitary
                
            # Simulate raw scores, then apply feature influence
            raw_scores = np.random.rand(len(self.tumor_types)) # Initial random scores
            raw_scores = raw_scores + feature_influence # Add feature influence
            
            # Ensure "No Tumor" is a strong prediction if features are low
            if features.get('largest_region_area', 0) < 100 and features.get('mean_intensity', 0) < 100:
                raw_scores[self.tumor_types.index("No Tumor")] += 0.5 # Boost "No Tumor"
                
            probabilities = np.exp(raw_scores) / np.sum(np.exp(raw_scores)) # Softmax-like normalization
            
            # Add model-specific noise to make them slightly different
            model_noise = np.random.normal(0, 0.02, len(probabilities)) # Small noise
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
        attention_map = np.random.random((height, width)) * 0.1 # Base low attention
        
        # Simulate different attention patterns for different tumor types
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        
        if predicted_class == "Glioma":
            # Gliomas often have irregular shapes and diffuse margins.
            # Simulate a more spread-out, less defined attention.
            sigma_x = width / (2 + np.random.rand() * 2) # Vary sigma for irregularity
            sigma_y = height / (2 + np.random.rand() * 2)
            main_region = np.exp(-(((x - center_x)**2 / (2 * sigma_x**2)) + ((y - center_y)**2 / (2 * sigma_y**2))))
            attention_map += main_region * confidence * 1.5 # Stronger activation, diffuse
            
            # Add some "irregular" hotspots
            for _ in range(3):
                hotspot_x, hotspot_y = np.random.randint(0, width), np.random.randint(0, height)
                hotspot_sigma = min(width, height) / (10 + np.random.rand() * 10)
                hotspot_mask = np.exp(-((x - hotspot_x)**2 + (y - hotspot_y)**2) / (2 * hotspot_sigma**2))
                attention_map += hotspot_mask * confidence * 0.5
                
        elif predicted_class == "Meningioma":
            # Meningiomas are often well-defined, pushing on brain tissue.
            # Simulate a more localized, distinct attention blob.
            radius = min(width, height) / (4 + np.random.rand())
            main_region = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * radius**2))
            attention_map += main_region * confidence * 2.0 # Stronger, more localized activation
            
        elif predicted_class == "Pituitary":
            # Pituitary tumors are centrally located at the base of the brain.
            # Simulate a very central, compact attention.
            radius = min(width, height) / (6 + np.random.rand() * 2)
            main_region = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * radius**2))
            attention_map += main_region * confidence * 2.5 # Very strong central activation
            
        else: # "No Tumor"
            # If no tumor, attention should be more diffuse or focus on general brain structures.
            # Simulate general low-level activity across the image.
            attention_map = np.random.random((height, width)) * 0.3 # Background noise-like
            # Add a slight diffuse center of attention as healthy brain regions
            main_region = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(width, height) / 2)**2))
            attention_map += main_region * 0.2
        
        # Add some realistic, subtle background noise
        noise = np.random.normal(0, 0.05, (height, width)) # Gaussian noise
        attention_map += noise
        
        # Normalize the attention map to [0, 1]
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-10)
        
        # Simulate different types of attention maps if more sophisticated logic were in place
        attention_maps['Grad-CAM'] = attention_map # Renamed for common terminology
        
        # Add placeholder for other attention types
        attention_maps['LIME'] = np.random.random((height, width)) * attention_map.max() # Example: LIME often highlights regions
        attention_maps['Integrated Gradients'] = np.random.random((height, width)) * attention_map.max()
        
        return attention_maps

def create_performance_dashboard():
    """Create a comprehensive performance dashboard using synthetic data."""
    st.subheader("📊 Model Performance Dashboard")
    
    # Generate synthetic performance data
    models = ["ResNet-50", "DenseNet-121", "EfficientNet-B0", "Vision Transformer", "ConvNeXt"]
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    
    # Create synthetic performance matrix
    np.random.seed(42) # For reproducibility of synthetic data
    
    # Simulate slightly varying performance, with some models "better" at certain metrics
    performance_data = {}
    for i, model in enumerate(models):
        perf_values = []
        for j, metric in enumerate(metrics):
            # Base performance
            score = np.random.uniform(0.78, 0.93)
            
            # Introduce small variations / strengths
            if "ResNet" in model and metric == "Accuracy":
                score = np.random.uniform(0.88, 0.95)
            elif "DenseNet" in model and metric == "Recall":
                score = np.random.uniform(0.85, 0.92)
            elif "EfficientNet" in model and metric == "F1-Score":
                score = np.random.uniform(0.87, 0.94)
            elif "Vision Transformer" in model and metric == "AUC-ROC":
                score = np.random.uniform(0.90, 0.96)
            elif "ConvNeXt" in model and metric == "Precision":
                score = np.random.uniform(0.86, 0.93)
            
            perf_values.append(score)
        performance_data[model] = perf_values
        
    performance_df = pd.DataFrame(performance_data, index=metrics).T # Transpose to have models as rows
    
    st.markdown("---")
    st.write("#### 🎯 Performance Metrics Heatmap")
    # Create heatmap using matplotlib and seaborn
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(performance_df, 
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                linewidths=.5,
                ax=ax)
    ax.set_title('Simulated Model Performance Metrics Heatmap')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    st.markdown("---")
    st.write("#### 📊 Model Performance Comparison (Bar Chart)")
    # Performance comparison chart using Plotly Express
    fig_comparison = px.bar(
        performance_df.reset_index().melt(id_vars='index'),
        x='index', 
        y='value',
        color='variable',
        barmode='group', # Group bars by model
        title="Simulated Model Performance Comparison by Metric",
        labels={'index': 'Model', 'value': 'Score', 'variable': 'Metric'},
        hover_data={'value': ':.3f'},
        height=500
    )
    fig_comparison.update_layout(xaxis_title="Model Architecture", yaxis_title="Score")
    st.plotly_chart(fig_comparison, use_container_width=True)

    st.markdown("---")
    st.write("#### 📈 Metric Trends (Line Chart)")
    # Line chart showing how each metric performs across models
    fig_line = px.line(
        performance_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score'),
        x='index',
        y='Score',
        color='Metric',
        markers=True,
        title="Simulated Metric Performance Across Models",
        labels={'index': 'Model', 'Score': 'Score'}
    )
    fig_line.update_layout(xaxis_title="Model Architecture", yaxis_title="Score")
    st.plotly_chart(fig_line, use_container_width=True)

def generate_report(analysis_data):
    """Generates a detailed PDF-like report from analysis data."""
    report_content = io.StringIO()
    
    report_content.write(f"# Medical Imaging Analysis Report\n")
    report_content.write(f"### Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report_content.write("---")
    report_content.write("## Patient & Scan Information\n")
    report_content.write(f"**Scan Filename:** {analysis_data['filename']}\n")
    report_content.write(f"**Analysis Timestamp:** {analysis_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report_content.write("\n---\n")
    report_content.write("## AI Prediction Summary\n")
    
    ensemble_scores = analysis_data['ensemble']
    predicted_class = max(ensemble_scores, key=ensemble_scores.get)
    confidence = ensemble_scores[predicted_class]
    uncertainty_metrics = analysis_data['uncertainty']
    
    report_content.write(f"**Primary Prediction:** **{predicted_class}**\n")
    report_content.write(f"**Ensemble Confidence:** {confidence:.2%}\n")
    report_content.write(f"**Prediction Uncertainty (Entropy):** {uncertainty_metrics['entropy']:.3f}\n")
    
    if predicted_class != "No Tumor":
        characteristics = AdvancedBrainTumorDetector().tumor_characteristics[predicted_class]
        report_content.write(f"\n### {predicted_class} Characteristics:\n")
        report_content.write(f"- **Description:** {characteristics['description']}\n")
        report_content.write(f"- **Typical Location:** {characteristics['typical_location']}\n")
        report_content.write(f"- **Prognosis:** {characteristics['prognosis']}\n")
        report_content.write(f"- **Treatment:** {characteristics['treatment']}\n")
    else:
        report_content.write("\nNo tumor detected. Normal brain tissue identified.\n")

    report_content.write("\n---\n")
    report_content.write("## Detailed Prediction Scores (Ensemble)\n")
    for tumor_type, score in ensemble_scores.items():
        report_content.write(f"- **{tumor_type}:** {score:.2%}\n")

    report_content.write("\n---\n")
    report_content.write("## Image Feature Analysis\n")
    features = analysis_data['features']
    for feature_name, value in features.items():
        if isinstance(value, (int, float)):
            report_content.write(f"- **{feature_name.replace('_', ' ').title()}:** {value:.2f}\n")
        else:
            report_content.write(f"- **{feature_name.replace('_', ' ').title()}:** {value}\n")

    report_content.write("\n---\n")
    report_content.write("## Model-Specific Predictions\n")
    for model_name, preds in analysis_data['predictions'].items():
        report_content.write(f"### {model_name}\n")
        for tumor_type, prob in preds.items():
            report_content.write(f"- {tumor_type}: {prob:.2%}\n")
        report_content.write("\n") # Add a newline after each model's predictions

    report_content.write("\n---\n")
    report_content.write("## Uncertainty Quantification\n")
    report_content.write(f"- **Ensemble Entropy:** {uncertainty_metrics['entropy']:.3f}\n")
    report_content.write(f"- **Mean Model Prediction Variance:** {uncertainty_metrics['mean_variance']:.5f}\n")
    
    report_content.write("\n---")
    report_content.write("\n**DISCLAIMER:** This report is generated by an AI system for research and informational purposes only. It is not intended for clinical diagnosis or as a substitute for professional medical advice. Always consult a qualified healthcare professional for any medical concerns.\n")

    return report_content.getvalue()

def main():
    st.markdown('<h1 class="main-header">🧠 Advanced Brain Tumor Detection System</h1>', unsafe_allow_html=True)
    
    # Enhanced Medical Disclaimer
    st.markdown("""
    <div class="alert-critical">
        <h3>⚠️ CRITICAL MEDICAL DISCLAIMER</h3>
        <p><strong>This application is for educational and research purposes ONLY.</strong></p>
        <ul>
            <li>🚫 NOT for clinical diagnosis or medical decision-making</li>
            <li>👨‍⚕️ Always consult qualified healthcare professionals</li>
            <li>🔬 AI predictions may contain errors - false positives/negatives possible</li>
            <li>🔒 Images processed locally - privacy protected</li>
            <li>📚 Use only for learning and research purposes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize detector
    detector = AdvancedBrainTumorDetector()
    
    # Create tabs for different functionalities
    tabs = st.tabs(["🔍 Analysis", "📊 Performance", "📈 Batch Processing", "🎯 Model Comparison", "📋 Reports"])
    
    with tabs[0]:  # Analysis Tab
        st.header("Single Image Analysis")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📁 Upload Brain Scan")
            
            uploaded_file = st.file_uploader(
                "Choose a brain scan image...",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                help="Upload MRI or CT scan images. Supported formats: JPG, PNG, JPEG, BMP, TIFF"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Enhanced image information
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📋 Image Information</h4>
                    <p><strong>Dimensions:</strong> {image.size[0]} × {image.size[1]} pixels</p>
                    <p><strong>Color Mode:</strong> {image.mode}</p>
                    <p><strong>Format:</strong> {image.format}</p>
                    <p><strong>File Size:</strong> {len(uploaded_file.getvalue())} bytes</p>
                    <p><strong>Aspect Ratio:</strong> {image.size[0]/image.size[1]:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Advanced processing options (currently just display, applied implicitly in preprocessing)
                st.subheader("🔧 Processing Options (Simulated)")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.checkbox("Enhanced Contrast", value=True, disabled=True, help="Applied during preprocessing.")
                    st.checkbox("Noise Reduction", value=True, disabled=True, help="Applied during preprocessing.")
                    st.checkbox("Edge Detection", value=True if OPENCV_AVAILABLE else False, disabled=True, help="Applied during preprocessing if OpenCV available.")
                
                with col_b:
                    st.checkbox("Morphological Operations", value=True if OPENCV_AVAILABLE else False, disabled=True, help="Applied during preprocessing if OpenCV available.")
                    st.checkbox("Texture Analysis", value=True if ADVANCED_PROCESSING else False, disabled=True, help="Applied during preprocessing if scikit-image/scipy available.")
                    st.checkbox("Uncertainty Quantification", value=True, disabled=True, help="Always calculated for ensemble predictions.")
                
                # Process button
                if st.button("🔍 Run Advanced Analysis", type="primary"):
                    with st.spinner("Running advanced AI analysis..."):
                        progress_bar = st.progress(0)
                        
                        # Advanced preprocessing
                        progress_bar.progress(20, text="Preprocessing image...")
                        processed_results = detector.advanced_preprocessing(image)
                        
                        # Feature extraction
                        progress_bar.progress(40, text="Extracting features...")
                        features = detector.extract_features(processed_results['resized'])
                        
                        # Predictions
                        progress_bar.progress(60, text="Simulating deep learning predictions...")
                        predictions = detector.simulate_advanced_prediction(processed_results['resized'], features)
                        ensemble_scores = detector.ensemble_prediction(predictions)
                        
                        # Uncertainty quantification
                        progress_bar.progress(80, text="Quantifying uncertainty...")
                        uncertainty_metrics = detector.uncertainty_quantification(predictions)
                        
                        # Attention maps
                        progress_bar.progress(90, text="Generating attention maps...")
                        attention_maps = detector.generate_detailed_attention_map(processed_results['resized'], ensemble_scores)
                        
                        progress_bar.progress(100, text="Analysis complete!")
                        
                        # Store results
                        st.session_state.processed_images.append({
                            'original': image,
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
                        st.experimental_rerun() # Rerun to display results in col2
            else:
                st.info("Please upload an image to start the analysis.")

        with col2:
            st.subheader("📊 Advanced Analysis Results")
            
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
                        <p>No tumor detected in the scan based on current analysis.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-warning">
                        <h3>⚠️ Prediction: {predicted_class}</h3>
                        <p><strong>Ensemble Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Uncertainty (Entropy):</strong> {uncertainty_metrics['entropy']:.3f}</p>
                        <p>Potential <strong>{predicted_class.lower()}</strong> detected. This requires immediate medical review.</p>
                        <p><strong>Recommendation:</strong> Consult a medical professional immediately with these findings.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Tumor characteristics
                if predicted_class != "No Tumor":
                    characteristics = detector.tumor_characteristics[predicted_class]
                    st.markdown(f"""
                    <div class="feature-box">
                        <h4>🏥 {predicted_class} Characteristics</h4>
                        <p><strong>Description:</strong> {characteristics['description']}</p>
                        <p><strong>Typical Location:</strong> {characteristics['typical_location']}</p>
                        <p><strong>Prognosis:</strong> {characteristics['prognosis']}</p>
                        <p><strong>Treatment:</strong> {characteristics['treatment']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence scores with uncertainty
                st.subheader("📈 Prediction Confidence")
                
                # Create confidence plot with uncertainty bars
                tumor_types = list(ensemble_scores.keys())
                confidences = [ensemble_scores[tt] for tt in tumor_types]
                uncertainties_variance_per_class = uncertainty_metrics['variance_per_class']
                
                fig_confidence = go.Figure()
                
                # Add bars for confidence
                fig_confidence.add_trace(go.Bar(
                    x=tumor_types,
                    y=confidences,
                    # Error bars using standard deviation (sqrt of variance)
                    error_y=dict(type='data', array=np.sqrt(uncertainties_variance_per_class)),
                    marker_color=['#ff7f0e' if t == predicted_class else '#1f77b4' for t in tumor_types],
                    text=[f'{c:.2%}' for c in confidences],
                    textposition='auto',
                    name='Confidence'
                ))
                
                fig_confidence.update_layout(
                    title="Ensemble Predictions with Model Disagreement (Error Bars)",
                    xaxis_title="Tumor Type",
                    yaxis_title="Confidence Score",
                    showlegend=False,
                    height=450,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_confidence, use_container_width=True)
                
                # Feature analysis
                st.subheader("🔍 Image Feature Analysis")
                
                feature_cols = st.columns(3)
                
                with feature_cols[0]:
                    st.metric("Mean Intensity", f"{features.get('mean_intensity', 0):.2f}")
                    st.metric("Std Intensity", f"{features.get('std_intensity', 0):.2f}")
                
                with feature_cols[1]:
                    st.metric("Hist. Entropy", f"{features.get('histogram_entropy', 0):.3f}")
                    st.metric("Hist. Variance", f"{features.get('histogram_variance', 0):.2f}")
                
                with feature_cols[2]:
                    if 'largest_region_area' in features:
                        st.metric("Largest Region Area", f"{features.get('largest_region_area', 0):.0f}")
                        st.metric("Region Eccentricity", f"{features.get('largest_region_eccentricity', 0):.3f}")
                        
                st.markdown("---")
                st.subheader("👁️‍🗨️ Attention Maps")
                st.info("These attention maps are simulated to illustrate where a model *might* focus, not actual Grad-CAM from a trained DL model.")
                
                # Display attention maps
                if attention_maps:
                    attention_map_types = list(attention_maps.keys())
                    selected_attention_map = st.selectbox("Select Attention Map Type", attention_map_types)
                    
                    if selected_attention_map:
                        att_map = attention_maps[selected_attention_map]
                        original_image_array = np.array(latest_result['original'].resize(att_map.shape[::-1])) # Resize original to match map
                        
                        # Overlay attention map on original image
                        fig_att = px.imshow(original_image_array, binary_string=True, title=f"Original Image with {selected_attention_map}")
                        
                        # Add heatmap layer (normalized attention map)
                        fig_att.add_trace(go.Heatmap(
                            z=att_map,
                            colorscale='Hot',
                            showscale=True,
                            opacity=0.4 # Adjust opacity for overlay effect
                        ))
                        fig_att.update_layout(coloraxis_showscale=False) # Hide color scale for heatmap
                        fig_att.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
                        st.plotly_chart(fig_att, use_container_width=True)
                        
                        # Show the attention map itself
                        st.image(att_map, caption=f"{selected_attention_map} (Raw)", use_column_width=True, clamp=True)
    
    with tabs[1]:  # Performance Tab
        create_performance_dashboard()
    
    with tabs[2]:  # Batch Processing Tab
        st.subheader("📈 Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple brain scan images for batch processing",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"📁 **{len(uploaded_files)}** files uploaded")
            
            if st.button("🚀 Process All Images in Batch", type="primary"):
                progress_bar_batch = st.progress(0, text="Starting batch processing...")
                batch_results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    file_name = uploaded_file.name
                    with st.spinner(f"Processing {file_name}..."):
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
                                'timestamp': datetime.now()
                            })
                            st.success(f"Processed: {file_name} -> Predicted: **{predicted_class}**")
                        except Exception as e:
                            st.error(f"Error processing {file_name}: {e}")
                            batch_results.append({
                                'filename': file_name,
                                'prediction': "Error",
                                'confidence': 0.0,
                                'uncertainty_entropy': float('inf'),
                                'timestamp': datetime.now()
                            })
                    
                    progress_bar_batch.progress((i + 1) / len(uploaded_files), text=f"Processing {i+1}/{len(uploaded_files)} images...")
                
                progress_bar_batch.empty()
                st.success("✅ Batch processing completed!")
                
                # Display batch results
                st.subheader("📊 Batch Results Summary")
                
                if batch_results:
                    batch_df = pd.DataFrame(batch_results)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Images", len(batch_df))
                    
                    with col2:
                        tumor_count = len(batch_df[batch_df['prediction'] != 'No Tumor'])
                        st.metric("Potential Tumors", tumor_count)
                    
                    with col3:
                        avg_confidence = batch_df['confidence'].mean()
                        st.metric("Avg Confidence (Tumor)", f"{avg_confidence:.2%}")
                    
                    with col4:
                        high_confidence_tumors = len(batch_df[(batch_df['prediction'] != 'No Tumor') & (batch_df['confidence'] > 0.7)])
                        st.metric("High Conf. Tumor", high_confidence_tumors)
                    
                    # Results table
                    st.write("#### Detailed Batch Results Table")
                    st.dataframe(batch_df.set_index('filename'), use_container_width=True)
                    
                    # Visualization of batch results
                    st.write("#### Prediction Distribution")
                    fig_batch_dist = px.histogram(
                        batch_df,
                        x='prediction',
                        color='prediction',
                        title="Distribution of Predictions in Batch",
                        labels={'prediction': 'Predicted Tumor Type', 'count': 'Number of Images'}
                    )
                    st.plotly_chart(fig_batch_dist, use_container_width=True)

                    st.write("#### Confidence vs. Uncertainty")
                    fig_scatter_conf = px.scatter(
                        batch_df,
                        x='confidence',
                        y='uncertainty_entropy',
                        color='prediction',
                        size='confidence', # Size points by confidence
                        hover_name='filename',
                        title="Confidence vs. Uncertainty for Batch Predictions",
                        labels={'confidence': 'Confidence Score', 'uncertainty_entropy': 'Uncertainty (Entropy)'}
                    )
                    st.plotly_chart(fig_scatter_conf, use_container_width=True)
                else:
                    st.info("No results to display for batch processing.")
    
    with tabs[3]:  # Model Comparison Tab
        st.subheader("🎯 Model Comparison for Last Analyzed Image")
        st.info("This section compares the simulated predictions from different model architectures for the **most recently analyzed single image**.")
        
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
            
            st.write("#### Individual Model Confidence Scores")
            st.dataframe(comparison_df.pivot_table(index='Model', columns='Tumor Type', values='Confidence').style.format("{:.3f}"), use_container_width=True)

            # Model agreement analysis
            st.markdown("---")
            st.write("#### 📊 Model Prediction Heatmap")
            
            # Create heatmap of model predictions
            pivot_df = comparison_df.pivot(index='Model', columns='Tumor Type', values='Confidence')
            
            fig_heatmap = px.imshow(
                pivot_df.values,
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale='Plasma', # Changed colormap for variety
                text_auto=True, # Show values on heatmap
                aspect="auto",
                title="Model Prediction Heatmap (Confidence per Class)"
            )
            fig_heatmap.update_layout(xaxis_title="Tumor Type", yaxis_title="Model Architecture")
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Model consensus
            st.markdown("---")
            st.write("#### 🤝 Model Consensus and Disagreement")
            
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

            st.write("Higher Standard Deviation indicates more disagreement among models for that specific tumor type.")

        else:
            st.info("Please run a single image analysis in the 'Analysis' tab first to see model comparisons.")

    with tabs[4]: # Reports Tab
        st.subheader("📋 Generate Analysis Report")
        st.info("Select a previous analysis from the history to generate a detailed report.")

        if st.session_state.processed_images:
            analysis_options = {
                f"{res['filename']} (Analyzed: {res['timestamp'].strftime('%Y-%m-%d %H:%M')})"
                for res in st.session_state.processed_images
            }
            selected_analysis_str = st.selectbox(
                "Choose an analysis to generate a report for:",
                options=list(analysis_options)
            )

            if selected_analysis_str:
                # Find the corresponding analysis data
                selected_analysis_data = next(
                    (res for res in st.session_state.processed_images if f"{res['filename']} (Analyzed: {res['timestamp'].strftime('%Y-%m-%d %H:%M')})" == selected_analysis_str),
                    None
                )

                if selected_analysis_data:
                    if st.button("📄 Generate Report", type="primary"):
                        report_text = generate_report(selected_analysis_data)
                        
                        st.download_button(
                            label="Download Report as Text File",
                            data=report_text,
                            file_name=f"Brain_Tumor_Report_{selected_analysis_data['filename'].split('.')[0]}_{selected_analysis_data['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            help="Downloads a comprehensive text report of the analysis."
                        )
                        st.markdown("---")
                        st.write("#### Preview of Generated Report:")
                        st.text_area("Report Content", report_text, height=500)
                else:
                    st.warning("Selected analysis data not found. Please re-select.")
            else:
                st.info("No analysis selected for report generation.")
        else:
            st.info("No analyses have been performed yet. Please go to the 'Analysis' tab and upload an image first.")

    st.markdown("---")
    st.markdown("Developed with ❤️ for educational and research purposes.")

if __name__ == "__main__":
    main()
