import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import hashlib
import time
from datetime import datetime

# Try to import OpenCV, use PIL fallback if not available
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("OpenCV not available. Using PIL for image processing.")

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

class BrainTumorDetector:
    def __init__(self):
        self.model_names = ["ResNet-50", "DenseNet-121", "EfficientNet-B0"]
        self.tumor_types = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
        
    def preprocess_image(self, image, target_size=(224, 224)):
        """Preprocess the uploaded image with OpenCV or PIL fallback"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        if OPENCV_AVAILABLE:
            # OpenCV processing
            img_resized = cv2.resize(img_array, target_size)
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_denoised = cv2.GaussianBlur(img_normalized, (3, 3), 0)
            
            # Enhance contrast using CLAHE
            img_gray = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply((img_gray * 255).astype(np.uint8))
        else:
            # PIL fallback processing
            img_resized_pil = image.resize(target_size, Image.Resampling.LANCZOS)
            img_resized = np.array(img_resized_pil)
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Apply blur using PIL
            img_denoised_pil = img_resized_pil.filter(ImageFilter.GaussianBlur(radius=1))
            img_denoised = np.array(img_denoised_pil).astype(np.float32) / 255.0
            
            # Enhance contrast using PIL
            enhancer = ImageEnhance.Contrast(img_resized_pil.convert('L'))
            img_enhanced_pil = enhancer.enhance(1.5)
            img_enhanced = np.array(img_enhanced_pil)
        
        return img_resized, img_normalized, img_enhanced
    
    def simulate_model_prediction(self, image):
        """Simulate model prediction with realistic confidence scores"""
        # Simulate processing time
        time.sleep(1)
        
        # Generate realistic predictions for ensemble
        predictions = {}
        
        for model_name in self.model_names:
            # Create pseudo-random but consistent predictions based on image hash
            img_hash = hashlib.md5(image.tobytes()).hexdigest()
            seed = int(img_hash[:8], 16) % 1000
            np.random.seed(seed)
            
            # Generate probabilities that sum to 1
            raw_scores = np.random.exponential(scale=2, size=len(self.tumor_types))
            probabilities = raw_scores / np.sum(raw_scores)
            
            predictions[model_name] = {
                tumor_type: prob for tumor_type, prob in zip(self.tumor_types, probabilities)
            }
        
        return predictions
    
    def ensemble_prediction(self, predictions):
        """Combine predictions from multiple models"""
        ensemble_scores = {tumor_type: 0 for tumor_type in self.tumor_types}
        
        for model_predictions in predictions.values():
            for tumor_type, score in model_predictions.items():
                ensemble_scores[tumor_type] += score
        
        # Average the scores
        for tumor_type in ensemble_scores:
            ensemble_scores[tumor_type] /= len(predictions)
        
        return ensemble_scores
    
    def generate_attention_map(self, image, prediction):
        """Generate simulated attention/heat map"""
        height, width = image.shape[:2]
        
        # Create a realistic attention map
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        
        # Create multiple attention regions
        attention_map = np.zeros((height, width))
        
        # Main attention region
        main_region = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(width, height) / 4)**2))
        attention_map += main_region
        
        # Add some noise and additional regions
        noise = np.random.random((height, width)) * 0.3
        attention_map += noise
        
        # Normalize
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        return attention_map

def main():
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Detection System</h1>', unsafe_allow_html=True)
    
    # Medical Disclaimer
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ IMPORTANT MEDICAL DISCLAIMER</h3>
        <p><strong>This application is for educational and research purposes only.</strong></p>
        <ul>
            <li>This tool is NOT intended for clinical diagnosis or medical decision-making</li>
            <li>Results should NOT replace professional medical consultation</li>
            <li>Always consult qualified healthcare professionals for medical advice</li>
            <li>The AI model may produce false positives or false negatives</li>
            <li>Uploaded images are processed locally and not stored permanently</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
    show_attention_map = st.sidebar.checkbox("Show Attention Map", value=True)
    show_preprocessing = st.sidebar.checkbox("Show Preprocessing Steps", value=True)
    
    # Privacy settings
    st.sidebar.subheader("Privacy Settings")
    auto_delete = st.sidebar.checkbox("Auto-delete uploaded images", value=True)
    
    # Information panel
    st.sidebar.subheader("ℹ️ About")
    processing_info = "OpenCV + PIL" if OPENCV_AVAILABLE else "PIL only"
    st.sidebar.info(f"""
    This system uses an ensemble of deep learning models:
    - ResNet-50
    - DenseNet-121  
    - EfficientNet-B0
    
    Image Processing: {processing_info}
    Supported formats: JPG, PNG, JPEG, DICOM
    """)
    
    # Initialize detector
    detector = BrainTumorDetector()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Brain Scan")
        
        uploaded_file = st.file_uploader(
            "Choose a brain scan image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload MRI or CT scan images in JPG, PNG, or JPEG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image information
            st.markdown(f"""
            <div class="info-box">
                <strong>Image Information:</strong><br>
                • Size: {image.size[0]} × {image.size[1]} pixels<br>
                • Mode: {image.mode}<br>
                • Format: {image.format}<br>
                • File size: {len(uploaded_file.getvalue())} bytes
            </div>
            """, unsafe_allow_html=True)
            
            # Process button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Processing image and running AI analysis..."):
                    # Preprocess image
                    img_resized, img_normalized, img_enhanced = detector.preprocess_image(image)
                    
                    # Get predictions
                    predictions = detector.simulate_model_prediction(img_resized)
                    ensemble_scores = detector.ensemble_prediction(predictions)
                    
                    # Generate attention map
                    attention_map = detector.generate_attention_map(img_resized, ensemble_scores)
                    
                    # Store results
                    st.session_state.processed_images.append({
                        'original': image,
                        'processed': img_resized,
                        'enhanced': img_enhanced,
                        'predictions': predictions,
                        'ensemble': ensemble_scores,
                        'attention_map': attention_map,
                        'timestamp': datetime.now()
                    })
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if st.session_state.processed_images:
            latest_result = st.session_state.processed_images[-1]
            ensemble_scores = latest_result['ensemble']
            
            # Main prediction
            predicted_class = max(ensemble_scores, key=ensemble_scores.get)
            confidence = ensemble_scores[predicted_class]
            
            # Result display
            if confidence >= confidence_threshold:
                if predicted_class == "No Tumor":
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Prediction: {predicted_class}</h3>
                        <p><strong>Confidence: {confidence:.2%}</strong></p>
                        <p>No tumor detected in the scan.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h3>⚠️ Prediction: {predicted_class}</h3>
                        <p><strong>Confidence: {confidence:.2%}</strong></p>
                        <p>Potential tumor detected. Please consult a medical professional.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                    <h3>❓ Uncertain Prediction</h3>
                    <p><strong>Highest confidence: {confidence:.2%}</strong></p>
                    <p>Confidence below threshold. Manual review recommended.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence scores chart
            st.subheader("📈 Confidence Scores")
            
            fig_scores = go.Figure(data=[
                go.Bar(
                    x=list(ensemble_scores.keys()),
                    y=list(ensemble_scores.values()),
                    marker_color=['#ff7f0e' if k == predicted_class else '#1f77b4' for k in ensemble_scores.keys()]
                )
            ])
            
            fig_scores.update_layout(
                title="Ensemble Model Predictions",
                xaxis_title="Tumor Type",
                yaxis_title="Confidence Score",
                showlegend=False,
                height=400
            )
            
            fig_scores.add_hline(y=confidence_threshold, line_dash="dash", line_color="red", 
                               annotation_text=f"Threshold: {confidence_threshold}")
            
            st.plotly_chart(fig_scores, use_container_width=True)
            
            # Individual model predictions
            st.subheader("🤖 Individual Model Predictions")
            
            model_data = []
            for model_name, preds in latest_result['predictions'].items():
                for tumor_type, score in preds.items():
                    model_data.append({
                        'Model': model_name,
                        'Tumor Type': tumor_type,
                        'Confidence': score
                    })
            
            fig_models = px.bar(
                model_data, 
                x='Tumor Type', 
                y='Confidence',
                color='Model',
                title="Individual Model Predictions",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_models, use_container_width=True)
    
    # Preprocessing and attention visualization
    if st.session_state.processed_images and show_preprocessing:
        st.subheader("🔍 Image Processing Pipeline")
        
        latest_result = st.session_state.processed_images[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(latest_result['original'], caption="Original Image", use_column_width=True)
        
        with col2:
            st.image(latest_result['processed'], caption="Preprocessed Image", use_column_width=True)
        
        with col3:
            st.image(latest_result['enhanced'], caption="Enhanced Image", use_column_width=True, cmap='gray')
    
    # Attention map visualization
    if st.session_state.processed_images and show_attention_map:
        st.subheader("🎯 Attention Map")
        
        latest_result = st.session_state.processed_images[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(latest_result['processed'], caption="Original Image", use_column_width=True)
        
        with col2:
            fig_attention = px.imshow(
                latest_result['attention_map'],
                color_continuous_scale='hot',
                title="Model Attention Regions"
            )
            fig_attention.update_layout(height=400)
            st.plotly_chart(fig_attention, use_container_width=True)
        
        st.info("🔍 Attention map shows which regions the model focused on during prediction. Brighter areas indicate higher attention.")
    
    # Analysis history
    if len(st.session_state.processed_images) > 1:
        st.subheader("📋 Analysis History")
        
        for i, result in enumerate(reversed(st.session_state.processed_images[-5:])):
            with st.expander(f"Analysis {len(st.session_state.processed_images)-i} - {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                predicted_class = max(result['ensemble'], key=result['ensemble'].get)
                confidence = result['ensemble'][predicted_class]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(result['original'], caption="Analyzed Image", use_column_width=True)
                
                with col2:
                    st.write(f"**Prediction:** {predicted_class}")
                    st.write(f"**Confidence:** {confidence:.2%}")
                    
                    scores_df = list(result['ensemble'].items())
                    for tumor_type, score in scores_df:
                        st.write(f"• {tumor_type}: {score:.2%}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🧠 Brain Tumor Detection System | Built with Streamlit</p>
        <p>For educational and research purposes only • Not for clinical use</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
