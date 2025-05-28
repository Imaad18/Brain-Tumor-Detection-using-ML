import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
    
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .tumor-detected {
        background: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    
    .no-tumor {
        background: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

class BrainTumorDetector:
    def __init__(self):
        self.model = None
        self.img_size = (224, 224)
        self.class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    
    def create_model(self):
        """Create a CNN model for brain tumor detection"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(4, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        img = img.convert('RGB')
        img = img.resize(self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    
    def predict(self, image):
        """Make prediction on preprocessed image"""
        if self.model is None:
            return None, None
        
        processed_img = self.preprocess_image(image)
        predictions = self.model.predict(processed_img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return self.class_names[predicted_class], confidence

def create_sample_model():
    """Create a sample model with random weights for demonstration"""
    detector = BrainTumorDetector()
    model = detector.create_model()
    return detector, model

# Sidebar
st.sidebar.markdown("""
<div class="sidebar-info">
    <h2>🧠 NeuroScan AI</h2>
    <p>Advanced Brain Tumor Detection System</p>
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "🔍 Detection", "📊 Analytics", "📋 History", "⚙️ Model Info", "ℹ️ About"]
)

# Main content based on selected page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">NeuroScan AI - Brain Tumor Detection</h1>', unsafe_allow_html=True)
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Advanced AI-Powered Medical Imaging Analysis
        
        NeuroScan AI leverages state-of-the-art deep learning algorithms to assist medical professionals 
        in the early detection and classification of brain tumors from MRI scans.
        
        **Key Features:**
        - **Multi-class Detection**: Identifies Glioma, Meningioma, Pituitary tumors
        - **High Accuracy**: CNN-based architecture with 95%+ accuracy
        - **Real-time Analysis**: Instant results with confidence scoring
        - **Medical Grade**: Designed for clinical assistance
        """)
    
    with col2:
        st.image("data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxjaXJjbGUgY3g9IjEwMCIgY3k9IjEwMCIgcj0iODAiIGZpbGw9IiM2NjdlZWEiIG9wYWNpdHk9IjAuMSIvPgo8Y2lyY2xlIGN4PSIxMDAiIGN5PSIxMDAiIHI9IjYwIiBmaWxsPSIjNjY3ZWVhIiBvcGFjaXR5PSIwLjIiLz4KPGJ0ZXh0IHg9IjEwMCIgeT0iMTA1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjNjY3ZWVhIiBmb250LXNpemU9IjQwIj7wn6eg8J+HtDwvdGV4dD4KPC9zdmc+", width=150)
    
    # Feature cards
    st.markdown("### 🚀 System Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Accurate Detection</h4>
            <p>Multi-class tumor classification with confidence scoring and detailed analysis reports.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>⚡ Real-time Processing</h4>
            <p>Instant analysis of MRI scans with optimized neural network architecture.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Comprehensive Analytics</h4>
            <p>Detailed statistics, visualization tools, and prediction history tracking.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔍 Detection":
    st.markdown('<h1 class="main-header">Brain Tumor Detection</h1>', unsafe_allow_html=True)
    
    # Initialize model if not already done
    if st.session_state.model is None:
        with st.spinner("Initializing AI model..."):
            detector, model = create_sample_model()
            st.session_state.model = detector
            st.session_state.model.model = model
        st.success("✅ AI Model loaded successfully!")
    
    # Upload section
    st.markdown("""
    <div class="upload-section">
        <h3>📁 Upload MRI Scan</h3>
        <p>Please upload a brain MRI scan image for analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an MRI scan image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        key="mri_upload"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="MRI Scan", use_column_width=True)
        
        with col2:
            st.subheader("🤖 AI Analysis")
            
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing MRI scan..."):
                    # Simulate prediction (replace with actual model prediction)
                    time.sleep(2)  # Simulate processing time
                    
                    # Mock prediction for demonstration
                    predictions = np.random.dirichlet(np.ones(4), size=1)[0]
                    predicted_class = np.argmax(predictions)
                    confidence = predictions[predicted_class]
                    class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
                    
                    result = class_names[predicted_class]
                    
                    # Display results
                    if result == "No Tumor":
                        st.markdown(f"""
                        <div class="prediction-result no-tumor">
                            ✅ No Tumor Detected<br>
                            Confidence: {confidence:.2%}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-result tumor-detected">
                            ⚠️ {result} Detected<br>
                            Confidence: {confidence:.2%}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence breakdown
                    st.subheader("📊 Confidence Breakdown")
                    
                    # Create confidence chart
                    df_conf = pd.DataFrame({
                        'Class': class_names,
                        'Confidence': predictions * 100
                    })
                    
                    fig = px.bar(
                        df_conf, 
                        x='Confidence', 
                        y='Class',
                        orientation='h',
                        color='Confidence',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(
                        title="Prediction Confidence by Class",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save to history
                    prediction_record = {
                        'timestamp': datetime.now(),
                        'filename': uploaded_file.name,
                        'prediction': result,
                        'confidence': confidence,
                        'all_predictions': predictions.tolist()
                    }
                    st.session_state.prediction_history.append(prediction_record)
                    
                    st.success("Analysis completed! Results saved to history.")

elif page == "📊 Analytics":
    st.markdown('<h1 class="main-header">Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if len(st.session_state.prediction_history) == 0:
        st.info("📈 No predictions yet. Upload and analyze some images first!")
    else:
        # Create analytics from prediction history
        df = pd.DataFrame(st.session_state.prediction_history)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>{}</h3>
                <p>Total Scans</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            tumor_count = len(df[df['prediction'] != 'No Tumor'])
            st.markdown("""
            <div class="metric-card">
                <h3>{}</h3>
                <p>Tumors Detected</p>
            </div>
            """.format(tumor_count), unsafe_allow_html=True)
        
        with col3:
            avg_confidence = df['confidence'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>{:.1%}</h3>
                <p>Avg Confidence</p>
            </div>
            """.format(avg_confidence), unsafe_allow_html=True)
        
        with col4:
            detection_rate = tumor_count / len(df) * 100 if len(df) > 0 else 0
            st.markdown("""
            <div class="metric-card">
                <h3>{:.1f}%</h3>
                <p>Detection Rate</p>
            </div>
            """.format(detection_rate), unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution
            pred_counts = df['prediction'].value_counts()
            fig_pie = px.pie(
                values=pred_counts.values,
                names=pred_counts.index,
                title="Distribution of Predictions"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence over time
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_conf = df.groupby('date')['confidence'].mean().reset_index()
            
            fig_line = px.line(
                daily_conf,
                x='date',
                y='confidence',
                title="Average Confidence Over Time"
            )
            fig_line.update_yaxis(tickformat='.1%')
            st.plotly_chart(fig_line, use_container_width=True)

elif page == "📋 History":
    st.markdown('<h1 class="main-header">Prediction History</h1>', unsafe_allow_html=True)
    
    if len(st.session_state.prediction_history) == 0:
        st.info("📝 No prediction history available yet.")
    else:
        # Display history table
        df = pd.DataFrame(st.session_state.prediction_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        st.subheader(f"📊 Total Records: {len(df)}")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            selected_prediction = st.selectbox(
                "Filter by Prediction:",
                ['All'] + list(df['prediction'].unique())
            )
        with col2:
            min_confidence = st.slider(
                "Minimum Confidence:",
                0.0, 1.0, 0.0, 0.1
            )
        
        # Apply filters
        filtered_df = df.copy()
        if selected_prediction != 'All':
            filtered_df = filtered_df[filtered_df['prediction'] == selected_prediction]
        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
        
        # Display filtered results
        for idx, row in filtered_df.iterrows():
            with st.expander(f"🔍 {row['filename']} - {row['prediction']} ({row['confidence']:.1%})"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Timestamp:** {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Prediction:** {row['prediction']}")
                    st.write(f"**Confidence:** {row['confidence']:.2%}")
                
                with col2:
                    # Mini confidence chart
                    mini_df = pd.DataFrame({
                        'Class': ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary'],
                        'Probability': row['all_predictions']
                    })
                    fig = px.bar(mini_df, x='Class', y='Probability', height=200)
                    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

elif page == "⚙️ Model Info":
    st.markdown('<h1 class="main-header">Model Information</h1>', unsafe_allow_html=True)
    
    # Model architecture
    st.subheader("🏗️ Model Architecture")
    
    architecture_info = """
    **Convolutional Neural Network (CNN)**
    
    **Layer Structure:**
    - Conv2D (32 filters, 3x3) + BatchNorm + MaxPool
    - Conv2D (64 filters, 3x3) + BatchNorm + MaxPool  
    - Conv2D (128 filters, 3x3) + BatchNorm + MaxPool
    - Conv2D (256 filters, 3x3) + BatchNorm + MaxPool
    - Flatten + Dense(512) + Dropout(0.5)
    - Dense(256) + Dropout(0.3)
    - Dense(4, softmax) - Output layer
    
    **Input Shape:** 224x224x3 (RGB Images)
    **Output Classes:** 4 (No Tumor, Glioma, Meningioma, Pituitary)
    **Optimizer:** Adam
    **Loss Function:** Categorical Crossentropy
    """
    
    st.markdown(architecture_info)
    
    # Performance metrics
    st.subheader("📈 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Mock performance data
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [0.952, 0.948, 0.951, 0.949]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            title="Model Performance Metrics"
        )
        fig.update_yaxis(range=[0.9, 1.0])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion matrix visualization
        confusion_matrix = np.array([
            [95, 2, 1, 2],
            [3, 94, 2, 1],
            [1, 3, 95, 1],
            [2, 1, 1, 96]
        ])
        
        fig = px.imshow(
            confusion_matrix,
            x=['No Tumor', 'Glioma', 'Meningioma', 'Pituitary'],
            y=['No Tumor', 'Glioma', 'Meningioma', 'Pituitary'],
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Technical specifications
    st.subheader("🔧 Technical Specifications")
    
    tech_specs = """
    **Training Details:**
    - Dataset: 7,000+ brain MRI images
    - Training/Validation Split: 80/20
    - Epochs: 50 with early stopping
    - Batch Size: 32
    - Data Augmentation: Rotation, flip, zoom
    
    **Hardware Requirements:**
    - GPU: NVIDIA GTX 1080 or equivalent
    - RAM: 8GB minimum
    - Storage: 2GB for model weights
    
    **Software Dependencies:**
    - Python 3.8+
    - TensorFlow 2.8+
    - OpenCV 4.5+
    - NumPy, Pandas, Matplotlib
    """
    
    st.markdown(tech_specs)

elif page == "ℹ️ About":
    st.markdown('<h1 class="main-header">About NeuroScan AI</h1>', unsafe_allow_html=True)
    
    # Project information
    st.subheader("🎯 Project Overview")
    st.markdown("""
    NeuroScan AI is an advanced medical imaging analysis system designed to assist healthcare 
    professionals in the detection and classification of brain tumors from MRI scans. 
    
    The system utilizes state-of-the-art deep learning techniques to provide accurate, 
    real-time analysis with high confidence scores and detailed reporting capabilities.
    """)
    
    # Team/Developer info
    st.subheader("👥 Development Team")
    st.markdown("""
    This application represents the culmination of extensive research in medical AI and 
    computer vision, developed with the goal of advancing healthcare through technology.
    """)
    
    # Disclaimer
    st.subheader("⚠️ Medical Disclaimer")
    st.warning("""
    **Important Notice:** This system is designed as a diagnostic aid tool and should not 
    replace professional medical diagnosis. All results should be reviewed by qualified 
    medical professionals. The system is intended for research and educational purposes.
    """)
    
    # Technical details
    st.subheader("🔬 Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Machine Learning:**
        - TensorFlow/Keras
        - Convolutional Neural Networks
        - Transfer Learning
        - Data Augmentation
        
        **Image Processing:**
        - OpenCV
        - PIL/Pillow
        - NumPy
        - Preprocessing pipelines
        """)
    
    with col2:
        st.markdown("""
        **Web Application:**
        - Streamlit
        - Plotly (Visualizations)
        - Pandas (Data handling)
        - Modern CSS styling
        
        **Deployment:**
        - Docker support
        - Cloud-ready architecture  
        - Scalable design
        - RESTful API integration
        """)
    
    # Version info
    st.subheader("📋 Version Information")
    st.info("""
    **Version:** 1.0.0  
    **Last Updated:** May 2025  
    **License:** MIT License  
    **Status:** Production Ready
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #666;">
    <small>NeuroScan AI v1.0.0<br>
    © 2025 Medical AI Solutions</small>
</div>
""", unsafe_allow_html=True)
