from logger import Logger
import streamlit as st 
import warnings
import numpy as np
from datetime import datetime
import time
from PIL import Image, ImageOps
import tensorflow as tf
from pymongo import MongoClient
import os
from dotenv import load_dotenv, find_dotenv
from keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from huggingface_hub import hf_hub_download
import requests
from pandas import DataFrame
import base64
from io import BytesIO

#Ignore the warnings 
warnings.filterwarnings("ignore")

load_dotenv(find_dotenv())

logger = Logger(name= "skin_disease")
logger.set_console_output(enabled= True)


user_name = os.getenv('HUGGINGFACE_USERNAME')
repository = os.getenv('HUGGINGFACE_REPO')

class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        super().__init__(*args, **kwargs)

@st.cache_resource(show_spinner="Loading AI model...")
def huggingface_load():
    """Load model from Huggingface. Cached using st.cache_resource to load only once."""
    try:
        # Download the model file from Hugging Face Hub
        model_path = hf_hub_download(
            repo_id=f"{user_name}/{repository}",
            filename="SDN5.h5",  
            repo_type="model"
        )
        logger.info(f"Model downloaded successfully from Hugging Face Hub: {model_path}")
        
        # Load the model with custom objects
        with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': PatchedDepthwiseConv2D}):
            model = load_model(model_path)
            
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        error_msg = str(e) if hasattr(e, '__str__') else "Unknown error occurred"
        logger.error(f"Error loading model from Hugging Face Hub: {error_msg}")
        
        # Fallback to local model if Hugging Face fails
        logger.info("Attempting to load model from local storage")
        try:
            with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': PatchedDepthwiseConv2D}):
                model = load_model("F:/Hitayu-PS1/SDN5/SDN5.h5")
            logger.info("Local model loaded successfully")
            return model
        except Exception as inner_e:
            error_msg = str(inner_e) if hasattr(inner_e, '__str__') else "Unknown error occurred"
            logger.error(f"Error loading local model: {error_msg}")
            # Wrap the error in a meaningful exception
            raise RuntimeError(f"Failed to load model: {error_msg}") from inner_e


logger.info("Started skin disease diagnosis")

# Initialize session state variables if they don't exist
if 'show_feedback_form' not in st.session_state:
    st.session_state.show_feedback_form = False
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'show_detailed_report' not in st.session_state:
    st.session_state.show_detailed_report = False

# Configure page
st.set_page_config(
    page_title="Skin Disease Diagnosis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Skin disease classification classes (matching your model's classes)
SKIN_DISEASE_CLASSES = [
    "Acne",
    "Eczema", 
    "Psoriasis",
    "FU-ringworm",
    "BA- cellulitis",
    "BA-impetigo",
    "Warts",
    "Lupus",
    "SkinCancer",
    "chickenpox"
]

# Custom CSS for modern, aesthetic theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles - Apply to all Streamlit containers */
    .stApp {
        background: 
            linear-gradient(135deg, rgba(15, 15, 35, 0.85) 0%, rgba(26, 26, 46, 0.85) 50%, rgba(22, 33, 62, 0.85) 100%),
            url('https://t4.ftcdn.net/jpg/14/27/04/15/360_F_1427041530_oayZEPrwEpm37Erazv441N3cvOo6G6BH.jpg') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        min-height: 100vh !important;
    }
    
    .main {
        font-family: 'Inter', sans-serif;
        background: 
            linear-gradient(135deg, rgba(15, 15, 35, 0.85) 0%, rgba(26, 26, 46, 0.85) 50%, rgba(22, 33, 62, 0.85) 100%),
            url('https://t4.ftcdn.net/jpg/14/27/04/15/360_F_1427041530_oayZEPrwEpm37Erazv441N3cvOo6G6BH.jpg') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        color: #e0e6ed;
        min-height: 100vh !important;
        position: relative;
    }
    
    /* Apply background to body and html */
    html, body {
        background: 
            linear-gradient(135deg, rgba(15, 15, 35, 0.85) 0%, rgba(26, 26, 46, 0.85) 50%, rgba(22, 33, 62, 0.85) 100%),
            url('https://t4.ftcdn.net/jpg/14/27/04/15/360_F_1427041530_oayZEPrwEpm37Erazv441N3cvOo6G6BH.jpg') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        min-height: 100vh !important;
    }
    
    /* Background blur overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            url('https://t4.ftcdn.net/jpg/14/27/04/15/360_F_1427041530_oayZEPrwEpm37Erazv441N3cvOo6G6BH.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        filter: blur(8px);
        z-index: -2;
    }
    
    /* Dark overlay for better readability */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.8) 0%, rgba(26, 26, 46, 0.8) 50%, rgba(22, 33, 62, 0.8) 100%);
        z-index: -1;
    }
    
    /* Ensure all content containers have transparent backgrounds */
    .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
    }
    
    /* Make sure sidebar doesn't interfere */
    .css-1d391kg {
        background: transparent !important;
    }
    
    /* Header Styling */
    .main-header {
        background: rgba(128, 128, 128, 0.2);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f2f6;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card Styling */
    .card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 15px 35px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2),
            0 0 0 1px rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(102, 126, 234, 0.6) 25%, 
            rgba(118, 75, 162, 0.8) 50%, 
            rgba(240, 147, 251, 0.6) 75%, 
            transparent 100%);
        opacity: 0.8;
    }
    
    .card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.1), 
            transparent);
        transition: left 0.8s ease;
        pointer-events: none;
    }
    
    .card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 25px 50px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.1),
            0 0 30px rgba(102, 126, 234, 0.2);
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    .card:hover::after {
        left: 100%;
    }
    
    .card-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .card-header::before {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 1px;
    }
    
    .card-header .emoji {
        font-size: 1.8rem;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Enhanced form sections within cards */
    .card .stMarkdown h3,
    .card .stMarkdown h4,
    .card strong {
        color: rgba(255, 255, 255, 0.95) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
        font-weight: 500 !important;
    }
    
    /* Section dividers within cards */
    .card .stMarkdown strong::after {
        content: '';
        display: block;
        width: 40px;
        height: 1px;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.6), transparent);
        margin-top: 0.3rem;
    }
    
    /* Form Styling */
    .stSelectbox label, .stTextInput label, .stDateInput label, .stNumberInput label, .stTextArea label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stSelectbox > div > div, .stTextInput > div > div, .stDateInput > div > div, 
    .stNumberInput > div > div, .stTextArea > div > div {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        color: white !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stSelectbox > div > div:focus, .stTextInput > div > div:focus, 
    .stDateInput > div > div:focus, .stNumberInput > div > div:focus, .stTextArea > div > div:focus {
        border-color: #f093fb !important;
        box-shadow: 0 0 20px rgba(240, 147, 251, 0.4) !important;
        transform: scale(1.02) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 3rem;
        font-weight: 600;
        font-size: 1.2rem;
        letter-spacing: 1px;
        transition: all 0.4s ease;
        box-shadow: 
            0 8px 25px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 
            0 15px 35px rgba(102, 126, 234, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }
    
    /* File Uploader */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 3px dashed rgba(255, 255, 255, 0.5) !important;
        border-radius: 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .uploadedFile:hover {
        border-color: #f093fb !important;
        background: rgba(255, 255, 255, 0.15) !important;
        transform: scale(1.02) !important;
    }
    
    /* Progress Bar */
    .stProgress .st-bo {
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    .stProgress .st-bp {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb) !important;
        border-radius: 10px !important;
        animation: progressGlow 2s ease-in-out infinite alternate !important;
    }
    
    @keyframes progressGlow {
        0% { box-shadow: 0 0 10px rgba(240, 147, 251, 0.5); }
        100% { box-shadow: 0 0 20px rgba(240, 147, 251, 0.8); }
    }
    
    /* Results Section */
    .result-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(240, 147, 251, 0.1));
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 25px;
        padding: 3rem;
        margin: 2rem 0;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        animation: resultShimmer 3s ease-in-out infinite;
    }
    
    @keyframes resultShimmer {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
    }
    
    .diagnosis-result {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: 1px;
    }
    
    .confidence-score {
        font-size: 1.5rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Metrics Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Image Display */
    .stImage > div {
        border-radius: 20px !important;
        overflow: hidden !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stImage > div:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Success/Error Messages */
    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #f093fb);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .card-header {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def generate_detailed_report(disease_info, predicted_class):
    """
    Generate a detailed medical report using Streamlit native components
    """
    logger.info(f"Generating styled report for: {predicted_class}")
    
    # Get disease information
    disease_name = disease_info.get('disease_name', predicted_class)
    description = disease_info.get('description', 'No description available.')
    entry_to_body = disease_info.get('entry_to_body', 'Development mechanism not specified.')
    spread = disease_info.get('spread', 'Transmission information not available.')
    
    def render_medical_report(disease_name: str, description: str, entry_to_body: str, spread: str) -> None:
        """Renders a styled medical report with disease information using glassmorphism design."""

        # Add custom CSS for styling with dark theme and blur effects
        st.markdown("""
        <style>
            .medical-report-container {
                background: linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
                backdrop-filter: blur(25px);
                border: 2px solid rgba(255, 255, 255, 0.2);
                border-radius: 25px;
                padding: 2.5rem;
                margin: 2rem 0;
                box-shadow: 
                    0 20px 40px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2),
                    0 0 0 1px rgba(255, 255, 255, 0.05);
                position: relative;
                overflow: hidden;
            }
            
            .medical-report-container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
                animation: reportHeaderShimmer 3s ease-in-out infinite;
            }
            
            @keyframes reportHeaderShimmer {
                0%, 100% { opacity: 0.8; }
                50% { opacity: 1; }
            }
            
            .medical-report-header {
                text-align: center;
                margin-bottom: 2.5rem;
                padding: 2rem;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.08));
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                box-shadow: 
                    0 10px 25px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
                position: relative;
                overflow: hidden;
            }
            
            .medical-report-header::after {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, 
                    transparent, 
                    rgba(255, 255, 255, 0.1), 
                    transparent);
                transition: left 0.8s ease;
                pointer-events: none;
            }
            
            .medical-report-header:hover::after {
                left: 100%;
            }
            
            .medical-report-title {
                font-size: 2.2rem;
                color: #ffffff;
                margin-bottom: 0.8rem;
                font-weight: 800;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                letter-spacing: 1px;
            }
            
            .medical-report-subtitle {
                font-size: 1.1rem;
                color: rgba(255, 255, 255, 0.8);
                margin: 0;
                font-weight: 500;
            }
            
            .medical-report-section {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.06));
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.15);
                padding: 2rem;
                border-radius: 18px;
                margin-bottom: 1.8rem;
                box-shadow: 
                    0 12px 28px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
                border-left: 4px solid;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            
            .medical-report-section::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, 
                    transparent, 
                    rgba(255, 255, 255, 0.08), 
                    transparent);
                transition: left 0.6s ease;
                pointer-events: none;
            }
            
            .medical-report-section:hover {
                transform: translateY(-5px) scale(1.01);
                box-shadow: 
                    0 18px 35px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.25),
                    0 0 0 1px rgba(255, 255, 255, 0.1);
                border-color: rgba(255, 255, 255, 0.25);
            }
            
            .medical-report-section:hover::before {
                left: 100%;
            }
            
            .medical-section-title {
                font-size: 1.4rem;
                font-weight: 700;
                margin-bottom: 1.2rem;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.8rem;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
                position: relative;
                padding-bottom: 0.5rem;
            }
            
            .medical-section-title::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 50px;
                height: 2px;
                border-radius: 1px;
                opacity: 0.8;
            }
            
            .medical-section-content {
                font-size: 1.05rem;
                line-height: 1.7;
                color: rgba(255, 255, 255, 0.9);
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                font-weight: 400;
            }
            
            .medical-section-icon {
                font-size: 1.6rem;
                filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
                animation: iconPulse 2.5s ease-in-out infinite;
            }
            
            @keyframes iconPulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }
            
            /* Color variations for different sections */
            .section-description {
                border-left-color: #4ecdc4;
            }
            .section-description .medical-section-title {
                color: #4ecdc4;
            }
            .section-description .medical-section-title::after {
                background: linear-gradient(90deg, #4ecdc4, transparent);
            }
            
            .section-development {
                border-left-color: #ff6b6b;
            }
            .section-development .medical-section-title {
                color: #ff6b6b;
            }
            .section-development .medical-section-title::after {
                background: linear-gradient(90deg, #ff6b6b, transparent);
            }
            
            .section-transmission {
                border-left-color: #ffa500;
            }
            .section-transmission .medical-section-title {
                color: #ffa500;
            }
            .section-transmission .medical-section-title::after {
                background: linear-gradient(90deg, #ffa500, transparent);
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                .medical-report-container {
                    padding: 1.5rem;
                    margin: 1rem 0;
                }
                
                .medical-report-header {
                    padding: 1.5rem;
                }
                
                .medical-report-title {
                    font-size: 1.8rem;
                }
                
                .medical-report-section {
                    padding: 1.5rem;
                }
                
                .medical-section-title {
                    font-size: 1.2rem;
                }
            }
        </style>
        """, unsafe_allow_html=True)

        # Define section configurations with enhanced styling
        sections = [
            {
                "title": "Medical Description",
                "content": description,
                "icon": "🏥",
                "class": "section-description"
            },
            {
                "title": "How It Develops", 
                "content": entry_to_body,
                "icon": "🔬",
                "class": "section-development"
            },
            {
                "title": "Transmission Information",
                "content": spread,
                "icon": "🦠",
                "class": "section-transmission"
            }
        ]

        # Render report container and header
        st.markdown(f"""
        <div class="medical-report-container">
            <div class="medical-report-header">
                <div class="medical-report-title">
                    📋 Medical Report: {disease_name}
                </div>
                <div class="medical-report-subtitle">
                    Comprehensive Medical Information & Analysis
                </div>
            </div>
        """, unsafe_allow_html=True)       

        # Render sections dynamically with enhanced styling and CENTERED content
        for section in sections:
            st.markdown(f"""
            <div class="medical-report-section {section['class']}">
                <div class="medical-section-title">
                    <span class="medical-section-icon">{section['icon']}</span>
                    {section['title']}
                </div>
                <div class="medical-section-content">
            """, unsafe_allow_html=True)
            
            # Enhanced content rendering with CENTERED typography and styling
            st.markdown(f"""
            <div style="
                font-size: 1.2rem;
                line-height: 1.8;
                color: rgba(255, 255, 255, 0.95);
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
                font-weight: 400;
                text-align: center;
                padding: 0.5rem 0;
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
                border-radius: 12px;
                padding: 1.5rem;
                margin: 0.5rem 0;
                border-left: 3px solid rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                box-shadow: 
                    0 4px 15px rgba(0, 0, 0, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 80px;
            ">
                {section['content']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)

        # Close report container
        st.markdown('</div>', unsafe_allow_html=True)

        # Close report container
        st.markdown('</div>', unsafe_allow_html=True)

    # Execute the function
    render_medical_report(disease_name, description, entry_to_body, spread)
    
    
    # Prevention section
    if disease_info.get('prevention'):
        prevention_items = '<br>'.join([f"• {prevention}" for prevention in disease_info['prevention']])
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(76, 175, 80, 0.08));
            backdrop-filter: blur(20px);
            border: 2px solid rgba(76, 175, 80, 0.3);
            border-radius: 25px;
            padding: 2.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        ">
            <h3 style="
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
                text-align: center;
                color: #4CAF50;
                margin-top: 0;
            ">How To Stay Safe</h3>
            <div style="
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.1rem;
                line-height: 1.8;
            ">{prevention_items}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Complications section
    if disease_info.get('complications'):
        complications_items = '<br>'.join([f"• {complication}" for complication in disease_info['complications']])
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(244, 67, 54, 0.15), rgba(244, 67, 54, 0.08));
            backdrop-filter: blur(20px);
            border: 2px solid rgba(244, 67, 54, 0.3);
            border-radius: 25px;
            padding: 2.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        ">
            <h3 style="
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
                text-align: center;
                color: #f44336;
                margin-top: 0;
            ">Things That Might Happen</h3>
            <div style="
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.1rem;
                line-height: 1.8;
            ">{complications_items}</div>
        </div>
        """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def preprocess_image(_image):
    """
    Preprocess image for model prediction. Cached to avoid reprocessing the same image.
    
    Args:
        image: PIL Image object
    Returns:
        tuple: preprocessed image array and original resized image
    """
    # Resize and preprocess image
    processed_image = ImageOps.fit(image=_image, size=(224,224), method=Image.Resampling.LANCZOS)
    image_array = np.asarray(processed_image)
    normalized_array = (image_array.astype(np.float32) / 127.5) - 1
    
    return np.array([normalized_array]), processed_image

@st.cache_data(show_spinner="Analyzing image...")
def load_labels():
    """Load and cache the labels file"""
    return open("E:/Hitayu-PS1/SDN5/sdn_labels.txt").readlines()

def predict_skin_disease(image) -> dict:
    """
    Predict skin disease type from uploaded image
    
    Args:
        image: PIL Image object
    Returns:
        dict: Prediction results with class and confidence
    """
    try:
        logger.info("Processing skin disease prediction")
        
        logger.info("Processing and analyzing image")
        
        # Get preprocessed image (cached)
        skin_data, processed_image = preprocess_image(image)
        
        # Load model (cached)
        skin_model = huggingface_load()
        
        # Load labels (cached)
        skin_labels = load_labels()

        logger.info('Making prediction initiated')
        prediction = skin_model.predict(skin_data)
        index = np.argmax(prediction)
        class_name = skin_labels[index]
        confidence_score = prediction[0][index]

        logger.info('Making prediction finished')

        return {
            "predicted_class": class_name[2:],
            "confidence_score": confidence_score,
            "all_predictions": prediction
        }
        
    except Exception as e:
        logger.error(f"Error in skin disease prediction: {str(e)}")
        return {
            "predicted_class": "Unknown",
            "confidence_score": 0.0,
            "error": str(e)
        }

def update_record_with_feedback(record_id: str, feedback_data: dict) -> dict:
    """
    Update an existing patient record with feedback data
    
    Args:
        record_id: The ID of the record to update
        feedback_data: The feedback data to add to the record
    Returns:
        dict: Update operation result
    """
    try:
        logger.info(f"Attempting to update record {record_id} with feedback")
        
        # Get environment variables
        mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
        mongo_database_name = os.getenv("MONGO_DATABASE_NAME", "skin_disease_db")
        collection_name = os.getenv("COLLECTION_NAME", "patient_diagnoses")
        
        if not mongo_connection_string:
            logger.warn("MongoDB connection not configured")
            return {
                "success": False,
                "message": "Database connection not available"
            }
        
        # Validate inputs
        if not record_id or not feedback_data:
            return {
                "success": False,
                "message": "Invalid record ID or feedback data"
            }
            
        # Connect to MongoDB
        client = MongoClient(mongo_connection_string, serverSelectionTimeoutMS=10000)
        database = client[mongo_database_name]
        collection = database[collection_name]
        
        # Test connection
        client.server_info()
        
        # Convert string ID to ObjectId if necessary
        from bson.objectid import ObjectId
        try:
            if isinstance(record_id, str) and len(record_id) == 24:
                record_id_obj = ObjectId(record_id)
            else:
                record_id_obj = record_id
        except Exception as e:
            logger.error(f"Invalid ObjectId format: {record_id}")
            return {
                "success": False,
                "message": f"Invalid record ID format: {record_id}"
            }
        
        # Check if record exists first
        existing_record = collection.find_one({"_id": record_id_obj})
        if not existing_record:
            logger.error(f"Record not found: {record_id}")
            return {
                "success": False,
                "message": f"Record not found: {record_id}"
            }
        
        # Update the record with feedback
        update_data = {
            "$set": {
                "user_feedback": feedback_data,
                "feedback_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        result = collection.update_one(
            {"_id": record_id_obj},
            update_data
        )
        
        if result.modified_count > 0:
            logger.info(f"Successfully updated record {record_id} with feedback")
            return {
                "success": True,
                "message": "Feedback successfully added to record",
                "record_id": str(record_id),
                "modified_count": result.modified_count
            }
        elif result.matched_count > 0:
            logger.info(f"Record {record_id} matched but no changes made (possibly duplicate feedback)")
            return {
                "success": True,
                "message": "Record found but no changes needed",
                "record_id": str(record_id),
                "modified_count": 0
            }
        else:
            logger.error(f"Record {record_id} not found or not modified")
            return {
                "success": False,
                "message": "Record not found or could not be modified"
            }
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error updating record with feedback: {error_msg}")
        return {
            "success": False,
            "message": f"Database error: {error_msg}"
        }
    finally:
        try:
            client.close() # Close the Database Connection
        except:
            pass
    

def save_patient_data(data: dict) -> dict:
    """
    Save patient data and diagnosis results to database
    
    Args:
        data: Patient information and diagnosis data
    Returns:
        dict: Save operation result
    """
    try:
        # Input validation
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
            
        # Validate required fields for initial save
        required_fields = ['patient_data', 'diagnosis_results']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
        logger.info("Attempting to save patient data to database")
        
        # Get environment variables
        mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")
        mongo_database_name = os.getenv("MONGO_DATABASE_NAME", "skin_disease_db")
        collection_name = os.getenv("COLLECTION_NAME", "patient_diagnoses")
        
        if not mongo_connection_string:
            logger.warn("MongoDB connection not configured - no connection string found")
            return {
                "success": False,
                "message": "Database connection not available",
                "record_id": f"local_save_{int(time.time())}"
            }
        
        try:
            # Connect to MongoDB
            logger.info("Connecting to MongoDB database")
            client = MongoClient(mongo_connection_string)
            database = client[mongo_database_name]
            collection = database[collection_name]
            
            # Test connection
            client.server_info()
            
            # Insert data
            logger.info("Inserting patient data into database")
            result = collection.insert_one(data)
            logger.info(f"Data saved to database successfully with ID: {result.inserted_id}")
            
            return {
            "success": True,
            "message": "Data successfully saved to database",
            "record_id": str(result.inserted_id)
        }

        except Exception as e:
            logger.error(f"Error inserting patient data into database: {str(e)}")
            return {
                "success": False,
                "message": f"Database error: {str(e)}",
                "record_id": f"error_save_{int(time.time())}"
            }

    except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            return {
                "success": False,
                "message": f"Database error: {str(e)}",
                "record_id": f"error_save_{int(time.time())}"
            }

def api_tool_call(class_name: str) -> dict:
    """
    API tool call for specific disease name

    Args:
        class_name: Predicted class name for skin disease
    Returns:
        dict: Save operation file
    """
    logger.info(f"Database tool call initiated for class: {class_name}")
    try:

        username = os.getenv('NAME')
        password = os.getenv("PASSWORD")
        api_key = os.getenv("API_KEY")


        logger.info(f' {username} | {password} | {class_name}')
        data = {
            "username": username.strip(),
            "password": password.strip(),
            "disease_name": class_name.strip()
        }

        result = requests.post(url=api_key, json=data)

        result = result.json()

        logger.info(f"Database tool call result: {result} | Disease: {class_name}")
        if result.get('status') == "success":
            logger.info(f"Database tool call successful for class: {class_name}")
            return result
        else:
            logger.error(f"Database tool call failed for class: {class_name}")
            return {"error": "Database tool call failed"}

    except Exception as e:
        logger.error(f"Error occurred during database tool call for class: {class_name}, Error: {str(e)}")
        return {"error": str(e)}


def display_combined_info(class_name: str) -> bool:
    """
    Display combined information from database and local prevention data with visual aesthetics
    
    Args:
        class_name: Label names for specific image
    Returns:
        bool: True if information was displayed successfully, False otherwise
    """
    # Input validation
    if not class_name or class_name == 'Unknown':
        st.error("❌ Error: Cannot generate report for unknown condition. Please perform skin analysis first.")
        return False
        
    # Check if we have valid prediction results
    if not st.session_state.get('prediction_results') or 'predicted_class' not in st.session_state.prediction_results:
        st.error("❌ Error: No valid prediction results found. Please perform skin analysis first.")
        return False
        
    try:
        # Get data from both sources
        api_data = api_tool_call(class_name)

        logger.info(f'Api data {api_data}')

        if api_data.get('error'):
            logger.error(f"Database tool call failed: {api_data['error']}")
            st.warning(f"⚠️ Could not fetch online database information. Using local data only.")
            
        prevention_data = info_generation(class_name)
        if not prevention_data or prevention_data.get('success') == False:
            error_msg = prevention_data.get('message', 'Unknown error occurred') if prevention_data else 'No data available'
            logger.error(f"Error getting prevention data: {error_msg}")
            st.error(f"❌ Error: {error_msg}")
            return False
            
        # Create modern container for the information
        st.markdown("""
        <style>
        .info-container {
            background: linear-gradient(135deg, rgba(25,25,50,0.9), rgba(45,45,80,0.9));
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        .section-title {
            color: #7B68EE;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .info-card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .highlight-text {
            color: #50C878;
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)
        
        
        if isinstance(api_data, dict):  # Changed to list since API data is a list
            st.markdown('<div class="info-container" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="text-align: center;">💊 Prescribed Medications</div>', unsafe_allow_html=True)

            for medication in api_data['data']:
                # Create a card for each medication
                # Get timing string
                timing = []
                if medication['morning'] == 'Yes':
                    timing.append("Morning")
                if medication['afternoon'] == 'Yes':
                    timing.append("Afternoon")
                if medication['evening'] == 'Yes':
                    timing.append("Evening")
                timing_str = ", ".join(timing) if timing else "As prescribed"
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(72, 49, 212, 0.05), rgba(72, 49, 212, 0.1));
                    border-radius: 15px;
                    padding: 1.5rem;
                    margin: 1rem auto; /* Changed to auto margins for horizontal centering */
                    border: 1px solid rgba(72, 49, 212, 0.2);
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    max-width: 800px; /* Added max-width for better readability */
                    text-align: center; /* Center the text content */
                ">
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 1rem;">
                        <div>
                            <div style="font-size: 1.5rem; color: #7B68EE; margin-bottom: 1rem; font-weight: 600;">
                                {medication['drug_name']}
                            </div>
                            <div style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 0.5rem;">
                                📋 Dosage: {medication['dosage'] if medication['dosage'] else 'As directed'}
                            </div>
                            <div style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 0.5rem;">
                                ⏱️ Duration: {medication['duration']}
                            </div>
                            <div style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 0.5rem;">
                                💊 Route: {medication['route']}
                            </div>
                            <div style="color: rgba(255,255,255,0.9); font-size: 1.1rem;">
                                🕒 Timing: {timing_str}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
            # Display Prevention Information Section
        if prevention_data and prevention_data.get('detailed_info'):
            try:
                # Create the prevention section container
                st.markdown('<div class="info-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🛡️ Prevention & Care Guide</div>', unsafe_allow_html=True)
                
                # Get disease info safely
                disease_info = prevention_data.get('detailed_info', {})
            except Exception as e:
                error_msg = str(e) if hasattr(e, '__str__') else "Unknown error occurred"
                logger.error(f"Error displaying prevention section: {error_msg}")
                st.error("An error occurred while displaying the prevention information.")
                return
            
            # Description, Entry to Body, and Spread
            disease_info = prevention_data.get('detailed_info', {})
            if disease_info:
                st.markdown("""
                <div class="info-card">
                    <div style="color: #7B68EE; font-size: 1.3rem; margin-bottom: 1.5rem; font-weight: 600;">
                        About the Condition
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Using st.write for reliable text display
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write("**Description:**")
                with col2:
                    st.write(disease_info.get('description', 'Information not available'))
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write("**How it Affects the Body:**")
                with col2:
                    st.write(disease_info.get('entry_to_body', 'Information not available'))
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write("**Transmission:**")
                with col2:
                    st.write(disease_info.get('spread', 'Information not available'))
                
            # Prevention Steps
            if prevention_data.get('secondary_prevention'):
                st.markdown("""
                <div class="info-card" style="
                    background: linear-gradient(135deg, rgba(80, 200, 120, 0.05), rgba(80, 200, 120, 0.1));
                    border: 1px solid rgba(80, 200, 120, 0.2);
                ">
                    <div style="
                        color: #50C878;
                        font-size: 1.3rem;
                        margin-bottom: 1.5rem;
                        font-weight: 600;
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                    ">
                        🛡️ Prevention & Management Steps
                    </div>
                """, unsafe_allow_html=True)

                for i, step in enumerate(prevention_data['secondary_prevention'], 1):
                    st.markdown(f"""
                    <div style="
                        display: flex;
                        align-items: center;
                        margin: 1rem 0;
                        padding: 1rem;
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 12px;
                        border: 1px solid rgba(80, 200, 120, 0.2);
                        transition: all 0.3s ease;
                    ">
                        <div style="
                            background: rgba(80, 200, 120, 0.1);
                            color: #50C878;
                            width: 28px;
                            height: 28px;
                            border-radius: 50%;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin-right: 1rem;
                            font-weight: 600;
                            border: 1px solid rgba(80, 200, 120, 0.3);
                        ">{i}</div>
                        <span style="
                            color: rgba(255,255,255,0.95);
                            font-size: 1.1rem;
                            line-height: 1.5;
                        ">{step}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Complications
            if prevention_data.get('complications'):
                st.markdown("""
                <div class="info-card">
                    <div style="color: #7B68EE; font-size: 1.1rem; margin-bottom: 0.5rem;">⚠️ Potential Complications</div>
                """, unsafe_allow_html=True)
                
                for complication in prevention_data['complications']:
                    st.markdown(f"""
                    <div style="display: flex; align-items: start; margin: 0.5rem 0; color: rgba(255,255,255,0.9);">
                        <span style="color: #FF6B6B; margin-right: 0.5rem;">•</span>
                        <span>{complication}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error displaying combined information: {error_msg}")
        st.error(f"❌ Error displaying information: {error_msg}")
        return False
        
    return True

def info_generation(class_name: str) -> dict:
    """
    Information generation for class name
    
    Args:
        class_name: Label names for specific image 
    Returns:
        dict: A dictionary with expected result 
    """
    try:
        logger.info(f"Generating comprehensive information for disease: {class_name}")
        
        # Comprehensive skin disease information from JSON structure
        skin_diseases_info = {
            "skin_diseases": [
                {
                    "disease": "Acne",
                    "description": "Acne is a common skin condition where pores become clogged with excess oil and dead skin cells, producing blackheads, whiteheads and pimples.",
                    "entry_to_body": "Acne begins when hair follicles (pores) get plugged by substances like sebum (oil), bacteria or dead skin cells, creating inflamed bumps (pimples).",
                    "spread": "Acne is not contagious; you cannot catch it from another person.",
                    "secondary_prevention": [
                        "Wash your face gently with warm water and a mild cleanser once or twice a day.",
                        "Use oil-free or non-comedogenic moisturizers and cosmetics to avoid clogging pores.",
                        "Avoid squeezing or picking at pimples; let the skin heal naturally to prevent scars."
                    ],
                    "complications": [
                        "Scarring: Deep or persistent acne can leave pitted scars or thick (keloid) scars.",
                        "Skin discoloration: After acne clears, the affected skin may remain darker or lighter than normal."
                    ]
                },
                {
                    "disease": "Eczema",
                    "description": "Atopic dermatitis (eczema) is a chronic, non-contagious skin condition that causes dry, itchy and inflamed skin. It often begins in childhood and can flare periodically.",
                    "entry_to_body": "Eczema arises when a person's skin barrier is weakened (often due to genetic factors), allowing irritants or allergens to penetrate and trigger inflammation.",
                    "spread": "Eczema is not contagious; it cannot be passed from person to person.",
                    "secondary_prevention": [
                        "Moisturize the skin at least twice daily (e.g. with creams or petroleum jelly) to keep it hydrated.",
                        "Take warm (not hot), brief showers or baths using gentle, fragrance-free cleansers, then pat the skin dry and apply moisturizer.",
                        "Identify and avoid triggers (such as rough fabrics, harsh soaps, extreme temperatures or known allergens)."
                    ],
                    "complications": [
                        "Asthma, hay fever and food allergies often develop in people with eczema.",
                        "Chronic scratching can thicken and discolor the skin (lichenification, hyperpigmentation or hypopigmentation).",
                        "Broken skin from scratching increases the risk of bacterial or viral skin infections.",
                        "Eczema can disrupt sleep and lead to anxiety or depression due to chronic itching."
                    ]
                },
                {
                    "disease": "Psoriasis",
                    "description": "Psoriasis is an autoimmune skin disease in which the immune system causes red, scaly patches (plaques) on the skin. The most common type (plaque psoriasis) produces raised, itchy areas covered with silvery scales.",
                    "entry_to_body": "In psoriasis, the immune system becomes overactive and attacks the skin, causing inflammation and rapid overgrowth of skin cells.",
                    "spread": "Psoriasis is not contagious; you cannot get it from another person.",
                    "secondary_prevention": [
                        "Maintain a healthy lifestyle (balanced diet, regular exercise, no smoking) and follow any treatment plan prescribed by a doctor to control symptoms.",
                        "Avoid known triggers of flares, such as severe stress, skin injury (cuts, sunburn), infections (strep throat) and certain medications.",
                        "Keep skin moisturized and protect it from trauma or sunburn to reduce flare-ups."
                    ],
                    "complications": [
                        "Psoriatic arthritis: many people with psoriasis develop inflammatory joint pain and swelling.",
                        "Metabolic and cardiovascular risks: psoriasis is linked to higher rates of obesity, diabetes, high cholesterol, heart attack and stroke.",
                        "Chronic discomfort and visibility can also impact emotional well-being (stress, self-esteem issues)."
                    ]
                },
                {
                    "disease": "FU-ringworm",
                    "description": "Ringworm (tinea) is a common, itchy fungal infection of the skin. It causes a ring-shaped red rash with clearer skin in the middle.",
                    "entry_to_body": "It occurs when fungi on the skin invade through small cuts or abrasions, infecting the outer layer of skin.",
                    "spread": "Ringworm is highly contagious. The fungus spreads by direct skin-to-skin contact with an infected person or animal, and by sharing contaminated items (such as towels, clothing or sports gear).",
                    "secondary_prevention": [
                        "Keep skin clean and dry; shower and change socks and underwear daily, especially after sweating.",
                        "Wear footwear in public showers, locker rooms and pool areas.",
                        "Avoid sharing personal items like towels, clothing or hairbrushes.",
                        "Treat infected pets (e.g. cats, dogs) and avoid contact with animals that have skin lesions."
                    ],
                    "complications": [
                        "If it infects the scalp, ringworm can cause scaly bald patches; if untreated, the hair loss can become permanent.",
                        "It can spread to fingernails or toenails, causing thickened, brittle nails (onychomycosis).",
                        "Repeated scratching may lead to secondary bacterial infection of the skin."
                    ]
                },
                {
                    "disease": "BA- cellulitis",
                    "description": "Cellulitis is a bacterial skin infection of the deeper layers of skin and underlying tissue. It causes redness, swelling, warmth and pain in the affected area, often on the legs.",
                    "entry_to_body": "Cellulitis happens when bacteria (usually strep or staph) enter through a crack or break in the skin (such as a cut, insect bite, surgical wound or rash).",
                    "spread": "Cellulitis itself is not spread from person to person; it is caused by bacteria entering the skin. However, the bacteria (strep/staph) can sometimes be spread through contact with infected wounds.",
                    "secondary_prevention": [
                        "Clean and cover any cuts, scrapes or insect bites right away; wash wounds with soap and water and apply antibiotic ointment, then keep them covered.",
                        "Keep skin moisturized to prevent cracks; check and care for skin daily if you have diabetes or poor circulation.",
                        "Treat underlying skin conditions (like athlete's foot) promptly and wear protective footwear or gloves to avoid injury."
                    ],
                    "complications": [
                        "Bacteria may spread to the bloodstream (bacteremia) causing sepsis.",
                        "Necrotizing fasciitis (flesh-eating disease) or infection of deeper tissues in severe cases.",
                        "Recurrent cellulitis can cause chronic swelling of the affected limb (lymphedema).",
                        "Infection can extend to bones (osteomyelitis) or heart valves (endocarditis) in rare cases."
                    ]
                },
                {
                    "disease": "BA-impetigo",
                    "description": "Impetigo is a highly contagious bacterial skin infection, most common in infants and young children. It appears as red sores or blisters (often around the nose and mouth) that burst and form yellowish-brown crusts.",
                    "entry_to_body": "Impetigo is caused by bacteria (usually staph or strep) entering the skin, often through minor cuts, insect bites or other breaks in the skin.",
                    "spread": "Impetigo spreads very easily by close contact. It can be transmitted through direct skin contact with the sores or by touching objects and surfaces (like towels, bedding, toys) that have the bacteria on them.",
                    "secondary_prevention": [
                        "Keep skin clean and treat cuts or scratches promptly: wash minor wounds right away and cover them with a bandage.",
                        "Do not share personal items (towels, clothing, toys, etc.) with infected individuals.",
                        "Wash the hands of infected individuals frequently and wear gloves when applying antibiotics.",
                        "Keep fingernails trimmed to minimize skin damage from scratching."
                    ],
                    "complications": [
                        "Cellulitis: the infection can spread to deeper skin layers and cause serious cellulitis.",
                        "Post-streptococcal glomerulonephritis: some strep bacteria can trigger kidney inflammation.",
                        "Scarring: if lesions extend deep into the skin (ecthyma), they may leave scars."
                    ]
                },
                {
                    "disease": "Warts",
                    "description": "Warts are common benign skin growths caused by certain strains of human papillomavirus (HPV). They often appear as rough, raised bumps on hands or feet, or fleshy nodules on the genitals.",
                    "entry_to_body": "The virus enters through tiny cuts or abrasions in the skin, causing extra cell growth that forms a wart.",
                    "spread": "Warts are contagious. HPV is spread by direct skin contact (touching another person's wart) and by indirect contact (using objects like towels or razors that have touched a wart).",
                    "secondary_prevention": [
                        "Avoid touching warts (yours or others') and do not pick or bite at warts.",
                        "Do not share personal items such as towels, socks, shoes, or nail clippers.",
                        "Keep skin moist and healthy; avoid cracked skin where the virus can enter.",
                        "Use the HPV vaccine (for genital warts) and wear shoes in public showers or pool areas."
                    ],
                    "complications": [
                        "Plantar warts (on the feet) can become painful when walking.",
                        "Warts may recur or spread to other areas if untreated.",
                        "Genital warts indicate HPV infection; appropriate monitoring for related cancers is important."
                    ]
                },
                {
                    "disease": "Lupus",
                    "description": "Lupus is an autoimmune disease in which the immune system attacks its own tissues, causing inflammation. It often affects the skin, joints, kidneys, brain and other organs. A common sign is a butterfly-shaped rash on the face over the cheeks and nose.",
                    "entry_to_body": "People with a genetic predisposition can develop lupus when exposed to triggers like sunlight (UV light), certain infections or medications.",
                    "spread": "Lupus is not contagious and cannot be passed from person to person.",
                    "secondary_prevention": [
                        "There is no known way to prevent lupus, but patients can reduce flares by avoiding triggers (e.g. sun exposure).",
                        "Manage infections promptly, avoid smoking and discuss safe medications with a doctor to minimize risk factors."
                    ],
                    "complications": [
                        "Kidney damage (lupus nephritis) leading to kidney failure.",
                        "Neurological problems: lupus can cause headaches, seizures, strokes, and cognitive difficulties.",
                        "Blood disorders: increased risk of blood clots, anemia and bleeding problems.",
                        "Heart and lung inflammation: higher risk of pericarditis, pleurisy, and heart disease.",
                        "Pregnancy complications: higher risk of miscarriage, preterm birth and high blood pressure."
                    ]
                },
                {
                    "disease": "SkinCancer",
                    "description": "Skin cancer is a disease in which skin cells grow abnormally and can form tumors. It is often caused by DNA damage from ultraviolet (UV) radiation (sunlight or tanning beds). The main types are basal cell carcinoma, squamous cell carcinoma, and melanoma (the most serious, as it can spread).",
                    "entry_to_body": "Exposure to UV light causes mutations in skin cell DNA, triggering cells to grow and divide in an uncontrolled way.",
                    "spread": "Skin cancer is not contagious. However, if malignant cells break away, they can invade nearby tissue or spread (metastasize) to lymph nodes or other parts of the body.",
                    "secondary_prevention": [
                        "Protect your skin from UV exposure: use broad-spectrum sunscreen (SPF 30+) daily, wear protective clothing and seek shade during peak sun hours.",
                        "Avoid indoor tanning and intentional sunburns.",
                        "Perform regular skin self-exams and get routine check-ups for any suspicious moles or lesions."
                    ],
                    "complications": [
                        "Advanced skin cancers, especially melanoma, can spread to lymph nodes and distant organs.",
                        "Late-stage skin cancer may require extensive surgery or radiation, which can be disfiguring.",
                        "Untreated melanoma can be fatal."
                    ]
                },
                {
                    "disease": "chickenpox",
                    "description": "Chickenpox is a highly contagious viral infection caused by the varicella-zoster virus. It causes an itchy rash of red bumps and blisters all over the body.",
                    "entry_to_body": "The virus enters the body through the respiratory tract or by direct contact with chickenpox blisters.",
                    "spread": "Chickenpox spreads very easily from person to person. It is transmitted through close contact, such as breathing in virus particles from coughs or sneezes, and by touching the rash of an infected person.",
                    "secondary_prevention": [
                        "Vaccination with the varicella (chickenpox) vaccine is the best prevention; two doses are recommended and prevent about 90% of cases.",
                        "Keep infected individuals isolated until all blisters have crusted to avoid spreading the virus to others.",
                        "Wash hands frequently and disinfect surfaces to reduce transmission."
                    ],
                    "complications": [
                        "Bacterial skin infections: scratching blisters can introduce bacteria into the skin.",
                        "Pneumonia (lung infection) or encephalitis (brain inflammation) can occur.",
                        "Dehydration or (in aspirin-treated children) Reye's syndrome.",
                        "In rare cases, severe infection can lead to hospitalization or death, especially in high-risk individuals."
                    ]
                }
            ]
        }
        
        # Clean class name for lookup (remove extra spaces and convert to lowercase for matching)
        clean_class_name = class_name.strip()
        logger.info(f"Looking up information for cleaned class name: {clean_class_name}")
        
        # Find matching disease information
        disease_info = None
        for disease in skin_diseases_info["skin_diseases"]:
            if disease["disease"].lower() == clean_class_name.lower():
                disease_info = disease
                logger.info(f"Found exact match for disease: {disease['disease']}")
                break
        
        if disease_info:
            logger.info(f"Successfully found comprehensive disease information for: {clean_class_name}")
            return {
                "success": True,
                "disease_name": disease_info["disease"],
                "source": "medical_knowledge_base",
                "description": disease_info["description"],
                "entry_to_body": disease_info["entry_to_body"],
                "spread": disease_info["spread"],
                "secondary_prevention": disease_info["secondary_prevention"],  # Fixed key name
                "complications": disease_info["complications"],
                "detailed_info": disease_info  # Include all detailed information
            }
        else:
            # Handle unknown diseases with general medical advice
            logger.warn(f"No specific information found for disease: {clean_class_name}. Providing general advice.")
            return {
                "success": True,
                "disease_name": clean_class_name,
                "source": "general_medical_knowledge",
                "description": f"Skin condition detected: {clean_class_name}. This appears to be a dermatological condition that requires professional evaluation.",
                "entry_to_body": "The exact mechanism of this condition may vary. Consult with a dermatologist for detailed information.",
                "spread": "Transmission characteristics unknown. Follow general hygiene practices and avoid direct contact until evaluated by a healthcare provider.",
                "secondary_prevention": [  # Fixed key name
                    "Maintain good skin hygiene",
                    "Avoid sharing personal items like towels and clothing",
                    "Keep the affected area clean and dry",
                    "Consult with a dermatologist for specific prevention strategies"
                ],
                "complications": [
                    "Potential for secondary bacterial infection if scratched",
                    "Possible scarring or skin discoloration",
                    "Consult healthcare provider to understand specific risks"
                ]
            }
            
    except Exception as e:
        error_message = str(e)
        logger.error(f'Error in information generation for class {class_name}: {error_message}')
        return {
            "success": False,
            "disease_name": class_name,
            "error": error_message,
            "message": "Failed to generate disease information. Please try again."
        }
def main():
    logger.info("Starting main application function")

    image = Image.open("E:/hitayu/captured_image.jpg")
    img_array = np.asarray(image)
    img_list = img_array.tolist()
    logger.info(f'Got an array of shape {img_array.shape}')

    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Skin Disease Diagnosis System</h1>
        <p>AI-powered dermatological analysis for accurate skin condition identification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'diagnosis_complete' not in st.session_state:
        st.session_state.diagnosis_complete = False
        logger.info("Initialized diagnosis_complete state")
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
        logger.info("Initialized uploaded_image state")
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = {}
        logger.info("Initialized patient_data state")
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = {}
        logger.info("Initialized prediction_results state")
    if 'show_detailed_report' not in st.session_state:
        st.session_state.show_detailed_report = False
        logger.info("Initialized show_detailed_report state")
    if "human_feedback" not in st.session_state:
        st.session_state.human_feedback = {}
        logger.info("Initialized human_feedback state")
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'show_feedback_form' not in st.session_state:
        st.session_state.show_feedback_form = False
    if 'patient_record_id' not in st.session_state:
        st.session_state.patient_record_id = None
    
    # Create two columns for layout
    col1 , col2 = st.columns([1,1])
    
    with col1:
        logger.info("Rendering patient information section")
        # Patient Information Card
        st.markdown("""
        <div class="card">
            <div class="card-header">
                👤 Patient Details & Medical History
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            # Basic Information
            patient_name = st.text_input("Full Name", placeholder="Enter patient's full name")
            
            col_age, col_gender = st.columns(2)
            with col_age:
                age = st.number_input("Age", min_value=0, max_value=150, value=30)
            with col_gender:
                gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
            
            # added email section 
            email = st.text_input("Email Address", placeholder="patient@example.com")
            
            # Medical History
            st.markdown("**Medical History**")
            medical_history = st.text_area(
                "General Medical History", 
                placeholder="Previous medical conditions, surgeries, medications...",
                height=100
            )
            
            # Skin-specific Information
            st.markdown("**Dermatological Information**")
            
            col_skin1, col_skin2 = st.columns(2)
            with col_skin1:
                skin_type = st.selectbox(
                    "Skin Type", 
                    ["Type I (Very Fair)", "Type II (Fair)", "Type III (Medium)", 
                     "Type IV (Olive)", "Type V (Brown)", "Type VI (Dark Brown/Black)"]
                )
                
                sun_exposure = st.selectbox(
                    "Sun Exposure History",
                    ["Minimal", "Moderate", "High", "Very High"]
                )
            
            with col_skin2:
                family_history_skin = st.selectbox(
                    "Family History of Skin Cancer",
                    ["None", "Acne", "Eczema", "Psoriasis", "FU-ringworm", "BA- cellulitis",
                     "BA-impetigo", "Warts", "Lupus", "SkinCancer", "chickenpox", "Other", "Unknown"]
                )
                
                previous_skin_issues = st.selectbox(
                    "Previous Skin Issues",
                    ["None", "Acne", "Eczema", "Psoriasis", "FU-ringworm", "BA- cellulitis",
                     "BA-impetigo", "Warts", "Lupus", "SkinCancer", "chickenpox", "Other"]
                )
            
            # Symptoms and Concerns
            current_symptoms = st.text_area(
                "Current Symptoms & Concerns",
                placeholder="Describe the skin lesion: size, color, texture, pain, itching, bleeding, changes over time...",
                height=120
            )
            
            # Lesion Details
            st.markdown("**Lesion Details**")
            lesion_location = st.text_input("Lesion Location", placeholder="e.g., left arm, face, back")
     
            # Store patient data
            st.session_state.patient_data = {
                'name': patient_name,
                'age': age,
                'gender': gender,
                'email': email, 
                'medical_history': medical_history,
                'skin_type': skin_type,
                'sun_exposure': sun_exposure,
                'family_history_skin': family_history_skin,
                'previous_skin_issues': previous_skin_issues,
                'current_symptoms': current_symptoms,
                'lesion_location': lesion_location,
                'array_of_image': img_list
            }
            logger.info(f"Updated patient data for: {patient_name if patient_name else 'unnamed patient'}")
    
    with col2:
        logger.info("Rendering image upload section")
        # Image Upload Card
        st.markdown("""
        <div class="card">
            <div class="card-header">
                📸 Skin Lesion Image Upload
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            uploaded_file = st.file_uploader(
                "Upload Skin Lesion Image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Supported formats: PNG, JPG, JPEG, BMP, TIFF. For best results, ensure good lighting and clear focus on the lesion."
            )
            
            if uploaded_file is not None:
                st.session_state.uploaded_image = uploaded_file
                logger.info(f"Image uploaded: {uploaded_file.name}, size: {uploaded_file.size} bytes")
                
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Skin Lesion Image", width= 'stretch')
                
                # Image details
                st.markdown("**Image Details:**")
                st.write(f"- **Filename:** {uploaded_file.name}")
                st.write(f"- **Size:** {uploaded_file.size / 1024:.2f} KB")
                st.write(f"- **Dimensions:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"- **Format:** {image.format}")
                
                # Image quality tips
                st.markdown("**📋 Image Quality Tips:**")
                st.info("""
                • Ensure good lighting (natural light preferred)
                • Keep the camera steady and focused
                • Include a ruler or coin for size reference if possible
                • Capture the entire lesion and surrounding skin
                • Avoid shadows and reflections
                """)

    st.markdown("<br>", unsafe_allow_html= True)

    # Analyze Button
    col_center = st.columns([2, 1, 2])[1]
    with col_center:
        analyze_button = st.button("🔍 Start Skin Analysis", type="primary", width= 'stretch')
    
    # Analysis and Results
    if analyze_button and st.session_state.uploaded_image and patient_name:
        logger.info(f"Starting skin analysis for patient: {patient_name}")
        st.session_state.diagnosis_complete = False
        st.session_state.show_detailed_report = False
        st.session_state.feedback_submitted = False  # Reset feedback status
        st.session_state.show_feedback_form = False
        
        # Progress Section
        st.markdown("""
        <div class="card">
            <div class="card-header">
                ⚡ Processing Analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Analysis steps
        steps = [
            "Preprocessing skin lesion image...",
            "Applying deep learning model...",
            "Analyzing dermatological patterns...",
            "Comparing with medical database...",
            "Generating diagnosis report..."
        ]
        
        start_time = time.perf_counter()
        logger.info("Starting analysis process")
        
        for i, step in enumerate(steps):
            status_text.text(step)
            logger.info(f"Analysis step {i+1}/5: {step}")
            time.sleep(0.8)
            progress_bar.progress((i + 1) / len(steps))
        
        status_text.text("✅ Analysis Complete!")
        logger.info("Analysis steps completed")
        
        # Get prediction results
        image = Image.open(st.session_state.uploaded_image)

        def compress_image_for_storage(image, max_size_kb=500, quality=85):
            """
            Compress image to reduce size before database storage
            """
            try:
                # Convert to RGB if necessary
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                
                # Resize if too large
                max_dimension = 800  # Maximum width or height
                if max(image.size) > max_dimension:
                    image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
                # Compress and convert to base64
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                
                # Check size and reduce quality if still too large
                while buffer.tell() > max_size_kb * 1024 and quality > 30:
                    buffer = BytesIO()
                    quality -= 10
                    image.save(buffer, format="JPEG", quality=quality, optimize=True)
                
                buffer.seek(0)
                image_b64 = base64.b64encode(buffer.read()).decode('utf-8')
                
                return {
                    'image_data': image_b64,
                    'compressed_size_kb': len(image_b64) / 1024,
                    'quality': quality,
                    'dimensions': image.size
                }
                
            except Exception as e:
                logger.error(f"Error compressing image: {str(e)}")
                return None

        compressed_image_info = compress_image_for_storage(image)
        if compressed_image_info:
            st.session_state.patient_data['compressed_image'] = compressed_image_info
            logger.info(f"Compressed image for storage: {compressed_image_info['compressed_size_kb']:.2f} KB at quality {compressed_image_info['quality']}")
        else:
            logger.warn("Image compression failed; proceeding without compressed image")
        
        logger.info("Loading image for prediction")
        results = predict_skin_disease(image= image)
        st.session_state.prediction_results = results
        
        end_time = time.time()
        analysis_time = end_time - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
        
        st.session_state.diagnosis_complete = True
        
        # Results Section
        st.markdown("""
        <div class="result-card">
            <div class="card-header">
                📊 Diagnosis Results
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if 'error' not in results:
            predicted_class = results['predicted_class']
            confidence_score = results['confidence_score']
            logger.info(f"Prediction successful: {predicted_class} with confidence {confidence_score:.2%}")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">CONFIDENCE</div>
                    <div style="color: white; font-size: 2rem; font-weight: bold; margin-bottom: 0.3rem;">{:.1%}</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">Prediction Accuracy</div>
                </div>
                """.format(confidence_score), unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">ANALYSIS TIME</div>
                    <div style="color: white; font-size: 2rem; font-weight: bold; margin-bottom: 0.3rem;">{:.1f}s</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">Processing Speed</div>
                </div>
                """.format(analysis_time), unsafe_allow_html=True)
                
            with col3:
                risk_level = "High" if confidence_score > 0.9 else "Medium" if confidence_score > 0.7 else "Low"
                st.markdown("""
                <div class="metric-card">
                    <div style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">RISK LEVEL</div>
                    <div style="color: white; font-size: 1.5rem; font-weight: bold; margin-bottom: 0.3rem;">{}</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">Assessment</div>
                </div>
                """.format(risk_level), unsafe_allow_html=True)
                
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <div style="color: white; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">CONDITION</div>
                    <div style="color: white; font-size: 1.2rem; font-weight: bold; margin-bottom: 0.3rem;">{}</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">Diagnosis</div>
                </div>
                """.format(predicted_class.split()[0]), unsafe_allow_html=True)
            
            # Create probability chart
            if 'all_predictions' in results:
                # Convert predictions to a format suitable for plotting
                predictions_flat = results['all_predictions'][0] if len(results['all_predictions'].shape) > 1 else results['all_predictions']
                
                # Create DataFrame for easier plotting
                df_predictions = DataFrame({
                    'Disease': SKIN_DISEASE_CLASSES,
                    'Probability': predictions_flat
                })
                df_predictions = df_predictions.sort_values('Probability', ascending=True)
                
                # Display top 3 predictions in a nice format
                st.markdown("### 🏆 Top 3 Predictions")
                top_3 = df_predictions.nlargest(3, 'Probability')
                
                cols = st.columns(3)
                for i, (idx, row) in enumerate(top_3.iterrows()):
                    with cols[i]:
                        rank_emoji = ["🥇", "🥈", "🥉"][i]
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0;">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{rank_emoji}</div>
                            <div style="font-weight: bold; color: white; margin-bottom: 0.5rem;">{row['Disease']}</div>
                            <div style="font-size: 1.5rem; color: #4CAF50; font-weight: bold;">{row['Probability']:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Prepare and save initial patient data
            logger.info("Preparing patient and diagnosis data for database save")
            
            # Convert numpy array to list for database compatibility
            all_predictions = results.get('all_predictions', [])
            if hasattr(all_predictions, 'tolist'):
                all_predictions = all_predictions.tolist()

            # Create initial data for database
            initial_data = {
                'patient_data': st.session_state.patient_data.copy(),
                'diagnosis_results': {
                    'diagnosis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_condition': predicted_class,
                    'confidence_score': float(confidence_score),
                    'analysis_time_seconds': round(analysis_time, 2),
                    'all_predictions': all_predictions
                }
            }
            
            # Save initial data to database
            save_result = save_patient_data(initial_data)
            
            if save_result.get('success'):
                logger.info(f"Initial data saved successfully with ID: {save_result['record_id']}")
                st.session_state.patient_record_id = save_result['record_id']
                st.success(f"✅ Patient record created successfully! Record ID: {save_result['record_id']}")
            else:
                logger.error(f"Failed to save initial data: {save_result.get('message', 'Unknown error')}")
                st.warning(f"⚠️ Error saving diagnosis data: {save_result.get('message', 'Unknown error')}")
            
            logger.info("Initial diagnosis data processed")
        
        else:
            logger.error(f"Analysis failed with error: {results['error']}")
            st.error(f"❌ Analysis failed: {results['error']}")
    
    elif analyze_button:
        logger.warn("Analysis button clicked but validation failed")
        if not st.session_state.uploaded_image:
            logger.error("No image uploaded for analysis")
            st.error("❌ Please upload a skin lesion image first.")
        if not patient_name:
            logger.error("No patient name provided for analysis")
            st.error("❌ Please enter patient name.")
    
    # Report generation section
    if st.session_state.diagnosis_complete and 'predicted_class' in st.session_state.prediction_results:
        logger.info("Diagnosis complete - showing report generation option")
        
        # Generate Report Button
        st.markdown("<br>", unsafe_allow_html=True)
        col_report = st.columns([2, 1, 2])[1]
        with col_report:
            generate_report_button = st.button("📄 Generate Detailed Report", width= 'stretch')

        # Handle report generation
        if generate_report_button:
            logger.info("Generate detailed report button clicked")
            predicted_class = st.session_state.prediction_results.get('predicted_class', 'Unknown')
            
            if predicted_class == 'Unknown':
                logger.error("No prediction results available")
                st.error("❌ Please perform skin disease analysis first before generating a report.")
            else:
                st.session_state.show_detailed_report = True
                
                # Display combined information
                if display_combined_info(predicted_class):
                    logger.info(f"Displayed combined information for: {predicted_class}")
                    
                    # Show feedback form if not already submitted
                    if not st.session_state.get('feedback_submitted', False):
                        st.session_state.show_feedback_form = True
                    
                    # Success message
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1));
                        backdrop-filter: blur(15px);
                        border: 2px solid rgba(76, 175, 80, 0.4);
                        border-radius: 20px;
                        padding: 1.5rem;
                        margin: 1.5rem 0;
                        text-align: center;
                        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                    ">
                        <div style="
                            font-size: 1.2rem;
                            font-weight: 600;
                            color: #4CAF50;
                            margin-bottom: 0.5rem;
                        ">
                            ✅ Detailed Medical Report Generated Successfully!
                        </div>
                        <div style="
                            font-size: 1rem;
                            color: rgba(255, 255, 255, 0.8);
                        ">
                            📧 Report ready for healthcare provider consultation
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Feedback Form Section
    if st.session_state.show_feedback_form and not st.session_state.get('feedback_submitted', False):
        st.markdown("<br>", unsafe_allow_html=True)

        # Feedback Section Header
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.15), rgba(255, 193, 7, 0.08));
            backdrop-filter: blur(20px);
            border: 2px solid rgba(255, 193, 7, 0.3);
            border-radius: 25px;
            padding: 2.5rem;
            margin: 2rem 0;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        ">
            <h3 style="
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
                text-align: center;
                color: #FFC107;
                margin-top: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.8rem;
            ">
                <span style="font-size: 1.8rem;">💬</span>
                Your Feedback Matters
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Use unique form key to prevent conflicts
        with st.form("feedback_submission_form", clear_on_submit=False):
            logger.info("Rendering feedback form")
            st.markdown("**Please help us improve our diagnosis system:**")
            
            # Multi-column feedback layout
            col_feedback1, col_feedback2 = st.columns(2)
            
            with col_feedback1:
                prediction_accuracy = st.selectbox(
                    "How accurate do you think this prediction is?",
                    ["Select Rating", "Very Accurate", "Mostly Accurate", "Somewhat Accurate", "Not Accurate", "Completely Wrong"],
                    key="pred_accuracy"
                )
                
                confidence = st.selectbox(
                    "How confident are you in this diagnosis?",
                    ["Select Level", "Very Confident", "Confident", "Neutral", "Not Confident", "No Confidence"],
                    key="conf_level"
                )
            
            with col_feedback2:
                report_usefulness = st.selectbox(
                    "How useful was the detailed medical report?",
                    ["Select Rating", "Extremely Useful", "Very Useful", "Moderately Useful", "Slightly Useful", "Not Useful"],
                    key="report_useful"
                )
                
                recommendation = st.selectbox(
                    "Would you recommend this system to others?",
                    ["Select Option", "Definitely Yes", "Probably Yes", "Maybe", "Probably No", "Definitely No"],
                    key="recommend"
                )
            
            # Additional comments
            additional_comments = st.text_area(
                "Additional Comments or Suggestions:",
                placeholder="Please share any additional thoughts, concerns, or suggestions for improvement...",
                height=100,
                key="comments"
            )
            
            # Submit feedback button
            submitted = st.form_submit_button("📝 Submit Feedback", width= 'stretch')
            
            # Handle form submission
            if submitted:
                logger.info("Feedback form submitted")

                # Check if all required fields are filled
                if (prediction_accuracy != "Select Rating" and 
                    confidence != "Select Level" and 
                    report_usefulness != "Select Rating" and 
                    recommendation != "Select Option"):
                    
                    try:
                        # Prepare feedback data
                        feedback_data = {
                            'prediction_accuracy': prediction_accuracy,
                            'confidence': confidence,
                            'report_usefulness': report_usefulness,
                            'recommendation': recommendation,
                            'additional_comments': additional_comments or "",
                        }

                        # Store feedback in session state
                        st.session_state.human_feedback = feedback_data
                        
                        # Update existing record with feedback if we have a record ID
                        if st.session_state.patient_record_id:
                            record_id = st.session_state.patient_record_id
                            logger.info(f"Updating existing record {record_id} with feedback")
                            
                            update_result = update_record_with_feedback(record_id, feedback_data)
                            
                            if update_result.get('success'):
                                logger.info(f"Feedback updated successfully for record: {record_id}")
                                st.session_state.feedback_submitted = True
                                st.success("✅ Thank you for your feedback! Your input has been saved successfully.")
                                time.sleep(2)
                                st.rerun()
                            else:
                                logger.warn(f"Failed to update record: {update_result.get('message')}")
                                # Try to save as new record
                                feedback_record = {
                                    'record_type': 'user_feedback',
                                    'feedback_data': feedback_data,
                                    'original_record_id': record_id,
                                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                                
                                feedback_save_result = save_patient_data(feedback_record)
                                if feedback_save_result.get('success'):
                                    logger.info(f"Feedback saved as new record: {feedback_save_result['record_id']}")
                                    st.session_state.feedback_submitted = True
                                    st.success("✅ Thank you for your feedback! Your input has been saved.")
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error("❌ Error saving feedback to database.")
                        else:
                            # No existing record - save feedback as new record
                            logger.info("Saving feedback as new record")
                            feedback_record = {
                                'record_type': 'user_feedback',
                                'feedback_data': feedback_data,
                                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            feedback_save_result = save_patient_data(feedback_record)
                            if feedback_save_result.get('success'):
                                logger.info(f"Feedback saved as new record: {feedback_save_result['record_id']}")
                                st.session_state.feedback_submitted = True
                                st.success("✅ Thank you for your feedback! Your input has been saved.")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("❌ Error saving feedback to database.")
                                
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Error processing feedback submission: {error_msg}")
                        st.error(f"❌ Error processing submission: {error_msg}")
                else:
                    st.error("❌ Please fill in all required fields before submitting.")

    # Show feedback success message if submitted
    if st.session_state.get('feedback_submitted', False):
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1));
            backdrop-filter: blur(15px);
            border: 2px solid rgba(76, 175, 80, 0.4);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        ">
            <div style="
                font-size: 1.3rem;
                font-weight: 600;
                color: #4CAF50;
                margin-bottom: 0.5rem;
            ">
                ✅ Thank You for Your Feedback!
            </div>
            <div style="
                font-size: 1rem;
                color: rgba(255, 255, 255, 0.8);
            ">
                Your input helps us improve our diagnostic accuracy and user experience.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("<br>" * 5, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.8); font-size: 0.9rem; padding: 2rem;">
        🔬 Skin Disease Diagnosis System | Powered by Advanced AI | 
        <strong>For Medical Professionals & Patients</strong><br>
        <em>⚠️ This system is for screening purposes only. Always consult with a dermatologist for final diagnosis.</em>
    </div>
    """, unsafe_allow_html=True)
    
    logger.info("Main application function completed")

if __name__ == "__main__":
    logger.info("Application starting")
    main()
    logger.info("Application finished")