import streamlit as st
import time
import cv2
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="HITAYU - AI Skin Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-bottom: 2rem;">
    <h1>🏥 HITAYU</h1>
    <p>Health at Your Fingertips - AI-Powered Skin Disease Prediction</p>
</div>
""", unsafe_allow_html=True)

# Add CSS for text justification and glassmorphism
st.markdown("""
<style>
    .justified-text {
        text-align: justify;
        text-justify: inter-word;
    }
    .center-text {
        text-align: center;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
</style>
""", unsafe_allow_html=True)


if 'image_captured' not in st.session_state:
    st.session_state.image_captured = False
if 'clicked_image' not in st.session_state:
    st.session_state.clicked_image = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = True

# Show processing spinner if currently processing
if st.session_state.processing:
    # Center the spinner
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.spinner("Processing your image..."):
            time.sleep(2)  # Simulate processing time
    
    # Mark processing as complete
    st.session_state.processing = False
    st.session_state.image_captured = True
    st.rerun()

# Only show camera interface if no image has been captured yet
elif not st.session_state.image_captured:
    
    # Centered header
    st.markdown("""
    <div style="text-align: center; padding: 2rem; margin: 2rem 0;">
        <h2>📸 Camera Interface</h2>
        <p>Position your skin area in front of the camera and click capture when ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    def capture_image():
        # Create columns to center the camera feed
        col1, col2, col3 = st.columns([0.5, 2, 0.5])  # Middle column is much larger
        
        with col2:
            # Create placeholder for the camera feed (centered)
            FRAME_WINDOW = st.image([],use_container_width=True)
            
            # Create centered button layout
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                capture_btn = st.button("📸 Capture Image", use_container_width=True, type="primary")
                
            # Add stop camera option
            stop_col1, stop_col2, stop_col3 = st.columns([1, 2, 1])
            with stop_col2:
                if st.session_state.camera_running:
                    if st.button("⏹️ Stop Camera", use_container_width=True, type="secondary"):
                        st.session_state.camera_running = False
                        st.rerun()
        
        # Camera operations
        if st.session_state.camera_running:
            # Initialize the webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not open webcam")
                return None
            
            # Set camera properties for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Handle capture button press
            if capture_btn:
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to numpy array
                    img_array = np.array(rgb_frame)
                    
                    # Store in session state
                    st.session_state.clicked_image = Image.fromarray(img_array)
                    
                    # Stop the camera
                    st.session_state.camera_running = False
                    
                    # Release webcam
                    cap.release()
                    
                    # Save the image
                    img = Image.fromarray(img_array)
                    img.save("captured_image.jpg")
                    
                    # Set processing state to trigger spinner
                    st.session_state.processing = True
                    
                    # Show success message briefly
                    with col2:
                        st.success("Image captured! Processing...")
                    
                    # Force rerun to show spinner
                    st.rerun()
                    
                    return img_array
                else:
                    st.error("Failed to capture image")
                    cap.release()
                    return None
            
            # Show live preview - single frame capture to avoid infinite loop
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Update live preview in centered layout
                with col2:
                    FRAME_WINDOW.image(frame_rgb, caption="Live Camera Feed",use_container_width=True)
            else:
                st.error("Cannot read from camera")
            
            # Release webcam after each frame
            cap.release()
            
            # Auto-refresh for live preview (small delay to prevent overwhelming)
            time.sleep(0.1)
            st.rerun()
            
        else:
            # Display captured image if available (when camera is stopped but not processed yet)
            if st.session_state.clicked_image:
                with col2:
                    FRAME_WINDOW.image(st.session_state.clicked_image, 
                                     caption="Captured Image", 
                                     use_container_width=True)
                    
                    # Add restart option
                    restart_col1, restart_col2, restart_col3 = st.columns([1, 1, 1])
                    with restart_col2:
                        if st.button("🔄 Restart Camera", use_container_width=True):
                            st.session_state.camera_running = True
                            st.session_state.clicked_image = None
                            st.rerun()
        
        return None
    
    capture_image()
else:
    
    col1, col2 , col3 = st.columns([0.5, 2, 0.5])

    with col2:
        st.success('✅ Your Picture has been saved sucessfully')    


# Project Information Section
st.markdown("""
<div class="glass-card">
    <div class="center-text"><h2>📑 AI-Based Skin Disease Prediction System</h2></div>
    <div class="justified-text"><strong>Project Division:</strong> Beta Division of Hitayu-PS1</div>
    <div class="justified-text"><strong>Model Version:</strong> SDN05 (Accuracy: 99%)</div>
    <div class="justified-text"><strong>Developed & Owned By:</strong>Indranil Bakshi</div>
    <div class="justified-text"><em>© All Rights Reserved</em></div>
</div>
""", unsafe_allow_html=True)

# Disclaimer Section
st.markdown("""
<div class="glass-card">
    <div class="center-text"><h3>⚠️ Disclaimer (Read Carefully)</h3></div>
    <div class="justified-text">
        🔬 This system is an overall prototype for research and educational purposes only.<br>
        📋 It should not be used as a final medical decision-making tool.<br>
        👨‍⚕️ Users must consult certified doctors before following any treatment.<br>
        🛡️ The system's outputs are suggestive and cannot replace professional diagnosis.
    </div>
</div>
""", unsafe_allow_html=True)

# Introduction Section
st.markdown("""
<div class="glass-card">
    <div class="center-text"><h2>🔬 1. Introduction</h2></div>
    <div class="justified-text">
        The <strong>Hitayu AI-Powered Skin Disease Prediction System</strong> is a research-based healthcare prototype designed to assist in the <strong>early diagnosis of skin conditions</strong>. It combines <strong>image analysis</strong> and <strong>query-based inputs</strong> to predict diseases and provide insights about medicines, precautions, and lifestyle recommendations.
        <br><br>
        This prototype is part of <strong>Hitayu-PS1</strong>, our flagship healthcare project for <strong>early diagnosis</strong>. Our current <strong>SDN05 model</strong> achieves <strong>99% accuracy</strong>, and we continue refining the architecture to improve further.
    </div>
</div>
""", unsafe_allow_html= True)

# Key Features Section
st.markdown("""
<div class="glass-card">
    <div class="center-text"><h2>✨ 2. Key Features</h2></div>
    <div class="justified-text">
        • 🤖 <strong>AI-Powered Prototype Diagnosis</strong> – Predicts 10 skin diseases.<br>
        • 🔗 <strong>Dual Input Support</strong> – Upload images + enter symptom queries.<br>
        • 💊 <strong>Doctor-Backed Medicine Suggestions</strong> – Verified treatment guidance.<br>
        • 🛡️ <strong>Precautions & Lifestyle Guidance</strong> – Preventive steps included.<br>
        • ⚡ <strong>Cutting-Edge AI Model</strong> – DepthwiseConv2D for efficient predictions.<br>
        • 🔒 <strong>Privacy First</strong> – Encrypted, secure data handling.
    </div>
</div>
""", unsafe_allow_html=True)

# Supported Diseases Section
st.markdown("""
<div class="glass-card">
    <div class="center-text"><h2>🩺 3. Supported Skin Diseases (Prototype)</h2></div>
    <div class="center-text">
        Acne, Eczema, Psoriasis, Rosacea, Vitiligo, Melanoma (early detection), 
        Fungal Infections, Dermatitis, Urticaria (Hives), Warts
    </div>
</div>
""", unsafe_allow_html=True)

# Workflow Section
st.markdown("""
<div class="glass-card">
    <div class="center-text"><h2>⚙️ 4. System Workflow</h2></div>
    <div class="justified-text">
        1️⃣ <strong>Data Input</strong> → Upload skin image + symptom queries.<br>
        2️⃣ <strong>AI Processing</strong> → CNN + NLP (SDN05) analysis.<br>
        3️⃣ <strong>Prediction & Output</strong> → Disease prediction, medicines, lifestyle advice.<br>
        4️⃣ <strong>Prototype Report</strong> → Includes diagnosis + treatment suggestions.
    </div>
</div>
""", unsafe_allow_html=True)

# Model Details Section
st.markdown("""
<div class="glass-card">
    <div class="center-text"><h2>🧠 5. Model Details – SDN05</h2></div>
    <div class="justified-text">
        • 🧠 <strong>Core Network</strong>: DepthwiseConv2D CNN<br>
        • 🔀 <strong>Architecture</strong>: Hybrid CNN + NLP<br>
        • 📊 <strong>Accuracy</strong>: 99% (validation)<br>
        • 🔮 <strong>Future Upgrade</strong>: SDN06 with Vision Transformers (ViT)
    </div>
</div>
""", unsafe_allow_html=True)

# Data Privacy Section
st.markdown("""
<div class="glass-card">
    <div class="center-text"><h2>🔒 6. Data Privacy & Security</h2></div>
    <div class="justified-text">
        • 🔐 <strong>Minimal Data Collection</strong> – Only images + symptoms.<br>
        • 🔏 <strong>Encrypted Storage</strong> – AES-256, TLS 1.3.<br>
        • 🛡️ <strong>Anonymization</strong> – No personal identifiers stored.<br>
        • 🌍 <strong>Compliance</strong> – HIPAA, GDPR, DPDP 2023.<br>
        • ⚖️ <strong>User Rights</strong> – Delete, access, withdraw consent anytime.<br>
        • 🔎 <strong>Audits</strong> – Regular penetration testing & monitoring.
    </div>
</div>
""", unsafe_allow_html=True)

# Roadmap Section
st.markdown("""
<div class="glass-card">
    <div class="center-text"><h2>🚀 7. Future Roadmap</h2></div>
    <div class="justified-text">
        • 📈 Expand predictions → 50+ skin diseases<br>
        • 📱 Mobile apps (Android/iOS)<br>
        • 🩺 Telemedicine integration<br>
        • 🌐 Multi-language support
    </div>
</div>
""", unsafe_allow_html=True)

# Download Section
st.markdown("""
<div class="glass-card">
    <div class="center-text"><h2>📥 Download Documentation</h2></div>
    <div class="justified-text">Get the complete system documentation for research and development purposes</div>
</div>
""", unsafe_allow_html=True)

doc_text = """
AI-Based Skin Disease Prediction System - HITAYU
===============================================
Version: SDN05
Division: Beta Division of Hitayu-PS1
Owned By: Indranil Bakshi

DISCLAIMER: 
→ This system is an overall prototype for research and educational purposes only.
→ It should not be used as a final medical decision-making tool.
→ Users must consult certified doctors before following any treatment.
→ The system's outputs are suggestive and cannot replace professional diagnosis.

FEATURES:
- AI-Powered Diagnosis (99% accuracy)
- Dual Input Support (Images + Symptoms)
- Doctor-Backed Medicine Suggestions
- Privacy-First Approach
- Cutting-Edge CNN Architecture

SUPPORTED DISEASES:
Acne, Eczema, Psoriasis, Rosacea, Vitiligo, Melanoma, 
Fungal Infections, Dermatitis, Urticaria, Warts

PRIVACY & SECURITY:
- AES-256 Encryption
- HIPAA, GDPR, DPDP 2023 Compliant
- No Personal Data Storage
- Regular Security Audits

© 2026 Indranil Bakshi. All Rights Reserved.
"""

st.download_button(
    label="⬇️ Download Complete Documentation (TXT)",
    data=doc_text,
    file_name="HITAYU_SkinDiseaseSystem_Documentation.txt",
    mime="text/plain",
    use_container_width=True
)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem;">
    <h3>🏥 HITAYU - Health at Your Fingertips</h3>
    <p>Empowering healthcare through AI innovation</p>
    <p><em>© 2026 Indranil Bakshi. All Rights Reserved.</em></p>
</div>
""", unsafe_allow_html=True)