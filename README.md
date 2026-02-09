# Hitayu-ICBP-3.0





Hitayu – AI-Powered Healthcare Platform Streamlit + FastAPI + Machine

Learning + Docker



Repository Structure - Hitayu-Fastapi-V1/ FastAPI backend services -

Hitayu-Streamlit-Prototype/ Streamlit frontend (image processing \& UI) -

requirements.txt Python dependencies - Dockerfile Docker configuration -

README.md Project documentation



Features Streamlit Frontend: - Medical image processing and analysis -

Interactive UI for experimentation - Visualization of prediction results



FastAPI Backend: - Disease prediction APIs - Conversational AI module

with multilingual support - Experimental endpoints for model evaluation



Machine Learning: - Disease prediction models - Confidence-based

predictions



Dockerized Architecture: - Easy deployment - Environment consistency



Installation Steps 1. Clone the repository 2. Create a virtual

environment 3. Install packages using requirements.txt



Command: pip install -r requirements.txt



Running Applications (Without Docker)



Run FastAPI: cd Hitayu-Fastapi-V1 uvicorn app:app –host 0.0.0.0 –port

8000 –reload



Swagger Docs: http://127.0.0.1:8000/docs



Run Streamlit: cd Hitayu-Streamlit-Prototype streamlit run app.py



Streamlit URL: http://localhost:8501



Docker Commands Build Image: docker build -t hitayu-ai .



Run Container: docker run -p 8000:8000 -p 8501:8501 hitayu-ai



Author Indranil Bakshi

ss

