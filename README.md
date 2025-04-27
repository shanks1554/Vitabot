# `Vitabot`

## _Your AI-Driven Healthcare Chatbot for Accessible and Personalized Medical Support._

VitaBot is an advanced healthcare chatbot designed to bridge the gap between expert medical resources and everyday users. It employs cutting-edge AI technologies to predict diseases, recommend preventive measures, and provide actionable insights to empower proactive health management.
___
# Features

- __`Health Query Responses:`__ Provides accurate and tailored answers to a range of health-related questions.

- __`Symptom-Based Disease Prediction:`__ Uses machine learning algorithms to predict potential diseases based on user-reported symptoms.

- __`Medication and Preventive Suggestions:`__ Offers evidence-based recommendations to guide users in managing their health.

- __`User-Friendly Interaction:`__ Simplifies complex medical terminology for clear communication.

- __`Scalable Deployment:`__ Deployable on local servers or cloud platforms like Streamlit.
___
# Technology Stack

-  __`Programming Language:`__ Python

-  __`Core Libraries:`__
    - Hugging Face (Mistral-7B-Instruct-v0.3)
    
    - SentenceTransformer
    
    - FAISS (Facebook AI Similarity Search)
    
    - Streamlit

-  __`Frameworks:`__ Streamlit for UI, FAISS for similarity search

-  __`Database:`__ FAISS for vector embedding storage

-  __`Deployment: Cloud (optional) or local`__
___
# Prerequisites 

- Python >= 3.9

- pip (Python Package Installer)

- __Required Libraries:__

   ```bash
    pip install -r requirements.txt
    ```

- GPU-enabled system (optional but recommended for efficiency)
___
# Installation

1. __Clone the Repository:__

    ```bash
    git clone https://github.com/shanks1554/vitabot.git
    cd vitabot
    ```

2. __Install Dependencies:__

    ```bash
    pip install -r requirements.txt
    ```

3. __Set Up Vector Embedding Storage:__

    - Preprocess raw data into chunks.

    - Generate vector embeddings using `SentenceTransformer`.

    - Store embeddings in FAISS for efficient retrieval.

4. __`Environment Variables:`__ Create a `.env` file with:

    ```bash 
    API_KEY=your_huggingface_api_key
    ```

5. __Run the Application:__

    ```bash
    streamlit run app.py
    ```
___
# Usage

1. __Start Interaction:__ Open `http://localhost:8501` in a web browser to access the chatbot interface.

2. __Query Examples:__

- "What are the common symptoms of diabetes?"

- "Suggest medications for mild fever."

3. __Real-Time Responses:__ Experience intuitive and context-aware answers through FAISS retrieval and Mistral-7B fallback.
___
# Limitations and Future Enhancements:

## Current Limitations:

* Text-based interaction only.

* Limited domain-specific knowledge.

* Latency during complex queries.

## Future Enhancements:

* Integrate voice-to-text capabilities.

* Deploy to scalable cloud environments.

* Expand domain expertise with specialized datasets.
___
Feel free to adapt this README to your specific deployment setup or project structure! Let me know if you'd like to expand on any section further. 🚀

