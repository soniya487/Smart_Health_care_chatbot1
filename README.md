# Smart_Health_care_chatbot
An intelligent, explainable healthcare chatbot that guides users through symptom-based conversations using a hybrid AI architecture combining machine learning, semantic understanding, and rule-based medical reasoning.

Project Overview

The Smart Healthcare Chatbot is designed to simulate how a real digital health assistant reasons through user-reported symptoms in a safe, structured, and human-like manner.

Unlike purely hardcoded chatbots or black-box models, this system follows a transparent, multi-stage reasoning pipeline that progressively refines vague user input into meaningful medical guidance through:

-Follow-up questioning

-Symptom prioritization

-Severity assessment

-Safety-aware triage

The goal is to move from unknown → known symptoms while maintaining explainability at every step.



**##  Key Features**

### 🔹 Hybrid AI Reasoning
- Machine Learning: `TF-IDF + Logistic Regression`
- Semantic Similarity Search: `MiniLM embeddings`
- Rule-based medical logic and safety checks

### 🔹 Multi-turn Conversational Flow
- Context-aware follow-up questions
- Session-level memory
- Progressive symptom refinement

### 🔹 Medical Safety & Triage
- Severe pain detection
- Blood pressure & metabolic red-flag checks
- Emergency escalation logic

### 🔹 Explainable AI
- “Why?” reasoning support
- Visibility into:
  - model confidence
  - symptom flags
  - decision flow

### 🔹 Interactive Web Interface
- Streamlit-based medical-style UI
- Optional voice input support

---
**##  System Architecture**

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/491f8c63-9d0e-4cb9-b04b-5826a5259917" />
---
  
**⚙️ Technology Stack**

The complete implementation was developed in Python, with:

**• Google Colab used for notebook-based experimentation and validation**

• VS Code + Streamlit used for deployment as an interactive web application 

🔹 Machine Learning Models

TF-IDF Vectorizer

Logistic Regression Classifier

MiniLM Sentence Transformer (semantic embeddings)

🔹 Frameworks & Libraries

Streamlit

scikit-learn

SentenceTransformers

PyTorch

📊 Dataset

The project uses a symptom–question–answer healthcare dataset to support:

Symptom classification

Follow-up question generation

Severity and safety triage

Explainable reasoning

✅ The full dataset is included in this repository and is used for:

Model training

Semantic similarity search

Rule-based medical reasoning

**All data is used strictly for educational and research purposes.**

▶️ Running the Application
1️⃣ Clone the repository
git clone https://github.com/soniya487/Smart-Health-care-chatbot.git
cd Smart-Health-care-chatbot

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py

**If not use Google colab notebbok**

The repository includes a fully structured Google Colab notebook explaining:

Data preprocessing

Model training

Rule-engine design

Hybrid decision logic

Output explainability

 ****This notebook demonstrates the core intelligence behind the chatbot independently of the web UI.**

**Future Improvements**

LLM-based medical summarization

Improve output quality

Clinical ontology integration

Cloud deployment

Enhanced multilingual support

