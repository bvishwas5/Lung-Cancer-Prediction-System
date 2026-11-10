# 🫁 Lung Cancer Prediction System

An educational Machine Learning project that uses a **Random Forest Classifier** and **Llama 3.2 (Ollama)** to demonstrate lung cancer risk prediction based on user-provided survey features.

## 🚀 Project Overview

This project combines Machine Learning and Local Large Language Models (LLMs) to build an interactive prediction application using Streamlit.

Users can enter patient-related information, receive a model-generated prediction with probability scores, and view an AI-generated educational explanation.

## ✨ Features

- Interactive Streamlit web application.
- Random Forest classification model.
- 15 input features.
- YES / NO prediction results.
- Prediction probability percentages.
- Reported model test accuracy of 91.94%.
- Local AI explanation using Llama 3.2.
- Privacy-conscious local LLM integration through Ollama.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| Scikit-learn | Machine Learning |
| Random Forest | Classification algorithm |
| Streamlit | Web application |
| Joblib | Model loading |
| Ollama | Local LLM integration |
| Llama 3.2 | AI-generated explanations |

## 📂 Project Structure

```text
Lung_Cancer_Prediction/
│
├── app.py
├── lung_cancer_model.pkl
├── lung_cancer.csv
├── Lung_Cancer_Prediction_Model.ipynb
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd Lung_Cancer_Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and configure Ollama

Download Ollama:

https://ollama.com/

Pull the Llama 3.2 model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running locally.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Model Information

- Algorithm: Random Forest Classifier
- Reported test accuracy: 91.94%
- Input features: 15
- Output: YES / NO
- Probability scores: YES and NO

## 🤖 AI Integration

The application integrates Llama 3.2 through Ollama to provide simple educational explanations of the machine learning output.

The AI explanation is generated locally using the Ollama model.

## ⚠️ Disclaimer

This project is developed strictly for educational and demonstration purposes.

It is not a clinically validated medical tool and must not be used for medical diagnosis, treatment decisions, or healthcare recommendations.

The reported model accuracy does not represent clinical diagnostic accuracy.

Always consult qualified healthcare professionals for medical concerns.

## 👨‍💻 Author

**Vishwas Basabathini**

GitHub: https://github.com/bvishwas5

---
