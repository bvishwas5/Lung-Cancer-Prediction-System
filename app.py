
import streamlit as st
import pandas as pd
import joblib
import ollama

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Lung Cancer Prediction System",
    page_icon="🫁",
    layout="centered"
)

# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("lung_cancer_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("🫁 Lung Cancer Prediction System")

st.write(
    "An educational project combining "
    "Machine Learning and Local AI."
)

st.caption(
    "Technology: Random Forest + Ollama + Llama 3.2"
)

st.warning(
    "⚠️ Educational purposes only. "
    "This application is not a medical diagnostic tool."
)

st.divider()

# ==========================================
# PATIENT INFORMATION
# ==========================================

st.subheader("👤 Patient Information")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)

def yes_no(label):
    return st.selectbox(
        label,
        ["No", "Yes"]
    )

smoking = yes_no("Smoking")
yellow_fingers = yes_no("Yellow Fingers")
anxiety = yes_no("Anxiety")
peer_pressure = yes_no("Peer Pressure")
chronic_disease = yes_no("Chronic Disease")
fatigue = yes_no("Fatigue")
allergy = yes_no("Allergy")
wheezing = yes_no("Wheezing")
alcohol_consuming = yes_no("Alcohol Consuming")
coughing = yes_no("Coughing")
shortness_of_breath = yes_no("Shortness of Breath")
swallowing_difficulty = yes_no("Swallowing Difficulty")
chest_pain = yes_no("Chest Pain")

st.divider()

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button(
    "🔍 Predict Lung Cancer",
    type="primary",
    use_container_width=True
):

    # ======================================
    # PREPARE INPUT DATA
    # ======================================

    patient = [[
        1 if gender == "Male" else 0,
        age,
        1 if smoking == "Yes" else 0,
        1 if yellow_fingers == "Yes" else 0,
        1 if anxiety == "Yes" else 0,
        1 if peer_pressure == "Yes" else 0,
        1 if chronic_disease == "Yes" else 0,
        1 if fatigue == "Yes" else 0,
        1 if allergy == "Yes" else 0,
        1 if wheezing == "Yes" else 0,
        1 if alcohol_consuming == "Yes" else 0,
        1 if coughing == "Yes" else 0,
        1 if shortness_of_breath == "Yes" else 0,
        1 if swallowing_difficulty == "Yes" else 0,
        1 if chest_pain == "Yes" else 0
    ]]

    columns = [
        "GENDER",
        "AGE",
        "SMOKING",
        "YELLOW_FINGERS",
        "ANXIETY",
        "PEER_PRESSURE",
        "CHRONIC DISEASE",
        "FATIGUE ",
        "ALLERGY ",
        "WHEEZING",
        "ALCOHOL CONSUMING",
        "COUGHING",
        "SHORTNESS OF BREATH",
        "SWALLOWING DIFFICULTY",
        "CHEST PAIN"
    ]

    patient_df = pd.DataFrame(
        patient,
        columns=columns
    )

    try:

        # ==================================
        # RANDOM FOREST PREDICTION
        # ==================================

        prediction = model.predict(patient_df)[0]

        probability = model.predict_proba(patient_df)[0]

        # Find probabilities using model class labels
        classes = list(model.classes_)

        yes_index = classes.index(1)
        no_index = classes.index(0)

        yes_probability = probability[yes_index] * 100
        no_probability = probability[no_index] * 100

        prediction_text = (
            "YES" if prediction == 1 else "NO"
        )

        # ==================================
        # DISPLAY PREDICTION RESULT
        # ==================================

        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.error("⚠️ Model Prediction: YES")
        else:
            st.success("✅ Model Prediction: NO")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Probability of YES",
                f"{yes_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Probability of NO",
                f"{no_probability:.2f}%"
            )

        st.info(
            "These values represent the model's estimated "
            "class probabilities, not a person's actual "
            "medical probability of having lung cancer."
        )

        # ==================================
        # MODEL PERFORMANCE
        # ==================================

        st.divider()
        st.subheader("📈 Model Performance")

        st.metric(
            "Random Forest Test Accuracy",
            "91.94%"
        )

        st.write(
            "The Random Forest model achieved 91.94% "
            "accuracy on the held-out test dataset."
        )

        st.info(
            "This reported accuracy is based on the "
            "project's test dataset. It is not clinical "
            "validation and does not establish real-world "
            "diagnostic accuracy."
        )

        # ==================================
        # LOCAL AI EXPLANATION
        # ==================================

        st.divider()
        st.subheader("🤖 AI Explanation")

        prompt = f"""
You are an educational AI assistant integrated
into a student machine learning project.

The project uses a Random Forest model to demonstrate
lung cancer prediction using survey-based input features.

INPUT INFORMATION:
Gender: {gender}
Age: {age}
Smoking: {smoking}
Yellow Fingers: {yellow_fingers}
Anxiety: {anxiety}
Peer Pressure: {peer_pressure}
Chronic Disease: {chronic_disease}
Fatigue: {fatigue}
Allergy: {allergy}
Wheezing: {wheezing}
Alcohol Consuming: {alcohol_consuming}
Coughing: {coughing}
Shortness of Breath: {shortness_of_breath}
Swallowing Difficulty: {swallowing_difficulty}
Chest Pain: {chest_pain}

MACHINE LEARNING OUTPUT:

Prediction: {prediction_text}
Probability of YES: {yes_probability:.2f}%
Probability of NO: {no_probability:.2f}%

TASK:

Explain the machine learning output in simple,
clear educational language.

Include:
1. What the model predicted.
2. What the two probabilities represent.
3. How the provided input features relate to
   the model's prediction.
4. Why this result should not be treated as
   a medical diagnosis.

IMPORTANT SAFETY RULES:

- Never diagnose the person.
- Never claim that the person definitely has cancer.
- Never claim that the person definitely does not
  have cancer.
- Do not interpret model probabilities as actual
  medical risk.
- Recommend professional medical evaluation for
  health concerns.

Keep the explanation concise and easy to understand.
"""

        # ==================================
        # OLLAMA - LLAMA 3.2 INTEGRATION
        # ==================================

        try:

            with st.spinner(
                "🤖 Llama 3.2 is generating AI explanation..."
            ):

                response = ollama.chat(
                    model="llama3.2:latest",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    options={
                        "temperature": 0.2
                    }
                )

            ai_explanation = response["message"]["content"]

            st.markdown(ai_explanation)

            st.caption(
                "Generated locally using Ollama - Llama 3.2"
            )

        except Exception as e:

            st.error(
                f"LLM Error: {e}"
            )

            st.info(
                "Please check whether Ollama is running "
                "and the Llama 3.2 model is available."
            )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Lung Cancer Prediction System | "
    "Random Forest + Ollama + Llama 3.2"
)

st.caption(
    "Educational demonstration only. "
    "Not intended for medical diagnosis."
)
