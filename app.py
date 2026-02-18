# -*- coding: utf-8 -*-
"""
Health Assistant App
- Disease Prediction Modules
- Gemini Medical Chatbot (Modern UI)
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from dotenv import load_dotenv

# LangChain (Gemini only)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# =====================================================
# CONFIG
# =====================================================

load_dotenv()

st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# =====================================================
# LOAD MODELS (cached for performance)
# =====================================================

@st.cache_resource
def load_models():
    diabetes = pickle.load(open("diabetes_model.sav", "rb"))
    heart = pickle.load(open("heartdisease_model.sav", "rb"))
    insurance = pickle.load(open("insurance_cost.sav", "rb"))
    calories = pickle.load(open("calories_model.sav", "rb"))
    return diabetes, heart, insurance, calories

diabetes_model, heart_disease_model, insurance_cost_model, calories_model = load_models()

# =====================================================
# GEMINI MODELS
# =====================================================

gemini_flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

gemini_pro = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2
)

gemini_flash_backup = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

MEDICAL_SYSTEM_PROMPT = """
You are a medical information assistant.

Rules:
- Provide informational guidance only.
- Never diagnose diseases.
- Suggest consulting a doctor for serious symptoms.
- Be concise, calm, and clear.
"""

# =====================================================
# CHATBOT LOGIC
# =====================================================

def get_medical_response(user_query, chat_history):

    messages = [SystemMessage(content=MEDICAL_SYSTEM_PROMPT)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_query))

    try:
        return gemini_flash.invoke(messages).content
    except Exception as e:
        print("Gemini 2.5 Flash failed:", e)

    try:
        return gemini_pro.invoke(messages).content
    except Exception as e:
        print("Gemini Pro failed:", e)

    try:
        return gemini_flash_backup.invoke(messages).content
    except Exception as e:
        print("All Gemini models failed:", e)
        return "Sorry — the assistant is currently unavailable."

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    selected = option_menu(
        "Health Care System",
        [
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Medical Insurance Cost Calculator",
            "Calories Burnt Calculator",
            "Medical Chatbot"
        ],
        icons=["activity", "heart", "currency-dollar", "fire", "chat-dots"],
        menu_icon="hospital-fill",
        default_index=0
    )

# =====================================================
# DIABETES PREDICTION
# =====================================================

if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")

    col1, col2, col3 = st.columns(3)

    with col1: Pregnancies = st.text_input("Pregnancies")
    with col2: Glucose = st.text_input("Glucose")
    with col3: BloodPressure = st.text_input("Blood Pressure")

    with col1: SkinThickness = st.text_input("Skin Thickness")
    with col2: Insulin = st.text_input("Insulin")
    with col3: BMI = st.text_input("BMI")

    with col1: DPF = st.text_input("Diabetes Pedigree Function")
    with col2: Age = st.text_input("Age")

    if st.button("Diabetes Test Result"):
        try:
            values = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DPF), float(Age)
            ]
            pred = diabetes_model.predict([values])
            st.success("The person is diabetic" if pred[0] == 1 else "The person is not diabetic")
        except:
            st.error("Please enter valid numeric values.")

# =====================================================
# HEART DISEASE PREDICTION
# =====================================================

if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1: age = st.text_input("Age")
    with col2: sex = st.text_input("Sex (0 male, 1 female)")
    with col3: cp = st.text_input("Chest Pain Type")

    with col1: trestbps = st.text_input("Resting BP")
    with col2: chol = st.text_input("Cholesterol")
    with col3: fbs = st.text_input("Fasting Blood Sugar")

    with col1: restecg = st.text_input("Rest ECG")
    with col2: thalach = st.text_input("Max Heart Rate")
    with col3: exang = st.text_input("Exercise Angina")

    with col1: oldpeak = st.text_input("ST Depression")
    with col2: slope = st.text_input("Slope")
    with col3: ca = st.text_input("Major Vessels")

    with col1: thal = st.text_input("Thal")

    if st.button("Heart Disease Test Result"):
        try:
            values = [
                float(age), float(sex), float(cp), float(trestbps),
                float(chol), float(fbs), float(restecg), float(thalach),
                float(exang), float(oldpeak), float(slope),
                float(ca), float(thal)
            ]
            pred = heart_disease_model.predict([values])
            st.success("Heart disease detected" if pred[0] == 1 else "No heart disease detected")
        except:
            st.error("Please enter valid numeric values.")

# =====================================================
# INSURANCE COST
# =====================================================

if selected == "Medical Insurance Cost Calculator":
    st.title("Medical Insurance Cost Calculator")

    col1, col2, col3 = st.columns(3)

    with col1: age = st.text_input("Age")
    with col2: sex = st.text_input("Sex (0 male, 1 female)")
    with col3: bmi = st.text_input("BMI")

    with col1: children = st.text_input("Children")
    with col2: smoker = st.text_input("Smoker (0 yes, 1 no)")
    with col3: region = st.text_input("Region (0-3)")

    if st.button("Calculate Insurance"):
        try:
            vals = [float(age), float(sex), float(bmi), float(children), float(smoker), float(region)]
            pred = insurance_cost_model.predict([vals])
            st.success(f"Estimated Insurance Cost: ${pred[0]:.2f}")
        except:
            st.error("Please enter valid numeric values.")

# =====================================================
# CALORIES BURNT
# =====================================================

if selected == "Calories Burnt Calculator":
    st.title("Calories Burnt Calculator")

    col1, col2, col3, col4 = st.columns(4)

    with col1: Gender = st.text_input("Gender (0 male, 1 female)")
    with col2: Age = st.text_input("Age")
    with col3: Height = st.text_input("Height (cm)")
    with col4: Weight = st.text_input("Weight (lbs)")

    with col1: Duration = st.text_input("Duration (mins)")
    with col2: Heart_Rate = st.text_input("Heart Rate")
    with col3: Body_Temp = st.text_input("Body Temp (°C)")

    if st.button("Calculate Calories Burnt"):
        try:
            vals = [0, float(Gender), float(Age), float(Height),
                    float(Weight), float(Duration),
                    float(Heart_Rate), float(Body_Temp)]
            pred = calories_model.predict([vals])
            st.success(f"Calories Burnt: {pred[0]:.2f}")
        except:
            st.error("Please enter valid numeric values.")

# =====================================================
# MEDICAL CHATBOT (MODERN UI)
# =====================================================

if selected == "Medical Chatbot":

    st.title("Medical Chatbot 💬")
    st.warning("⚠️ Informational only — not a medical diagnosis.")

    # ---------- Chat CSS ----------
    st.markdown("""
    <style>
    .chat-container {
        max-height: 520px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 12px;
        background: #0E1117;
    }
    .user-msg {
        background:#2E7DFF;
        color:white;
        padding:10px 14px;
        border-radius:16px 16px 0 16px;
        margin:8px 0;
        max-width:70%;
        margin-left:auto;
    }
    .bot-msg {
        background:#262730;
        color:white;
        padding:10px 14px;
        border-radius:16px 16px 16px 0;
        margin:8px 0;
        max-width:70%;
        margin-right:auto;
    }
    </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(f'<div class="user-msg">{msg.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">{msg.content}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Input row
    col1, col2 = st.columns([8, 1])

    with col1:
        user_input = st.text_input(
            "Type message",
            label_visibility="collapsed",
            placeholder="Ask a health question..."
        )

    with col2:
        send = st.button("➤")

    if send and user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        with st.spinner("Thinking..."):
            reply = get_medical_response(
                user_input,
                st.session_state.chat_history[:-1]
            )

        st.session_state.chat_history.append(AIMessage(content=reply))
        st.rerun()

       