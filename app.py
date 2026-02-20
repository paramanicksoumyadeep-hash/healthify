# -*- coding: utf-8 -*-

import pickle
import streamlit as st
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

@st.cache_resource
def load_models():
    diabetes = pickle.load(open("diabetes_model.sav", "rb"))
    heart = pickle.load(open("heartdisease_model.sav", "rb"))
    insurance = pickle.load(open("insurance_cost.sav", "rb"))
    calories = pickle.load(open("calories_model.sav", "rb"))
    return diabetes, heart, insurance, calories

diabetes_model, heart_disease_model, insurance_cost_model, calories_model = load_models()

gemini_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
gemini_pro = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
gemini_flash_backup = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

MEDICAL_SYSTEM_PROMPT = """
You are a medical information assistant.
Provide informational guidance only.
Never diagnose diseases.
Suggest consulting a doctor for serious symptoms.
Be concise and clear.
"""

def get_medical_response(user_query, chat_history):
    messages = [SystemMessage(content=MEDICAL_SYSTEM_PROMPT)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_query))
    try:
        return gemini_flash.invoke(messages).content
    except:
        try:
            return gemini_pro.invoke(messages).content
        except:
            try:
                return gemini_flash_backup.invoke(messages).content
            except:
                return "Assistant unavailable."

def animated_speedometer(title, value, min_val, max_val, median_val):
    green_zone = min_val + (max_val - min_val) * 0.4
    yellow_zone = min_val + (max_val - min_val) * 0.7

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={"reference": median_val},
        title={"text": f"<b>{title}</b>"},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": "white", "thickness": 0.3},
            "steps": [
                {"range": [min_val, green_zone], "color": "#2ecc71"},
                {"range": [green_zone, yellow_zone], "color": "#f1c40f"},
                {"range": [yellow_zone, max_val], "color": "#e74c3c"}
            ],
            "threshold": {
                "line": {"color": "cyan", "width": 4},
                "thickness": 0.8,
                "value": median_val
            }
        }
    ))

    fig.update_layout(
        height=420,
        margin=dict(t=50, b=0, l=0, r=0),
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        transition={'duration': 800}
    )

    return fig

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

if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")

    col1, col2, col3 = st.columns(3)

    with col1: Pregnancies = st.number_input("Pregnancies", 0, 20)
    with col2: Glucose = st.number_input("Glucose", 0.0, 200.0)
    with col3: BloodPressure = st.number_input("Blood Pressure", 0.0, 130.0)

    with col1: SkinThickness = st.number_input("Skin Thickness", 0.0, 100.0)
    with col2: Insulin = st.number_input("Insulin", 0.0, 900.0)
    with col3: BMI = st.number_input("BMI", 10.0, 70.0)

    with col1: DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    with col2: Age = st.number_input("Age", 18, 100)

    if st.button("Diabetes Test Result"):
        values = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
        pred = diabetes_model.predict([values])
        st.success("The person is diabetic" if pred[0] == 1 else "The person is not diabetic")

    if st.checkbox("Show Health Gauge"):
        metrics = {
            "Glucose": {"value": Glucose, "min": 0, "max": 200, "median": 117},
            "Blood Pressure": {"value": BloodPressure, "min": 0, "max": 130, "median": 72},
            "BMI": {"value": BMI, "min": 10, "max": 70, "median": 32},
            "Age": {"value": Age, "min": 18, "max": 100, "median": 29}
        }

        metric = st.selectbox("Select Metric", list(metrics.keys()))
        data = metrics[metric]
        st.plotly_chart(animated_speedometer(metric, data["value"], data["min"], data["max"], data["median"]), use_container_width=True)

if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1: age = st.number_input("Age", 20, 100)
    with col2: sex = st.number_input("Sex (0 male, 1 female)", 0, 1)
    with col3: cp = st.number_input("Chest Pain Type", 0, 3)

    with col1: trestbps = st.number_input("Resting BP", 80.0, 220.0)
    with col2: chol = st.number_input("Cholesterol", 100.0, 600.0)
    with col3: fbs = st.number_input("Fasting Blood Sugar", 0, 1)

    with col1: restecg = st.number_input("Rest ECG", 0, 2)
    with col2: thalach = st.number_input("Max Heart Rate", 60.0, 220.0)
    with col3: exang = st.number_input("Exercise Angina", 0, 1)

    with col1: oldpeak = st.number_input("ST Depression", 0.0, 7.0)
    with col2: slope = st.number_input("Slope", 0, 2)
    with col3: ca = st.number_input("Major Vessels", 0, 4)

    with col1: thal = st.number_input("Thal", 0, 3)

    if st.button("Heart Disease Test Result"):
        values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                  exang, oldpeak, slope, ca, thal]
        pred = heart_disease_model.predict([values])
        st.success("Heart disease detected" if pred[0] == 1 else "No heart disease detected")

    if st.checkbox("Show Health Gauge"):
        metrics = {
            "Cholesterol": {"value": chol, "min": 100, "max": 600, "median": 245},
            "Max Heart Rate": {"value": thalach, "min": 60, "max": 220, "median": 150},
            "ST Depression": {"value": oldpeak, "min": 0, "max": 7, "median": 0.8}
        }

        metric = st.selectbox("Select Metric", list(metrics.keys()))
        data = metrics[metric]
        st.plotly_chart(animated_speedometer(metric, data["value"], data["min"], data["max"], data["median"]), use_container_width=True)

if selected == "Medical Chatbot":
    st.title("Medical Chatbot 💬")
    st.warning("Informational only — not a medical diagnosis.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    user_input = st.chat_input("Ask a health question...")

    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = get_medical_response(user_input, st.session_state.chat_history[:-1])
                st.markdown(reply)
        st.session_state.chat_history.append(AIMessage(content=reply))
