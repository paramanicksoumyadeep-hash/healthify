# -*- coding: utf-8 -*-

import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

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

    if st.checkbox("Show Dataset Metrics"):
        df = pd.DataFrame({
            "Feature":["Pregnancies","Glucose","BloodPressure","BMI","Age"],
            "Median":[3,117,72,32,29]
        })
        fig, ax = plt.subplots()
        ax.bar(df["Feature"], df["Median"])
        ax.set_title("Median Values - Diabetes Dataset")
        st.pyplot(fig)

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

    if st.checkbox("Show Dataset Metrics"):
        df = pd.DataFrame({
            "Metric":["Age","Cholesterol","MaxHR"],
            "Median":[55,245,150]
        })
        fig, ax = plt.subplots()
        ax.bar(df["Metric"], df["Median"])
        ax.set_title("Median Values - Heart Dataset")
        st.pyplot(fig)

if selected == "Medical Insurance Cost Calculator":
    st.title("Medical Insurance Cost Calculator")

    col1, col2, col3 = st.columns(3)

    with col1: age = st.number_input("Age", 18, 100)
    with col2: sex = st.number_input("Sex (0 male, 1 female)", 0, 1)
    with col3: bmi = st.number_input("BMI", 10.0, 60.0)

    with col1: children = st.number_input("Children", 0, 10)
    with col2: smoker = st.number_input("Smoker (0 yes, 1 no)", 0, 1)
    with col3: region = st.number_input("Region (0=NE, 1=NW, 2=SE, 3=SW)", 0, 3)

    if st.button("Calculate Insurance"):
        vals = [age, sex, bmi, children, smoker, region]
        pred = insurance_cost_model.predict([vals])
        st.success(f"Estimated Insurance Cost: ${pred[0]:.2f}")

    if st.checkbox("Show Dataset Metrics"):
        df = pd.DataFrame({
            "Metric":["Age","BMI","Children"],
            "Median":[39,30.4,1]
        })
        fig, ax = plt.subplots()
        ax.bar(df["Metric"], df["Median"])
        ax.set_title("Median Values - Insurance Dataset")
        st.pyplot(fig)

if selected == "Calories Burnt Calculator":
    st.title("Calories Burnt Calculator")

    col1, col2, col3, col4 = st.columns(4)

    with col1: Gender = st.number_input("Gender (0 male, 1 female)", 0, 1)
    with col2: Age = st.number_input("Age", 15, 100)
    with col3: Height = st.number_input("Height (cm)", 100.0, 250.0)
    with col4: Weight = st.number_input("Weight (lbs)", 30.0, 300.0)

    with col1: Duration = st.number_input("Duration (mins)", 1.0, 180.0)
    with col2: Heart_Rate = st.number_input("Heart Rate", 50.0, 200.0)
    with col3: Body_Temp = st.number_input("Body Temp (°C)", 35.0, 42.0)

    if st.button("Calculate Calories Burnt"):
        weight_kg = Weight * 0.453592
        vals = [0, Gender, Age, Height, weight_kg, Duration, Heart_Rate, Body_Temp]
        pred = calories_model.predict([vals])
        st.success(f"Calories Burnt: {pred[0]:.2f}")

    if st.checkbox("Show Dataset Metrics"):
        df = pd.DataFrame({
            "Metric":["Age","Height","Duration","HeartRate"],
            "Median":[42,170,15,103]
        })
        fig, ax = plt.subplots()
        ax.bar(df["Metric"], df["Median"])
        ax.set_title("Median Values - Calories Dataset")
        st.pyplot(fig)

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
       
