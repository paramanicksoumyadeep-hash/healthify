# 🩺 HealthifyX – Healthcare Analytics & Medical Chatbot

**HealthifyX** is an interactive **healthcare analytics application** built using Python and data science techniques.  
It leverages **machine learning models** for health-related predictions and includes an **AI-powered medical chatbot** that provides basic medical guidance to users.

🚀 **Live App:** https://healthifyx.streamlit.app/

---

## 🔍 Overview

HealthifyX combines **data analytics, machine learning, and conversational AI** to deliver insights related to healthcare and wellness.  
The application focuses on three major predictive analytics tasks along with an intelligent chatbot interface.

---

## 🧠 Key Functionalities

### 📊 Healthcare Predictions
The app uses **regression and classification models** to predict:

- ❤️ **Heart Disease Detection** (Classification)
- 🔥 **Calories Burnt Estimation** (Linear Regression)
- 💰 **Medical Insurance Cost Prediction** (Linear Regression)

These models are trained using structured healthcare datasets and implemented using **Python and Pandas**.

---

### 🤖 Medical Chatbot
- Built using **LangChain**
- Powered by **Gemini-2.5-Flash**
- Provides **basic medical advice**, health tips, and general wellness guidance
- Designed strictly for **informational purposes**, not medical diagnosis

---

## 🧰 Tech Stack

- **Programming Language:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **App Framework:** :contentReference[oaicite:0]{index=0}  
- **LLM Integration:** LangChain  
- **Medical Chatbot LLM:** :contentReference[oaicite:1]{index=1} (Gemini-2.5-Flash)  
- **Deployment:** Streamlit Community Cloud  

---

## ⚙️ Application Workflow

### 1️⃣ Data Processing
- Healthcare datasets are cleaned and processed using **Pandas**
- Feature engineering is applied where required

### 2️⃣ Model Training
- **Linear Regression** for:
  - Calories Burnt Prediction
  - Insurance Cost Prediction
- **Classification Models** for:
  - Heart Disease Detection

### 3️⃣ Predictions
- Users input health-related parameters
- Models generate real-time predictions through the Streamlit interface

### 4️⃣ Medical Chatbot
- User queries are passed through LangChain
- Gemini-2.5-Flash generates contextual medical responses

---

## ✨ Features

- 📈 Real-time healthcare predictions  
- ❤️ Heart disease risk classification  
- 🔥 Calories burn estimation  
- 💰 Insurance cost forecasting  
- 🤖 AI-based medical chatbot  
- 🌐 Fully web-based & interactive UI  

---

## 🖥️ Local Installation

```bash
# Clone the repository
git clone https://github.com/your-username/healthifyx.git
cd healthifyx

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
