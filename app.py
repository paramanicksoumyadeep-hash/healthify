import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Luxury Risk Dashboard", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.lux-card {
    background: linear-gradient(145deg, #111827, #1f2937);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 0 30px rgba(255, 215, 0, 0.15);
}
h1, h2, h3, h4 {
    color: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)

def create_dark_luxury_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            "font": {"size": 36, "color": "#FFD700"},
        },
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 22, "color": "white"}
        },
        gauge={
            "axis": {
                "range": [0, 39],
                "tickcolor": "#888",
                "tickwidth": 1,
                "tickfont": {"color": "white", "size": 12}
            },
            "bar": {
                "color": "#FFD700",
                "thickness": 0.3
            },
            "bgcolor": "#111827",
            "borderwidth": 3,
            "bordercolor": "#333",
            "steps": [
                {"range": [0, 13], "color": "#0f5132"},
                {"range": [13, 26], "color": "#664d03"},
                {"range": [26, 39], "color": "#842029"}
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="#0e1117",
        font={"family": "Arial"}
    )

    return fig

data = {
    "Heart Rate": np.random.randint(0, 39),
    "Blood Pressure": np.random.randint(0, 39),
    "Cholesterol": np.random.randint(0, 39),
    "Glucose": np.random.randint(0, 39),
    "BMI": np.random.randint(0, 39)
}

st.markdown("<h1 style='text-align:center; color:#FFD700;'>Luxury Health Risk Dashboard</h1>", unsafe_allow_html=True)

selected_metric = st.selectbox("Select Metric", list(data.keys()))

selected_value = data[selected_metric]

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.plotly_chart(
        create_dark_luxury_gauge(selected_value, selected_metric),
        use_container_width=True
    )
