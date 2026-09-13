import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_ticket


st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)


st.title("🎫 Customer Support Ticket Classifier")

st.write(
    "Enter a customer support ticket to predict "
    "its support queue and priority."
)


subject = st.text_input(
    "Ticket Subject",
    placeholder="Example: Internet connection problem"
)


body = st.text_area(
    "Ticket Description",
    placeholder="Describe the customer's problem here...",
    height=200
)


if st.button("Predict"):

    if not subject.strip() and not body.strip():

        st.warning(
            "Please enter a ticket subject or description."
        )

    else:

        result = predict_ticket(
            subject=subject,
            body=body
        )

        st.subheader("Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Support Queue",
                result["queue"]
            )

        with col2:
            st.metric(
                "Priority",
                result["priority"].upper()
            )