import streamlit as st
import requests

st.title("Toxic Comment Classifier")

comment = st.text_area("Enter your comment:")

if st.button("Classify"):
    if comment.strip():
        print("HERE")
        response = requests.post("http://127.0.0.1:8000/predict", json={"text": comment})
        if response.status_code == 200:
            labels = response.json()["labels"]
            st.success("Predicted Labels: " + ", ".join(labels) if labels else "Not toxic")
        else:
            st.error("API error.")
    else:
        st.warning("Please enter a comment.")