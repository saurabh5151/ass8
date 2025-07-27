import streamlit as st

st.set_page_config(page_title="Loan Chatbot", layout="centered")

st.title("Loan Approval Chatbot")

# Simple input and response
user_input = st.text_input("Ask about loan approval:")

if user_input:
    st.write("This is where your chatbot response will appear.")
