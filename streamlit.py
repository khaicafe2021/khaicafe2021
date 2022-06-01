import streamlit as st
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
from io import BytesIO, StringIO


fileTypes = ["csv", "png", "jpg"]
ten = st.text_input('nhap ten')
st.write('tenbana: ', ten)
file = st.file_uploader("Upload file", type=fileTypes)
show_file = st.empty()
if not file:
    show_file.info("Please upload a file of type: " + ", ".join(["csv", "png", "jpg"]))


if isinstance(file, BytesIO):
    show_file.image(file)

