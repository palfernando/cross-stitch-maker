import base64
import streamlit as st

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_data
def get_base64_cached(bin_file):
    return get_base64(bin_file)

def set_bg(png_file):
    st.set_page_config(page_title="Stitch Pattern Generator", page_icon=":blossom:")
    bin_str = get_base64_cached(png_file)
    page_bg_img = """
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
    """ % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
