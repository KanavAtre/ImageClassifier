import streamlit as st
from bs4 import BeautifulSoup
import requests
import spacy
import pandas as pd
import json
import pandas as pd


dates = []

st.title("MLA Format Citation Generator")
url = st.text_area("Enter the website/article URL")
button = st.button("Generate APA Citation")

if "en_core_web_sm" not in spacy.util.get_installed_models():
    spacy.cli.download("en_core_web_sm")
