import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in project root
load_dotenv(Path(__file__).parent.parent / ".env")

st.set_page_config(
    page_title="Universe Context Generator",
    page_icon="🌍",
    layout="wide",
)

st.title("Universe Context Generator")
st.markdown(
    """
Welcome to the Universe Context Generator! This tool helps you create and manage universe contexts and belief evaluations for those universes.

### Navigation
- **Universe Context Generation**: Use the Universe Context page to create and manage universe contexts
- **Belief Evaluation Generation**: Use the Belief Eval page to generate various types of evaluations

Please select a page from the sidebar to get started.
"""
)
