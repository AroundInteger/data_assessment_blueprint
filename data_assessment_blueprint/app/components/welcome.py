import streamlit as st
import json
from pathlib import Path

def load_glossary():
    """Load statistical glossary from JSON file."""
    glossary_path = Path(__file__).parent.parent / "data" / "glossary.json"
    if glossary_path.exists():
        with open(glossary_path, 'r') as f:
            return json.load(f)
    return {}

def show_welcome_screen():
    """Display the welcome screen with app overview and navigation."""
    st.title("Data Assessment Blueprint")
    
    # App Overview
    st.markdown("""
    Welcome to the Data Assessment Blueprint! This tool helps you create robust statistical analysis plans
    through an interactive, step-by-step process.
    """)
    
    # Quick Start Guide
    with st.expander("Quick Start Guide", expanded=True):
        st.markdown("""
        1. **Upload Your Data**: Start by uploading your dataset in CSV, Excel, or JSON format
        2. **Data Preview**: Review your data structure and basic statistics
        3. **Analysis Planning**: Follow the interactive guide to select appropriate statistical tests
        4. **Generate Plan**: Get a comprehensive analysis plan with code examples
        """)
    
    # Session Management
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start New Analysis"):
            st.session_state.clear()
            st.session_state['current_step'] = 'data_upload'
            st.experimental_rerun()
    
    with col2:
        if st.button("Load Previous Session"):
            # TODO: Implement session loading
            st.info("Session loading coming soon!")
    
    # Statistical Glossary
    with st.sidebar:
        st.header("Statistical Glossary")
        glossary = load_glossary()
        if glossary:
            for term, definition in glossary.items():
                with st.expander(term):
                    st.markdown(definition)
        else:
            st.info("Glossary will be available soon!")

def initialize_session():
    """Initialize session state variables."""
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = 'welcome'
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'analysis_plan' not in st.session_state:
        st.session_state['analysis_plan'] = {} 