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
    """Display the welcome screen with project overview and navigation options."""
    st.markdown('<h1 class="main-header">Data Assessment Blueprint</h1>', unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("""
    Welcome to the Data Assessment Blueprint! This tool helps you:
    
    - Understand your data structure
    - Perform distributional analysis
    - Select appropriate statistical tests
    - Generate comprehensive analysis plans
    """)
    
    # Navigation Options
    st.markdown("### Get Started")
    if st.button("Begin Data Analysis"):
        st.session_state['current_step'] = 'data_input'
        st.rerun()
    
    # Project Information
    with st.expander("About This Project"):
        st.markdown("""
        This tool is designed to guide you through the process of statistical analysis,
        from data preparation to interpretation. It provides:
        
        - Interactive data exploration
        - Automated statistical test selection
        - Comprehensive analysis planning
        - Clear visualization options
        """)
    
    # Quick Start Guide
    with st.expander("Quick Start Guide"):
        st.markdown("""
        1. **Data Input**: Upload your data or use example datasets
        2. **Data Understanding**: Explore your data structure and characteristics
        3. **Distributional Analysis**: Examine data distributions and normality
        4. **Test Selection**: Choose appropriate statistical tests
        5. **Analysis Plan**: Generate a comprehensive analysis plan
        """)

def initialize_session():
    """Initialize session state variables if they don't exist."""
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = 'welcome'
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'analysis_type' not in st.session_state:
        st.session_state['analysis_type'] = None

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