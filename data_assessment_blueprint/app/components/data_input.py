import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
from utils.data_generation import get_data_examples

def load_data_format_requirements():
    """Load data format requirements from JSON file."""
    requirements_path = Path(__file__).parent.parent / "data" / "format_requirements.json"
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            return json.load(f)
    return {}

def get_column_types(df):
    """Analyze and return column types with basic statistics."""
    column_info = {}
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            stats = {
                'type': 'numeric',
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isnull().sum()
            }
        elif pd.api.types.is_datetime64_dtype(col_type):
            stats = {
                'type': 'datetime',
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isnull().sum()
            }
        else:
            stats = {
                'type': 'categorical',
                'unique_values': df[col].nunique(),
                'missing': df[col].isnull().sum()
            }
        column_info[col] = stats
    return column_info

def show_data_input():
    """Display the data input interface with file upload and preview."""
    st.markdown('<h2 class="section-header">Data Input</h2>', unsafe_allow_html=True)
    
    # File Upload
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=['csv', 'xlsx', 'json'],
        help="Supported formats: CSV, Excel, JSON"
    )
    
    if uploaded_file is not None:
        try:
            # Read data based on file type
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            
            # Store data in session state
            st.session_state['data'] = data
            
            # Data Preview
            st.markdown("### Data Preview")
            st.dataframe(data.head())
            
            # Column Information
            st.markdown("### Column Information")
            column_info = get_column_types(data)
            
            for col, info in column_info.items():
                with st.expander(f"Column: {col}"):
                    st.write(f"Type: {info['type']}")
                    if info['type'] == 'numeric':
                        st.write(f"Mean: {info['mean']:.2f}")
                        st.write(f"Std: {info['std']:.2f}")
                        st.write(f"Range: [{info['min']:.2f}, {info['max']:.2f}]")
                    elif info['type'] == 'datetime':
                        st.write(f"Range: [{info['min']}, {info['max']}]")
                    else:
                        st.write(f"Unique values: {info['unique_values']}")
                    st.write(f"Missing values: {info['missing']}")
            
            # Data Cleaning Options
            st.markdown("### Data Cleaning Options")
            
            # Column Selection
            selected_columns = st.multiselect(
                "Select columns to keep",
                data.columns,
                default=data.columns.tolist()
            )
            
            if selected_columns:
                data = data[selected_columns]
                st.session_state['data'] = data
            
            # Missing Value Handling
            st.markdown("#### Handle Missing Values")
            missing_strategy = st.selectbox(
                "Select strategy for missing values",
                ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"]
            )
            
            if st.button("Apply Missing Value Strategy"):
                if missing_strategy == "Drop rows":
                    data = data.dropna()
                elif missing_strategy == "Fill with mean":
                    data = data.fillna(data.mean())
                elif missing_strategy == "Fill with median":
                    data = data.fillna(data.median())
                else:  # Fill with mode
                    data = data.fillna(data.mode().iloc[0])
                
                st.session_state['data'] = data
                st.success("Missing value strategy applied!")
                st.dataframe(data.head())
            
            # Data Type Conversion
            st.markdown("#### Convert Data Types")
            for col in data.columns:
                current_type = str(data[col].dtype)
                new_type = st.selectbox(
                    f"Convert {col} ({current_type}) to:",
                    ["Keep as is", "numeric", "datetime", "categorical"],
                    key=f"type_{col}"
                )
                
                if new_type != "Keep as is":
                    try:
                        if new_type == "numeric":
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                        elif new_type == "datetime":
                            data[col] = pd.to_datetime(data[col], errors='coerce')
                        else:  # categorical
                            data[col] = data[col].astype('category')
                        st.session_state['data'] = data
                    except Exception as e:
                        st.error(f"Error converting {col}: {str(e)}")
            
            # Save cleaned data
            if st.button("Save Cleaned Data"):
                st.session_state['data'] = data
                st.success("Data cleaned and saved!")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Example Data
    st.markdown("### Example Data")
    examples = get_data_examples()
    
    # Select data structure
    data_structure = st.selectbox(
        "Select data structure",
        list(examples.keys())
    )
    
    # Select distribution type
    distribution_type = st.radio(
        "Select distribution type",
        ["normal", "skewed"],
        format_func=lambda x: "Normal Distribution" if x == "normal" else "Skewed Distribution"
    )
    
    if st.button("Load Example Data"):
        data = examples[data_structure]["examples"][distribution_type]["data"]
        st.session_state['data'] = data
        st.session_state['data_structure'] = data_structure
        st.success(f"Loaded {examples[data_structure]['description']} with {distribution_type} distribution")
        
        # Show data preview
        st.markdown("### Data Preview")
        st.write(data.head())
        
        # Show basic statistics
        st.markdown("### Basic Statistics")
        st.write(data.describe()) 