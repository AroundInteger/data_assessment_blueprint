import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import io
from pathlib import Path
from components.welcome import show_welcome_screen, initialize_session

# Page configuration
st.set_page_config(
    page_title="Data Assessment Blueprint - Phase 3",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
initialize_session()

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main text color - light gray for better contrast against dark background */
    .stMarkdown, .stText, .stDataFrame {
        color: #E0E0E0;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1rem;
        border-bottom: 3px solid #9C27B0;
        padding-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #FFFFFF;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Color-blind friendly info box */
    .info-box {
        background: #1B5E20;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #E0E0E0;
    }
    
    /* Color-blind friendly warning box */
    .warning-box {
        background: #BF360C;
        border-left: 4px solid #FF5722;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #FFFFFF;
    }
    
    /* Color-blind friendly code box */
    .code-box {
        background: #424242;
        border: 1px solid #757575;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #E0E0E0;
    }
    
    /* Color-blind friendly example box */
    .example-box {
        background: #1565C0;
        border-left: 4px solid #64B5F6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #FFFFFF;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #1B5E20;
        color: #E0E0E0;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    /* Error message styling */
    .stError {
        background-color: #B71C1C;
        color: #FFFFFF;
        border-left: 4px solid #F44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #424242;
        border: 1px solid #757575;
        border-radius: 5px;
        padding: 1rem;
        color: #E0E0E0;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #9C27B0;
        color: #FFFFFF;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
    }
    
    .stButton>button:hover {
        background-color: #BA68C8;
    }
    
    /* Selectbox styling */
    .stSelectbox {
        color: #E0E0E0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #424242;
        color: #E0E0E0;
        border-radius: 4px;
    }
    
    /* Ensure all text has good contrast */
    p, li, td, th {
        color: #E0E0E0;
    }
    
    /* Ensure links are visible */
    a {
        color: #64B5F6;
    }
    
    a:hover {
        color: #90CAF9;
    }
    
    /* Table styling */
    .stDataFrame td, .stDataFrame th {
        color: #E0E0E0;
        background-color: #424242;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        color: #E0E0E0;
        background-color: #424242;
    }
    
    /* Number input */
    .stNumberInput>div>div>input {
        color: #E0E0E0;
        background-color: #424242;
    }
    
    /* Selectbox dropdown */
    .stSelectbox>div>div>select {
        color: #E0E0E0;
        background-color: #424242;
    }
</style>
""", unsafe_allow_html=True)

def get_data_examples():
    """Get example datasets for different data structures"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Repeated measures example
    n_subjects = 10
    n_timepoints = 4
    n_total = n_subjects * n_timepoints
    
    repeated_measures = pd.DataFrame({
        'athlete_id': np.repeat(range(1, n_subjects + 1), n_timepoints),
        'session': np.tile(['baseline', 'week2', 'week4', 'week6'], n_subjects),
        'performance': np.random.normal(100, 10, n_total) + 
                      np.repeat(np.random.normal(0, 5, n_subjects), n_timepoints) +  # Individual effects
                      np.tile([0, 5, 8, 10], n_subjects)  # Training effect
    })
    
    # Independent groups example
    n_groups = 3
    n_per_group = 10
    n_total = n_groups * n_per_group
    
    independent_groups = pd.DataFrame({
        'treatment': np.repeat(['Control', 'Treatment A', 'Treatment B'], n_per_group),
        'growth': np.concatenate([
            np.random.normal(10, 2, n_per_group),  # Control
            np.random.normal(12, 2, n_per_group),  # Treatment A
            np.random.normal(15, 2, n_per_group)   # Treatment B
        ])
    })
    
    # Hierarchical example
    n_schools = 2
    n_classes = 2
    n_students = 10
    n_total = n_schools * n_classes * n_students
    
    # Generate school effects (2 schools)
    school_effects = np.repeat(np.random.normal(0, 5, n_schools), n_classes * n_students)
    # Generate class effects (4 classes total)
    class_effects = np.repeat(np.random.normal(0, 3, n_classes), n_students)
    # Repeat class effects for each school
    class_effects = np.tile(class_effects, n_schools)
    
    hierarchical = pd.DataFrame({
        'school': np.repeat(['School A', 'School B'], n_classes * n_students),
        'class': np.tile(np.repeat(['Class 1', 'Class 2'], n_students), n_schools),
        'student': range(1, n_total + 1),
        'score': np.random.normal(70, 10, n_total) + school_effects + class_effects
    })
    
    # Time series example
    n_days = 30
    dates = pd.date_range(start='2023-01-01', periods=n_days)
    
    time_series = pd.DataFrame({
        'date': dates,
        'temperature': np.sin(np.linspace(0, 4*np.pi, n_days)) * 10 + 20 + 
                      np.random.normal(0, 2, n_days)  # Seasonal pattern + noise
    })
    
    return {
        "Repeated measures": {
            "description": "Athlete performance data across training sessions",
            "example": repeated_measures
        },
        "Independent groups": {
            "description": "Treatment effects on plant growth",
            "example": independent_groups
        },
        "Hierarchical": {
            "description": "Student performance across schools and classes",
            "example": hierarchical
        },
        "Time series": {
            "description": "Daily temperature measurements",
            "example": time_series
        }
    }

def get_data_format_requirements():
    """Get data format requirements for different analysis types"""
    return {
        "Repeated measures": {
            "description": "Data should be in long format with the following columns:",
            "required_columns": [
                "subject_id: Unique identifier for each subject",
                "time_point: Measurement time point (e.g., 'baseline', 'week1', etc.)",
                "measurement: The actual measurement value"
            ],
            "example": pd.DataFrame({
                'subject_id': ['S1', 'S1', 'S2', 'S2'],
                'time_point': ['baseline', 'week1', 'baseline', 'week1'],
                'measurement': [10.5, 11.2, 9.8, 10.5]
            }),
            "notes": [
                "Each row represents a single measurement",
                "Include all time points for each subject",
                "Use consistent time point labels"
            ]
        },
        "Independent groups": {
            "description": "Data should be in long format with the following columns:",
            "required_columns": [
                "group: Group identifier (e.g., 'control', 'treatment')",
                "measurement: The actual measurement value"
            ],
            "example": pd.DataFrame({
                'group': ['control', 'control', 'treatment', 'treatment'],
                'measurement': [10.5, 11.2, 9.8, 10.5]
            }),
            "notes": [
                "Each row represents a single measurement",
                "Include group labels for all measurements",
                "Ensure balanced groups if possible"
            ]
        },
        "Hierarchical": {
            "description": "Data should be in long format with the following columns:",
            "required_columns": [
                "level1_id: Identifier for the highest level (e.g., 'school')",
                "level2_id: Identifier for the second level (e.g., 'class')",
                "subject_id: Identifier for the individual subject",
                "measurement: The actual measurement value"
            ],
            "example": pd.DataFrame({
                'school': ['School A', 'School A', 'School B', 'School B'],
                'class': ['Class 1', 'Class 1', 'Class 2', 'Class 2'],
                'student': ['S1', 'S2', 'S3', 'S4'],
                'score': [85, 88, 82, 90]
            }),
            "notes": [
                "Each row represents a single measurement",
                "Include all hierarchical levels for each measurement",
                "Use consistent identifiers across levels"
            ]
        },
        "Time series": {
            "description": "Data should be in long format with the following columns:",
            "required_columns": [
                "timestamp: Date/time of measurement",
                "measurement: The actual measurement value"
            ],
            "example": pd.DataFrame({
                'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'measurement': [20.5, 21.2, 19.8]
            }),
            "notes": [
                "Each row represents a single measurement",
                "Timestamps should be in a consistent format",
                "Include all time points in chronological order"
            ]
        }
    }

def validate_data_format(data, data_type):
    """Validate the uploaded data format against requirements"""
    requirements = get_data_format_requirements()[data_type]
    validation_results = {
        "is_valid": True,
        "messages": [],
        "warnings": []
    }
    
    # Check required columns
    required_columns = [col.split(':')[0].strip() for col in requirements["required_columns"]]
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        validation_results["is_valid"] = False
        validation_results["messages"].append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check for missing values
    if data.isnull().any().any():
        validation_results["warnings"].append("Data contains missing values")
    
    # Check for duplicate entries
    if data.duplicated().any():
        validation_results["warnings"].append("Data contains duplicate entries")
    
    # Data type specific checks
    if data_type == "Repeated measures":
        # Check if each subject has all time points
        subject_timepoints = data.groupby('subject_id')['time_point'].nunique()
        if not (subject_timepoints == subject_timepoints.iloc[0]).all():
            validation_results["warnings"].append("Some subjects have missing time points")
    
    elif data_type == "Independent groups":
        # Check group balance
        group_counts = data['group'].value_counts()
        if not (group_counts == group_counts.iloc[0]).all():
            validation_results["warnings"].append("Groups are not balanced")
    
    elif data_type == "Time series":
        # Check timestamp order
        if not pd.to_datetime(data['timestamp']).is_monotonic_increasing:
            validation_results["warnings"].append("Timestamps are not in chronological order")
    
    return validation_results

def analyze_data(data, analysis_type):
    """Analyze the provided data based on the selected analysis type"""
    results = {}
    
    if analysis_type == "Visual Distribution Assessment":
        # Create distribution plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram with KDE
        sns.histplot(data=data, x=data.columns[-1], kde=True, ax=ax1)
        ax1.set_title('Distribution of Measurements')
        
        # Q-Q plot
        stats.probplot(data[data.columns[-1]], dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        results['plots'] = fig
        
        # Basic statistics
        results['statistics'] = {
            'mean': data[data.columns[-1]].mean(),
            'std': data[data.columns[-1]].std(),
            'skew': data[data.columns[-1]].skew(),
            'kurtosis': data[data.columns[-1]].kurtosis()
        }
        
    elif analysis_type == "Statistical Tests for Normality":
        # Perform multiple normality tests
        data_series = data[data.columns[-1]]
        
        # Shapiro-Wilk
        shapiro_stat, shapiro_p = stats.shapiro(data_series)
        results['shapiro_wilk'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p
        }
        
        # Anderson-Darling
        anderson_result = stats.anderson(data_series)
        results['anderson_darling'] = {
            'statistic': anderson_result.statistic,
            'critical_values': anderson_result.critical_values,
            'significance_level': anderson_result.significance_level
        }
        
        # Create Q-Q plot
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(data_series, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot for Normality Assessment')
        results['qq_plot'] = fig
    
    return results

def get_statistical_analysis_workflow():
    """Get the interactive statistical analysis workflow structure"""
    return {
        "Data Understanding": {
            "description": "First, let's understand your data structure and characteristics",
            "questions": [
                {
                    "id": "data_structure",
                    "text": "What is the structure of your data?",
                    "options": [
                        "Repeated measures (same subjects over time)",
                        "Independent groups (different subjects)",
                        "Hierarchical (nested) structure",
                        "Time series",
                        "Other"
                    ],
                    "follow_up": {
                        "Repeated measures": [
                            "How many time points?",
                            "Are measurements taken at regular intervals?",
                            "Do you expect learning or fatigue effects?"
                        ],
                        "Independent groups": [
                            "How many groups?",
                            "Are group sizes balanced?",
                            "Are there any matching or pairing considerations?"
                        ],
                        "Hierarchical": [
                            "What are the levels of nesting?",
                            "How many units at each level?",
                            "Are you interested in effects at specific levels?"
                        ],
                        "Time series": [
                            "How many time points?",
                            "What is the sampling frequency?",
                            "Do you expect seasonal patterns?"
                        ]
                    },
                    "examples": {
                        "Repeated measures": "Athlete performance measured across multiple training sessions",
                        "Independent groups": "Different treatment groups in an experiment",
                        "Hierarchical": "Students nested within classes within schools",
                        "Time series": "Daily temperature measurements over a year"
                    }
                },
                {
                    "id": "measurement_type",
                    "text": "What type of measurements are you analyzing?",
                    "options": [
                        "Continuous measurements",
                        "Counts or frequencies",
                        "Binary outcomes",
                        "Ordinal categories",
                        "Multiple measurements per subject"
                    ],
                    "examples": {
                        "Continuous measurements": "Height, weight, reaction time",
                        "Counts or frequencies": "Number of events, frequency of behaviors",
                        "Binary outcomes": "Success/failure, present/absent",
                        "Ordinal categories": "Likert scale responses, severity ratings",
                        "Multiple measurements per subject": "Multiple trials, repeated observations"
                    }
                },
                {
                    "id": "sample_size",
                    "text": "What is your sample size?",
                    "type": "number",
                    "follow_up": {
                        "small": "Consider non-parametric methods or exact tests",
                        "medium": "Check normality assumptions carefully",
                        "large": "Consider practical significance in addition to statistical significance"
                    },
                    "examples": {
                        "small": "n < 30: Use non-parametric tests or exact tests",
                        "medium": "30 ≤ n < 100: Check normality assumptions carefully",
                        "large": "n ≥ 100: Consider practical significance"
                    }
                }
            ]
        },
        "Distributional Analysis": {
            "description": "Let's examine the distribution of your data",
            "steps": [
                {
                    "id": "visual_inspection",
                    "text": "Visual Distribution Assessment",
                    "methods": [
                        "Histogram",
                        "Q-Q plot",
                        "Box plot",
                        "Violin plot"
                    ],
                    "code_example": """
# Create distribution plots
import seaborn as sns
import matplotlib.pyplot as plt

# Histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='measurement', kde=True)
plt.title('Distribution of Measurements')
plt.show()

# Q-Q plot
from scipy import stats
stats.probplot(df['measurement'], dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()
"""
                },
                {
                    "id": "normality_tests",
                    "text": "Statistical Tests for Normality",
                    "methods": [
                        "Shapiro-Wilk (n < 5000)",
                        "Anderson-Darling (sensitive to tails)",
                        "Kolmogorov-Smirnov",
                        "Jarque-Bera"
                    ],
                    "code_example": """
# Multiple normality tests
from scipy import stats

# Shapiro-Wilk
stat, p_value = stats.shapiro(data)
print(f"Shapiro-Wilk: p-value = {p_value:.4f}")

# Anderson-Darling
result = stats.anderson(data)
print(f"Anderson-Darling: statistic = {result.statistic:.4f}")
"""
                }
            ]
        },
        "Statistical Test Selection": {
            "description": "Based on your data characteristics, let's choose appropriate statistical tests",
            "decision_tree": {
                "repeated_measures": {
                    "normal": {
                        "two_groups": "Paired t-test",
                        "multiple_groups": "Repeated measures ANOVA",
                        "multiple_factors": "Mixed ANOVA"
                    },
                    "non_normal": {
                        "two_groups": "Wilcoxon signed-rank test",
                        "multiple_groups": "Friedman test",
                        "multiple_factors": "Non-parametric mixed ANOVA"
                    }
                },
                "independent_groups": {
                    "normal": {
                        "two_groups": "Independent t-test",
                        "multiple_groups": "One-way ANOVA",
                        "multiple_factors": "Factorial ANOVA"
                    },
                    "non_normal": {
                        "two_groups": "Mann-Whitney U test",
                        "multiple_groups": "Kruskal-Wallis test",
                        "multiple_factors": "Non-parametric factorial ANOVA"
                    }
                },
                "hierarchical": {
                    "normal": "Mixed-effects models",
                    "non_normal": "Generalized linear mixed models"
                },
                "time_series": {
                    "normal": "ARIMA models",
                    "non_normal": "GARCH models"
                }
            }
        },
        "Analysis Plan Generation": {
            "description": "Let's create a comprehensive analysis plan",
            "components": [
                {
                    "id": "data_preparation",
                    "text": "Data Preparation Steps",
                    "items": [
                        "Handle missing values",
                        "Check for outliers",
                        "Transform data if needed",
                        "Create necessary dummy variables"
                    ]
                },
                {
                    "id": "statistical_tests",
                    "text": "Statistical Tests",
                    "items": [
                        "Primary hypothesis test",
                        "Post-hoc tests if needed",
                        "Effect size calculations",
                        "Power analysis"
                    ]
                },
                {
                    "id": "visualization",
                    "text": "Visualization Plan",
                    "items": [
                        "Distribution plots",
                        "Comparison plots",
                        "Effect size plots",
                        "Residual plots"
                    ]
                },
                {
                    "id": "interpretation",
                    "text": "Interpretation Framework",
                    "items": [
                        "Statistical significance",
                        "Practical significance",
                        "Effect size interpretation",
                        "Limitations and assumptions"
                    ]
                }
            ]
        }
    }

def main():
    # Show welcome screen if no step is set
    if st.session_state['current_step'] == 'welcome':
        show_welcome_screen()
        return

    # Main content
    st.title("Phase 3: Statistical Characterization")
    
    # Get the workflow structure
    workflow = get_statistical_analysis_workflow()
    data_examples = get_data_examples()
    data_requirements = get_data_format_requirements()
    
    # Data Understanding Section
    st.markdown('<h2 class="section-header">1. Data Understanding</h2>', unsafe_allow_html=True)
    st.markdown(workflow["Data Understanding"]["description"])
    
    # Interactive questionnaire
    for question in workflow["Data Understanding"]["questions"]:
        st.markdown(f"### {question['text']}")
        
        if question.get("type") == "number":
            value = st.number_input("Enter your sample size", min_value=1, value=100)
            if value < 30:
                st.info(question["follow_up"]["small"])
                st.markdown(f'<div class="example-box"><strong>Example:</strong> {question["examples"]["small"]}</div>', unsafe_allow_html=True)
            elif value < 100:
                st.info(question["follow_up"]["medium"])
                st.markdown(f'<div class="example-box"><strong>Example:</strong> {question["examples"]["medium"]}</div>', unsafe_allow_html=True)
            else:
                st.info(question["follow_up"]["large"])
                st.markdown(f'<div class="example-box"><strong>Example:</strong> {question["examples"]["large"]}</div>', unsafe_allow_html=True)
        else:
            selected = st.selectbox("Select an option", question["options"])
            
            # Show example for selected option
            if selected in question.get("examples", {}):
                st.markdown(f'<div class="example-box"><strong>Example:</strong> {question["examples"][selected]}</div>', unsafe_allow_html=True)
            
            if selected in question.get("follow_up", {}):
                st.markdown("#### Follow-up questions:")
                for follow_up in question["follow_up"][selected]:
                    st.text_input(follow_up)
                
                # Show example data if available
                if selected in data_examples:
                    with st.expander("View Example Data"):
                        st.write(data_examples[selected]["description"])
                        st.dataframe(data_examples[selected]["example"])
                        
                        # Add option to analyze example data
                        if st.button(f"Analyze {selected} Example Data"):
                            results = analyze_data(data_examples[selected]["example"], "Visual Distribution Assessment")
                            st.pyplot(results['plots'])
                            st.write("Basic Statistics:", results['statistics'])
    
    # Distributional Analysis Section
    st.markdown('<h2 class="section-header">2. Distributional Analysis</h2>', unsafe_allow_html=True)
    st.markdown(workflow["Distributional Analysis"]["description"])
    
    # Method selection
    selected_method = st.selectbox(
        "Choose a distribution analysis method",
        [step["text"] for step in workflow["Distributional Analysis"]["steps"]]
    )
    
    # Show code example for selected method
    for step in workflow["Distributional Analysis"]["steps"]:
        if step["text"] == selected_method:
            with st.expander("View Code Example"):
                st.code(step["code_example"], language="python")
            
            # Add option to upload and analyze data
            st.markdown("### Upload Your Data")
            
            # Show data format requirements
            if selected in data_requirements:
                with st.expander("View Data Format Requirements"):
                    st.write(data_requirements[selected]["description"])
                    st.write("Required columns:")
                    for col in data_requirements[selected]["required_columns"]:
                        st.write(f"- {col}")
                    st.write("Example data format:")
                    st.dataframe(data_requirements[selected]["example"])
                    st.write("Important notes:")
                    for note in data_requirements[selected]["notes"]:
                        st.write(f"- {note}")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.write("Preview of your data:")
                st.dataframe(data.head())
                
                # Validate data format
                if selected in data_requirements:
                    validation_results = validate_data_format(data, selected)
                    
                    if not validation_results["is_valid"]:
                        st.error("Data format validation failed:")
                        for message in validation_results["messages"]:
                            st.error(message)
                    
                    if validation_results["warnings"]:
                        st.warning("Data format warnings:")
                        for warning in validation_results["warnings"]:
                            st.warning(warning)
                
                if st.button("Analyze Data"):
                    results = analyze_data(data, selected_method)
                    
                    if "plots" in results:
                        st.pyplot(results["plots"])
                    if "qq_plot" in results:
                        st.pyplot(results["qq_plot"])
                    if "statistics" in results:
                        st.write("Basic Statistics:", results["statistics"])
                    if "shapiro_wilk" in results:
                        st.write("Shapiro-Wilk Test Results:", results["shapiro_wilk"])
                    if "anderson_darling" in results:
                        st.write("Anderson-Darling Test Results:", results["anderson_darling"])
    
    # Statistical Test Selection
    st.markdown('<h2 class="section-header">3. Statistical Test Selection</h2>', unsafe_allow_html=True)
    st.markdown(workflow["Statistical Test Selection"]["description"])
    
    # Interactive test selection
    data_type = st.selectbox(
        "Select your data structure",
        list(workflow["Statistical Test Selection"]["decision_tree"].keys())
    )
    
    if data_type in workflow["Statistical Test Selection"]["decision_tree"]:
        distribution = st.selectbox(
            "Is your data normally distributed?",
            ["normal", "non_normal"]
        )
        
        if distribution in workflow["Statistical Test Selection"]["decision_tree"][data_type]:
            group_type = st.selectbox(
                "How many groups are you comparing?",
                ["two_groups", "multiple_groups", "multiple_factors"]
            )
            
            if group_type in workflow["Statistical Test Selection"]["decision_tree"][data_type][distribution]:
                recommended_test = workflow["Statistical Test Selection"]["decision_tree"][data_type][distribution][group_type]
                st.success(f"Recommended test: {recommended_test}")
                
                # Show example data and analysis for the recommended test
                if data_type in data_examples:
                    with st.expander("View Example Analysis"):
                        st.write("Example data for", data_type)
                        st.dataframe(data_examples[data_type]["example"])
                        
                        if st.button(f"Analyze {data_type} Example"):
                            results = analyze_data(data_examples[data_type]["example"], "Visual Distribution Assessment")
                            st.pyplot(results['plots'])
                            st.write("Basic Statistics:", results['statistics'])
    
    # Analysis Plan Generation
    st.markdown('<h2 class="section-header">4. Analysis Plan Generation</h2>', unsafe_allow_html=True)
    st.markdown(workflow["Analysis Plan Generation"]["description"])
    
    # Generate plan based on previous selections
    if st.button("Generate Analysis Plan"):
        plan = workflow["Analysis Plan Generation"]["components"]
        
        # Display plan with checkboxes
        for component in plan:
            st.markdown(f"### {component['text']}")
            for item in component["items"]:
                st.checkbox(item)
        
        # Add export option
        if st.button("Export Analysis Plan"):
            st.download_button(
                "Download Plan",
                data=json.dumps(plan, indent=2),
                file_name="statistical_analysis_plan.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main() 