import streamlit as st
import plotly.express as px
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
from components.data_input import show_data_input
from utils.data_generation import get_data_examples

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

def initialize_session():
    """Initialize session state variables if they don't exist."""
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = 'welcome'
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'analysis_type' not in st.session_state:
        st.session_state['analysis_type'] = None
    if 'analysis_decisions' not in st.session_state:
        st.session_state['analysis_decisions'] = {
            'use_nonparametric': None,
            'transformation_applied': False,
            'transformation_type': None,
            'reasoning': None,
            'preferred_tests': {}
        }

def get_nonparametric_test_info():
    """Get detailed information about non-parametric tests."""
    return {
        "Mann-Whitney U": {
            "description": "Tests whether two independent samples are from populations with the same distribution.",
            "assumptions": [
                "Independent observations",
                "Ordinal or continuous data",
                "No assumption of normal distribution"
            ],
            "effect_size": "r = Z/√N (where Z is the standardized test statistic and N is the total sample size)",
            "interpretation": {
                "small": "r < 0.1",
                "medium": "0.1 ≤ r < 0.3",
                "large": "r ≥ 0.3"
            },
            "code_example": """
from scipy import stats
statistic, p_value = stats.mannwhitneyu(group1, group2)
# Calculate effect size
n1, n2 = len(group1), len(group2)
z = stats.norm.ppf(1 - p_value/2)
r = z / np.sqrt(n1 + n2)
"""
        },
        "Wilcoxon signed-rank": {
            "description": "Tests whether two related samples are from populations with the same distribution.",
            "assumptions": [
                "Paired observations",
                "Ordinal or continuous data",
                "Symmetric distribution of differences"
            ],
            "effect_size": "r = Z/√N (where Z is the standardized test statistic and N is the number of pairs)",
            "interpretation": {
                "small": "r < 0.1",
                "medium": "0.1 ≤ r < 0.3",
                "large": "r ≥ 0.3"
            },
            "code_example": """
from scipy import stats
statistic, p_value = stats.wilcoxon(group1, group2)
# Calculate effect size
n = len(group1)
z = stats.norm.ppf(1 - p_value/2)
r = z / np.sqrt(n)
"""
        },
        "Kruskal-Wallis": {
            "description": "Tests whether multiple independent samples are from populations with the same distribution.",
            "assumptions": [
                "Independent observations",
                "Ordinal or continuous data",
                "No assumption of normal distribution"
            ],
            "effect_size": "η² = (H - k + 1)/(n - k) (where H is the test statistic, k is number of groups, n is total sample size)",
            "interpretation": {
                "small": "η² < 0.06",
                "medium": "0.06 ≤ η² < 0.14",
                "large": "η² ≥ 0.14"
            },
            "code_example": """
from scipy import stats
statistic, p_value = stats.kruskal(*groups)
# Calculate effect size
n = sum(len(g) for g in groups)
k = len(groups)
eta_squared = (statistic - k + 1) / (n - k)
"""
        },
        "Friedman": {
            "description": "Tests whether multiple related samples are from populations with the same distribution.",
            "assumptions": [
                "Repeated measures",
                "Ordinal or continuous data",
                "No assumption of normal distribution"
            ],
            "effect_size": "W = χ²/(N(k-1)) (where χ² is the test statistic, N is number of subjects, k is number of conditions)",
            "interpretation": {
                "small": "W < 0.1",
                "medium": "0.1 ≤ W < 0.3",
                "large": "W ≥ 0.3"
            },
            "code_example": """
from scipy import stats
statistic, p_value = stats.friedmanchisquare(*groups)
# Calculate effect size
n = len(groups[0])
k = len(groups)
w = statistic / (n * (k-1))
"""
        }
    }

def generate_analysis_plan():
    """Generate a comprehensive analysis plan based on previous decisions and findings."""
    plan = {
        "approach": "non-parametric" if st.session_state['analysis_decisions']['use_nonparametric'] else "parametric",
        "data_structure": st.session_state.get('data_structure', 'Unknown'),
        "evidence": [],
        "decision_tree": [],
        "recommended_tests": [],
        "alternative_tests": [],
        "code_template": "",
        "visualization_plan": [],
        "interpretation_guidelines": []
    }
    
    # Add evidence from distributional analysis
    if 'distribution_analysis' in st.session_state:
        evidence = st.session_state['distribution_analysis']
        plan["evidence"].extend([
            f"Skewness: {evidence.get('skewness', 'N/A')}",
            f"Kurtosis: {evidence.get('kurtosis', 'N/A')}",
            f"Shapiro-Wilk p-value: {evidence.get('shapiro_p', 'N/A')}",
            f"Anderson-Darling statistic: {evidence.get('anderson_stat', 'N/A')}"
        ])
    
    # Build decision tree
    if st.session_state['analysis_decisions']['use_nonparametric']:
        plan["decision_tree"].extend([
            "1. Data Distribution Assessment",
            "   → Non-normal distribution detected",
            "   → Decision: Use non-parametric methods",
            f"   → Reasoning: {st.session_state['analysis_decisions']['reasoning']}"
        ])
    else:
        plan["decision_tree"].extend([
            "1. Data Distribution Assessment",
            "   → Normal distribution confirmed",
            "   → Decision: Use parametric methods",
            f"   → Reasoning: {st.session_state['analysis_decisions']['reasoning']}"
        ])
    
    if st.session_state['analysis_decisions']['transformation_applied']:
        plan["decision_tree"].extend([
            "2. Data Transformation",
            f"   → Applied: {st.session_state['analysis_decisions']['transformation_type']}",
            "   → Verification: Distribution improved"
        ])
    
    # Add recommended and alternative tests based on data structure
    data_structure = st.session_state.get('data_structure', 'Unknown')
    
    if data_structure == "Repeated Measures":
        if st.session_state['analysis_decisions']['use_nonparametric']:
            plan["recommended_tests"].extend([
                "Primary: Wilcoxon signed-rank test (2 groups) or Friedman test (3+ groups)",
                "Effect Size: r (2 groups) or W (3+ groups)",
                "Post-hoc: Dunn's test with Bonferroni correction"
            ])
            plan["alternative_tests"].extend([
                "Robust t-test (if distribution is only slightly non-normal)",
                "Bootstrap-based tests (for more precise p-values)",
                "Permutation tests (for exact significance levels)",
                "Mixed-effects models with robust estimation",
                "Quantile regression for repeated measures"
            ])
        else:
            plan["recommended_tests"].extend([
                "Primary: Paired t-test (2 groups) or Repeated measures ANOVA (3+ groups)",
                "Effect Size: Cohen's d (2 groups) or η² (3+ groups)",
                "Post-hoc: Tukey's HSD or Bonferroni correction"
            ])
            plan["alternative_tests"].extend([
                "Mixed-effects models (for more complex designs)",
                "MANOVA (for multiple dependent variables)",
                "Multilevel modeling (for nested data)",
                "Generalized estimating equations (GEE)",
                "Robust repeated measures ANOVA"
            ])
    
    elif data_structure == "Independent Groups":
        if st.session_state['analysis_decisions']['use_nonparametric']:
            plan["recommended_tests"].extend([
                "Primary: Mann-Whitney U test (2 groups) or Kruskal-Wallis test (3+ groups)",
                "Effect Size: r (2 groups) or η² (3+ groups)",
                "Post-hoc: Dunn's test with Bonferroni correction"
            ])
            plan["alternative_tests"].extend([
                "Robust t-test (Yuen's t-test)",
                "Bootstrap-based tests",
                "Permutation tests",
                "Quantile regression",
                "Robust ANOVA (trimmed means)"
            ])
        else:
            plan["recommended_tests"].extend([
                "Primary: Independent t-test (2 groups) or One-way ANOVA (3+ groups)",
                "Effect Size: Cohen's d (2 groups) or η² (3+ groups)",
                "Post-hoc: Tukey's HSD or Bonferroni correction"
            ])
            plan["alternative_tests"].extend([
                "Welch's t-test (unequal variances)",
                "Robust ANOVA (trimmed means)",
                "Generalized linear models",
                "Mixed-effects models",
                "Bayesian t-test/ANOVA"
            ])
    
    elif data_structure == "Hierarchical":
        if st.session_state['analysis_decisions']['use_nonparametric']:
            plan["recommended_tests"].extend([
                "Primary: Non-parametric mixed-effects models",
                "Effect Size: Pseudo R²",
                "Post-hoc: Rank-based multiple comparisons"
            ])
            plan["alternative_tests"].extend([
                "Robust mixed-effects models",
                "Quantile mixed-effects models",
                "Bootstrap-based hierarchical models",
                "Permutation tests for nested data",
                "Rank-based hierarchical models"
            ])
        else:
            plan["recommended_tests"].extend([
                "Primary: Mixed-effects models",
                "Effect Size: Conditional and marginal R²",
                "Post-hoc: Estimated marginal means"
            ])
            plan["alternative_tests"].extend([
                "Generalized linear mixed models",
                "Robust mixed-effects models",
                "Bayesian hierarchical models",
                "Multilevel structural equation models",
                "Generalized estimating equations (GEE)"
            ])
    
    elif data_structure == "Time Series":
        if st.session_state['analysis_decisions']['use_nonparametric']:
            plan["recommended_tests"].extend([
                "Primary: Non-parametric trend analysis",
                "Effect Size: Trend strength measures",
                "Post-hoc: Seasonal decomposition"
            ])
            plan["alternative_tests"].extend([
                "Robust time series models",
                "Quantile regression for time series",
                "Bootstrap-based trend analysis",
                "Permutation tests for time series",
                "Wavelet analysis"
            ])
        else:
            plan["recommended_tests"].extend([
                "Primary: ARIMA models",
                "Effect Size: Model fit metrics",
                "Post-hoc: Seasonal decomposition"
            ])
            plan["alternative_tests"].extend([
                "GARCH models",
                "State space models",
                "Bayesian time series models",
                "Robust ARIMA models",
                "Dynamic regression models"
            ])
    
    # Generate code template based on data structure and approach
    if data_structure == "Repeated Measures":
        if st.session_state['analysis_decisions']['use_nonparametric']:
            plan["code_template"] = """
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scikit_posthocs import posthoc_nemenyi_friedman

# Load and prepare data
df = pd.read_csv('your_data.csv')  # Replace with your data loading code

# Primary test
if len(df['group'].unique()) == 2:
    # Wilcoxon signed-rank test for 2 groups
    statistic, p_value = stats.wilcoxon(
        df[df['group'] == 'group1']['measurement'],
        df[df['group'] == 'group2']['measurement']
    )
    # Calculate effect size
    n = len(df['subject'].unique())
    z = stats.norm.ppf(1 - p_value/2)
    r = z / np.sqrt(n)
else:
    # Friedman test for 3+ groups
    statistic, p_value = stats.friedmanchisquare(
        *[group['measurement'] for name, group in df.groupby('group')]
    )
    # Calculate effect size
    n = len(df['subject'].unique())
    k = len(df['group'].unique())
    w = statistic / (n * (k-1))

# Post-hoc analysis
if len(df['group'].unique()) > 2:
    posthoc_results = posthoc_nemenyi_friedman(
        df.pivot(index='subject', columns='group', values='measurement')
    )

# Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='group', y='measurement')
sns.stripplot(data=df, x='group', y='measurement', color='black', alpha=0.5)
plt.title('Measurement by Group')
plt.show()

# Print results
print(f"Test statistic: {statistic:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"Effect size: {r if len(df['group'].unique()) == 2 else w:.3f}")

# Alternative robust analysis
from statsmodels.robust.robust_linear_model import RLM
model = RLM.from_formula('measurement ~ group', data=df).fit()
print("\\nRobust regression results:")
print(model.summary())
"""
        else:
            plan["code_template"] = """
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load and prepare data
df = pd.read_csv('your_data.csv')  # Replace with your data loading code

# Primary test
if len(df['group'].unique()) == 2:
    # Paired t-test for 2 groups
    statistic, p_value = stats.ttest_rel(
        df[df['group'] == 'group1']['measurement'],
        df[df['group'] == 'group2']['measurement']
    )
    # Calculate effect size (Cohen's d)
    d = statistic / np.sqrt(len(df['subject'].unique()))
else:
    # Repeated measures ANOVA for 3+ groups
    aovrm = AnovaRM(
        df, 'measurement', 'subject', within=['group']
    ).fit()
    # Calculate effect size (η²)
    eta_squared = aovrm.anova_table['sum_sq'][0] / (
        aovrm.anova_table['sum_sq'][0] + aovrm.anova_table['sum_sq'][1]
    )

# Post-hoc analysis
if len(df['group'].unique()) > 2:
    tukey = pairwise_tukeyhsd(
        df['measurement'], df['group']
    )

# Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='group', y='measurement')
sns.stripplot(data=df, x='group', y='measurement', color='black', alpha=0.5)
plt.title('Measurement by Group')
plt.show()

# Print results
print(f"Test statistic: {statistic:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"Effect size: {d if len(df['group'].unique()) == 2 else eta_squared:.3f}")

# Alternative mixed-effects analysis
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fit mixed-effects model
model = smf.mixedlm('measurement ~ group', df, groups=df['subject']).fit()
print("\\nMixed-effects model results:")
print(model.summary())
"""
    
    elif data_structure == "Independent Groups":
        if st.session_state['analysis_decisions']['use_nonparametric']:
            plan["code_template"] = """
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from scikit_posthocs import posthoc_dunn

# Load and prepare data
df = pd.read_csv('your_data.csv')  # Replace with your data loading code

# Primary test
if len(df['group'].unique()) == 2:
    # Mann-Whitney U test for 2 groups
    statistic, p_value = stats.mannwhitneyu(
        df[df['group'] == 'group1']['measurement'],
        df[df['group'] == 'group2']['measurement']
    )
    # Calculate effect size
    n1 = len(df[df['group'] == 'group1'])
    n2 = len(df[df['group'] == 'group2'])
    z = stats.norm.ppf(1 - p_value/2)
    r = z / np.sqrt(n1 + n2)
else:
    # Kruskal-Wallis test for 3+ groups
    statistic, p_value = stats.kruskal(
        *[group['measurement'] for name, group in df.groupby('group')]
    )
    # Calculate effect size
    n = len(df)
    k = len(df['group'].unique())
    eta_squared = (statistic - k + 1) / (n - k)

# Post-hoc analysis
if len(df['group'].unique()) > 2:
    posthoc_results = posthoc_dunn(
        df, val_col='measurement', group_col='group'
    )

# Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='group', y='measurement')
sns.stripplot(data=df, x='group', y='measurement', color='black', alpha=0.5)
plt.title('Measurement by Group')
plt.show()

# Print results
print(f"Test statistic: {statistic:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"Effect size: {r if len(df['group'].unique()) == 2 else eta_squared:.3f}")

# Alternative robust analysis
from scipy import stats
# Yuen's t-test for 2 groups
if len(df['group'].unique()) == 2:
    from scipy.stats import trim_mean
    trimmed_stat, trimmed_p = stats.ttest_ind(
        df[df['group'] == 'group1']['measurement'],
        df[df['group'] == 'group2']['measurement'],
        trim=0.2  # 20% trimming
    )
    print("\\nYuen's t-test results:")
    print(f"Trimmed t-statistic: {trimmed_stat:.3f}")
    print(f"p-value: {trimmed_p:.3f}")
"""
        else:
            plan["code_template"] = """
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load and prepare data
df = pd.read_csv('your_data.csv')  # Replace with your data loading code

# Primary test
if len(df['group'].unique()) == 2:
    # Independent t-test for 2 groups
    statistic, p_value = stats.ttest_ind(
        df[df['group'] == 'group1']['measurement'],
        df[df['group'] == 'group2']['measurement']
    )
    # Calculate effect size (Cohen's d)
    n1 = len(df[df['group'] == 'group1'])
    n2 = len(df[df['group'] == 'group2'])
    d = statistic * np.sqrt((n1 + n2)/(n1 * n2))
else:
    # One-way ANOVA for 3+ groups
    groups = [group['measurement'] for name, group in df.groupby('group')]
    statistic, p_value = stats.f_oneway(*groups)
    # Calculate effect size (η²)
    n = len(df)
    k = len(df['group'].unique())
    eta_squared = statistic / (statistic + n - k)

# Post-hoc analysis
if len(df['group'].unique()) > 2:
    tukey = pairwise_tukeyhsd(
        df['measurement'], df['group']
    )

# Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='group', y='measurement')
sns.stripplot(data=df, x='group', y='measurement', color='black', alpha=0.5)
plt.title('Measurement by Group')
plt.show()

# Print results
print(f"Test statistic: {statistic:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"Effect size: {d if len(df['group'].unique()) == 2 else eta_squared:.3f}")

# Alternative robust analysis
from scipy import stats
# Welch's t-test for 2 groups
if len(df['group'].unique()) == 2:
    welch_stat, welch_p = stats.ttest_ind(
        df[df['group'] == 'group1']['measurement'],
        df[df['group'] == 'group2']['measurement'],
        equal_var=False
    )
    print("\\nWelch's t-test results:")
    print(f"t-statistic: {welch_stat:.3f}")
    print(f"p-value: {welch_p:.3f}")
"""
    
    # Add visualization plan
    plan["visualization_plan"].extend([
        "1. Distribution Plots",
        "   - Histogram with KDE",
        "   - Q-Q plot",
        "   - Box plot with individual points",
        "2. Effect Size Visualization",
        "   - Forest plot of effect sizes",
        "   - Confidence interval plot",
        "3. Post-hoc Analysis",
        "   - Multiple comparison plot",
        "   - Group comparison matrix"
    ])
    
    # Add interpretation guidelines
    plan["interpretation_guidelines"].extend([
        "1. Statistical Significance",
        "   - Report exact p-values",
        "   - Consider multiple testing correction",
        "2. Effect Size Interpretation",
        "   - Report effect size with confidence intervals",
        "   - Compare to field-specific benchmarks",
        "3. Practical Significance",
        "   - Consider minimum important difference",
        "   - Evaluate clinical/practical relevance",
        "4. Limitations",
        "   - Note any assumption violations",
        "   - Consider alternative explanations",
        "   - Discuss generalizability"
    ])
    
    return plan

def main():
    # Initialize session state first
    initialize_session()
    
    # Show welcome screen if no step is set
    if st.session_state['current_step'] == 'welcome':
        show_welcome_screen()
        return
    
    # Main content
    st.title("Phase 3: Statistical Characterization")
    
    # Navigation
    st.sidebar.title("Navigation")
    current_step = st.sidebar.radio(
        "Select Step",
        ["Data Input", "Data Understanding", "Distributional Analysis", "Statistical Test Selection", "Analysis Plan"]
    )
    
    if current_step == "Data Input":
        show_data_input()
    
    elif current_step == "Data Understanding":
        st.markdown('<h2 class="section-header">Data Understanding</h2>', unsafe_allow_html=True)
        if 'data' not in st.session_state or st.session_state['data'] is None:
            st.warning("Please upload or load example data first!")
            return
        
        data = st.session_state['data']
        
        # Data Structure Selection
        st.markdown("### Data Structure")
        data_structure = st.selectbox(
            "Select your data structure",
            ["Repeated Measures", "Independent Groups", "Hierarchical", "Time Series"]
        )
        st.session_state['data_structure'] = data_structure
        
        st.write(f"Number of rows: {len(data)}")
        st.write(f"Number of columns: {len(data.columns)}")
        
        # Column Types
        st.markdown("### Column Types")
        for col in data.columns:
            st.write(f"- {col}: {data[col].dtype}")
        
        # Basic Statistics
        st.markdown("### Basic Statistics")
        st.write(data.describe())
        
        # Missing Values
        st.markdown("### Missing Values")
        missing = data.isnull().sum()
        if missing.any():
            st.write(missing[missing > 0])
        else:
            st.success("No missing values found!")
    
    elif current_step == "Distributional Analysis":
        st.markdown('<h2 class="section-header">Distributional Analysis</h2>', unsafe_allow_html=True)
        if 'data' not in st.session_state or st.session_state['data'] is None:
            st.warning("Please upload or load example data first!")
            return
        
        data = st.session_state['data']
        
        # Select column for analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for distribution analysis", numeric_cols)
            
            # Distribution Plot
            st.markdown("### Distribution Plot")
            fig = px.histogram(data, x=selected_col, marginal="box")
            st.plotly_chart(fig)
            
            # Q-Q Plot
            st.markdown("### Q-Q Plot")
            qq_fig = px.scatter(x=stats.probplot(data[selected_col], dist="norm")[0][0],
                              y=stats.probplot(data[selected_col], dist="norm")[0][1])
            qq_fig.add_scatter(x=[-3, 3], y=[-3, 3], mode='lines', name='Normal')
            st.plotly_chart(qq_fig)
            
            # Normality Assessment
            st.markdown("### Normality Assessment")
            
            # Calculate skewness and kurtosis
            skewness = stats.skew(data[selected_col])
            kurtosis = stats.kurtosis(data[selected_col])
            
            # Shapiro-Wilk Test
            shapiro_stat, shapiro_p = stats.shapiro(data[selected_col])
            
            # Anderson-Darling Test
            anderson = stats.anderson(data[selected_col])
            
            # Store distribution analysis results
            st.session_state['distribution_analysis'] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'shapiro_p': shapiro_p,
                'anderson_stat': anderson.statistic
            }
            
            # Display normality metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Skewness", f"{skewness:.3f}")
                st.metric("Kurtosis", f"{kurtosis:.3f}")
            with col2:
                st.metric("Shapiro-Wilk p-value", f"{shapiro_p:.4f}")
                st.metric("Anderson-Darling Statistic", f"{anderson.statistic:.4f}")
            
            # Interpret normality
            st.markdown("#### Normality Interpretation")
            normality_assessment = []
            
            # Check skewness
            if abs(skewness) > 1:
                normality_assessment.append(f"Strong {'positive' if skewness > 0 else 'negative'} skewness detected")
            elif abs(skewness) > 0.5:
                normality_assessment.append(f"Moderate {'positive' if skewness > 0 else 'negative'} skewness detected")
            
            # Check kurtosis
            if abs(kurtosis) > 2:
                normality_assessment.append("Heavy tails detected (excess kurtosis)")
            elif abs(kurtosis) > 1:
                normality_assessment.append("Moderate deviation from normal kurtosis")
            
            # Check Shapiro-Wilk
            if shapiro_p < 0.05:
                normality_assessment.append("Shapiro-Wilk test suggests non-normality")
            
            # Check Anderson-Darling
            if anderson.statistic > anderson.critical_values[2]:  # Using 5% significance level
                normality_assessment.append("Anderson-Darling test suggests non-normality")
            
            # Display assessment
            if normality_assessment:
                st.warning("### Non-normality detected:")
                for assessment in normality_assessment:
                    st.markdown(f"- {assessment}")
                
                # Analysis Decision
                st.markdown("### Analysis Decision")
                analysis_choice = st.radio(
                    "Choose your analysis approach:",
                    ["Transform the data", "Use non-parametric methods"]
                )
                
                if analysis_choice == "Transform the data":
                    st.session_state['analysis_decisions']['use_nonparametric'] = False
                    st.session_state['analysis_decisions']['reasoning'] = "Data transformation chosen to achieve normality"
                    
                    # Transformation options
                    st.markdown("#### Data Transformation Options:")
                    if skewness > 0:
                        st.markdown("""
                        - Log transformation: `np.log1p(data)`
                        - Square root transformation: `np.sqrt(data)`
                        - Box-Cox transformation: `stats.boxcox(data)`
                        """)
                    elif skewness < 0:
                        st.markdown("""
                        - Square transformation: `data ** 2`
                        - Cube transformation: `data ** 3`
                        - Yeo-Johnson transformation: `stats.yeojohnson(data)`
                        """)
                    
                    # Transformation Preview
                    st.markdown("### Transformation Preview")
                    transform_option = st.selectbox(
                        "Select transformation to preview",
                        ["Log", "Square Root", "Box-Cox", "Square", "Cube", "Yeo-Johnson"]
                    )
                    
                    if transform_option == "Log":
                        transformed_data = np.log1p(data[selected_col])
                        transform_name = "Log-transformed"
                    elif transform_option == "Square Root":
                        transformed_data = np.sqrt(data[selected_col])
                        transform_name = "Square root-transformed"
                    elif transform_option == "Box-Cox":
                        transformed_data, _ = stats.boxcox(data[selected_col])
                        transform_name = "Box-Cox-transformed"
                    elif transform_option == "Square":
                        transformed_data = data[selected_col] ** 2
                        transform_name = "Square-transformed"
                    elif transform_option == "Cube":
                        transformed_data = data[selected_col] ** 3
                        transform_name = "Cube-transformed"
                    else:  # Yeo-Johnson
                        transformed_data, _ = stats.yeojohnson(data[selected_col])
                        transform_name = "Yeo-Johnson-transformed"
                    
                    # Plot transformed data
                    fig = px.histogram(x=transformed_data, marginal="box",
                                     title=f"{transform_name} Distribution")
                    st.plotly_chart(fig)
                    
                    # Show normality metrics for transformed data
                    st.markdown(f"#### {transform_name} Data Normality Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Skewness", f"{stats.skew(transformed_data):.3f}")
                        st.metric("Kurtosis", f"{stats.kurtosis(transformed_data):.3f}")
                    with col2:
                        sw_stat, sw_p = stats.shapiro(transformed_data)
                        st.metric("Shapiro-Wilk p-value", f"{sw_p:.4f}")
                        ad_stat = stats.anderson(transformed_data).statistic
                        st.metric("Anderson-Darling Statistic", f"{ad_stat:.4f}")
                    
                    # Store transformation decision
                    st.session_state['analysis_decisions']['transformation_applied'] = True
                    st.session_state['analysis_decisions']['transformation_type'] = transform_option
                    
                else:  # Use non-parametric methods
                    st.session_state['analysis_decisions']['use_nonparametric'] = True
                    st.session_state['analysis_decisions']['transformation_applied'] = False
                    st.session_state['analysis_decisions']['reasoning'] = "Non-parametric methods chosen due to non-normality"
                    
                    # Non-parametric alternatives
                    st.markdown("#### Recommended Non-parametric Tests:")
                    st.markdown("""
                    - Mann-Whitney U test (instead of t-test)
                    - Kruskal-Wallis test (instead of ANOVA)
                    - Wilcoxon signed-rank test (instead of paired t-test)
                    - Friedman test (instead of repeated measures ANOVA)
                    """)
                    
                    # Store test preferences
                    st.session_state['analysis_decisions']['preferred_tests'] = {
                        'two_groups': 'Mann-Whitney U test',
                        'multiple_groups': 'Kruskal-Wallis test',
                        'paired': 'Wilcoxon signed-rank test',
                        'repeated_measures': 'Friedman test'
                    }
            else:
                st.success("Data appears to be normally distributed. Parametric tests are appropriate.")
                st.session_state['analysis_decisions']['use_nonparametric'] = False
                st.session_state['analysis_decisions']['reasoning'] = "Data is normally distributed"
                st.session_state['analysis_decisions']['transformation_applied'] = False
            
            # Display current analysis decisions
            with st.sidebar:
                st.markdown("### Analysis Decisions")
                if st.session_state['analysis_decisions']['use_nonparametric'] is not None:
                    st.markdown(f"**Approach:** {'Non-parametric' if st.session_state['analysis_decisions']['use_nonparametric'] else 'Parametric'}")
                    if st.session_state['analysis_decisions']['transformation_applied']:
                        st.markdown(f"**Transformation:** {st.session_state['analysis_decisions']['transformation_type']}")
                    st.markdown(f"**Reasoning:** {st.session_state['analysis_decisions']['reasoning']}")
    
    elif current_step == "Statistical Test Selection":
        st.markdown('<h2 class="section-header">Statistical Test Selection</h2>', unsafe_allow_html=True)
        if 'data' not in st.session_state or st.session_state['data'] is None:
            st.warning("Please upload or load example data first!")
            return
        
        if 'data_structure' not in st.session_state:
            st.warning("Please select your data structure in the Data Understanding step!")
            return
        
        data = st.session_state['data']
        data_structure = st.session_state['data_structure']
        
        # Test Selection based on data structure and previous decisions
        if st.session_state['analysis_decisions']['use_nonparametric']:
            st.markdown("### Recommended Non-parametric Tests")
            test_info = get_nonparametric_test_info()
            
            if data_structure == "Repeated Measures":
                st.markdown("#### Wilcoxon signed-rank test (2 groups)")
                with st.expander("Test Details"):
                    info = test_info["Wilcoxon signed-rank"]
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown("**Assumptions:**")
                    for assumption in info['assumptions']:
                        st.markdown(f"- {assumption}")
                    st.markdown(f"**Effect Size:** {info['effect_size']}")
                    st.markdown("**Interpretation:**")
                    for size, value in info['interpretation'].items():
                        st.markdown(f"- {size}: {value}")
                    st.markdown("**Code Example:**")
                    st.code(info['code_example'], language="python")
                
                st.markdown("#### Friedman test (3+ groups)")
                with st.expander("Test Details"):
                    info = test_info["Friedman"]
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown("**Assumptions:**")
                    for assumption in info['assumptions']:
                        st.markdown(f"- {assumption}")
                    st.markdown(f"**Effect Size:** {info['effect_size']}")
                    st.markdown("**Interpretation:**")
                    for size, value in info['interpretation'].items():
                        st.markdown(f"- {size}: {value}")
                    st.markdown("**Code Example:**")
                    st.code(info['code_example'], language="python")
            
            elif data_structure == "Independent Groups":
                st.markdown("#### Mann-Whitney U test (2 groups)")
                with st.expander("Test Details"):
                    info = test_info["Mann-Whitney U"]
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown("**Assumptions:**")
                    for assumption in info['assumptions']:
                        st.markdown(f"- {assumption}")
                    st.markdown(f"**Effect Size:** {info['effect_size']}")
                    st.markdown("**Interpretation:**")
                    for size, value in info['interpretation'].items():
                        st.markdown(f"- {size}: {value}")
                    st.markdown("**Code Example:**")
                    st.code(info['code_example'], language="python")
                
                st.markdown("#### Kruskal-Wallis test (3+ groups)")
                with st.expander("Test Details"):
                    info = test_info["Kruskal-Wallis"]
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown("**Assumptions:**")
                    for assumption in info['assumptions']:
                        st.markdown(f"- {assumption}")
                    st.markdown(f"**Effect Size:** {info['effect_size']}")
                    st.markdown("**Interpretation:**")
                    for size, value in info['interpretation'].items():
                        st.markdown(f"- {size}: {value}")
                    st.markdown("**Code Example:**")
                    st.code(info['code_example'], language="python")
            
            # ... similar expansions for other data structures ...
        
        # Display analysis decision context
        with st.expander("Analysis Decision Context"):
            st.markdown(f"**Current Approach:** {'Non-parametric' if st.session_state['analysis_decisions']['use_nonparametric'] else 'Parametric'}")
            if st.session_state['analysis_decisions']['transformation_applied']:
                st.markdown(f"**Applied Transformation:** {st.session_state['analysis_decisions']['transformation_type']}")
            st.markdown(f"**Reasoning:** {st.session_state['analysis_decisions']['reasoning']}")
    
    else:  # Analysis Plan
        st.markdown('<h2 class="section-header">Analysis Plan</h2>', unsafe_allow_html=True)
        if 'data' not in st.session_state or st.session_state['data'] is None:
            st.warning("Please upload or load example data first!")
            return
        
        if 'data_structure' not in st.session_state:
            st.warning("Please select your data structure in the Data Understanding step!")
            return
        
        # Generate comprehensive analysis plan
        plan = generate_analysis_plan()
        
        # Display the plan
        st.markdown("### Evidence-Based Analysis Plan")
        
        # Evidence Section
        with st.expander("Evidence for Analysis Decisions", expanded=True):
            st.markdown("#### Distributional Analysis Evidence")
            for evidence in plan["evidence"]:
                st.markdown(f"- {evidence}")
        
        # Decision Tree
        with st.expander("Decision Tree", expanded=True):
            for step in plan["decision_tree"]:
                st.markdown(step)
        
        # Recommended Tests
        st.markdown("### Recommended Statistical Tests")
        for test in plan["recommended_tests"]:
            st.markdown(f"- {test}")
        
        # Alternative Tests
        with st.expander("Alternative Approaches"):
            st.markdown("#### Robust Alternatives")
            for test in plan["alternative_tests"]:
                st.markdown(f"- {test}")
        
        # Code Template
        with st.expander("Implementation Code"):
            st.code(plan["code_template"], language="python")
        
        # Visualization Plan
        st.markdown("### Visualization Plan")
        for viz in plan["visualization_plan"]:
            st.markdown(viz)
        
        # Interpretation Guidelines
        st.markdown("### Interpretation Guidelines")
        for guideline in plan["interpretation_guidelines"]:
            st.markdown(guideline)
        
        # Export Plan
        if st.button("Export Analysis Plan"):
            st.download_button(
                "Download Plan",
                data=json.dumps(plan, indent=2),
                file_name="statistical_analysis_plan.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main() 