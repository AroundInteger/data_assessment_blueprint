import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="PhD Data Analysis Blueprint", 
    page_icon="🔬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 1rem;
    }
    
    .phase-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .phase-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .phase-description {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .question-box {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .ai-response {
        background: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .data-type-badge {
        background: #3498db;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = "Overview"
if 'data_type' not in st.session_state:
    st.session_state.data_type = "General"
if 'analysis_log' not in st.session_state:
    st.session_state.analysis_log = []

# Phase definitions
PHASES = {
    "Overview": {
        "title": "Analysis Blueprint Overview",
        "description": "Understanding the complete data analysis workflow",
        "color": "#34495e",
        "key_questions": [
            "What is the overall structure of rigorous data analysis?",
            "How do I know which phase I should focus on first?",
            "What are the common pitfalls in data analysis workflows?",
            "How do I document my analysis process effectively?"
        ],
        "best_practices": [
            "Always start with a clear research question",
            "Document every step of your analysis process",
            "Use version control for your analysis code",
            "Create reproducible workflows"
        ]
    },
    "Phase 1": {
        "title": "Initial Data Exploration & Documentation",
        "description": "Data provenance, structure assessment, and metadata review",
        "color": "#3498db",
        "key_questions": [
            "How do I assess the quality and completeness of my metadata?",
            "What should I document about the data collection methodology?",
            "How do I identify potential issues in data structure or format?",
            "What are the key questions to ask about data provenance?",
            "How do I handle missing data patterns in initial exploration?"
        ],
        "best_practices": [
            "Create a data dictionary before analysis",
            "Document all data transformations",
            "Use exploratory data analysis (EDA) tools",
            "Implement data quality checks",
            "Maintain a data lineage record"
        ]
    },
    "Phase 2": {
        "title": "Measurement Quality & Uncertainty Analysis",
        "description": "Precision, accuracy evaluation, and signal quality assessment",
        "color": "#9b59b6",
        "key_questions": [
            "How do I estimate measurement uncertainty in my data?",
            "What methods can I use to assess signal-to-noise ratio?",
            "How do I identify systematic biases or drift in measurements?",
            "What are appropriate ways to handle outliers and artifacts?",
            "How do measurement uncertainties propagate through calculations?"
        ],
        "best_practices": [
            "Calculate and document measurement uncertainties",
            "Use calibration standards regularly",
            "Implement quality control procedures",
            "Monitor for systematic biases",
            "Document all measurement protocols"
        ]
    },
    "Phase 3": {
        "title": "Statistical Characterization",
        "description": "Distributional analysis, pattern recognition, and pre-analysis testing",
        "color": "#e74c3c",
        "key_questions": [
            "How do I systematically test for normality in my data?",
            "When should I use parametric vs non-parametric methods?",
            "How do I handle violations of statistical assumptions?",
            "What transformations are appropriate for my data type?",
            "How do I test for temporal or spatial autocorrelation?",
            "What sample size considerations affect my statistical choices?"
        ],
        "best_practices": [
            "Use multiple statistical tests",
            "Document all statistical assumptions",
            "Implement appropriate data transformations",
            "Consider effect sizes, not just p-values",
            "Use visualization to complement statistical tests"
        ]
    },
    "Phase 4": {
        "title": "Validation & Cross-Checking",
        "description": "Internal consistency, external validation, and benchmark comparison",
        "color": "#f39c12",
        "key_questions": [
            "How do I validate my results against known benchmarks?",
            "What internal consistency checks should I perform?",
            "How do I compare my findings with published literature?",
            "What cross-validation approaches are appropriate?",
            "How do I assess the reliability of my analytical pipeline?"
        ],
        "best_practices": [
            "Use multiple validation methods",
            "Compare against established benchmarks",
            "Implement cross-validation techniques",
            "Document all validation steps",
            "Consider biological/physical plausibility"
        ]
    },
    "Phase 5": {
        "title": "Limitation Documentation",
        "description": "Constraint identification, uncertainty propagation, and boundary conditions",
        "color": "#27ae60",
        "key_questions": [
            "How do I systematically identify limitations in my analysis?",
            "What are the boundaries of validity for my conclusions?",
            "How do I quantify and communicate uncertainty in results?",
            "What assumptions underlie my analytical approach?",
            "How do I prepare limitations for publication or presentation?"
        ],
        "best_practices": [
            "Document all assumptions explicitly",
            "Quantify uncertainties in final results",
            "Consider alternative interpretations",
            "Identify scope of applicability",
            "Prepare clear limitations section"
        ]
    }
}

# Data type specific guidance
DATA_TYPE_GUIDANCE = {
    "Temporal (Race Measurements)": {
        "phase_specific": {
            "Phase 1": "Focus on sampling rates, temporal resolution, timing accuracy, and synchronization across measurement systems.",
            "Phase 2": "Assess measurement precision across time, identify drift or calibration issues, evaluate temporal artifacts.",
            "Phase 3": "Test for stationarity, autocorrelation patterns, trend analysis, and appropriate time series models.",
            "Phase 4": "Compare against established performance benchmarks, validate timing accuracy, cross-check with video analysis.",
            "Phase 5": "Document temporal resolution limits, measurement frequency constraints, and performance context dependencies."
        }
    },
    "Spatiotemporal (Sports Motion)": {
        "phase_specific": {
            "Phase 1": "Examine coordinate system consistency, spatial-temporal sampling rates, motion capture completeness.",
            "Phase 2": "Evaluate spatial accuracy, marker tracking reliability, assess motion artifacts and occlusions.",
            "Phase 3": "Test for spatial autocorrelation, repeated measures assumptions, hierarchical data structures.",
            "Phase 4": "Validate against biomechanical models, compare with established movement patterns, cross-check kinematic consistency.",
            "Phase 5": "Document spatial resolution limits, movement complexity boundaries, environmental constraint effects."
        }
    },
    "Fluorescence (Biological Timelapse)": {
        "phase_specific": {
            "Phase 1": "Assess imaging parameters, temporal sampling adequacy, spatial resolution, and experimental design completeness.",
            "Phase 2": "Evaluate photobleaching effects, background fluorescence, signal saturation, and measurement photophysics.",
            "Phase 3": "Consider Poisson vs negative binomial distributions, hierarchical structures (cells/wells/plates), time-dependent effects.",
            "Phase 4": "Validate against positive/negative controls, compare with established cellular responses, verify biological plausibility.",
            "Phase 5": "Document photophysical limitations, temporal resolution trade-offs, biological system constraints."
        }
    }
}

def create_phase_flowchart():
    """Create an interactive flowchart of the analysis phases"""
    fig = go.Figure()
    
    # Define positions for each phase with more spacing
    phases = list(PHASES.keys())[1:]  # Exclude Overview
    y_positions = [5, 3.5, 2, 0.5, -1]  # Increased vertical spacing
    
    # Add phase boxes with improved layout
    for i, (phase, pos_y) in enumerate(zip(phases, y_positions)):
        phase_info = PHASES[phase]
        
        # Add phase box
        fig.add_trace(go.Scatter(
            x=[-0.8], y=[pos_y],  # Moved circles further left
            mode='markers+text',
            marker=dict(
                size=120,
                color=phase_info['color'],
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=phase,
            textposition='middle center',
            textfont=dict(size=14, color='white', family='Arial Black'),
            name=phase,
            hovertemplate=f"<b>{phase_info['title']}</b><br>{phase_info['description']}<extra></extra>"
        ))
        
        # Add phase title and description to the right of the circle
        fig.add_annotation(
            x=-0.6,  # Moved text closer to circle
            y=pos_y,
            text=f"<b>{phase_info['title']}</b><br>{phase_info['description']}",
            showarrow=False,
            font=dict(size=14, color='#2c3e50'),
            xanchor='left',
            yanchor='middle',
            align='left',
            xref='x',
            yref='y'
        )
        
        # Add arrows between phases with improved styling
        if i < len(phases) - 1:
            fig.add_annotation(
                x=-0.8,  # Align with circles
                y=pos_y - 0.4,
                ax=-0.8,
                ay=pos_y - 0.6,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='#2c3e50'
            )
    
    # Update layout with improved styling
    fig.update_layout(
        title=dict(
            text="Interactive Analysis Blueprint Flow",
            font=dict(size=20, color='#2c3e50'),
            y=0.95
        ),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.0, 0.2]  # Adjusted range to ensure all text is visible
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-2, 6]
        ),
        height=600,
        plot_bgcolor='white',
        margin=dict(t=100, b=50, l=50, r=50),
        paper_bgcolor='white'
    )
    
    return fig

def simulate_ai_response(question, phase, data_type):
    """Simulate an AI response based on the question, phase, and data type"""
    
    # This is a simplified simulation - in practice, you'd integrate with OpenAI API, Anthropic, etc.
    responses = {
        "Phase 1": {
            "general": "For initial data exploration, start by examining the basic structure and completeness of your dataset. Key steps include: 1) Documenting data source and collection methods, 2) Assessing data completeness and missing value patterns, 3) Validating data formats and consistency, 4) Creating initial summary statistics.",
            "specific": f"For {data_type} data, pay special attention to the guidance provided in your data type section. Ensure you understand the specific measurement characteristics and potential issues relevant to your domain."
        },
        "Phase 2": {
            "general": "Measurement quality assessment involves evaluating both precision and accuracy of your measurements. Consider: 1) Signal-to-noise ratio analysis, 2) Identification of systematic biases, 3) Uncertainty quantification, 4) Assessment of measurement artifacts.",
            "specific": f"With {data_type} data, focus on domain-specific quality metrics and common sources of measurement error in your field."
        },
        "Phase 3": {
            "general": "Statistical characterization requires systematic evaluation of your data distributions and patterns. Key steps: 1) Visual inspection (histograms, Q-Q plots), 2) Multiple normality tests, 3) Assessment of statistical assumptions for planned analyses, 4) Selection of appropriate statistical methods.",
            "specific": f"For {data_type} data, consider the specific distributional characteristics and correlation structures typical in your domain."
        },
        "Phase 4": {
            "general": "Validation involves checking your results against known standards and expectations. Include: 1) Internal consistency checks, 2) Comparison with literature values, 3) Cross-validation approaches, 4) Sensitivity analysis.",
            "specific": f"In {data_type} analysis, validation should include domain-specific benchmarks and biological/physical plausibility checks."
        },
        "Phase 5": {
            "general": "Thorough limitation documentation enhances the credibility of your work. Address: 1) Methodological constraints, 2) Data quality boundaries, 3) Uncertainty propagation, 4) Scope of applicability.",
            "specific": f"For {data_type} data, be explicit about temporal/spatial resolution limits and measurement-specific constraints."
        }
    }
    
    # Get base response
    base_response = responses.get(phase, {}).get("general", "I'd be happy to help with your question about data analysis!")
    specific_response = responses.get(phase, {}).get("specific", "")
    
    # Combine and personalize
    full_response = f"{base_response}\n\n{specific_response}"
    
    # Add contextual guidance based on question content
    if "normality" in question.lower():
        full_response += "\n\n**Specific guidance on normality testing**: Use multiple approaches - visual inspection (Q-Q plots, histograms) combined with statistical tests (Shapiro-Wilk for n<5000, Anderson-Darling for tail sensitivity). Remember that large samples will often reject normality even for minor deviations that may not be practically significant."
    
    if "uncertainty" in question.lower():
        full_response += "\n\n**Uncertainty considerations**: Quantify both random and systematic uncertainties. Consider how uncertainties propagate through your calculations and impact your final conclusions. Document uncertainty sources and their relative contributions."
    
    return full_response

def create_decision_diagram(phase, data_type):
    """Create an interactive decision diagram for the selected phase"""
    phase_info = PHASES[phase]
    
    # Create a new figure
    fig = go.Figure()
    
    # Define node positions with more space for complex flows
    nodes = {
        "start": {"x": 0, "y": 0},
        "decision": {"x": 0, "y": -1.5},
        "action1": {"x": -2, "y": -3},
        "action2": {"x": 0, "y": -3},
        "action3": {"x": 2, "y": -3},
        "subdecision1": {"x": -2, "y": -4.5},
        "subdecision2": {"x": 2, "y": -4.5},
        "end1": {"x": -2.5, "y": -6},
        "end2": {"x": -1.5, "y": -6},
        "end3": {"x": 1.5, "y": -6},
        "end4": {"x": 2.5, "y": -6}
    }
    
    # Add nodes with hover information
    node_colors = {
        "start": "#2ecc71",  # Green
        "decision": "#3498db",  # Blue
        "action": "#e74c3c",  # Red
        "end": "#2ecc71"  # Green
    }
    
    # Node sizes
    node_sizes = {
        "start": 100,  # Increased size
        "decision": 100,  # Increased size
        "action": 100,
        "subdecision": 80,
        "end": 80
    }
    
    # Start node
    fig.add_trace(go.Scatter(
        x=[nodes["start"]["x"]],
        y=[nodes["start"]["y"]],
        mode='markers+text',
        marker=dict(size=node_sizes["start"], color=node_colors["start"]),
        text=["Start"],
        textposition='middle center',
        textfont=dict(color='#2c3e50', size=14, family='Arial Black'),
        name='Start',
        hovertemplate="Begin your analysis process<extra></extra>"
    ))
    
    # Decision node with wrapped text
    fig.add_trace(go.Scatter(
        x=[nodes["decision"]["x"]],
        y=[nodes["decision"]["y"]],
        mode='markers+text',
        marker=dict(size=node_sizes["decision"], color=node_colors["decision"]),
        text=["Initial<br>Assessment"],  # Wrapped text
        textposition='middle center',
        textfont=dict(color='#2c3e50', size=14, family='Arial Black'),
        name='Decision',
        hovertemplate="Evaluate your data and choose appropriate actions<extra></extra>"
    ))
    
    # Action nodes with best practices
    for i, practice in enumerate(phase_info["best_practices"][:3]):
        x = nodes[f"action{i+1}"]["x"]
        y = nodes[f"action{i+1}"]["y"]
        
        # Add the node with a short label
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=node_sizes["action"], color=node_colors["action"]),
            text=["Action"],
            textposition='middle center',
            textfont=dict(color='#2c3e50', size=14, family='Arial Black'),
            name=f'Action {i+1}',
            hovertemplate=f"<b>{practice}</b><br>Click for detailed guidance<extra></extra>"
        ))
        
        # Add the practice text to the side with wrapping
        text_x = x + 0.5 if x >= 0 else x - 0.5
        # Split text into 2-3 lines
        words = practice.split()
        if len(words) > 4:
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                if len(current_line) >= 4:
                    lines.append(' '.join(current_line))
                    current_line = []
            if current_line:
                lines.append(' '.join(current_line))
            wrapped_text = '<br>'.join(lines)
        else:
            # Special case for "Document all data transformations"
            if practice == "Document all data transformations":
                wrapped_text = "Document all<br>data<br>transformations"
            else:
                wrapped_text = practice
            
        fig.add_annotation(
            x=text_x,
            y=y,
            text=wrapped_text,
            showarrow=False,
            font=dict(size=12, color='#2c3e50'),
            xanchor='left' if x >= 0 else 'right',
            yanchor='middle',
            align='left' if x >= 0 else 'right',
            xref='x',
            yref='y'
        )
    
    # Sub-decision nodes
    for i in range(1, 3):
        x = nodes[f"subdecision{i}"]["x"]
        y = nodes[f"subdecision{i}"]["y"]
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=node_sizes["subdecision"], color=node_colors["decision"]),
            text=["Next<br>Steps"],  # Wrapped text
            textposition='middle center',
            textfont=dict(color='#2c3e50', size=14, family='Arial Black'),
            name=f'Sub-decision {i}',
            hovertemplate="Choose your next action based on results<extra></extra>"
        ))
    
    # End nodes
    for i in range(1, 5):
        x = nodes[f"end{i}"]["x"]
        y = nodes[f"end{i}"]["y"]
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=node_sizes["end"], color=node_colors["end"]),
            text=["Complete"],
            textposition='middle center',
            textfont=dict(color='#2c3e50', size=14, family='Arial Black'),
            name=f'End {i}',
            hovertemplate="Phase completion point<extra></extra>"
        ))
    
    # Add edges with arrows
    edges = [
        ("start", "decision"),
        ("decision", "action1"),
        ("decision", "action2"),
        ("decision", "action3"),
        ("action1", "subdecision1"),
        ("action3", "subdecision2"),
        ("subdecision1", "end1"),
        ("subdecision1", "end2"),
        ("subdecision2", "end3"),
        ("subdecision2", "end4"),
        ("action2", "end2")
    ]
    
    # Calculate arrow positions to connect at node perimeters
    def get_arrow_positions(start_node, end_node):
        # Get node positions
        start_x, start_y = nodes[start_node]["x"], nodes[start_node]["y"]
        end_x, end_y = nodes[end_node]["x"], nodes[end_node]["y"]
        
        # Calculate direction vector
        dx = end_x - start_x
        dy = end_y - start_y
        length = (dx**2 + dy**2)**0.5
        
        # Normalize direction vector
        if length > 0:
            dx /= length
            dy /= length
        
        # Calculate exact radius for start and end nodes
        start_radius = node_sizes.get(start_node, 50) / 200  # Convert to plot units (radius = diameter/2)
        end_radius = node_sizes.get(end_node, 50) / 200
        
        # Calculate arrow start and end positions exactly at the circle perimeters
        start_x_adj = start_x + dx * start_radius
        start_y_adj = start_y + dy * start_radius
        end_x_adj = end_x - dx * end_radius
        end_y_adj = end_y - dy * end_radius
        
        return start_x_adj, start_y_adj, end_x_adj, end_y_adj
    
    for start, end in edges:
        start_x, start_y, end_x, end_y = get_arrow_positions(start, end)
        fig.add_annotation(
            x=end_x,
            y=end_y,
            ax=start_x,
            ay=start_y,
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='#2c3e50'
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{phase} Decision Flow",
            font=dict(size=20, color='#2c3e50'),
            y=0.95
        ),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-4, 4]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-7, 1]
        ),
        height=800,
        plot_bgcolor='white',
        margin=dict(t=100, b=50, l=50, r=50),
        paper_bgcolor='white',
        hovermode='closest'
    )
    
    return fig

def get_llm_guidance(phase, data_type, node_type, node_content):
    """Get LLM guidance for specific decision points"""
    guidance = {
        "start": f"Begin your {phase} analysis by reviewing the key objectives and requirements.",
        "decision": f"Consider your {data_type} specific needs when making decisions in {phase}.",
        "action": f"For {node_content}, consider these specific steps:\n1. Review relevant documentation\n2. Apply appropriate methods\n3. Document your process",
        "end": f"Complete your {phase} analysis by ensuring all requirements are met."
    }
    return guidance.get(node_type, "Continue with your analysis process.")

def create_statistical_flow_diagram():
    """Create an interactive statistical analysis flow diagram for repeated measurements"""
    fig = go.Figure()
    
    # Define node positions
    nodes = {
        "start": {"x": 0, "y": 0, "text": "Repeated<br>Measurements"},
        "normality": {"x": 0, "y": -2, "text": "Test for<br>Normality"},
        "normal_yes": {"x": -2, "y": -4, "text": "Data is<br>Normal"},
        "normal_no": {"x": 2, "y": -4, "text": "Data is<br>Not Normal"},
        "parametric": {"x": -2, "y": -6, "text": "Parametric<br>Tests"},
        "nonparametric": {"x": 2, "y": -6, "text": "Non-parametric<br>Tests"},
        "t_test": {"x": -3, "y": -8, "text": "Paired t-test"},
        "anova": {"x": -1, "y": -8, "text": "Repeated<br>Measures<br>ANOVA"},
        "wilcoxon": {"x": 1, "y": -8, "text": "Wilcoxon<br>Signed-Rank"},
        "friedman": {"x": 3, "y": -8, "text": "Friedman<br>Test"},
        "end1": {"x": -3, "y": -10, "text": "Compare<br>Means"},
        "end2": {"x": -1, "y": -10, "text": "Compare<br>Multiple<br>Groups"},
        "end3": {"x": 1, "y": -10, "text": "Compare<br>Medians"},
        "end4": {"x": 3, "y": -10, "text": "Compare<br>Multiple<br>Groups"}
    }
    
    # Node colors
    node_colors = {
        "start": "#2ecc71",  # Green
        "normality": "#3498db",  # Blue
        "normal_yes": "#2ecc71",  # Green
        "normal_no": "#e74c3c",  # Red
        "parametric": "#3498db",  # Blue
        "nonparametric": "#3498db",  # Blue
        "t_test": "#9b59b6",  # Purple
        "anova": "#9b59b6",  # Purple
        "wilcoxon": "#9b59b6",  # Purple
        "friedman": "#9b59b6",  # Purple
        "end": "#2ecc71"  # Green
    }
    
    # Node sizes
    node_sizes = {
        "start": 100,
        "normality": 100,
        "normal_yes": 80,
        "normal_no": 80,
        "parametric": 80,
        "nonparametric": 80,
        "t_test": 80,
        "anova": 80,
        "wilcoxon": 80,
        "friedman": 80,
        "end": 80
    }
    
    # Add nodes
    for node_id, node_info in nodes.items():
        fig.add_trace(go.Scatter(
            x=[node_info["x"]],
            y=[node_info["y"]],
            mode='markers+text',
            marker=dict(
                size=node_sizes.get(node_id, 80),
                color=node_colors.get(node_id, "#3498db"),
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=[node_info["text"]],
            textposition='middle center',
            textfont=dict(color='#2c3e50', size=14, family='Arial Black'),
            name=node_id,
            hovertemplate=f"<b>{node_info['text']}</b><br>Click for detailed guidance<extra></extra>"
        ))
    
    # Define edges
    edges = [
        ("start", "normality"),
        ("normality", "normal_yes"),
        ("normality", "normal_no"),
        ("normal_yes", "parametric"),
        ("normal_no", "nonparametric"),
        ("parametric", "t_test"),
        ("parametric", "anova"),
        ("nonparametric", "wilcoxon"),
        ("nonparametric", "friedman"),
        ("t_test", "end1"),
        ("anova", "end2"),
        ("wilcoxon", "end3"),
        ("friedman", "end4")
    ]
    
    # Calculate arrow positions to connect at node perimeters
    def get_arrow_positions(start_node, end_node):
        start_x, start_y = nodes[start_node]["x"], nodes[start_node]["y"]
        end_x, end_y = nodes[end_node]["x"], nodes[end_node]["y"]
        
        dx = end_x - start_x
        dy = end_y - start_y
        length = (dx**2 + dy**2)**0.5
        
        if length > 0:
            dx /= length
            dy /= length
        
        start_radius = node_sizes.get(start_node, 50) / 200
        end_radius = node_sizes.get(end_node, 50) / 200
        
        start_x_adj = start_x + dx * start_radius
        start_y_adj = start_y + dy * start_radius
        end_x_adj = end_x - dx * end_radius
        end_y_adj = end_y - dy * end_radius
        
        return start_x_adj, start_y_adj, end_x_adj, end_y_adj
    
    # Add edges
    for start, end in edges:
        start_x, start_y, end_x, end_y = get_arrow_positions(start, end)
        fig.add_annotation(
            x=end_x,
            y=end_y,
            ax=start_x,
            ay=start_y,
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='#2c3e50'
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Statistical Analysis Flow for Repeated Measurements",
            font=dict(size=20, color='#2c3e50'),
            y=0.95
        ),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-4, 4]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-11, 1]
        ),
        height=1000,
        plot_bgcolor='white',
        margin=dict(t=100, b=50, l=50, r=50),
        paper_bgcolor='white',
        hovermode='closest'
    )
    
    return fig

def get_statistical_guidance(node_type, test_type=None):
    """Get detailed guidance for statistical analysis steps"""
    guidance = {
        "start": {
            "title": "Repeated Measurements Analysis",
            "description": "Begin by examining your repeated measurements data. Consider:\n- Number of measurements per subject\n- Time points or conditions\n- Missing data patterns\n- Outliers"
        },
        "normality": {
            "title": "Normality Testing",
            "description": "Choose appropriate normality tests:\n- Shapiro-Wilk (n < 5000)\n- Anderson-Darling (sensitive to tails)\n- Q-Q plots for visual assessment\n- Histograms for distribution shape"
        },
        "normal_yes": {
            "title": "Normal Distribution Confirmed",
            "description": "Proceed with parametric tests:\n- Check for equal variances\n- Consider sample size\n- Look for outliers"
        },
        "normal_no": {
            "title": "Non-normal Distribution",
            "description": "Consider non-parametric alternatives:\n- Check for transformations\n- Assess sample size\n- Consider robust methods"
        },
        "parametric": {
            "title": "Parametric Tests",
            "description": "Choose based on:\n- Number of groups\n- Time points\n- Experimental design"
        },
        "nonparametric": {
            "title": "Non-parametric Tests",
            "description": "Choose based on:\n- Number of groups\n- Time points\n- Experimental design"
        },
        "t_test": {
            "title": "Paired t-test",
            "description": "Use when:\n- Comparing two time points\n- Data is normally distributed\n- Samples are paired"
        },
        "anova": {
            "title": "Repeated Measures ANOVA",
            "description": "Use when:\n- Comparing multiple time points\n- Data is normally distributed\n- Sphericity assumption is met"
        },
        "wilcoxon": {
            "title": "Wilcoxon Signed-Rank Test",
            "description": "Use when:\n- Comparing two time points\n- Data is not normally distributed\n- Samples are paired"
        },
        "friedman": {
            "title": "Friedman Test",
            "description": "Use when:\n- Comparing multiple time points\n- Data is not normally distributed\n- Samples are related"
        }
    }
    
    return guidance.get(node_type, {}).get("description", "No guidance available for this step.")

def get_statistical_code_example(test_type, data_type="repeated_measures"):
    """Get Python code examples for statistical tests"""
    code_examples = {
        "normality_shapiro": """
# Shapiro-Wilk test for normality
from scipy import stats

# For a single group of measurements
statistic, p_value = stats.shapiro(data)

# For multiple groups
results = {}
for group_name, group_data in data_dict.items():
    statistic, p_value = stats.shapiro(group_data)
    results[group_name] = {"statistic": statistic, "p_value": p_value}

print("Normality test results:")
for group, result in results.items():
    print(f"{group}: p-value = {result['p_value']:.4f}")
""",
        "normality_qq": """
# Q-Q plot for normality assessment
import matplotlib.pyplot as plt
import scipy.stats as stats

# Create Q-Q plot
stats.probplot(data, dist="norm", plot=plt)
plt.title("Q-Q Plot for Normality Assessment")
plt.show()
""",
        "paired_ttest": """
# Paired t-test for repeated measurements
from scipy import stats

# For two time points
t_stat, p_value = stats.ttest_rel(timepoint1_data, timepoint2_data)

print(f"Paired t-test results:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
""",
        "repeated_anova": """
# Repeated measures ANOVA
import pandas as pd
from statsmodels.stats.anova import AnovaRM

# Prepare data in long format
data_long = pd.DataFrame({
    'subject': subjects,
    'timepoint': timepoints,
    'value': values
})

# Perform repeated measures ANOVA
aovrm = AnovaRM(data_long, 'value', 'subject', within=['timepoint'])
res = aovrm.fit()

print(res.summary())
""",
        "wilcoxon": """
# Wilcoxon signed-rank test
from scipy import stats

# For two time points
statistic, p_value = stats.wilcoxon(timepoint1_data, timepoint2_data)

print(f"Wilcoxon signed-rank test results:")
print(f"statistic: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")
""",
        "friedman": """
# Friedman test for multiple time points
from scipy import stats

# For multiple time points
statistic, p_value = stats.friedmanchisquare(*[data for timepoint in timepoints])

print(f"Friedman test results:")
print(f"statistic: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")
"""
    }
    return code_examples.get(test_type, "Code example not available for this test.")

def get_data_guidance_questions():
    """Get questions to help users understand their data"""
    return [
        "How many measurements do you have per subject?",
        "How many time points or conditions are you comparing?",
        "Are your measurements taken at regular intervals?",
        "Do you have any missing data?",
        "Have you noticed any outliers in your data?",
        "What is the range of your measurements?",
        "Are your measurements continuous or discrete?",
        "Do you expect any learning or fatigue effects?",
        "Are there any known sources of variability in your measurements?",
        "What is your primary research question?"
    ]

def generate_analysis_plan(data_description, test_type, normality_test=None):
    """Generate a structured analysis plan with checkboxes and scoring"""
    plan = {
        "Data Quality Assessment": {
            "items": [
                "Check for missing values",
                "Identify and handle outliers",
                "Assess measurement reliability",
                "Verify data collection consistency",
                "Review measurement protocols"
            ],
            "weight": 20  # 20% of total score
        },
        "Statistical Assumptions": {
            "items": [
                "Test for normality",
                "Check for equal variances",
                "Assess sample size adequacy",
                "Verify independence of observations",
                "Check for sphericity (if applicable)"
            ],
            "weight": 25  # 25% of total score
        },
        "Analysis Execution": {
            "items": [
                "Perform primary statistical test",
                "Conduct post-hoc analysis (if needed)",
                "Calculate effect sizes",
                "Generate appropriate visualizations",
                "Document test assumptions"
            ],
            "weight": 30  # 30% of total score
        },
        "Results Interpretation": {
            "items": [
                "Interpret statistical significance",
                "Evaluate practical significance",
                "Consider effect sizes",
                "Document limitations",
                "Compare with previous findings"
            ],
            "weight": 25  # 25% of total score
        }
    }
    
    # Add specific items based on test type
    if test_type == "Normality Testing":
        plan["Statistical Assumptions"]["items"].append(f"Perform {normality_test}")
    elif "Time Points" in test_type:
        if "Two" in test_type:
            plan["Analysis Execution"]["items"].append("Perform paired comparison")
        else:
            plan["Analysis Execution"]["items"].append("Perform multiple comparisons")
    
    return plan

def calculate_plan_score(plan, completed_items):
    """Calculate the completeness score of the analysis plan"""
    total_score = 0
    max_score = 0
    
    for section, details in plan.items():
        section_score = 0
        section_max = len(details["items"])
        
        for item in details["items"]:
            if item in completed_items:
                section_score += 1
        
        # Calculate weighted section score
        section_percentage = (section_score / section_max) * details["weight"]
        total_score += section_percentage
        max_score += details["weight"]
    
    return (total_score / max_score) * 100

def update_completed_items(item):
    """Update the set of completed items in session state"""
    if "completed_items" not in st.session_state:
        st.session_state.completed_items = set()
    
    if item in st.session_state.completed_items:
        st.session_state.completed_items.remove(item)
    else:
        st.session_state.completed_items.add(item)

def get_case_studies():
    """Get example case studies for statistical analysis"""
    return {
        "Repeated Measures in Sports Performance": {
            "description": "Analysis of sprint times across multiple training sessions",
            "data_type": "Temporal (Race Measurements)",
            "scenario": """
            A sports scientist is analyzing the effect of a new training program on sprint performance.
            Data includes:
            - 20 athletes
            - 4 time points (baseline, 2 weeks, 4 weeks, 6 weeks)
            - 3 trials per time point
            - Measured: 40m sprint time
            """,
            "thought_process": [
                "1. Data Quality Assessment",
                "   - Check for learning effects across trials",
                "   - Identify any outliers or measurement errors",
                "   - Verify timing equipment consistency",
                "   - Assess environmental conditions",
                "",
                "2. Statistical Considerations",
                "   - Repeated measures design (same athletes)",
                "   - Multiple time points (4)",
                "   - Need to account for individual differences",
                "   - Consider training adaptation patterns",
                "",
                "3. Analysis Approach",
                "   - Test for normality at each time point",
                "   - Check for sphericity assumption",
                "   - Use repeated measures ANOVA",
                "   - Post-hoc tests for specific comparisons",
                "   - Calculate effect sizes for practical significance"
            ],
            "code_example": """
# Example code for sports performance analysis
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM

# Create example data
np.random.seed(42)
n_athletes = 20
n_timepoints = 4
n_trials = 3

# Generate data with improvement trend
baseline = np.random.normal(5.5, 0.3, (n_athletes, n_trials))
week2 = baseline - np.random.normal(0.2, 0.1, (n_athletes, n_trials))
week4 = baseline - np.random.normal(0.4, 0.1, (n_athletes, n_trials))
week6 = baseline - np.random.normal(0.5, 0.1, (n_athletes, n_trials))

# Create DataFrame
data = []
for athlete in range(n_athletes):
    for trial in range(n_trials):
        data.append({
            'athlete': athlete,
            'timepoint': 'baseline',
            'trial': trial,
            'time': baseline[athlete, trial]
        })
        data.append({
            'athlete': athlete,
            'timepoint': 'week2',
            'trial': trial,
            'time': week2[athlete, trial]
        })
        data.append({
            'athlete': athlete,
            'timepoint': 'week4',
            'trial': trial,
            'time': week4[athlete, trial]
        })
        data.append({
            'athlete': athlete,
            'timepoint': 'week6',
            'trial': trial,
            'time': week6[athlete, trial]
        })

df = pd.DataFrame(data)

# 1. Visualize data
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='timepoint', y='time')
plt.title('Sprint Times Across Training Period')
plt.xlabel('Time Point')
plt.ylabel('Sprint Time (s)')
plt.show()

# 2. Test for normality
for timepoint in df['timepoint'].unique():
    data = df[df['timepoint'] == timepoint]['time']
    stat, p = stats.shapiro(data)
    print(f"{timepoint}: p-value = {p:.4f}")

# 3. Perform repeated measures ANOVA
# Calculate mean for each athlete at each timepoint
means = df.groupby(['athlete', 'timepoint'])['time'].mean().reset_index()

# Perform ANOVA
aovrm = AnovaRM(means, 'time', 'athlete', within=['timepoint'])
res = aovrm.fit()
print(res.summary())

# 4. Post-hoc analysis
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(means['time'], means['timepoint'])
print(tukey.summary())

# 5. Calculate effect sizes
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_se

# Calculate effect sizes between timepoints
timepoints = means['timepoint'].unique()
for i in range(len(timepoints)-1):
    d = cohens_d(
        means[means['timepoint'] == timepoints[i]]['time'],
        means[means['timepoint'] == timepoints[i+1]]['time']
    )
    print(f"Effect size {timepoints[i]} to {timepoints[i+1]}: {d:.2f}")
""",
            "expected_outcomes": {
                "visualizations": [
                    "Box plots showing improvement trend",
                    "Individual athlete progress lines",
                    "Effect size comparisons"
                ],
                "statistical_results": [
                    "Significant main effect of time (p < 0.05)",
                    "Post-hoc tests showing improvement at each timepoint",
                    "Large effect sizes (>0.8) for training effects"
                ],
                "interpretation": [
                    "Clear improvement in sprint performance",
                    "Individual variation in response to training",
                    "Practical significance of improvements"
                ]
            }
        },
        "Biological Time Series": {
            "description": "Analysis of cell growth measurements over time",
            "data_type": "Fluorescence (Biological Timelapse)",
            "scenario": """
            A biologist is studying the effect of a drug on cell growth.
            Data includes:
            - 3 treatment groups (Control, Low dose, High dose)
            - 6 time points (0, 24, 48, 72, 96, 120 hours)
            - 8 replicates per group
            - Measured: Cell count and fluorescence intensity
            """,
            "thought_process": [
                "1. Data Quality Assessment",
                "   - Check for photobleaching effects",
                "   - Assess background fluorescence",
                "   - Verify cell counting accuracy",
                "   - Check for plate effects",
                "",
                "2. Statistical Considerations",
                "   - Hierarchical structure (cells/wells/plates)",
                "   - Time series nature of data",
                "   - Multiple measurements per well",
                "   - Need to account for growth curves",
                "",
                "3. Analysis Approach",
                "   - Test for normality in residuals",
                "   - Consider mixed effects models",
                "   - Analyze growth curves",
                "   - Compare treatment effects"
            ],
            "code_example": """
# Example code for biological time series analysis
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
from scipy.optimize import curve_fit

# Create example data
np.random.seed(42)
n_groups = 3
n_timepoints = 6
n_replicates = 8

# Generate growth curve data
def growth_curve(t, a, b, c):
    return a / (1 + np.exp(-b * (t - c)))

# Parameters for different groups
params = {
    'Control': (100, 0.1, 48),
    'Low_dose': (80, 0.15, 36),
    'High_dose': (60, 0.2, 24)
}

# Generate data
data = []
timepoints = np.array([0, 24, 48, 72, 96, 120])
for group in params.keys():
    a, b, c = params[group]
    for rep in range(n_replicates):
        for t in timepoints:
            # Add some noise
            value = growth_curve(t, a, b, c) + np.random.normal(0, 5)
            data.append({
                'group': group,
                'timepoint': t,
                'replicate': rep,
                'value': value
            })

df = pd.DataFrame(data)

# 1. Visualize growth curves
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='timepoint', y='value', hue='group')
plt.title('Cell Growth Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('Cell Count')
plt.show()

# 2. Fit growth curves
def fit_group_curve(group_data):
    popt, _ = curve_fit(growth_curve, 
                       group_data['timepoint'], 
                       group_data['value'],
                       p0=[100, 0.1, 48])
    return popt

# Fit curves for each group
group_params = {}
for group in df['group'].unique():
    group_data = df[df['group'] == group]
    group_params[group] = fit_group_curve(group_data)

# 3. Compare parameters
# Create comparison plot
plt.figure(figsize=(12, 6))
for group, params in group_params.items():
    t = np.linspace(0, 120, 100)
    y = growth_curve(t, *params)
    plt.plot(t, y, label=group)
plt.title('Fitted Growth Curves')
plt.xlabel('Time (hours)')
plt.ylabel('Cell Count')
plt.legend()
plt.show()

# 4. Statistical analysis
# Test for differences at each timepoint
results = []
for t in timepoints:
    time_data = df[df['timepoint'] == t]
    f_stat, p_val = stats.f_oneway(
        time_data[time_data['group'] == 'Control']['value'],
        time_data[time_data['group'] == 'Low_dose']['value'],
        time_data[time_data['group'] == 'High_dose']['value']
    )
    results.append({
        'timepoint': t,
        'f_statistic': f_stat,
        'p_value': p_val
    })

results_df = pd.DataFrame(results)
print("ANOVA results at each timepoint:")
print(results_df)

# 5. Calculate effect sizes
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_se

# Calculate effect sizes between groups at final timepoint
final_data = df[df['timepoint'] == 120]
for group1 in ['Control', 'Low_dose']:
    for group2 in ['Low_dose', 'High_dose']:
        if group1 != group2:
            d = cohens_d(
                final_data[final_data['group'] == group1]['value'],
                final_data[final_data['group'] == group2]['value']
            )
            print(f"Effect size {group1} vs {group2}: {d:.2f}")
""",
            "expected_outcomes": {
                "visualizations": [
                    "Growth curves for each treatment",
                    "Parameter comparison plots",
                    "Effect size comparisons"
                ],
                "statistical_results": [
                    "Significant group differences (p < 0.05)",
                    "Dose-dependent effects",
                    "Time-dependent changes in group differences"
                ],
                "interpretation": [
                    "Clear dose-response relationship",
                    "Different growth rates between groups",
                    "Practical significance of treatment effects"
                ]
            }
        }
    }

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
                    ]
                },
                {
                    "id": "sample_size",
                    "text": "What is your sample size?",
                    "type": "number",
                    "follow_up": {
                        "small": "Consider non-parametric methods or exact tests",
                        "medium": "Check normality assumptions carefully",
                        "large": "Consider practical significance in addition to statistical significance"
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

# Main app layout
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 PhD Data Analysis Blueprint</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation and settings
    with st.sidebar:
        st.header("Navigation & Settings")
        
        # Phase selection
        selected_phase = st.selectbox(
            "Select Analysis Phase:",
            options=list(PHASES.keys()),
            index=list(PHASES.keys()).index(st.session_state.current_phase)
        )
        st.session_state.current_phase = selected_phase
        
        # Data type selection
        data_types = ["General", "Temporal (Race Measurements)", "Spatiotemporal (Sports Motion)", "Fluorescence (Biological Timelapse)"]
        selected_data_type = st.selectbox(
            "Your Data Type:",
            options=data_types,
            index=data_types.index(st.session_state.data_type)
        )
        st.session_state.data_type = selected_data_type
        
        st.divider()
        
        # Analysis log
        st.subheader("Analysis Progress")
        if st.session_state.analysis_log:
            for entry in st.session_state.analysis_log[-5:]:  # Show last 5 entries
                st.write(f"✓ {entry}")
        else:
            st.write("No progress logged yet")
        
        if st.button("Clear Progress Log"):
            st.session_state.analysis_log = []
            st.rerun()
    
    # Main content area
    if selected_phase == "Overview":
        # Overview page with flowchart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Interactive Analysis Flow")
            fig = create_phase_flowchart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Getting Started")
            st.write("""
            This interactive blueprint guides you through rigorous data analysis for your PhD research.
            
            **How to use:**
            1. Select your data type in the sidebar
            2. Choose the relevant analysis phase
            3. Ask specific questions for guidance
            4. Track your progress as you work
            
            **Key Principles:**
            - Always document your process
            - Question assumptions systematically  
            - Validate results at each stage
            - Consider limitations explicitly
            """)
            
            if st.button("Start Analysis Journey"):
                st.session_state.current_phase = "Phase 1"
                st.rerun()
    
    elif selected_phase == "Phase 3":
        st.markdown("""
        # Statistical Characterization
        
        This phase helps you develop a comprehensive statistical analysis plan based on your data characteristics.
        """)
        
        # Get the workflow structure
        workflow = get_statistical_analysis_workflow()
        
        # Data Understanding Section
        st.markdown("## 1. Data Understanding")
        st.markdown(workflow["Data Understanding"]["description"])
        
        # Interactive questionnaire
        for question in workflow["Data Understanding"]["questions"]:
            st.markdown(f"### {question['text']}")
            
            if question.get("type") == "number":
                value = st.number_input("Enter your sample size", min_value=1, value=100)
                if value < 30:
                    st.info(question["follow_up"]["small"])
                elif value < 100:
                    st.info(question["follow_up"]["medium"])
                else:
                    st.info(question["follow_up"]["large"])
            else:
                selected = st.selectbox("Select an option", question["options"])
                
                if selected in question.get("follow_up", {}):
                    st.markdown("#### Follow-up questions:")
                    for follow_up in question["follow_up"][selected]:
                        st.text_input(follow_up)
        
        # Distributional Analysis Section
        st.markdown("## 2. Distributional Analysis")
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
        
        # Statistical Test Selection
        st.markdown("## 3. Statistical Test Selection")
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
        
        # Analysis Plan Generation
        st.markdown("## 4. Analysis Plan Generation")
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
    
    else:
        # Phase-specific content
        phase_info = PHASES[selected_phase]
        
        # Phase header
        st.markdown(f"""
        <div class="phase-card">
            <div class="phase-title">{phase_info['title']}</div>
            <div class="phase-description">{phase_info['description']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Best Practices Section with interactive elements
        st.subheader("Best Practices")
        for i, practice in enumerate(phase_info["best_practices"]):
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown("✅")
            with col2:
                if st.button(practice, key=f"practice_{i}"):
                    st.info(get_llm_guidance(selected_phase, selected_data_type, "action", practice))
        
        # Interactive Decision Diagram
        st.subheader("Interactive Decision Flow")
        decision_fig = create_decision_diagram(selected_phase, selected_data_type)
        st.plotly_chart(decision_fig, use_container_width=True)
        
        # Add click event handling for the diagram
        if st.button("Get Guidance for Current Step"):
            st.info("Click on any node in the diagram above to get specific guidance for that step.")
        
        # Data type specific guidance
        if selected_data_type != "General" and selected_phase in DATA_TYPE_GUIDANCE.get(selected_data_type, {}).get("phase_specific", {}):
            st.markdown(f"""
            <div class="warning-box">
                <strong>🎯 {selected_data_type} Specific Guidance:</strong><br>
                {DATA_TYPE_GUIDANCE[selected_data_type]["phase_specific"][selected_phase]}
            </div>
            """, unsafe_allow_html=True)
        
        # Key questions for this phase
        st.subheader("Common Questions for This Phase")
        key_questions = phase_info["key_questions"]
        
        # Display questions as clickable buttons
        cols = st.columns(2)
        for i, question in enumerate(key_questions):
            with cols[i % 2]:
                if st.button(question, key=f"q_{i}"):
                    # Add to conversation
                    response = simulate_ai_response(question, selected_phase, selected_data_type)
                    st.session_state.conversation_history.append({
                        "question": question,
                        "response": response,
                        "phase": selected_phase,
                        "data_type": selected_data_type,
                        "timestamp": datetime.now()
                    })
                    # Log progress
                    st.session_state.analysis_log.append(f"{selected_phase}: Asked about {question[:50]}...")
        
        st.divider()
        
        # Custom question input
        st.subheader("Ask Your Own Question")
        custom_question = st.text_area(
            "What specific guidance do you need for this phase?",
            placeholder="e.g., How should I handle non-normal distributions in my fluorescence data?",
            height=100
        )
        
        if st.button("Get AI Guidance", type="primary"):
            if custom_question.strip():
                response = simulate_ai_response(custom_question, selected_phase, selected_data_type)
                st.session_state.conversation_history.append({
                    "question": custom_question,
                    "response": response,
                    "phase": selected_phase,
                    "data_type": selected_data_type,
                    "timestamp": datetime.now()
                })
                st.session_state.analysis_log.append(f"{selected_phase}: Custom question about {custom_question[:30]}...")
                st.rerun()
        
        # Display conversation history for current phase
        st.subheader("Your Questions & Guidance")
        phase_conversations = [conv for conv in st.session_state.conversation_history 
                             if conv["phase"] == selected_phase]
        
        if phase_conversations:
            for i, conv in enumerate(reversed(phase_conversations[-3:])):  # Show last 3
                with st.expander(f"Q: {conv['question'][:80]}...", expanded=(i==0)):
                    st.markdown(f"""
                    <div class="question-box">
                        <strong>Question:</strong> {conv['question']}<br>
                        <small>Phase: {conv['phase']} | Data Type: {conv['data_type']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="ai-response">
                        <strong>AI Guidance:</strong><br>
                        {conv['response']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No questions asked for this phase yet. Try clicking on one of the common questions above or ask your own!")
    
    # Footer with export options
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Analysis Log"):
            if st.session_state.conversation_history:
                # Create export data
                export_data = {
                    "analysis_log": st.session_state.analysis_log,
                    "conversations": st.session_state.conversation_history,
                    "data_type": st.session_state.data_type,
                    "export_timestamp": datetime.now().isoformat()
                }
                st.download_button(
                    "Download JSON",
                    data=json.dumps(export_data, indent=2, default=str),
                    file_name=f"analysis_blueprint_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
    
    with col2:
        if st.button("🔄 Reset Session"):
            for key in ['conversation_history', 'analysis_log']:
                if key in st.session_state:
                    st.session_state[key] = []
            st.rerun()
    
    with col3:
        st.write(f"**Session Summary:** {len(st.session_state.conversation_history)} questions asked")

if __name__ == "__main__":
    main()