# Data Assessment Blueprint

A comprehensive tool for planning and executing statistical data analysis with a focus on proper methodology and validation.

## Overview

The Data Assessment Blueprint is designed to help researchers and data analysts create robust statistical analysis plans. It provides an interactive interface for:

- Understanding data structure and characteristics
- Performing distributional analysis
- Selecting appropriate statistical tests
- Generating comprehensive analysis plans
- Validating data formats and requirements

## Framework Structure

### 1. User Interface (UI) Layer
- **Dashboard/Welcome Screen**
  - App purpose explanation
  - New analysis/load session options
  - Statistical glossary and learning resources
- **Data Upload/Input Module**
  - Support for CSV, Excel, JSON formats
  - Data preview and validation
  - Basic data cleaning tools
- **Interactive Q&A Module**
  - LLM-driven question flow
  - Multiple choice and open-ended inputs
  - Dynamic visualizations
- **Statistical Pathway Visualization**
  - Progress tracking
  - Decision tree visualization
- **Results & Recommendations**
  - Test recommendations with explanations
  - Code snippets (Python/R)
  - Interpretation guidance

### 2. Core Logic & Statistical Engine
- **Data Preprocessing & Validation**
  - Quality checks
  - Missing value handling
  - Outlier detection
- **Statistical Decision Engine**
  - Pre-defined pathways
  - LLM-driven branching
  - Assumption checking
- **Code Generation**
  - Python/R implementation
  - Documentation and examples

### 3. LLM Integration
- **Prompt Engineering**
  - System and user prompts
  - Context management
  - Response parsing
- **Error Handling**
  - Fallback mechanisms
  - Clarification requests
  - Session management

### 4. Knowledge Base
- **Statistical Concepts**
  - Term definitions
  - Test assumptions
  - Best practices
- **Code Repository**
  - Implementation examples
  - Common pitfalls
  - External resources

## Features

### Phase 3: Statistical Characterization
- Interactive data structure assessment
- Distributional analysis tools
- Statistical test selection guidance
- Analysis plan generation
- Data format validation
- Color-blind friendly interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AroundInteger/data_assessment_blueprint.git
cd data_assessment_blueprint
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the Phase 3 application:

```bash
streamlit run data_assessment_blueprint/app/app_p3.py
```

The application will be available at:
- Local URL: http://localhost:8501
- Network URL: http://192.168.4.24:8501

## Data Format Requirements

The application supports various data structures:

### Repeated Measures
- Required columns: `subject_id`, `time_point`, `measurement`
- Each subject should have measurements at all time points

### Independent Groups
- Required columns: `group`, `measurement`
- Each group should have multiple measurements

### Hierarchical Data
- Required columns: `level1_id`, `level2_id`, `subject_id`, `measurement`
- Supports nested data structures (e.g., students within classes within schools)

### Time Series
- Required columns: `timestamp`, `measurement`
- Data should be chronologically ordered

## Planned Enhancements

### UI Improvements
- [ ] Interactive decision tree visualization
- [ ] Session management and export
- [ ] Enhanced data preview and cleaning tools
- [ ] Statistical glossary integration

### LLM Integration
- [ ] Dynamic question generation
- [ ] Context-aware recommendations
- [ ] Code snippet generation
- [ ] Error handling and fallbacks

### Knowledge Base
- [ ] Comprehensive statistical concepts database
- [ ] Code repository expansion
- [ ] External resource links
- [ ] Best practices documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web application framework
- Plotly for interactive visualizations
- Pandas and NumPy for data manipulation
- SciPy for statistical functions 