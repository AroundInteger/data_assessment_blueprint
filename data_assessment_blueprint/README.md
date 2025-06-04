# Data Assessment Blueprint

A Streamlit application for comprehensive statistical data analysis and assessment.

## Overview

This application provides a structured approach to statistical analysis, guiding users through:
1. Data Input and Understanding
2. Distributional Analysis
3. Statistical Test Selection
4. Analysis Plan Generation

## Features

### Data Input
- Upload your own data or use example datasets
- Support for various data structures:
  - Repeated measures
  - Independent groups
  - Hierarchical data
  - Time series

### Distributional Analysis
- Visual distribution assessment
  - Histogram with KDE
  - Q-Q plot
  - Box plot
- Statistical tests for normality
  - Shapiro-Wilk test
  - Anderson-Darling test
  - Skewness and kurtosis analysis

### Statistical Test Selection
- Automatic test recommendations based on:
  - Data structure
  - Distribution characteristics
  - Sample size
- Support for both parametric and non-parametric tests
- Detailed test information and assumptions

### Analysis Plan
- Comprehensive analysis plan generation
- Evidence-based decision making
- Alternative approaches and transformations
- Code templates for implementation
- Visualization recommendations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AroundInteger/data_assessment_blueprint.git
cd data_assessment_blueprint
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app/app_p3.py
```

## Project Structure

```
data_assessment_blueprint/
├── app/
│   ├── app_p3.py              # Main application file
│   ├── components/            # UI components
│   │   ├── welcome.py        # Welcome screen
│   │   └── data_input.py     # Data input handling
│   └── utils/
│       └── data_generation.py # Example data generation
├── Data/                      # Data storage
└── tests/                     # Test files
```

## Usage

1. **Data Input**
   - Upload your data or select an example dataset
   - Specify the data structure and measurement column

2. **Distributional Analysis**
   - Review visual distribution plots
   - Check normality test results
   - Understand distribution characteristics

3. **Statistical Test Selection**
   - Review recommended tests based on your data
   - Learn about test assumptions and requirements
   - Choose appropriate statistical methods

4. **Analysis Plan**
   - Generate a comprehensive analysis plan
   - Review alternative approaches
   - Get code templates for implementation

## Recent Updates

- Moved transformation options to Analysis Plan section
- Improved normality assessment logic
- Added detailed test information and assumptions
- Enhanced code templates with error handling
- Fixed variable scope issues in transformation preview

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 