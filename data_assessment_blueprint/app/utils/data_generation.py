import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_data_examples():
    """Generate example datasets for different data structures and distributions."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    examples = {
        "Repeated Measures": {
            "description": "Athlete performance data across training sessions",
            "examples": {
                "normal": {
                    "data": pd.DataFrame({
                        'athlete_id': np.repeat(range(1, 41), 5),  # 40 athletes × 5 sessions = 200
                        'session': np.tile(range(1, 6), 40),  # 5 sessions × 40 athletes = 200
                        'performance': np.random.normal(100, 5, 200)  # 200 samples
                    }),
                    "measurement_col": "performance"  # Specify the column for statistical analysis
                },
                "skewed": {
                    "data": pd.DataFrame({
                        'athlete_id': np.repeat(range(1, 41), 5),  # 40 athletes × 5 sessions = 200
                        'session': np.tile(range(1, 6), 40),  # 5 sessions × 40 athletes = 200
                        'performance': np.random.lognormal(4.5, 0.3, 200)  # 200 samples
                    }),
                    "measurement_col": "performance"  # Specify the column for statistical analysis
                }
            }
        },
        "Independent Groups": {
            "description": "Plant growth data for different treatment groups",
            "examples": {
                "normal": {
                    "data": pd.DataFrame({
                        'treatment': np.repeat(['A', 'B', 'C'], [67, 67, 66]),  # 67 + 67 + 66 = 200 samples
                        'growth': np.random.normal(50, 3, 200)  # 200 samples
                    }),
                    "measurement_col": "growth"  # Specify the column for statistical analysis
                },
                "skewed": {
                    "data": pd.DataFrame({
                        'treatment': np.repeat(['A', 'B', 'C'], [67, 67, 66]),  # 67 + 67 + 66 = 200 samples
                        'growth': np.random.lognormal(3.8, 0.4, 200)  # 200 samples
                    }),
                    "measurement_col": "growth"  # Specify the column for statistical analysis
                }
            }
        },
        "Hierarchical": {
            "description": "Student scores across schools and classes",
            "examples": {
                "normal": {
                    "data": pd.DataFrame({
                        'school': np.repeat(['School A', 'School B'], 100),  # 2 schools × 100 students = 200
                        'class': np.tile(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], 40),  # 5 classes × 40 students = 200
                        'score': np.random.normal(75, 4, 200)  # 200 samples
                    }),
                    "measurement_col": "score"  # Specify the column for statistical analysis
                },
                "skewed": {
                    "data": pd.DataFrame({
                        'school': np.repeat(['School A', 'School B'], 100),  # 2 schools × 100 students = 200
                        'class': np.tile(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], 40),  # 5 classes × 40 students = 200
                        'score': np.random.lognormal(4.3, 0.2, 200)  # 200 samples
                    }),
                    "measurement_col": "score"  # Specify the column for statistical analysis
                }
            }
        },
        "Time Series": {
            "description": "Daily temperature measurements",
            "examples": {
                "normal": {
                    "data": pd.DataFrame({
                        'date': pd.date_range(start='2024-01-01', periods=200),  # 200 days
                        'temperature': np.random.normal(20, 2, 200)  # 200 samples
                    }),
                    "measurement_col": "temperature"  # Specify the column for statistical analysis
                },
                "skewed": {
                    "data": pd.DataFrame({
                        'date': pd.date_range(start='2024-01-01', periods=200),  # 200 days
                        'temperature': np.random.lognormal(3.0, 0.3, 200)  # 200 samples
                    }),
                    "measurement_col": "temperature"  # Specify the column for statistical analysis
                }
            }
        }
    }
    return examples 