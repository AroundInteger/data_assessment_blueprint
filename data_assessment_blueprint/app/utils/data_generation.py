import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_data_examples():
    """Generate example datasets for different data structures and distributions."""
    examples = {
        "Repeated Measures": {
            "description": "Athlete performance data across training sessions",
            "examples": {
                "normal": {
                    "data": pd.DataFrame({
                        'athlete_id': np.repeat(range(1, 11), 5),
                        'session': np.tile(range(1, 6), 10),
                        'performance': np.random.normal(100, 10, 50)
                    })
                },
                "skewed": {
                    "data": pd.DataFrame({
                        'athlete_id': np.repeat(range(1, 11), 5),
                        'session': np.tile(range(1, 6), 10),
                        'performance': np.random.lognormal(4.5, 0.3, 50)
                    })
                }
            }
        },
        "Independent Groups": {
            "description": "Plant growth data for different treatment groups",
            "examples": {
                "normal": {
                    "data": pd.DataFrame({
                        'treatment': np.repeat(['A', 'B', 'C'], 10),
                        'growth': np.random.normal(50, 5, 30)
                    })
                },
                "skewed": {
                    "data": pd.DataFrame({
                        'treatment': np.repeat(['A', 'B', 'C'], 10),
                        'growth': np.random.lognormal(3.8, 0.4, 30)
                    })
                }
            }
        },
        "Hierarchical": {
            "description": "Student scores across schools and classes",
            "examples": {
                "normal": {
                    "data": pd.DataFrame({
                        'school': np.repeat(['School A', 'School B'], 15),
                        'class': np.repeat(['Class 1', 'Class 2', 'Class 3'], 10),
                        'score': np.random.normal(75, 8, 30)
                    })
                },
                "skewed": {
                    "data": pd.DataFrame({
                        'school': np.repeat(['School A', 'School B'], 15),
                        'class': np.repeat(['Class 1', 'Class 2', 'Class 3'], 10),
                        'score': np.random.lognormal(4.3, 0.2, 30)
                    })
                }
            }
        },
        "Time Series": {
            "description": "Daily temperature measurements",
            "examples": {
                "normal": {
                    "data": pd.DataFrame({
                        'date': pd.date_range(start='2024-01-01', periods=30),
                        'temperature': np.random.normal(20, 5, 30)
                    })
                },
                "skewed": {
                    "data": pd.DataFrame({
                        'date': pd.date_range(start='2024-01-01', periods=30),
                        'temperature': np.random.lognormal(3.0, 0.3, 30)
                    })
                }
            }
        }
    }
    return examples 