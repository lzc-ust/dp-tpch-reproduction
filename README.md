# TPC-H User-Level Differential Privacy Reproduction Project

## Project Introduction

This project implements a user-level differential privacy protected query system based on the TPC-H dataset. It aims to research and compare the performance of different differential privacy mechanisms in real-world database query scenarios. The project primarily focuses on the privacy-preserving implementation of TPC-H Query 3 (Q3), protecting individual user privacy through carefully designed noise mechanisms while maintaining the validity and usability of query results.

## Features

- Implementation of three differential privacy mechanisms: Naive Laplace, R2T (Randomized Response with Truncation), and Shifted Inverse
- Comprehensive experimental evaluation framework supporting multiple evaluation metrics (Jaccard similarity, relative error, Kendall Tau)
- Data-driven parameter auto-selection system that optimizes differential privacy parameters based on query pattern characteristics
- Complete visualization report generation functionality to intuitively display performance comparisons between different mechanisms
- Support for market segment analysis, enabling differentiated privacy protection strategies for different customer groups

## Project Structure

```
dp-tpch-reproduction/
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
└── src/                 # Source code directory
    ├── core/            # Core differential privacy mechanism implementation
    │   ├── naive_laplace.py       # Naive Laplace mechanism
    │   ├── r2t.py                 # R2T mechanism
    │   ├── shifted_inverse.py     # Shifted Inverse mechanism
    │   └── data_loader.py         # Data loading and processing
    ├── evaluation/      # Evaluation module
    │   └── evaluator.py           # Performance evaluation tools
    ├── experiments/     # Experiment module
    │   ├── experiment_runner.py   # Experiment runner
    │   └── data_driven_experiment.py # Data-driven experiment
    ├── utils/           # Utility module
    │   ├── config.py              # Configuration file
    │   ├── data_analyzer.py       # Data analyzer
    │   └── data_driven_parameter_selector.py # Parameter selector
    ├── final_project_summary.py   # Project summary
    ├── final_comparison.py        # Mechanism comparison
    └── report_table_build.py      # Report generation
```

## Installation Dependencies

The project is implemented in Python and requires the following dependencies:

```bash
pip install -r requirements.txt
```

requirements.txt includes the following dependencies:
- pandas>=1.5.0
- numpy>=1.21.0
- scipy>=1.7.0
- sqlalchemy>=1.4.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- jupyter>=1.0.0

## Database Configuration

The project needs to connect to a MySQL database to obtain TPC-H data. Please configure the database connection information in the `src/utils/config.py` file:

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',      # Replace with your MySQL username
    'password': '123456', # Replace with your MySQL password
    'database': 'tpc_h'  # Replace with your database name
}
```

## Usage

### 1. Running Basic Experiments

```python
from src.experiments.experiment_runner import ExperimentRunner
from src.utils.config import get_db_connection_string

# Create experiment runner
runner = ExperimentRunner(get_db_connection_string())

# Run comprehensive experiment (multiple epsilon values)
epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
results = runner.run_comprehensive_experiment(epsilon_values)

# Run single experiment (specific epsilon value)
single_result = runner.run_single_experiment(epsilon=1.0)
```

### 2. Running Data-Driven Experiments

```python
from src.experiments.data_driven_experiment import DataDrivenExperiment
from src.utils.config import get_db_connection_string

# Create data-driven experiment
 experiment = DataDrivenExperiment(get_db_connection_string())

# Run data-driven analysis
experiment.run_data_driven_analysis()
```

### 3. Using Evaluation Tools

```python
from src.evaluation.evaluator import Evaluator

# Create evaluator
evaluator = Evaluator()

# Calculate evaluation metrics
jaccard_score = evaluator.calculate_jaccard_similarity(true_top10, dp_top10)
relative_error = evaluator.calculate_relative_error(true_total, dp_total)
kendall_tau = evaluator.calculate_kendall_tau(true_ranking, dp_ranking)
```

### 4. Generating Reports and Visualizations

```python
from src.report_table_build import create_all_visualizations

# Create all visualization reports
create_all_visualizations()
```

## Differential Privacy Mechanisms

### 1. Naive Laplace Mechanism

The most basic differential privacy mechanism that directly adds Laplace noise to the total revenue. Suitable for simple scenarios but sensitive to extreme values.

```python
from src.core.naive_laplace import NaiveLaplaceMechanism

# Create mechanism instance
mechanism = NaiveLaplaceMechanism(epsilon=1.0)

# Run mechanism
dp_result = mechanism.run_mechanism(customer_contributions)
```

### 2. R2T Mechanism (Randomized Response with Truncation)

An advanced mechanism that combines truncation technology with Laplace noise. It first uses the exponential mechanism to select the optimal truncation parameter T, then adds noise to the truncated data. Performs best in most cases.

```python
from src.core.r2t import R2TMechanism

# Create mechanism instance
mechanism = R2TMechanism(epsilon=1.0)

# Run mechanism
dp_result = mechanism.run_mechanism(customer_contributions)
```

### 3. Shifted Inverse Mechanism

A sampling mechanism based on customer contribution distribution, achieving differential privacy through probabilistic sampling and weighted aggregation. Particularly suitable for scenarios with uneven data distribution.

```python
from src.core.shifted_inverse import ShiftedInverseMechanism

# Create mechanism instance
mechanism = ShiftedInverseMechanism(epsilon=1.0)

# Run mechanism
dp_result = mechanism.run_mechanism(customer_contributions)
```

## Data-Driven Parameter Selection

The project implements a data feature-based parameter auto-selection system that can dynamically adjust differential privacy parameters based on characteristics such as query pattern complexity and data richness to achieve a better privacy-utility tradeoff.

```python
from src.utils.data_analyzer import DataAnalyzer
from src.utils.data_driven_parameter_selector import DataDrivenParameterSelector
from src.utils.config import get_db_connection_string

# Create data analyzer
analyzer = DataAnalyzer(get_db_connection_string())

# Create parameter selector
selector = DataDrivenParameterSelector(analyzer)
selector.initialize_global_analysis()

# Get recommended parameters
market_segment = 'BUILDING'
date = '1995-03-15'
r2t_parameters = selector.suggest_parameters_for_query(market_segment, date, 'R2T')
```

## Experimental Results Analysis

According to the experimental results, the performance ranking of the three mechanisms is:

1. **R2T Mechanism (After Improvement)**: Error reduced from 4.46 to 0.635, improved by 85.7%
2. **Naive Laplace Mechanism**: Stable performance but sensitive to extreme values
3. **Shifted Inverse Mechanism**: Limited improvement, error reduced from 0.94 to 0.937

Key findings:
- Parameter tuning has a significant impact on the R2T mechanism, which can greatly improve performance
- Different market segments require different privacy protection strategies
- Data complexity and richness are key factors affecting differential privacy performance

## Notes

1. Database connection information needs to be configured according to the actual environment
2. Experiments may take a long time to run, it is recommended to execute in a high-performance environment
3. The selection of privacy budget (epsilon value) needs to be balanced according to specific privacy requirements and data characteristics
4. For different market segments and query patterns, it is recommended to use the data-driven parameter selection system to obtain optimal parameters

## License

This project is for academic research and learning purposes only.

## Contact

For any questions or suggestions, please contact the project maintainers.