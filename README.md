## 🎯 DataFrameExpectations

![CI](https://github.com/getyourguide/dataframe-expectations/workflows/CI/badge.svg)
![Publish to PyPI](https://github.com/getyourguide/dataframe-expectations/workflows/Publish%20to%20PyPI/badge.svg)
[![PyPI version](https://badge.fury.io/py/dataframe-expectations.svg)](https://badge.fury.io/py/dataframe-expectations)
![PyPI downloads](https://img.shields.io/pypi/dm/dataframe-expectations)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-available-blue.svg)](https://code.getyourguide.com/dataframe-expectations/)

**DataFrameExpectations** is a Python library designed to validate **Pandas** and **PySpark** DataFrames using customizable, reusable expectations. It simplifies testing in data pipelines and end-to-end workflows by providing a standardized framework for DataFrame validation.

Instead of using different validation approaches for DataFrames, this library provides a
standardized solution for this use case. As a result, any contributions made here—such as adding new expectations—can be leveraged by all users of the library.

📚 **[View Documentation](https://code.getyourguide.com/dataframe-expectations/)** | 📋 **[List of Expectations](https://code.getyourguide.com/dataframe-expectations/expectations.html)**


### Installation:
```bash
pip install dataframe-expectations
```

### Requirements

* Python 3.10+
* pandas >= 1.5.0
* pydantic >= 2.12.4
* pyspark >= 3.3.0
* tabulate >= 0.8.9

### Quick Start

#### Pandas Example
```python
from dataframe_expectations.suite import DataFrameExpectationsSuite
import pandas as pd

# Build a suite with expectations
suite = (
    DataFrameExpectationsSuite()
    .expect_min_rows(min_rows=3)
    .expect_max_rows(max_rows=10)
    .expect_value_greater_than(column_name="age", value=18)
    .expect_value_less_than(column_name="salary", value=100000)
    .expect_value_not_null(column_name="name")
)

# Create a runner
runner = suite.build()

# Validate a DataFrame
df = pd.DataFrame({
    "age": [25, 15, 45, 22],
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "salary": [50000, 60000, 80000, 45000]
})
runner.run(df)
```

#### PySpark Example
```python
from dataframe_expectations.suite import DataFrameExpectationsSuite
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("example").getOrCreate()

# Build a validation suite (same API as Pandas!)
suite = (
    DataFrameExpectationsSuite()
    .expect_min_rows(min_rows=3)
    .expect_max_rows(max_rows=10)
    .expect_value_greater_than(column_name="age", value=18)
    .expect_value_less_than(column_name="salary", value=100000)
    .expect_value_not_null(column_name="name")
)

# Build the runner
runner = suite.build()

# Create a PySpark DataFrame
data = [
    {"age": 25, "name": "Alice", "salary": 50000},
    {"age": 15, "name": "Bob", "salary": 60000},
    {"age": 45, "name": "Charlie", "salary": 80000},
    {"age": 22, "name": "Diana", "salary": 45000}
]
df = spark.createDataFrame(data)

# Validate
runner.run(df)
```

### Validation Patterns

#### Manual Validation
Use `runner.run()` to explicitly validate DataFrames:

```python
# Run validation and raise exception on failure
runner.run(df)

# Run validation without raising exception
result = runner.run(df, raise_on_failure=False)
```

#### Decorator-Based Validation
Automatically validate function return values using decorators:

```python
from dataframe_expectations.suite import DataFrameExpectationsSuite
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("example").getOrCreate()

suite = (
    DataFrameExpectationsSuite()
    .expect_min_rows(min_rows=3)
    .expect_max_rows(max_rows=10)
    .expect_value_greater_than(column_name="age", value=18)
    .expect_value_less_than(column_name="salary", value=100000)
    .expect_value_not_null(column_name="name")
)

# Build the runner
runner = suite.build()

# Apply decorator to automatically validate function output
@runner.validate
def load_employee_data():
    """Load and return employee data - automatically validated."""
    return spark.createDataFrame(
        [
            {"age": 25, "name": "Alice", "salary": 50000},
            {"age": 15, "name": "Bob", "salary": 60000},
            {"age": 45, "name": "Charlie", "salary": 80000},
            {"age": 22, "name": "Diana", "salary": 45000}
        ]
    )

# Function execution automatically validates the returned DataFrame
df = load_employee_data()  # Raises DataFrameExpectationsSuiteFailure if validation fails

# Allow functions that may return None
@runner.validate(allow_none=True)
def conditional_load(should_load: bool):
    """Conditionally load data - validation only runs when DataFrame is returned."""
    if should_load:
        return spark.createDataFrame([{"age": 25, "name": "Alice", "salary": 50000}])
    return None  # No validation when None is returned
```

##### Validation Output
When validation runs, you'll see output like this:

```python
========================== Running expectations suite ==========================
ExpectationMinRows (DataFrame contains at least 3 rows) ... OK
ExpectationMaxRows (DataFrame contains at most 10 rows) ... OK
ExpectationValueGreaterThan ('age' is greater than 18) ... FAIL
ExpectationValueLessThan ('salary' is less than 100000) ... OK
ExpectationValueNotNull ('name' is not null) ... OK
============================ 4 success, 1 failures =============================

ExpectationSuiteFailure: (1/5) expectations failed.

================================================================================
List of violations:
--------------------------------------------------------------------------------
[Failed 1/1] ExpectationValueGreaterThan ('age' is greater than 18): Found 1 row(s) where 'age' is not greater than 18.
Some examples of violations:
+-----+------+--------+
| age | name | salary |
+-----+------+--------+
| 15  | Bob  | 60000  |
+-----+------+--------+
================================================================================
```

#### Programmatic Result Inspection
Get detailed validation results without raising exceptions:

```python
# Get detailed results without raising exceptions
result = runner.run(df, raise_on_failure=False)

# Inspect validation outcomes
print(f"Total: {result.total_expectations}, Passed: {result.total_passed}, Failed: {result.total_failed}")
print(f"Pass rate: {result.pass_rate:.2%}")
print(f"Duration: {result.total_duration_seconds:.2f}s")
print(f"Applied filters: {result.applied_filters}")

# Access individual results
for exp_result in result.results:
    if exp_result.status == "failed":
        print(f"Failed: {exp_result.description} - {exp_result.violation_count} violations")
```

### Advanced Features

#### Tag-Based Filtering
Filter which expectations to run using tags:

```python
from dataframe_expectations import DataFrameExpectationsSuite, TagMatchMode

# Tag expectations with priorities and environments
suite = (
    DataFrameExpectationsSuite()
    .expect_value_greater_than(column_name="age", value=18, tags=["priority:high", "env:prod"])
    .expect_value_not_null(column_name="name", tags=["priority:high"])
    .expect_min_rows(min_rows=1, tags=["priority:low", "env:test"])
)

# Run only high-priority checks (OR logic - matches ANY tag)
runner = suite.build(tags=["priority:high"], tag_match_mode=TagMatchMode.ANY)
runner.run(df)

# Run production-critical checks (AND logic - matches ALL tags)
runner = suite.build(tags=["priority:high", "env:prod"], tag_match_mode=TagMatchMode.ALL)
runner.run(df)
```

## Development Setup

To set up the development environment:

```bash
# 1. Fork and clone the repository
git clone https://github.com/getyourguide/dataframe-expectations.git
cd dataframe-expectations

# 2. Install UV package manager
pip install uv

# 3. Install development dependencies (this will automatically create a virtual environment)
uv sync --group dev

# 4. Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 5. Verify your setup
uv run pytest tests/ -n auto --cov=dataframe_expectations

# 6. Install pre-commit hooks
pre-commit install
# This will automatically run checks before each commit
```

## Contributing

We welcome contributions! Whether you're adding new expectations, fixing bugs, or improving documentation, your help is appreciated.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup instructions
- How to add new expectations
- Code style guidelines
- Testing requirements
- Pull request process

## Security

For security vulnerabilities, please see our [Security Policy](SECURITY.md) or contact security@getyourguide.com.

### Legal
dataframe-expectations is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE.txt) for the full text.
