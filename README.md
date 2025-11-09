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

### Development setup

To set up the development environment:

```bash
# 1. Clone the repository
git clone https://github.com/getyourguide/dataframe-expectations.git
cd dataframe-expectations

# 2. Install UV package manager
pip install uv

# 3. Install development dependencies (this will automatically create a virtual environment)
uv sync --group dev

# 4. (Optional) To explicitly activate the virtual environment:
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 5. Run tests (this will run the tests in the virtual environment)
uv run pytest tests/ --cov=dataframe_expectations
```

### Using the library

**Basic usage with Pandas:**
```python
from dataframe_expectations.expectations_suite import DataFrameExpectationsSuite
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

**PySpark example:**
```python
from dataframe_expectations.expectations_suite import DataFrameExpectationsSuite
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

**Decorator pattern for automatic validation:**
```python
from dataframe_expectations.expectations_suite import DataFrameExpectationsSuite
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

**Output:**
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

### How to contribute?
Contributions are welcome! You can enhance the library by adding new expectations, refining existing ones, or improving the testing framework.

### Versioning

This project follows [Semantic Versioning](https://semver.org/) (SemVer) and uses [Release Please](https://github.com/googleapis/release-please) for automated version management.

Versions are automatically determined based on [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature → **MINOR** version bump (0.1.0 → 0.2.0)
- `fix:` - Bug fix → **PATCH** version bump (0.1.0 → 0.1.1)
- `feat!:` or `BREAKING CHANGE:` - Breaking change → **MAJOR** version bump (0.1.0 → 1.0.0)
- `chore:`, `docs:`, `style:`, `refactor:`, `test:`, `ci:` - No version bump

**Example commits:**
```bash
git commit -m "feat: add new expectation for null values"
git commit -m "fix: correct validation logic in expect_value_greater_than"
git commit -m "feat!: remove deprecated API methods"
```

When changes are pushed to the main branch, Release Please automatically:
1. Creates or updates a Release PR with version bump and changelog
2. When merged, creates a GitHub Release and publishes to PyPI

No manual version updates needed - just use conventional commit messages!

### Security
For security issues please contact security@getyourguide.com.

### Legal
dataframe-expectations is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE.txt) for the full text.
