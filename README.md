## 🎯 DataFrameExpectations

**DataFrameExpectations** is a Python library designed to validate **Pandas** and **PySpark** DataFrames using customizable, reusable expectations. It simplifies testing in data pipelines and end-to-end workflows by providing a standardized framework for DataFrame validation.

Instead of using different validation approaches for DataFrames, this library provides a
standardized solution for this use case. As a result, any contributions made here—such as adding new expectations—can be leveraged by all users of the library.
You can find the complete list of expectations [here](docs/build/html/expectations.html).


### Installation:
```bash
pip install dataframe-expectations
```

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

**Pandas example:**
```python
from dataframe_expectations.expectations_suite import DataFameExpectationsSuite

suite = (
    DataFrameExpectationsSuite()
    .expect_value_greater_than("age", 18)
    .expect_value_less_than("age", 10)
)

# Create a Pandas DataFrame
import pandas as pd
test_pandas_df = pd.DataFrame({"age": [20, 15, 30], "name": ["Alice", "Bob", "Charlie"]})

suite.run(test_pandas_df)

```


**PySpark example:**
```python
from dataframe_expectations.expectations_suite import DataFrameExpectationsSuite

suite = (
    DataFrameExpectationsSuite()
    .expect_value_greater_than("age", 18)
    .expect_value_less_than("age", 40)
)

# Create a PySpark DataFrame
test_spark_df = spark.createDataFrame(
    [
        {"name": "Alice", "age": 20},
        {"name": "Bob", "age": 15},
        {"name": "Charlie", "age": 30},
    ]
)

suite.run(test_spark_df)

```

**Output:**
```python
========================== Running expectations suite ==========================
ExpectationValueGreaterThan ('age' greater than 18) ... FAIL
ExpectationValueLessThan ('age' less than 40) ... OK
============================ 1 success, 1 failures =============================

ExpectationSuiteFailure: (1/2) expectations failed.

================================================================================
List of violations:
--------------------------------------------------------------------------------
[Failed 1/1] ExpectationValueGreaterThan ('age' greater than 18): Found 1 row(s) where 'age' is not greater than 18.
Some examples of violations:
+-----+------+
| age | name |
+-----+------+
| 15  | Bob  |
+-----+------+
================================================================================

```

### How to contribute?
Contributions are welcome! You can enhance the library by adding new expectations, refining existing ones, or improving the testing framework.

### Versioning

This project follows [Semantic Versioning](https://semver.org/) (SemVer):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backward-compatible functionality additions
- **PATCH** version for backward-compatible bug fixes

To update the version, manually edit the version field in `pyproject.toml`:

```toml
[project]
version = "0.2.0"  # Update this line
```

Alternatively, you can use uv to bump version number:

```bash
# Bump patch version (0.1.0 -> 0.1.1)
uv version --bump patch

# Bump minor version (0.1.0 -> 0.2.0)
uv version --bump minor

# Bump major version (0.1.0 -> 1.0.0)
uv version --bump major

# Set specific version
uv version 0.2.0
```

Don't forget to update the [CHANGELOG.md](CHANGELOG.md) file with a detailed description of the changes you've introduced.

### Security
For security issues please contact security@getyourguide.com.

### Legal
dataframe-expectations is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE.txt) for the full text.
