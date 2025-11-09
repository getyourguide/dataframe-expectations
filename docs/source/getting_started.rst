Getting Started
===============

Welcome to DataFrame Expectations! This guide will help you get up and running quickly with validating your Pandas and PySpark DataFrames.

Installation
------------

Install DataFrame Expectations using pip:

.. code-block:: bash

   pip install dataframe-expectations

Requirements
~~~~~~~~~~~~

* Python 3.10+
* pandas >= 1.5.0
* pydantic >= 2.12.4
* pyspark >= 3.3.0
* tabulate >= 0.8.9

Basic Usage
-----------

DataFrame Expectations provides a fluent API for building validation suites. Here's how to get started:

Basic Usage with Pandas
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from dataframe_expectations.expectations_suite import DataFrameExpectationsSuite

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


PySpark Example
~~~~~~~~~~~~~~~

.. code-block:: python

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

Decorator Pattern for Automatic Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

Example Output
~~~~~~~~~~~~~~

When validations fail, you'll see detailed output like this:

.. code-block:: text

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

How to contribute?
------------------
Contributions are welcome! You can enhance the library by adding new expectations, refining existing ones, or improving
the testing framework or the documentation.
