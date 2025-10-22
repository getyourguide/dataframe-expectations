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
* pyspark >= 3.3.0
* tabulate >= 0.8.9

Basic Usage
-----------

DataFrame Expectations provides a fluent API for building validation suites. Here's how to get started:

Pandas Example
~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from dataframe_expectations.expectations_suite import DataFrameExpectationsSuite

    # Create a sample DataFrame
    df = pd.DataFrame({
         "age": [25, 15, 45, 22],
         "name": ["Alice", "Bob", "Charlie", "Diana"],
         "salary": [50000, 60000, 80000, 45000]
    })

    # Build a validation suite
    suite = (
         DataFrameExpectationsSuite()
         .expect_min_rows(3)  # At least 3 rows
         .expect_max_rows(10)  # At most 10 rows
         .expect_value_greater_than("age", 18)  # All ages > 18
         .expect_value_less_than("salary", 100000)  # All salaries < 100k
         .expect_value_not_null("name")  # No null names
    )

    # Run validation
     suite.run(df)


PySpark Example
~~~~~~~~~~~~~~~

.. code-block:: python

    from pyspark.sql import SparkSession
    from dataframe_expectations.expectations_suite import DataFrameExpectationsSuite

    # Initialize Spark
    spark = SparkSession.builder.appName("DataFrameExpectations").getOrCreate()

    # Create a sample DataFrame
    data = [
         {"age": 25, "name": "Alice", "salary": 50000},
         {"age": 15, "name": "Bob", "salary": 60000},
         {"age": 45, "name": "Charlie", "salary": 80000},
         {"age": 22, "name": "Diana", "salary": 45000}
    ]
    df = spark.createDataFrame(data)

    # Build a validation suite (same API as Pandas!)
    suite = (
         DataFrameExpectationsSuite()
         .expect_min_rows(3)
         .expect_max_rows(10)
         .expect_value_greater_than("age", 18)
         .expect_value_less_than("salary", 100000)
         .expect_value_not_null("name")
    )

    # Run validation
    suite.run(df)

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
