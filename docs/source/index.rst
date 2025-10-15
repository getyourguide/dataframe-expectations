DataFrame Expectations Documentation
====================================

**DataFrameExpectations** is a Python library designed to validate **Pandas** and **PySpark**
DataFrames using customizable, reusable expectations. It simplifies testing in data pipelines
and end-to-end workflows by providing a standardized framework for DataFrame validation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   expectations
   api_reference
   contributing

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install dataframe-expectations

Basic usage with Pandas:

.. code-block:: python

   from dataframe_expectations.expectations_suite import DataframeExpectationsSuite
   import pandas as pd

   # Create a suite of expectations
   suite = (
       DataframeExpectationsSuite()
       .expect_value_greater_than("age", 18)
       .expect_value_less_than("age", 65)
   )

   # Create a DataFrame to validate
   df = pd.DataFrame({"age": [25, 30, 45], "name": ["Alice", "Bob", "Charlie"]})

   # Run the validation
   suite.run(df)

Basic usage with PySpark:

.. code-block:: python

   from dataframe_expectations.expectations_suite import DataframeExpectationsSuite
   from pyspark.sql import SparkSession

   # Initialize Spark session
   spark = SparkSession.builder.appName("DataFrameExpectations").getOrCreate()

   # Create a suite of expectations
   suite = (
       DataframeExpectationsSuite()
       .expect_value_greater_than("age", 18)
       .expect_value_less_than("age", 65)
   )

   # Create a PySpark DataFrame to validate
   data = [{"age": 25, "name": "Alice"}, {"age": 30, "name": "Bob"}, {"age": 45, "name": "Charlie"}]
   df = spark.createDataFrame(data)

   # Run the validation
   suite.run(df)
