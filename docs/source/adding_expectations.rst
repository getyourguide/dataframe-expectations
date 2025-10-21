Adding Your Expectations
========================

This guide will walk you through the process of creating custom expectations for DataFrame validation.
There are three main approaches depending on your use case.

Defining Your Expectations
--------------------------

Most use cases that involve validating a single column in the dataframe can be covered by the initialising the
``DataFrameColumnExpectation`` class with the correct parameters. Expectations implemented by initialising
``DataFrameColumnExpectation`` can be found in the ``column_expectations`` module, categorised based on the data-type of
the column value.

If you want to go ahead with implementing ``DataFrameColumnExpectation``, you first need to identify the data-type of
the column value. Existing expectations are already categorised into ``string``, ``numerical`` or ``any_value``
expectations. Create a new category in column_expectations if you think existing categories don't fit your use case.
Once you have decided where the expectation needs to be added, you can define it as follows:

.. code-block:: python

    from dataframe_expectations.expectations.expectation_registry import (
        register_expectation,
    )
    from dataframe_expectations.expectations.utils import requires_params


    @register_expectation("ExpectIsDivisible")
    @requires_params("column_name", "value", types={"column_name": str, "value": int})
    def create_expectation_do_something_unexpected(**kwargs) -> DataFrameColumnExpectation:
        column_name = kwargs["column_name"]
        value = kwargs["value"]

        return DataFrameColumnExpectation(
            expectation_name="ExpectIsDivisible",
            column_name=column_name,
            fn_violations_pandas=lambda df: df[df[column_name] % value != 0], # function that finds violations
            fn_violations_pyspark=lambda df: df.filter(F.col(column_name) % value != 0), # function that finds violations
            description=f"'{column_name}' divisible by {value}",
            error_message=f"'{column_name}' not divisible by {value}.",
        )

For additional guidance, you can refer to the implementation of ``ExpectationValueGreaterThan`` and
``ExpectationValueLessThan`` in ``column_expectation_factory.py``. These examples demonstrate how to initialise the
``DataFrameColumnExpectation`` class with the right parameters and define filtering logic for different dataframes.
The ``@register_expectation`` decorator is needed to add your expectation to the library. ``@requires_params`` decorator
is a utility that helps you validate the input parameters.

Adding Aggregation-Based Expectations
--------------------------------------

Just like the column expectations, you can find the aggregation-based expectations in the ``aggregation_expectations``
module. For expectations that require aggregation operations (such as row counts, distinct value counts, null
percentages, etc.), you should implement custom expectation classes by inheriting from
``DataFrameAggregationExpectation``. These types of expectations cannot be easily covered
by the ``DataFrameColumnExpectation`` class because they involve DataFrame-level or column-level aggregations rather
than row-by-row validations.

Existing expectations are already categorised into ``string``, ``numerical`` or ``any_value``
expectations. Before you implement your aggregation-based expectation, infer the category of the aggregation operation
and add it to the right category. Feel free to create a new category if needed.

Here's an example of how to implement an aggregation-based expectation:

.. code-block:: python

    from dataframe_expectations import DataFrameLike, DataFrameType
    from dataframe_expectations.expectations.aggregation_expectation import (
        DataFrameAggregationExpectation,
    )
    from dataframe_expectations.expectations.expectation_registry import register_expectation
    from dataframe_expectations.expectations.utils import requires_params
    from dataframe_expectations.result_message import (
        DataFrameExpectationFailureMessage,
        DataFrameExpectationResultMessage,
        DataFrameExpectationSuccessMessage,
    )
    import pandas as pd
    from pyspark.sql import functions as F


    class ExpectationMinRows(DataFrameAggregationExpectation):
        """
        Expectation that validates the DataFrame has at least a minimum number of rows.
        """

        def __init__(self, min_count: int):
            description = f"DataFrame has at least {min_count} row(s)"
            self.min_count = min_count

            super().__init__(
                expectation_name="ExpectationMinRows",
                column_names=[],  # Empty list since this operates on entire DataFrame
                description=description,
            )

        def aggregate_and_validate_pandas(
            self, data_frame: DataFrameLike, **kwargs
        ) -> DataFrameExpectationResultMessage:
            """Validate minimum row count in a pandas DataFrame."""
            # Note: Parent class already checks if the column is present when column_names is not empty
            try:
                row_count = len(data_frame)

                if row_count >= self.min_count:
                    return DataFrameExpectationSuccessMessage(
                        expectation_name=self.get_expectation_name()
                    )
                else:
                    return DataFrameExpectationFailureMessage(
                        expectation_str=str(self),
                        data_frame_type=DataFrameType.PANDAS,
                        message=f"DataFrame has {row_count} row(s), expected at least {self.min_count}.",
                    )
            except Exception as e:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Error counting rows: {str(e)}",
                )

        def aggregate_and_validate_pyspark(
            self, data_frame: DataFrameLike, **kwargs
        ) -> DataFrameExpectationResultMessage:
            """Validate minimum row count in a PySpark DataFrame."""
            # Note: Parent class already checks if the column is present when column_names is not empty
            try:
                row_count = data_frame.count()

                if row_count >= self.min_count:
                    return DataFrameExpectationSuccessMessage(
                        expectation_name=self.get_expectation_name()
                    )
                else:
                    return DataFrameExpectationFailureMessage(
                        expectation_str=str(self),
                        data_frame_type=DataFrameType.PYSPARK,
                        message=f"DataFrame has {row_count} row(s), expected at least {self.min_count}.",
                    )
            except Exception as e:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Error counting rows: {str(e)}",
                )


    @register_expectation("ExpectationMinRows")
    @requires_params("min_count", types={"min_count": int})
    def create_expectation_min_rows(**kwargs) -> ExpectationMinRows:
        """
        Create an ExpectationMinRows instance.

        Args:
            min_count (int): Minimum required number of rows.

        Returns:
            ExpectationMinRows: A configured expectation instance.
        """
        return ExpectationMinRows(min_count=kwargs["min_count"])

Key differences for aggregation-based expectations:

1. **Inherit from** ``DataFrameAggregationExpectation``: This base class provides the framework for aggregation operations and automatically handles column validation.

2. **Implement** ``aggregate_and_validate_pandas`` **and** ``aggregate_and_validate_pyspark``: These methods are specifically designed for aggregation operations rather than the generic ``validate_pandas`` and ``validate_pyspark`` methods.

3. **Call** ``super().__init__()``: Initialize the parent class with expectation metadata including ``expectation_name``, ``column_names``, and ``description``.

4. **Automatic column validation**: The parent class automatically validates that required columns exist before calling your implementation methods. You don't need to manually check for column existence.

5. **Error handling**: Wrap aggregation operations in try-catch blocks since aggregations can fail due to data type issues or other DataFrame problems.

Example of a column-based aggregation expectation:

.. code-block:: python

    class ExpectationColumnMeanBetween(DataFrameAggregationExpectation):
        """
        Expectation that validates the mean value of a column falls within a specified range.
        """

        def __init__(self, column_name: str, min_value: float, max_value: float):
            description = f"column '{column_name}' mean value between {min_value} and {max_value}"

            self.column_name = column_name
            self.min_value = min_value
            self.max_value = max_value

            super().__init__(
                expectation_name="ExpectationColumnMeanBetween",
                column_names=[column_name],  # List of columns this expectation requires
                description=description,
            )

        def aggregate_and_validate_pandas(
            self, data_frame: DataFrameLike, **kwargs
        ) -> DataFrameExpectationResultMessage:
            """Validate column mean in a pandas DataFrame."""
            # Column validation is automatically handled by the parent class
            try:
                mean_val = data_frame[self.column_name].mean()

                if pd.isna(mean_val):
                    return DataFrameExpectationFailureMessage(
                        expectation_str=str(self),
                        data_frame_type=DataFrameType.PANDAS,
                        message=f"Column '{self.column_name}' contains only null values.",
                    )

                if self.min_value <= mean_val <= self.max_value:
                    return DataFrameExpectationSuccessMessage(
                        expectation_name=self.get_expectation_name()
                    )
                else:
                    return DataFrameExpectationFailureMessage(
                        expectation_str=str(self),
                        data_frame_type=DataFrameType.PANDAS,
                        message=f"Column '{self.column_name}' mean value {mean_val} is not between {self.min_value} and {self.max_value}.",
                    )
            except Exception as e:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Error calculating mean for column '{self.column_name}': {str(e)}",
                )

        def aggregate_and_validate_pyspark(
            self, data_frame: DataFrameLike, **kwargs
        ) -> DataFrameExpectationResultMessage:
            """Validate column mean in a PySpark DataFrame."""
            # Column validation is automatically handled by the parent class
            try:
                mean_result = data_frame.select(F.avg(self.column_name).alias("mean_val")).collect()
                mean_val = mean_result[0]["mean_val"]

                if mean_val is None:
                    return DataFrameExpectationFailureMessage(
                        expectation_str=str(self),
                        data_frame_type=DataFrameType.PYSPARK,
                        message=f"Column '{self.column_name}' contains only null values.",
                    )

                if self.min_value <= mean_val <= self.max_value:
                    return DataFrameExpectationSuccessMessage(
                        expectation_name=self.get_expectation_name()
                    )
                else:
                    return DataFrameExpectationFailureMessage(
                        expectation_str=str(self),
                        data_frame_type=DataFrameType.PYSPARK,
                        message=f"Column '{self.column_name}' mean value {mean_val} is not between {self.min_value} and {self.max_value}.",
                    )
            except Exception as e:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Error calculating mean for column '{self.column_name}': {str(e)}",
                )

Key considerations for aggregation-based expectations:

1. **Performance**: Aggregation operations can be expensive, especially on large datasets in PySpark. Consider the performance implications of your aggregation logic.

2. **Different DataFrame types**: Ensure your implementation works correctly for both Pandas and PySpark DataFrames, as aggregation APIs may differ (e.g., ``df.mean()`` vs ``F.avg()``).

3. **Error handling**: Include proper error handling for edge cases like empty DataFrames or all-null columns.

4. **Message clarity**: Provide clear, informative error messages that help users understand what went wrong.

5. **Automatic column validation**: The ``DataFrameAggregationExpectation`` base class automatically validates that required columns exist before calling your ``aggregate_and_validate_*`` methods. Simply specify the required columns in the ``column_names`` parameter during initialization.

6. **Focus on aggregation logic**: Since column validation is handled automatically, you can focus purely on implementing your aggregation and validation logic without worrying about column existence checks.

Examples of aggregation-based expectations include:

- ``ExpectationMinRows`` / ``ExpectationMaxRows``: Validate row count limits
- ``ExpectationDistinctColumnValuesEquals``: Validate the number of distinct values in a column
- ``ExpectationMaxNullPercentage``: Validate the percentage of null values in a column
- ``ExpectationUniqueRows``: Validate that rows are unique across specified columns
- ``ExpectationColumnMeanBetween``: Validate that column mean falls within a range
- ``ExpectationColumnQuantileBetween``: Validate that column quantiles fall within ranges

For more examples, check the aggregation_expectations module.

Custom Expectations with Full Control
--------------------------------------

While the ``DataFrameColumnExpectation`` covers most use cases there might be other instances where you need more control
over the validation logic. For such instances you can define a new expectation by inheriting the ``DataFrameExpectation``
class.

To help you get started, here's a template you can customize to fit your specific use case:

.. code-block:: python

    from typing import Callable

    from dataframe_expectations import DataFrameLike, DataFrameType
    from dataframe_expectations.expectations import DataFrameExpectation
    from dataframe_expectations.result_message import (
        DataFrameExpectationFailureMessage,
        DataFrameExpectationResultMessage,
        DataFrameExpectationSuccessMessage,
    )

    class ExpectTheUnexpected(DataFrameExpectation):
        """
        Description of the expectation
        """

        def __init__(self, <add your initialization variables here>):
            """
            Initialize the expectation. For example:
            - column_name: The name of the column to validate.
            - value: The expected threshold for validation.
            """
            <initialise your class variables here>
            pass

        def get_description(self) -> str:
            """
            Returns a description of the expectation.
            """
            return <Add the description of your expectation. Include class variables if needed>

        def validate_pandas(
            self, data_frame: DataFrameLike, **kwargs
        ) -> DataFrameExpectationResultMessage:
            """
            Validate a pandas DataFrame against the expectation.
            """
            <Add your validation logic here for Pandas DataFrame. Return either DataFrameExpectationSuccessMessage or DataFrameExpectationFailureMessage>

        def validate_pyspark(
            self, data_frame: DataFrameLike, **kwargs
        ) -> DataFrameExpectationResultMessage:
            """
            Validate a PySpark DataFrame against the expectation.
            """
            <Add your validation logic here for PySpark DataFrame. Return either DataFrameExpectationSuccessMessage or DataFrameExpectationFailureMessage>

Adding to DataFrameExpectationsSuite
-------------------------------------

The ``DataFrameExpectationsSuite`` encapsulates all the expectations that are provided by this library.
After defining and testing your expectation, integrate it into the ``DataFrameExpectationsSuite`` by creating a new
method with a descriptive name starting with the prefix ``expect_`` (this is needed to generate documentation later).
Here's an example:

.. code-block:: python

    class DataFrameExpectationsSuite:
        """
        A suite of expectations for validating DataFrames.
        """
        ...

        def expect_is_divisible(
            self,
            column_name: str,
            value: float,
            # You can add more parmeters here
        ):
            """
            Define what the expectation does
            :param column_name: The name of the column to check.
            :param value: The value to compare against.
            :return: An instance of DataFrameExpectationsSuite.
            """

            expectation = DataFrameExpectationRegistry.get_expectation(
                expectation_name="ExpectIsDivisible",
                column_name=column_name,
                value=value,
            )

            logger.info(f"Adding expectation: {expectation}")
            self.__expectations.append(expectation)
            return self

Adding Unit Tests
-----------------

To ensure your expectations work as expected (pun intended), make sure to add unit tests in the
``tests/data_engineering/dataframe_expectations/expectations_implemented`` folder. Here's a template to get you started:

.. code-block:: python

    import pytest
    import pandas as pd

    from dataframe_expectations import DataFrameType
    from dataframe_expectations.expectations.expectation_registry import (
        DataFrameExpectationRegistry,
    )
    from dataframe_expectations.result_message import (
        DataFrameExpectationFailureMessage,
        DataFrameExpectationSuccessMessage,
    )


    def test_expectation_name():
        """
        Test that the expectation name is correctly returned.
        This method should be implemented in the subclass.
        """
        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationDoesSomeCheck",
            column_name="col1",
            value=5,
        )
        assert expectation.get_expectation_name() == "ExpectationDoesSomeCheck", f"Expected 'ExpectationDoesSomeCheck' but got: {expectation.get_expectation_name()}"

    def test_expectation_pandas_success():
        """
        Test the expectation for pandas DataFrame with no violations.
        This method should be implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def test_expectation_pandas_violations():
        """
        Test the expectation for pandas DataFrame with violations.
        This method should be implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def test_expectation_pyspark_success(spark):
        """
        Test the expectation for PySpark DataFrame with no violations.
        This method should be implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def test_expectation_pyspark_violations(spark):
        """
        Test the expectation for PySpark DataFrame with violations.
        This method should be implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def test_suite_pandas_success():
        """
        Test the expectation suite for pandas DataFrame with no violations.
        This method should be implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def test_suite_pandas_violations():
        """
        Test the expectation suite for pandas DataFrame with violations.
        This method should be implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def test_suite_pyspark_success(spark):
        """
        Test the expectation suite for PySpark DataFrame with no violations.
        This method should be implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def test_suite_pyspark_violations(spark):
        """
        Test the expectation suite for PySpark DataFrame with violations.
        This method should be implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

For concrete examples of unit tests, check for tests in the ``expectations_implemented`` folder. You can also
find the unit test template here.

Updating the Documentation
--------------------------

After the expectation is ready for use, the last thing remaining is adding your expectation to the documentation. The documentation is automatically generated using a CI pipeline with the ``uv`` package manager and is available at ``docs/build/html/expectations.html``.

Make sure to add the docstring for the function you added to ``DataFrameExpectationsSuite`` before submitting your changes. The CI pipeline will automatically update the documentation using the make targets in the ``docs`` folder when your changes are merged.

If you need to build the documentation locally for testing, you can use the make targets available in the ``docs`` folder.

.. code-block:: bash

    cd docs
    uv run sphinx-build source build/html

or use the make command

.. code-block:: bash

    cd docs
    make html
