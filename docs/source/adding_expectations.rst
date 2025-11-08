Adding Your Expectations
========================

This guide will walk you through the process of creating custom expectations for DataFrame validation.
There are three main approaches depending on your use case.

Understanding Expectation Categories
-------------------------------------

All expectations must be registered with a category and subcategory to organize them properly. The available options are:

**Categories:**

- ``ExpectationCategory.COLUMN``: For expectations that validate individual column values row-by-row
- ``ExpectationCategory.AGGREGATION``: For expectations that require aggregation operations (counts, means, etc.)

**Subcategories:**

- ``ExpectationSubcategory.NUMERICAL``: For expectations dealing with numeric data
- ``ExpectationSubcategory.STRING``: For expectations dealing with string/text data
- ``ExpectationSubcategory.ANY_VALUE``: For expectations that work with any data type
- ``ExpectationSubcategory.UNIQUE``: For expectations related to uniqueness constraints

Choose the category and subcategory that best describes your expectation's purpose.

Defining Your Expectations
--------------------------

Most use cases that involve validating a single column in the dataframe can be covered by initialising the
``DataFrameColumnExpectation`` class with the correct parameters. Expectations implemented by initialising
``DataFrameColumnExpectation`` can be found in the ``column_expectations`` module, categorised based on the data-type of
the column value.

If you want to go ahead with implementing ``DataFrameColumnExpectation``, you first need to identify the data-type of
the column value. Existing expectations are already categorised into ``string``, ``numerical`` or ``any_value``
expectations. Create a new category in column_expectations if you think existing categories don't fit your use case.
Once you have decided where the expectation needs to be added, you can define it as follows:

.. code-block:: python

    from dataframe_expectations.expectations.column_expectation import DataFrameColumnExpectation
    from dataframe_expectations.expectations.expectation_registry import (
        ExpectationCategory,
        ExpectationSubcategory,
        register_expectation,
    )
    from dataframe_expectations.expectations.utils import requires_params
    from pyspark.sql import functions as F


    @register_expectation(
        expectation_name="ExpectIsDivisible",
        category=ExpectationCategory.COLUMN,
        subcategory=ExpectationSubcategory.NUMERICAL,
        pydoc="Expect values in column to be divisible by a specified value.",
        params=["column_name", "value"],
        params_doc={
            "column_name": "The name of the column to validate",
            "value": "The divisor value to check against"
        },
        param_types={"column_name": str, "value": int}
    )
    @requires_params("column_name", "value", types={"column_name": str, "value": int})
    def create_expectation_is_divisible(**kwargs) -> DataFrameColumnExpectation:
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
``ExpectationValueLessThan`` in the ``column_expectations`` module. These examples demonstrate how to initialise the
``DataFrameColumnExpectation`` class with the right parameters and define filtering logic for different dataframes.

The ``@register_expectation`` decorator is required and has the following mandatory parameters:

- ``expectation_name``: The class name of your expectation (e.g., "ExpectationIsDivisible")
- ``category``: Use ``ExpectationCategory.COLUMN`` or ``ExpectationCategory.AGGREGATION``
- ``subcategory``: Choose from ``ExpectationSubcategory.NUMERICAL``, ``ExpectationSubcategory.STRING``, or ``ExpectationSubcategory.ANY_VALUE``
- ``pydoc``: A brief description of what the expectation does
- ``params``: List of parameter names (e.g., ["column_name", "value"])
- ``params_doc``: Dictionary mapping parameter names to their descriptions
- ``param_types``: Dictionary mapping parameter names to their Python types

The ``@requires_params`` decorator is a utility that helps you validate the input parameters at runtime.

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
    from dataframe_expectations.expectations.expectation_registry import (
        ExpectationCategory,
        ExpectationSubcategory,
        register_expectation,
    )
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


    @register_expectation(
        expectation_name="ExpectationMinRows",
        category=ExpectationCategory.AGGREGATION,
        subcategory=ExpectationSubcategory.ANY_VALUE,
        pydoc="Expect DataFrame to have at least a minimum number of rows.",
        params=["min_count"],
        params_doc={"min_count": "Minimum required number of rows"},
        param_types={"min_count": int}
    )
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

Automatic Integration with DataFrameExpectationsSuite
-----------------------------------------------------

The ``DataFrameExpectationsSuite`` provides access to all registered expectations through dynamically generated methods.
When you register an expectation using the ``@register_expectation`` decorator, the suite automatically creates a
corresponding ``expect_*`` method. For example:

- ``ExpectationIsDivisible`` → ``suite.expect_is_divisible(column_name="...", value=...)``
- ``ExpectationMinRows`` → ``suite.expect_min_rows(min_count=...)``
- ``ExpectationValueGreaterThan`` → ``suite.expect_value_greater_than(column_name="...", value=...)``

The method names are automatically derived by:

1. Removing the "Expectation" prefix from your expectation name
2. Converting from PascalCase to snake_case
3. Adding the "expect\_" prefix

No manual integration is required! Simply register your expectation and it will be available in the suite.

Generating Type Stubs for IDE Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To provide IDE autocomplete and type hints for all expect methods, run the stub generator:

.. code-block:: bash

    uv run python scripts/generate_suite_stubs.py

This creates ``expectations_suite.pyi`` with type hints for all registered expectations. The stub file is automatically
validated by the sanity check script and pre-commit hooks.

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

Updating the Documentation and Stubs
------------------------------------

After implementing your expectation, you need to generate the type stubs and documentation:

**1. Generate Type Stubs**

Run the stub generator to create IDE autocomplete support:

.. code-block:: bash

    uv run python scripts/generate_suite_stubs.py

This updates ``dataframe_expectations/expectations_suite.pyi`` with type hints for your new expectation method.

**2. Build Documentation**

The documentation is automatically generated using the ``pydoc`` parameter you provided in ``@register_expectation``.
To build the documentation locally for testing:

.. code-block:: bash

    cd docs
    uv run sphinx-build source build/html

or use the make command:

.. code-block:: bash

    cd docs
    make html

The documentation will be available at ``docs/build/html/expectations.html``.

**3. Run Sanity Checks**

Before committing, run the sanity check to ensure everything is properly registered:

.. code-block:: bash

    uv run python scripts/sanity_checks.py

This validates that:

- Your expectation is registered in the registry
- The stub file is up-to-date
- Test files exist for your expectation
- Everything is consistent across the framework

**4. Pre-commit Hooks**

The pre-commit hooks will automatically check:

- Code formatting (black, isort)
- Type checking (mypy)
- Stub file is up-to-date
- Linting (ruff)

When your changes are merged, the CI pipeline will automatically build and deploy the updated documentation to GitHub Pages.
