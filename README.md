## 🎯 DataFrameExpectations

**DataFrameExpectations** is a Python library designed to validate **Pandas** and **PySpark** DataFrames using customizable, reusable expectations. It simplifies testing in data pipelines and end-to-end workflows by providing a standardized framework for DataFrame validation.

Instead of using different validation approaches for DataFrames across repositories, this library provides a
standardized solution for this use case. As a result, any contributions made here—such as adding new expectations—can be leveraged by all users of the library.
You can find the complete list of expectations [here](docs/build/html/expectations.html).

**Pandas example:**
```python
from dataframe_expectations.expectations_suite import DataframeExpectationsSuite

suite = (
    DataframeExpectationsSuite()
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
from dataframe_expectations.expectations_suite import DataframeExpectationsSuite

suite = (
    DataframeExpectationsSuite()
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

## How to contribute?
Contributions are welcome! You can enhance the library by adding new expectations, refining existing ones, or improving the testing framework.

### Adding a new expectation

#### Step 1: Defining your expectations

Most use cases that involve validating a single column in the dataframe can be covered by the initialising the
`DataframeColumnExpectation` class with the correct parameters. Expectations implemented by initialising `DataframeColumnExpectation` can be found in the [column_expectations](expectations/column_expectations) module, categorised based on the column type.

If you want to go ahead with implementing `DataframeColumnExpectation`, you first need to identify the category based on the column value. Existing expectations are already categorised into `string`, `numerical` or `any_value` expectations. Create a new category in [column_expectations](expectations/column_expectations) if you think existing categories don't fit your use case. Once you have decided where the expectation needs to be added, you can define it as follows:

```python
from dataframe_expectations.expectations.expectation_registry import (
    register_expectation,
)
from dataframe_expectations.expectations.utils import requires_params


@register_expectation("ExpectIsDivisible")
@requires_params("column_name", "value", types={"column_name": str, "value": int})
def create_expectation_do_something_unexpected(**kwargs) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    value = kwargs["value"]

    return DataframeColumnExpectation(
        expectation_name="ExpectIsDivisible",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name] % value != 0], # function that finds violations
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name) % value != 0), # function that finds violations
        description=f"'{column_name}' divisible by {value}",
        error_message=f"'{column_name}' not divisible by {value}.",
    )
```

For additional guidance, you can refer to the implementation of `ExpectationValueGreaterThan` and `ExpectationValueLessThan`
in [column_expectation_factory.py](expectations/column_expectations/numerical_expectations.py). These examples demonstrate how to
initialise the `DataframeColumnExpectation` class with the right parameters and define filtering logic for different conditions. The `@register_expectation` decorator is needed to add your expectation to the library. `@requires_params` decorator is a utility that helps you validate the input parameters.

#### Adding aggregation-based expectations

For expectations that require aggregation operations (such as row counts, distinct value counts, null percentages, etc.), you should implement custom expectation classes by inheriting from `DataframeAggregationExpectation`. These types of expectations cannot be easily covered by the `DataframeColumnExpectation` class because they involve DataFrame-level or column-level aggregations rather than row-by-row validations.

Here's an example of how to implement an aggregation-based expectation:

```python
from dataframe_expectations import DataFrameLike, DataFrameType
from dataframe_expectations.expectations.aggregation_expectation import (
    DataframeAggregationExpectation,
)
from dataframe_expectations.expectations.expectation_registry import register_expectation
from dataframe_expectations.expectations.utils import requires_params
from dataframe_expectations.result_message import (
    DataframeExpectationFailureMessage,
    DataframeExpectationResultMessage,
    DataframeExpectationSuccessMessage,
)
import pandas as pd
from pyspark.sql import functions as F


class ExpectationMinRows(DataframeAggregationExpectation):
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
    ) -> DataframeExpectationResultMessage:
        """Validate minimum row count in a pandas DataFrame."""
        # Note: Column validation is automatically handled by the parent class
        try:
            row_count = len(data_frame)

            if row_count >= self.min_count:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"DataFrame has {row_count} row(s), expected at least {self.min_count}.",
                )
        except Exception as e:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error counting rows: {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """Validate minimum row count in a PySpark DataFrame."""
        # Note: Column validation is automatically handled by the parent class
        try:
            row_count = data_frame.count()

            if row_count >= self.min_count:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"DataFrame has {row_count} row(s), expected at least {self.min_count}.",
                )
        except Exception as e:
            return DataframeExpectationFailureMessage(
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
```

Key differences for aggregation-based expectations:

1. **Inherit from `DataframeAggregationExpectation`**: This base class provides the framework for aggregation operations and automatically handles column validation.

2. **Implement `aggregate_and_validate_pandas` and `aggregate_and_validate_pyspark`**: These methods are specifically designed for aggregation operations rather than the generic `validate_pandas` and `validate_pyspark` methods.

3. **Call `super().__init__()`**: Initialize the parent class with expectation metadata including `expectation_name`, `column_names`, and `description`.

4. **Automatic column validation**: The parent class automatically validates that required columns exist before calling your implementation methods. You don't need to manually check for column existence.

5. **Error handling**: Wrap aggregation operations in try-catch blocks since aggregations can fail due to data type issues or other DataFrame problems.

Example of a column-based aggregation expectation:

```python
class ExpectationColumnMeanBetween(DataframeAggregationExpectation):
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
    ) -> DataframeExpectationResultMessage:
        """Validate column mean in a pandas DataFrame."""
        # Column validation is automatically handled by the parent class
        try:
            mean_val = data_frame[self.column_name].mean()

            if pd.isna(mean_val):
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Column '{self.column_name}' contains only null values.",
                )

            if self.min_value <= mean_val <= self.max_value:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Column '{self.column_name}' mean value {mean_val} is not between {self.min_value} and {self.max_value}.",
                )
        except Exception as e:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error calculating mean for column '{self.column_name}': {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """Validate column mean in a PySpark DataFrame."""
        # Column validation is automatically handled by the parent class
        try:
            mean_result = data_frame.select(F.avg(self.column_name).alias("mean_val")).collect()
            mean_val = mean_result[0]["mean_val"]

            if mean_val is None:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' contains only null values.",
                )

            if self.min_value <= mean_val <= self.max_value:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' mean value {mean_val} is not between {self.min_value} and {self.max_value}.",
                )
        except Exception as e:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=f"Error calculating mean for column '{self.column_name}': {str(e)}",
            )
```

Key considerations for aggregation-based expectations:

1. **Performance**: Aggregation operations can be expensive, especially on large datasets in PySpark. Consider the performance implications of your aggregation logic.

2. **Different DataFrame types**: Ensure your implementation works correctly for both Pandas and PySpark DataFrames, as aggregation APIs may differ (e.g., `df.mean()` vs `F.avg()`).

3. **Error handling**: Include proper error handling for edge cases like empty DataFrames or all-null columns.

4. **Message clarity**: Provide clear, informative error messages that help users understand what went wrong.

5. **Automatic column validation**: The `DataframeAggregationExpectation` base class automatically validates that required columns exist before calling your `aggregate_and_validate_*` methods. Simply specify the required columns in the `column_names` parameter during initialization.

6. **Focus on aggregation logic**: Since column validation is handled automatically, you can focus purely on implementing your aggregation and validation logic without worrying about column existence checks.

Examples of aggregation-based expectations include:
- `ExpectationMinRows` / `ExpectationMaxRows`: Validate row count limits
- `ExpectationDistinctColumnValuesEquals`: Validate the number of distinct values in a column
- `ExpectationMaxNullPercentage`: Validate the percentage of null values in a column
- `ExpectationUniqueRows`: Validate that rows are unique across specified columns
- `ExpectationColumnMeanBetween`: Validate that column mean falls within a range
- `ExpectationColumnQuantileBetween`: Validate that column quantiles fall within ranges

For more examples, check the [aggregation_expectations](expectations/aggregation_expectations) module.

#### Custom expectations with full control over the validation

While the `DataframeColumnExpectation` covers most use cases there might be other instances where you need more control
over the validation logic. For such instances you can define a new expectation by inheriting the `DataframeExpectation`
class.

To help you get started, here's a template you can customize to fit your specific use case:

```python
from typing import Callable

from dataframe_expectations import DataFrameLike, DataFrameType
from dataframe_expectations.expectations import DataframeExpectation
from dataframe_expectations.result_message import (
    DataframeExpectationFailureMessage,
    DataframeExpectationResultMessage,
    DataframeExpectationSuccessMessage,
)

class ExpectTheUnexpected(DataframeExpectation):
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
    ) -> DataframeExpectationResultMessage:
        """
        Validate a pandas DataFrame against the expectation.
        """
        <Add your validation logic here for Pandas DataFrame. Return either DataframeExpectationSuccessMessage or DataframeExpectationFailureMessage>

    def validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """
        Validate a PySpark DataFrame against the expectation.
        """
        <Add your validation logic here for PySpark DataFrame. Return either DataframeExpectationSuccessMessage or DataframeExpectationFailureMessage>
```

#### Step 2: Adding your expectation to `DataframeExpectationsSuite`
The `DataframeExpectationsSuite` encapsulates all the expectations that are provided by this library.
After defining and testing your expectation, integrate it into the [DataframeExpectationsSuite](expectations_suite.py) by creating a new method with a descriptive name starting with the prefix `expect_` (this is needed to generate documentation later). Here’s an example:

```python
class DataframeExpectationsSuite:
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
        :return: An instance of DataframeExpectationsSuite.
        """

        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectIsDivisible",
            column_name=column_name,
            value=value,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

```

#### Step 3: Adding unit tests
To ensure your expectations work as expected (pun intended), make sure to add unit tests in the
`tests/data_engineering/dataframe_expectations/expectations_implemented` folder. Here's a template to get you started:

```python
import pytest
import pandas as pd

from dataframe_expectations import DataFrameType
from dataframe_expectations.expectations.expectation_registry import (
    DataframeExpectationRegistry,
)
from dataframe_expectations.result_message import (
    DataframeExpectationFailureMessage,
    DataframeExpectationSuccessMessage,
)


def test_expectation_name():
    """
    Test that the expectation name is correctly returned.
    This method should be implemented in the subclass.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
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
```

For concrete examples of unit tests, check for tests in the [expectations_implemented](../../tests/dataframe_expectations/expectations_implemented/column_expectations) folder. You can also
find the unit test template [here](../../tests/dataframe_expectations/expectations_implemented/template_test_expectation.py).

#### Step 4: Updating the documentation
After the expectation is ready for use, the last thing remaining is adding your expectation to the documentation. The documentation is automatically generated using a CI pipeline with the `uv` package manager and is available at `docs/build/html/expectations.html`.

Make sure to add the docstring for the function you added to `DataframeExpectationsSuite` before submitting your changes. The CI pipeline will automatically update the documentation using the make targets in the `docs` folder when your changes are merged.

If you need to build the documentation locally for testing, you can use the make targets available in the `docs` folder.

```bash
cd docs
uv run sphinx-build source build/html
```

or use the make command

```bash
cd docs
make html
```
