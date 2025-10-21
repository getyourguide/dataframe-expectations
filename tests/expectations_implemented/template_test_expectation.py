from dataframe_expectations.expectations.expectation_registry import (
    DataFrameExpectationRegistry,
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
    assert (
        expectation.get_expectation_name() == "ExpectationDoesSomeCheck"
    ), f"Expected 'ExpectationDoesSomeCheck' but got: {expectation.get_expectation_name()}"


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
