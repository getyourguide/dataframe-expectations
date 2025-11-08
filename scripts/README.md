# Stub Generator for IDE Autocomplete

## Overview

The `generate_suite_stubs.py` script generates a `.pyi` stub file for `DataFrameExpectationsSuite` to provide IDE autocomplete and type hints. The actual implementation can use `__getattr__` for dynamic method generation without any boilerplate.

## Clean Separation Approach

**Runtime (.py file)**: Clean implementation with `__getattr__` or manual methods
**IDE Support (.pyi file)**: Type stubs with all method signatures for autocomplete

✅ **Full IDE Autocomplete** - All 31 `expect_*` methods visible in IDE
✅ **Type Hints** - Parameter types (`str`, `int`, `Union[int, float]`, etc.)
✅ **Clean .py File** - No code duplication or generated boilerplate
✅ **Registry-Based** - Stubs auto-generated from expectation metadata

## Usage

### Generate/Update Stub File

```bash
# From project root
python scripts/generate_suite_stubs.py
```

This will:
1. Read all expectation metadata from the registry
2. Generate method signatures with full docstrings and type hints
3. Create/update `dataframe_expectations/expectations_suite.pyi`

The `.pyi` file is automatically discovered by IDEs (VS Code, PyCharm, etc.) and type checkers (mypy, pyright).

### Check if Stub File is Up-to-Date

```bash
# Useful for CI/CD pipelines
python scripts/generate_suite_stubs.py --check
```

Returns exit code 0 if up-to-date, 1 if stubs need regeneration.

### Print Generated Stubs

```bash
# Preview without writing files
python scripts/generate_suite_stubs.py --print
```

## When to Run

Run the script whenever you:

1. **Add a new expectation** - After registering a new expectation with metadata
2. **Update metadata** - After changing descriptions, categories, or parameters
3. **IDE autocomplete missing** - If your IDE doesn't show expect_* methods

## How It Works

The `.pyi` file contains type stubs that IDEs use for autocomplete:

```python
# expectations_suite.pyi
class DataFrameExpectationsSuite:
    def expect_value_equals(
        self,
        column_name: str,
        value: object
    ) -> DataFrameExpectationsSuite:
        """Check if the values in a column equal a specified value..."""
        ...
```

Python's type system automatically uses `.pyi` files when they exist alongside `.py` files, providing IDE autocomplete without affecting runtime behavior.

## How It Works

### 1. Registry Metadata

Each expectation is registered with metadata:

```python
@register_expectation(
    "ExpectationValueEquals",
    description="Check if the values in a column equal a specified value",
    category="Column Expectations",
    subcategory="Any Value",
    params_doc={
        "column_name": "The name of the column to check",
        "value": "The value to compare against",
    },
)
@requires_params("column_name", "value", types={"column_name": str, "value": object})
def create_expectation_value_equals(**kwargs) -> ExpectationValueEquals:
    ...
```

### 2. Script Generates Stubs

The script reads the metadata and generates:

```python
def expect_value_equals(self, column_name: str, value: object):
    """
    Check if the values in a column equal a specified value

    Categories:
      category: Column Expectations
      subcategory: Any Value

    :param column_name: The name of the column to check
    :param value: The value to compare against
    :return: an instance of DataFrameExpectationsSuite.
    """
    return self._add_expectation("ExpectationValueEquals", {"column_name": column_name, "value": value})
```

### 3. Single Implementation Point

All stubs delegate to:

```python
def _add_expectation(self, expectation_name: str, kwargs: dict):
    """Helper method to add an expectation using the registry."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name=expectation_name, **kwargs
    )
    logger.info(f"Adding expectation: {expectation}")
    self.__expectations.append(expectation)
    return self
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Code: suite.expect_value_equals(column="age", value=5) │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Generated Stub: def expect_value_equals(self, ...)          │
│    return self._add_expectation("ExpectationValueEquals", ) │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Helper: def _add_expectation(self, name, kwargs)           │
│    expectation = Registry.get_expectation(name, **kwargs)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Registry: Returns expectation instance from factory         │
└─────────────────────────────────────────────────────────────┘
```

## Integration with CI/CD

Add to your pre-commit hook or CI pipeline:

```yaml
# .github/workflows/ci.yml
- name: Check stub methods are up-to-date
  run: |
    python scripts/generate_suite_stubs.py --check
```

## Maintenance

The script is self-contained and requires no maintenance. When adding new expectations:

1. Register with metadata in your expectation file
2. Run `python scripts/generate_suite_stubs.py`
3. Commit the updated `expectations_suite.py`

That's it! 🎉

## Example: Adding a New Expectation

### Step 1: Create the expectation

```python
# In dataframe_expectations/expectations/column_expectations/my_expectations.py
@register_expectation(
    "ExpectationMyCheck",
    description="Check if my custom condition is met",
    category="Column Expectations",
    subcategory="Custom",
    params_doc={
        "column_name": "The column to check",
        "threshold": "The threshold value",
    },
)
@requires_params("column_name", "threshold", types={"column_name": str, "threshold": int})
def create_expectation_my_check(**kwargs) -> ExpectationMyCheck:
    return ExpectationMyCheck(**kwargs)
```

### Step 2: Generate stubs

```bash
python scripts/generate_suite_stubs.py
```

### Step 3: Use with autocomplete

```python
suite = DataFrameExpectationsSuite()
suite.expect_my_check(column_name="score", threshold=100)  # IDE autocomplete works!
```

## Troubleshooting

**Q: Script fails with "ModuleNotFoundError"**
A: Run with `PYTHONPATH=. python scripts/generate_suite_stubs.py`

**Q: Stubs not updating**
A: Check that `self.__expectations = []` exists in `__init__` and `def run(` exists

**Q: Type hints not showing in IDE**
A: Ensure your IDE is using the project's Python interpreter

**Q: Want to customize generated stubs**
A: Modify the `generate_stub_method()` function in the script
