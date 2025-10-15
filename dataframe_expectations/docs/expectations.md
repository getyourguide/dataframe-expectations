# DataFrame Expectations Documentation

## Expectations Summary

| Category | Subcategory | Expectations |
|----------|-------------|-------------|
| [Column Aggregation Expectations](#column-aggregation-expectations) | [Any Value](#any-value) | [expect_distinct_column_values_between](#-expect_distinct_column_values_between), [expect_distinct_column_values_equals](#-expect_distinct_column_values_equals), [expect_distinct_column_values_greater_than](#-expect_distinct_column_values_greater_than), [expect_distinct_column_values_less_than](#-expect_distinct_column_values_less_than), [expect_max_null_count](#-expect_max_null_count), [expect_max_null_percentage](#-expect_max_null_percentage), [expect_unique_rows](#-expect_unique_rows) |
| [Column Aggregation Expectations](#column-aggregation-expectations) | [Numerical](#numerical) | [expect_column_max_between](#-expect_column_max_between), [expect_column_mean_between](#-expect_column_mean_between), [expect_column_median_between](#-expect_column_median_between), [expect_column_min_between](#-expect_column_min_between), [expect_column_quantile_between](#-expect_column_quantile_between) |
| [Column Expectations](#column-expectations) | [Any Value](#any-value-1) | [expect_value_equals](#-expect_value_equals), [expect_value_in](#-expect_value_in), [expect_value_not_equals](#-expect_value_not_equals), [expect_value_not_in](#-expect_value_not_in), [expect_value_not_null](#-expect_value_not_null), [expect_value_null](#-expect_value_null) |
| [Column Expectations](#column-expectations) | [Numerical](#numerical-1) | [expect_value_between](#-expect_value_between), [expect_value_greater_than](#-expect_value_greater_than), [expect_value_less_than](#-expect_value_less_than) |
| [Column Expectations](#column-expectations) | [String](#string) | [expect_string_contains](#-expect_string_contains), [expect_string_ends_with](#-expect_string_ends_with), [expect_string_length_between](#-expect_string_length_between), [expect_string_length_equals](#-expect_string_length_equals), [expect_string_length_greater_than](#-expect_string_length_greater_than), [expect_string_length_less_than](#-expect_string_length_less_than), [expect_string_not_contains](#-expect_string_not_contains), [expect_string_starts_with](#-expect_string_starts_with) |
| [DataFrame Aggregation Expectations](#dataframe-aggregation-expectations) | [Any Value](#any-value-2) | [expect_max_rows](#-expect_max_rows), [expect_min_rows](#-expect_min_rows) |

---

## Detailed Documentation

### Column Aggregation Expectations

#### Any Value

##### 🎯 `expect_distinct_column_values_between`

**Description**: Add an expectation to check if the number of distinct values in a column falls within a range.

**Parameters**:

- `column_name`: <class 'str'>
- `min_value`: <class 'int'>
- `max_value`: <class 'int'>

**Signature**:

```python
expect_distinct_column_values_between(self, column_name: str, min_value: int, max_value: int)
```

---

##### 🎯 `expect_distinct_column_values_equals`

**Description**: Add an expectation to check if the number of distinct values in a column equals an expected count.

**Parameters**:

- `column_name`: <class 'str'>
- `expected_value`: <class 'int'>

**Signature**:

```python
expect_distinct_column_values_equals(self, column_name: str, expected_value: int)
```

---

##### 🎯 `expect_distinct_column_values_greater_than`

**Description**: Add an expectation to check if the number of distinct values in a column is greater than a threshold.

**Parameters**:

- `column_name`: <class 'str'>
- `threshold`: <class 'int'>

**Signature**:

```python
expect_distinct_column_values_greater_than(self, column_name: str, threshold: int)
```

---

##### 🎯 `expect_distinct_column_values_less_than`

**Description**: Add an expectation to check if the number of distinct values in a column is less than a threshold.

**Parameters**:

- `column_name`: <class 'str'>
- `threshold`: <class 'int'>

**Signature**:

```python
expect_distinct_column_values_less_than(self, column_name: str, threshold: int)
```

---

##### 🎯 `expect_max_null_count`

**Description**: Add an expectation to check if the count of null/NaN values in a specific column is below a threshold.

**Parameters**:

- `column_name`: <class 'str'>
- `max_count`: <class 'int'>

**Signature**:

```python
expect_max_null_count(self, column_name: str, max_count: int)
```

---

##### 🎯 `expect_max_null_percentage`

**Description**: Add an expectation to check if the percentage of null/NaN values in a specific column is below a threshold.

**Parameters**:

- `column_name`: <class 'str'>
- `max_percentage`: <class 'float'>

**Signature**:

```python
expect_max_null_percentage(self, column_name: str, max_percentage: float)
```

---

##### 🎯 `expect_unique_rows`

**Description**: Add an expectation to check if the rows in the DataFrame are unique based on specified columns.

**Parameters**:

- `column_names`: typing.List[str]

**Signature**:

```python
expect_unique_rows(self, column_names: List[str])
```

---

#### Numerical

##### 🎯 `expect_column_max_between`

**Description**: Add an expectation to check if the maximum value of a column falls within a specified range.

**Parameters**:

- `column_name`: <class 'str'>
- `min_value`: typing.Union[int, float]
- `max_value`: typing.Union[int, float]

**Signature**:

```python
expect_column_max_between(self, column_name: str, min_value: Union[int, float], max_value: Union[int, float])
```

---

##### 🎯 `expect_column_mean_between`

**Description**: Add an expectation to check if the mean value of a column falls within a specified range.

**Parameters**:

- `column_name`: <class 'str'>
- `min_value`: typing.Union[int, float]
- `max_value`: typing.Union[int, float]

**Signature**:

```python
expect_column_mean_between(self, column_name: str, min_value: Union[int, float], max_value: Union[int, float])
```

---

##### 🎯 `expect_column_median_between`

**Description**: Add an expectation to check if the median value of a column falls within a specified range.

**Parameters**:

- `column_name`: <class 'str'>
- `min_value`: typing.Union[int, float]
- `max_value`: typing.Union[int, float]

**Signature**:

```python
expect_column_median_between(self, column_name: str, min_value: Union[int, float], max_value: Union[int, float])
```

---

##### 🎯 `expect_column_min_between`

**Description**: Add an expectation to check if the minimum value of a column falls within a specified range.

**Parameters**:

- `column_name`: <class 'str'>
- `min_value`: typing.Union[int, float]
- `max_value`: typing.Union[int, float]

**Signature**:

```python
expect_column_min_between(self, column_name: str, min_value: Union[int, float], max_value: Union[int, float])
```

---

##### 🎯 `expect_column_quantile_between`

**Description**: Add an expectation to check if a quantile of a column falls within a specified range.

**Parameters**:

- `column_name`: <class 'str'>
- `quantile`: <class 'float'>
- `min_value`: typing.Union[int, float]
- `max_value`: typing.Union[int, float]

**Signature**:

```python
expect_column_quantile_between(self, column_name: str, quantile: float, min_value: Union[int, float], max_value: Union[int, float])
```

---

### Column Expectations

#### Any Value

##### 🎯 `expect_value_equals`

**Description**: Add an expectation to check if the values in a column equal a specified value.

**Parameters**:

- `column_name`: <class 'str'>
- `value`: <class 'object'>

**Signature**:

```python
expect_value_equals(self, column_name: str, value: object)
```

---

##### 🎯 `expect_value_in`

**Description**: Add an expectation to check if the values in a column are in a specified list of values.

**Parameters**:

- `column_name`: <class 'str'>
- `values`: typing.List[object]

**Signature**:

```python
expect_value_in(self, column_name: str, values: List[object])
```

---

##### 🎯 `expect_value_not_equals`

**Description**: Add an expectation to check if the values in a column do not equal a specified value.

**Parameters**:

- `column_name`: <class 'str'>
- `value`: <class 'object'>

**Signature**:

```python
expect_value_not_equals(self, column_name: str, value: object)
```

---

##### 🎯 `expect_value_not_in`

**Description**: Add an expectation to check if the values in a column are not in a specified list of values.

**Parameters**:

- `column_name`: <class 'str'>
- `values`: typing.List[object]

**Signature**:

```python
expect_value_not_in(self, column_name: str, values: List[object])
```

---

##### 🎯 `expect_value_not_null`

**Description**: Add an expectation to check if the values in a column are not null.

**Parameters**:

- `column_name`: <class 'str'>

**Signature**:

```python
expect_value_not_null(self, column_name: str)
```

---

##### 🎯 `expect_value_null`

**Description**: Add an expectation to check if the values in a column are null.

**Parameters**:

- `column_name`: <class 'str'>

**Signature**:

```python
expect_value_null(self, column_name: str)
```

---

#### Numerical

##### 🎯 `expect_value_between`

**Description**: Add an expectation to check if the values in a column are between two specified values.

**Parameters**:

- `column_name`: <class 'str'>
- `min_value`: <class 'float'>
- `max_value`: <class 'float'>

**Signature**:

```python
expect_value_between(self, column_name: str, min_value: float, max_value: float)
```

---

##### 🎯 `expect_value_greater_than`

**Description**: Add an expectation to check if the values in a column are greater than a specified value.

**Parameters**:

- `column_name`: <class 'str'>
- `value`: <class 'float'>

**Signature**:

```python
expect_value_greater_than(self, column_name: str, value: float)
```

---

##### 🎯 `expect_value_less_than`

**Description**: Add an expectation to check if the values in a column are less than a specified value.

**Parameters**:

- `column_name`: <class 'str'>
- `value`: <class 'float'>

**Signature**:

```python
expect_value_less_than(self, column_name: str, value: float)
```

---

#### String

##### 🎯 `expect_string_contains`

**Description**: Add an expectation to check if the values in a string column contain a specified substring.

**Parameters**:

- `column_name`: <class 'str'>
- `substring`: <class 'str'>

**Signature**:

```python
expect_string_contains(self, column_name: str, substring: str)
```

---

##### 🎯 `expect_string_ends_with`

**Description**: Add an expectation to check if the values in a string column end with a specified suffix.

**Parameters**:

- `column_name`: <class 'str'>
- `suffix`: <class 'str'>

**Signature**:

```python
expect_string_ends_with(self, column_name: str, suffix: str)
```

---

##### 🎯 `expect_string_length_between`

**Description**: Add an expectation to check if the length of the values in a string column is between two specified lengths.

**Parameters**:

- `column_name`: <class 'str'>
- `min_length`: <class 'int'>
- `max_length`: <class 'int'>

**Signature**:

```python
expect_string_length_between(self, column_name: str, min_length: int, max_length: int)
```

---

##### 🎯 `expect_string_length_equals`

**Description**: Add an expectation to check if the length of the values in a string column equals a specified length.

**Parameters**:

- `column_name`: <class 'str'>
- `length`: <class 'int'>

**Signature**:

```python
expect_string_length_equals(self, column_name: str, length: int)
```

---

##### 🎯 `expect_string_length_greater_than`

**Description**: Add an expectation to check if the length of the values in a string column is greater than a specified length.

**Parameters**:

- `column_name`: <class 'str'>
- `length`: <class 'int'>

**Signature**:

```python
expect_string_length_greater_than(self, column_name: str, length: int)
```

---

##### 🎯 `expect_string_length_less_than`

**Description**: Add an expectation to check if the length of the values in a string column is less than a specified length.

**Parameters**:

- `column_name`: <class 'str'>
- `length`: <class 'int'>

**Signature**:

```python
expect_string_length_less_than(self, column_name: str, length: int)
```

---

##### 🎯 `expect_string_not_contains`

**Description**: Add an expectation to check if the values in a string column do not contain a specified substring.

**Parameters**:

- `column_name`: <class 'str'>
- `substring`: <class 'str'>

**Signature**:

```python
expect_string_not_contains(self, column_name: str, substring: str)
```

---

##### 🎯 `expect_string_starts_with`

**Description**: Add an expectation to check if the values in a string column start with a specified prefix.

**Parameters**:

- `column_name`: <class 'str'>
- `prefix`: <class 'str'>

**Signature**:

```python
expect_string_starts_with(self, column_name: str, prefix: str)
```

---

### DataFrame Aggregation Expectations

#### Any Value

##### 🎯 `expect_max_rows`

**Description**: Add an expectation to check if the DataFrame has at most a maximum number of rows.

**Parameters**:

- `max_rows`: <class 'int'>

**Signature**:

```python
expect_max_rows(self, max_rows: int)
```

---

##### 🎯 `expect_min_rows`

**Description**: Add an expectation to check if the DataFrame has at least a minimum number of rows.

**Parameters**:

- `min_rows`: <class 'int'>

**Signature**:

```python
expect_min_rows(self, min_rows: int)
```

---
