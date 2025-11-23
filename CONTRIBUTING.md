# Contributing to DataFrameExpectations

Thank you for your interest in contributing to DataFrameExpectations! We welcome contributions from the community, whether it's adding new expectations, fixing bugs, improving documentation, or enhancing the testing framework.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Adding New Expectations](#adding-new-expectations)
- [Running Tests](#running-tests)
- [Code Style Guidelines](#code-style-guidelines)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Versioning and Commits](#versioning-and-commits)

## Getting Started

Before you begin:
1. Check existing [issues](https://github.com/getyourguide/dataframe-expectations/issues) and [pull requests](https://github.com/getyourguide/dataframe-expectations/pulls) to avoid duplicates
2. For major changes, open an issue first to discuss your proposal
3. Ensure you agree with the [Apache 2.0 License](LICENSE.txt)

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/getyourguide/dataframe-expectations.git
   cd dataframe-expectations
   ```

2. **Install UV package manager:**
   ```bash
   pip install uv
   ```

3. **Install development dependencies:**
   ```bash
   # This will automatically create a virtual environment
   uv sync --group dev
   ```

4. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

5. **Verify your setup:**
   ```bash
   uv run pytest tests/ -n auto --cov=dataframe_expectations
   ```

6. **(Optional) Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```
   This will automatically run checks before each commit.

## How to Contribute

### Reporting Bugs
Open an [issue](https://github.com/getyourguide/dataframe-expectations/issues) with a clear description, steps to reproduce, expected vs. actual behavior, and relevant environment details.

### Documentation
Fix typos, clarify docs, add examples, or improve the README.

### Features
Open an issue first to discuss new features, explain the use case, and consider backward compatibility.

### Adding Expectations
See the **[Adding Expectations Guide](https://code.getyourguide.com/dataframe-expectations/adding_expectations.html)** for detailed instructions.


## Running Tests

```bash
# Run all tests with parallelization
uv run pytest tests/ -n auto

# Run with coverage and parallelization
uv run pytest tests/ -n auto --cov=dataframe_expectations

# Run specific test file
uv run pytest tests/test_expectations_suite.py -n auto

# Run tests matching a pattern
uv run pytest tests/ -n auto -k "test_expect_min_rows"
```

## Code Style Guidelines

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints for all function parameters and return values
- Maximum line length: 120 characters
- Use meaningful variable and function names

### Docstrings
- Use Google-style docstrings
- Include parameter descriptions and return types
- Add usage examples for complex functions

### Code Quality
- Write clear, self-documenting code
- Add comments for complex logic
- Keep functions focused and single-purpose
- Avoid deep nesting (max 3-4 levels)

### Testing
- Maintain or improve test coverage
- Test expected behavior (happy paths) and error conditions (edge cases)
- Use descriptive test names

## Submitting a Pull Request

1. **Create a branch** and make your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Run tests:**
   ```bash
   uv run pytest tests/ -n auto --cov=dataframe_expectations
   ```

3. **Commit using [Conventional Commits](https://www.conventionalcommits.org/)** (see [Versioning](#versioning-and-commits))
   ```bash
   git commit -m "feat: your feature description"
   ```

4. **Push and open a PR** with a clear description referencing any related issues

## Versioning and Commits

This project follows [Semantic Versioning](https://semver.org/) and uses [Conventional Commits](https://www.conventionalcommits.org/).

### Commit Message Format

```
<type>: <description>

[optional body]

[optional footer]
```

### Commit Types

- `feat:` - New feature → **MINOR** version bump (0.1.0 → 0.2.0)
- `fix:` - Bug fix → **PATCH** version bump (0.1.0 → 0.1.1)
- `feat!:` or `BREAKING CHANGE:` - Breaking change → **MAJOR** version bump (0.1.0 → 1.0.0)
- `docs:` - Documentation changes (no version bump)
- `test:` - Test changes (no version bump)
- `chore:` - Maintenance tasks (no version bump)
- `refactor:` - Code refactoring (no version bump)
- `style:` - Code style changes (no version bump)
- `ci:` - CI/CD changes (no version bump)

### Examples

```bash
# Adding a new feature
git commit -m "feat: add expect_column_sum_equals expectation"

# Fixing a bug
git commit -m "fix: correct validation logic in expect_value_greater_than"

# Breaking change
git commit -m "feat!: remove deprecated API methods"

# With body
git commit -m "feat: add tag filtering support

Allow expectations to be filtered by tags at runtime.
This enables selective execution of validation rules."

# Documentation update
git commit -m "docs: update README with new examples"
```

### What Happens Next

When your PR is merged to main:
1. [Release Please](https://github.com/googleapis/release-please) automatically creates/updates a Release PR
2. The Release PR includes version bump and changelog
3. When the Release PR is merged, a GitHub Release is created
4. The maintainer manually publishes the package to PyPI

## Questions?

If you have questions or need help:
- Open an [issue](https://github.com/getyourguide/dataframe-expectations/issues)
- Review the [documentation](https://code.getyourguide.com/dataframe-expectations/)

Thank you for contributing! 🎉
