# OpenPlaceRecognition Tests

This directory contains tests for the OpenPlaceRecognition (OPR) project.

## Running Tests

To run all tests:

```bash
pytest
```

To run specific test categories:

```bash
pytest -m unit         # Run only unit tests
pytest -m integration  # Run only integration tests
pytest -m e2e          # Run only end-to-end tests
```

## Test Categories

We use pytest markers to categorize tests:

- `unit`: Fast tests that check individual components in isolation
- `integration`: Tests that check how components work together
- `e2e`: End-to-end tests that test the system as a whole
- `model`: Tests for model implementations
- `dataset`: Tests for dataset loading and preprocessing
- `metrics`: Tests for evaluation metrics
- `gpu`: Tests requiring GPU resources
- `slow`: Tests that take a long time to run

## Contribution Guidelines

When contributing new tests:

1. **Choose the right category:** Add the appropriate pytest marker to your test
2. **Follow naming conventions:** Name test files as `test_*.py` and test functions as `test_*`
3. **Write clear assertions:** Make sure failure messages are descriptive
4. **Include docstrings:** Document what your test is checking
5. **Test edge cases:** Consider boundary conditions and error cases
6. **Avoid test interdependence:** Tests should be able to run independently
7. **Use fixtures:** For common setup/teardown logic

Example:

```python
import pytest

@pytest.mark.unit
def test_your_feature():
    """Test description of what you're testing."""
    # Arrange
    expected = "expected result"

    # Act
    actual = your_function()

    # Assert
    assert actual == expected, f"Expected {expected}, got {actual}"
```
