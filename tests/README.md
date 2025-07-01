# OpenPlaceRecognition Tests

This directory contains tests for the OpenPlaceRecognition (OPR) project, organized by test type for optimal development workflow and CI/CD execution.

## Directory Structure

```
tests/
├── README.md                    # This file
├── conftest.py                 # Shared pytest configuration and fixtures
├── fixtures/                   # Shared test fixtures and utilities
│   ├── minkowski_fixtures.py   # MinkowskiEngine-related test fixtures
│   ├── dataset_fixtures.py     # Dataset loading test fixtures
│   └── mock_fixtures.py        # Mock objects and stubs
├── unit/                       # Fast, isolated component tests
│   ├── test_optional_deps.py   # Optional dependency management tests
│   ├── modules/                # Tests for src/opr/modules/
│   ├── datasets/               # Tests for src/opr/datasets/
│   ├── pipelines/              # Tests for src/opr/pipelines/
│   ├── models/                 # Tests for src/opr/models/
│   └── utils/                  # Tests for src/opr/utils/
├── integration/                # Cross-component interaction tests
│   ├── test_minkowski_integration.py   # MinkowskiEngine integration tests
│   ├── test_multimodal_pipeline.py     # Multi-modal feature integration
│   └── test_dataset_model_integration.py
├── e2e/                       # End-to-end system tests
│   ├── test_full_pipeline.py     # Complete pipeline execution
│   ├── test_notebook_examples.py # Jupyter notebook compatibility
│   └── test_cli_tools.py          # Command-line interface tests
└── performance/               # Performance and benchmarking tests
    ├── test_startup_time.py      # Import and initialization performance
    ├── test_memory_usage.py      # Memory consumption tests
    └── test_inference_speed.py   # Model inference benchmarks
```

## Running Tests

### Run All Tests
```bash
pytest                           # Run all tests (excluding slow/gpu by default)
pytest --slow                   # Include slow tests
pytest --gpu                    # Include GPU-requiring tests
```

### Run by Test Type
```bash
pytest tests/unit/              # Fast unit tests only
pytest tests/integration/       # Integration tests only
pytest tests/e2e/              # End-to-end tests only
pytest tests/performance/       # Performance tests only
```

### Run by Markers (Legacy Support)
```bash
pytest -m unit                  # Unit tests (equivalent to tests/unit/)
pytest -m integration          # Integration tests
pytest -m e2e                  # End-to-end tests
pytest -m "not slow"           # Exclude slow tests
pytest -m "gpu"                # Only GPU tests
```

### Run by Source Component
```bash
pytest tests/unit/modules/      # All module unit tests
pytest tests/*/test_*optional*  # All optional dependency tests
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Fast tests that check individual components in isolation
- **Duration**: < 1 second per test
- **Dependencies**: Minimal; use mocks for external dependencies
- **Scope**: Single function, class, or module
- **Example**: Testing `OptionalDependencyManager.exists_on_path()`

### Integration Tests (`tests/integration/`)
- **Purpose**: Test how components work together
- **Duration**: 1-10 seconds per test
- **Dependencies**: May require optional packages (MinkowskiEngine, etc.)
- **Scope**: Multiple components, cross-module interactions
- **Example**: Testing lazy import patterns with actual MinkowskiEngine

### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete user workflows and system behavior
- **Duration**: 10+ seconds per test
- **Dependencies**: Full system with datasets, models, etc.
- **Scope**: Complete pipelines, CLI tools, notebook examples
- **Example**: Running a full place recognition pipeline

### Performance Tests (`tests/performance/`)
- **Purpose**: Validate non-functional requirements
- **Duration**: Variable (often slow)
- **Dependencies**: Real system components
- **Scope**: Startup time, memory usage, inference speed
- **Example**: Measuring import time with/without optional dependencies

## Test Markers

We use pytest markers for additional categorization:

- `unit`: Fast isolated component tests
- `integration`: Cross-component interaction tests
- `e2e`: End-to-end system tests
- `gpu`: Tests requiring GPU resources
- `slow`: Tests taking > 10 seconds
- `minkowski`: Tests requiring MinkowskiEngine
- `model`: Tests for model implementations
- `dataset`: Tests for dataset loading
- `performance`: Performance benchmarking tests

## Contribution Guidelines

When contributing new tests:

### 1. Choose the Right Directory
- **Unit tests**: Place in `tests/unit/` - for testing individual functions/classes in isolation
- **Integration tests**: Place in `tests/integration/` - for testing component interactions
- **End-to-end tests**: Place in `tests/e2e/` - for testing complete user workflows
- **Performance tests**: Place in `tests/performance/` - for benchmarking and performance validation

### 2. Mirror Source Structure
Within each test type directory, mirror the source code structure:
```bash
# Testing src/opr/modules/eca.py
tests/unit/modules/test_eca.py           # Unit tests
tests/integration/test_eca_integration.py # Integration tests (if needed)
```

### 3. Follow Naming Conventions
- **Test files**: `test_*.py` (e.g., `test_eca.py`)
- **Test functions**: `test_*` (e.g., `test_eca_layer_initialization`)
- **Test classes**: `Test*` (e.g., `TestECALayer`)

### 4. Use Appropriate Markers
Add pytest markers to help with test discovery and filtering:
```python
import pytest

@pytest.mark.unit
@pytest.mark.minkowski  # If test requires MinkowskiEngine
def test_mink_eca_layer():
    """Test MinkowskiEngine ECA layer functionality."""
    pass
```

### 5. Write Clear Documentation
- **File docstrings**: Describe what module/component is being tested
- **Function docstrings**: Describe the specific behavior being tested
- **Assertion messages**: Provide helpful failure messages

### 6. Test Structure (AAA Pattern)
Follow the Arrange-Act-Assert pattern:

```python
@pytest.mark.unit
def test_optional_dependency_exists():
    """Test that existing packages are correctly detected."""
    # Arrange
    package_name = "os"  # Standard library package that always exists

    # Act
    result = OptionalDependencyManager.exists_on_path(package_name)

    # Assert
    assert result is True, f"Expected {package_name} to be detected as available"
```

### 7. Handle Optional Dependencies
For tests requiring optional dependencies:

```python
@pytest.mark.integration
@pytest.mark.minkowski
@pytest.mark.skipif(not has_minkowski(), reason="MinkowskiEngine not available")
def test_sparse_convolution_integration():
    """Test integration with MinkowskiEngine sparse convolutions."""
    # Test implementation here
    pass
```

### 8. Use Fixtures Appropriately
- **Simple setup**: Use fixtures from `tests/fixtures/`
- **Complex setup**: Create local fixtures in `conftest.py` within the test directory
- **Shared mocks**: Place in `tests/fixtures/mock_fixtures.py`

### 9. Test Independence
- Tests should be able to run in any order
- Use `setUp`/`tearDown` or fixtures for test isolation
- Don't rely on side effects from other tests

### 10. Performance Considerations
- **Unit tests**: Should complete in < 1 second
- **Integration tests**: Should complete in < 10 seconds
- **Slow tests**: Mark with `@pytest.mark.slow`
- **GPU tests**: Mark with `@pytest.mark.gpu`

## Examples

### Unit Test Example
```python
# tests/unit/test_optional_deps.py
import pytest
from unittest.mock import patch
from opr.optional_deps import OptionalDependencyManager

@pytest.mark.unit
class TestOptionalDependencyManager:
    """Unit tests for OptionalDependencyManager."""

    def test_exists_on_path_with_real_package(self):
        """Test exists_on_path correctly identifies real packages."""
        # Arrange
        package = "os"  # Always available

        # Act
        result = OptionalDependencyManager.exists_on_path(package)

        # Assert
        assert result is True, f"Standard library package {package} should be detected"

    @patch('importlib.util.find_spec')
    def test_exists_on_path_with_missing_package(self, mock_find_spec):
        """Test exists_on_path correctly identifies missing packages."""
        # Arrange
        mock_find_spec.return_value = None
        package = "nonexistent_package"

        # Act
        result = OptionalDependencyManager.exists_on_path(package)

        # Assert
        assert result is False, f"Nonexistent package {package} should not be detected"
```

### Integration Test Example
```python
# tests/integration/test_minkowski_integration.py
import pytest
from opr.optional_deps import has_minkowski, require_minkowski

@pytest.mark.integration
@pytest.mark.minkowski
@pytest.mark.skipif(not has_minkowski(), reason="MinkowskiEngine required")
class TestMinkowskiIntegration:
    """Integration tests for MinkowskiEngine functionality."""

    def test_lazy_import_pattern(self):
        """Test that lazy import pattern works with real MinkowskiEngine."""
        # Arrange
        feature_name = "test sparse convolution"

        # Act
        available = require_minkowski(feature_name)

        # Assert
        assert available is True, "MinkowskiEngine should be available for testing"

        # Act - perform actual lazy import
        import MinkowskiEngine as ME

        # Assert
        assert hasattr(ME, 'SparseTensor'), "MinkowskiEngine should have SparseTensor"
```

### End-to-End Test Example
```python
# tests/e2e/test_pipeline_e2e.py
import pytest
from opr.pipelines.place_recognition import PlaceRecognitionPipeline

@pytest.mark.e2e
@pytest.mark.slow
class TestPipelineE2E:
    """End-to-end tests for complete pipeline execution."""

    def test_pipeline_with_sparse_fallback(self):
        """Test pipeline gracefully falls back when MinkowskiEngine unavailable."""
        # Arrange
        pipeline = PlaceRecognitionPipeline(use_sparse=None)  # Auto-detect

        # Act & Assert - should not raise regardless of MinkowskiEngine availability
        assert pipeline is not None
        assert hasattr(pipeline, 'use_sparse')
```

## Migration Guide

If you have existing tests in the flat structure, here's how to migrate them:

1. **Identify test type**:
   - Fast, isolated → `tests/unit/`
   - Cross-component → `tests/integration/`
   - Full system → `tests/e2e/`

2. **Move to appropriate directory**:
   ```bash
   # Old location
   tests/test_optional_deps.py

   # New location
   tests/unit/test_optional_deps.py
   ```

3. **Update imports if needed** (usually not required)

4. **Add appropriate markers**:
   ```python
   @pytest.mark.unit  # Add this marker
   def test_existing_function():
       pass
   ```

5. **Update CI/CD scripts** to use new directory structure
