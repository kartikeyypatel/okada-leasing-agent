"""
Pytest configuration and shared fixtures for ChromaDB integration tests.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
from unittest.mock import patch
import pandas as pd

# Configure pytest for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Test settings configuration."""
    test_config = {
        "CHROMA_PERSIST_DIRECTORY": tempfile.mkdtemp(),
        "CHROMA_HOST": None,
        "CHROMA_PORT": None,
        "CHROMA_COLLECTION_PREFIX": "test_okada_user_"
    }
    
    with patch.multiple('app.config.settings', **test_config):
        yield test_config
    
    # Cleanup
    if os.path.exists(test_config["CHROMA_PERSIST_DIRECTORY"]):
        shutil.rmtree(test_config["CHROMA_PERSIST_DIRECTORY"])


@pytest.fixture
def sample_property_data():
    """Sample property data for testing."""
    return [
        {
            "property address": "123 Test St",
            "monthly rent": "2500",
            "size (sf)": "1200",
            "bedrooms": "2",
            "bathrooms": "1"
        },
        {
            "property address": "456 Demo Ave",
            "monthly rent": "3000",
            "size (sf)": "1500",
            "bedrooms": "3",
            "bathrooms": "2"
        },
        {
            "property address": "789 Sample Rd",
            "monthly rent": "2000",
            "size (sf)": "1000",
            "bedrooms": "1",
            "bathrooms": "1"
        }
    ]


@pytest.fixture
def create_test_csv():
    """Factory fixture to create test CSV files."""
    created_files = []
    
    def _create_csv(data, filename=None):
        if filename is None:
            fd, filename = tempfile.mkstemp(suffix='.csv')
            os.close(fd)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        created_files.append(filename)
        return filename
    
    yield _create_csv
    
    # Cleanup
    for filename in created_files:
        if os.path.exists(filename):
            os.remove(filename)


# Pytest markers for test categorization
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test class or function names
        if "Performance" in item.name:
            item.add_marker(pytest.mark.performance)
        elif "Integration" in item.name:
            item.add_marker(pytest.mark.integration)
        elif "Test" in item.name and "Integration" not in item.name and "Performance" not in item.name:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if "concurrent" in item.name.lower() or "large" in item.name.lower():
            item.add_marker(pytest.mark.slow) 