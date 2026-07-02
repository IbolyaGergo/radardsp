import pytest
import numpy as np
from radarsig.filters import Filter
from radarsig.processing import SignalPipeline
from pathlib import Path

@pytest.fixture
def pipeline():
    # Load a filter - using the toml we created earlier
    # Note: Assumes filters/bp_2845_8MHz.toml exists
    filter_path = Path("filters/bp_2845_8MHz.toml")
    filt = Filter.from_toml(filter_path)
    return SignalPipeline(filters={"bp_filter": filt})

@pytest.fixture
def mock_data():
    return np.sin(np.linspace(0, 10, 100))

def test_pipeline_chaining(pipeline, mock_data):
    # Test chaining with load_data
    pipeline.load_data(mock_data) \
            .apply_filter("bp_filter", key="step1") \
            .apply_filter("bp_filter", key="step2") \
            .downsample(2, key="step3")
    
    assert "step1" in pipeline.history
    assert "step2" in pipeline.history
    assert "step3" in pipeline.history
    assert len(pipeline.history) == 3
    assert isinstance(pipeline.current_data, np.ndarray)
    
def test_pipeline_no_data_error(pipeline):
    # Ensure it fails if no data is loaded
    with pytest.raises(ValueError, match="No data provided or loaded"):
        pipeline.apply_filter("bp_filter", key="step1")
