"""
Unit tests for data fetcher module.
Tests data acquisition, processing, and caching functionality.
"""

import pytest
import pandas as pd
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_fetcher import LifeExpectancyDataFetcher, get_life_expectancy_data


class TestLifeExpectancyDataFetcher:
    """Test cases for LifeExpectancyDataFetcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.fetcher = LifeExpectancyDataFetcher(cache_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test fetcher initialization."""
        assert self.fetcher.cache_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)
        assert len(self.fetcher.indicators) == 3
        assert 'SP.DYN.LE00.IN' in self.fetcher.indicators
    
    def test_should_update_data_no_metadata(self):
        """Test update check when no metadata exists."""
        assert self.fetcher.should_update_data() == True
    
    def test_should_update_data_recent(self):
        """Test update check with recent data."""
        # Create recent metadata
        metadata = {
            'last_update': datetime.now().isoformat(),
            'total_records': 100
        }
        
        with open(self.fetcher.metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        assert self.fetcher.should_update_data(max_age_hours=24) == False
    
    def test_should_update_data_old(self):
        """Test update check with old data."""
        # Create old metadata
        old_time = datetime.now() - timedelta(hours=25)
        metadata = {
            'last_update': old_time.isoformat(),
            'total_records': 100
        }
        
        with open(self.fetcher.metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        assert self.fetcher.should_update_data(max_age_hours=24) == True
    
    def test_get_country_regions(self):
        """Test country regions mapping."""
        regions = self.fetcher.get_country_regions()
        
        assert isinstance(regions, dict)
        assert 'Spain' in regions
        assert 'United States' in regions
        assert regions['Spain'] == 'Europe & Central Asia'
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        sample_data = self.fetcher._generate_sample_data(2020, 2022)
        
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) > 0
        assert 'Country' in sample_data.columns
        assert 'Year' in sample_data.columns
        assert 'Value' in sample_data.columns
        assert sample_data['Year'].min() >= 2020
        assert sample_data['Year'].max() <= 2022
    
    def test_process_and_save_data(self):
        """Test data processing and saving."""
        # Create sample data
        sample_data = self.fetcher._generate_sample_data(2020, 2021)
        
        # Process and save
        processed_data = self.fetcher.process_and_save_data(sample_data)
        
        # Verify processing
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        assert 'Country' in processed_data.columns
        assert 'Year' in processed_data.columns
        assert 'Region' in processed_data.columns
        
        # Verify files were created
        assert os.path.exists(self.fetcher.data_file)
        assert os.path.exists(self.fetcher.metadata_file)
        
        # Verify metadata
        with open(self.fetcher.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert 'last_update' in metadata
        assert 'total_records' in metadata
        assert metadata['total_records'] == len(processed_data)
    
    def test_load_cached_data(self):
        """Test loading cached data."""
        # First, ensure no cached data
        assert self.fetcher.load_cached_data() is None
        
        # Create and save sample data
        sample_data = self.fetcher._generate_sample_data(2020, 2021)
        processed_data = self.fetcher.process_and_save_data(sample_data)
        
        # Load cached data
        cached_data = self.fetcher.load_cached_data()
        
        assert cached_data is not None
        assert isinstance(cached_data, pd.DataFrame)
        assert len(cached_data) == len(processed_data)
    
    @patch('sys.path')
    def test_get_data_force_update(self, mock_path):
        """Test get_data with force update."""
        # Mock the World Bank data call to use sample data
        with patch.object(self.fetcher, 'fetch_world_bank_data') as mock_fetch:
            mock_fetch.return_value = self.fetcher._generate_sample_data(2020, 2021)
            
            data = self.fetcher.get_data(force_update=True)
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            mock_fetch.assert_called_once()
    
    def test_get_data_use_cache(self):
        """Test get_data using cached data."""
        # Create cached data first
        sample_data = self.fetcher._generate_sample_data(2020, 2021)
        self.fetcher.process_and_save_data(sample_data)
        
        # Get data (should use cache)
        data = self.fetcher.get_data(force_update=False)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('src.data_fetcher.LifeExpectancyDataFetcher')
    def test_get_life_expectancy_data(self, mock_fetcher_class):
        """Test get_life_expectancy_data convenience function."""
        # Mock the fetcher instance
        mock_fetcher = MagicMock()
        mock_data = pd.DataFrame({'Country': ['Spain'], 'Year': [2020], 'Life_Expectancy_Total': [82.0]})
        mock_fetcher.get_data.return_value = mock_data
        mock_fetcher_class.return_value = mock_fetcher
        
        # Call function
        result = get_life_expectancy_data(force_update=True)
        
        # Verify
        mock_fetcher_class.assert_called_once()
        mock_fetcher.get_data.assert_called_once_with(force_update=True)
        assert isinstance(result, pd.DataFrame)


@pytest.fixture
def sample_data():
    """Fixture providing sample life expectancy data."""
    return pd.DataFrame({
        'Country': ['Spain', 'Spain', 'Germany', 'Germany'],
        'Year': [2020, 2021, 2020, 2021],
        'Life_Expectancy_Total': [82.0, 82.2, 81.0, 81.1],
        'Life_Expectancy_Female': [85.0, 85.2, 84.0, 84.1],
        'Life_Expectancy_Male': [79.0, 79.2, 78.0, 78.1],
        'Region': ['Europe & Central Asia', 'Europe & Central Asia', 
                  'Europe & Central Asia', 'Europe & Central Asia']
    })


class TestDataQuality:
    """Test data quality and validation."""
    
    def test_data_completeness(self, sample_data):
        """Test that data has all required columns."""
        required_columns = ['Country', 'Year', 'Life_Expectancy_Total']
        
        for col in required_columns:
            assert col in sample_data.columns, f"Missing required column: {col}"
    
    def test_data_types(self, sample_data):
        """Test that data types are correct."""
        assert pd.api.types.is_numeric_dtype(sample_data['Year'])
        assert pd.api.types.is_numeric_dtype(sample_data['Life_Expectancy_Total'])
        assert pd.api.types.is_string_dtype(sample_data['Country'])
    
    def test_data_ranges(self, sample_data):
        """Test that data values are within expected ranges."""
        # Life expectancy should be reasonable
        assert sample_data['Life_Expectancy_Total'].min() > 30
        assert sample_data['Life_Expectancy_Total'].max() < 100
        
        # Years should be reasonable
        assert sample_data['Year'].min() >= 1960
        assert sample_data['Year'].max() <= datetime.now().year
    
    def test_no_negative_values(self, sample_data):
        """Test that there are no negative life expectancy values."""
        numeric_columns = ['Life_Expectancy_Total', 'Life_Expectancy_Female', 'Life_Expectancy_Male']
        
        for col in numeric_columns:
            if col in sample_data.columns:
                assert (sample_data[col] >= 0).all(), f"Negative values found in {col}"
    
    def test_gender_consistency(self, sample_data):
        """Test that gender data is consistent."""
        if all(col in sample_data.columns for col in ['Life_Expectancy_Female', 'Life_Expectancy_Male']):
            # Female life expectancy should generally be higher than male
            gender_diff = sample_data['Life_Expectancy_Female'] - sample_data['Life_Expectancy_Male']
            assert gender_diff.mean() > 0, "Female life expectancy should generally be higher than male"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
