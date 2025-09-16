"""
Unit tests for predictive models module.
Tests ML model training, prediction, and performance evaluation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predictive_models import LifeExpectancyPredictor, train_predictive_models


@pytest.fixture
def sample_data():
    """Fixture providing sample life expectancy data for testing."""
    np.random.seed(42)
    
    countries = ['Spain', 'Germany', 'France', 'Italy', 'Portugal']
    years = list(range(2000, 2021))
    
    data = []
    for country in countries:
        base_le = np.random.normal(80, 2)  # Base life expectancy
        for year in years:
            # Simulate realistic trends
            trend = (year - 2000) * 0.1  # Slight upward trend
            noise = np.random.normal(0, 0.5)
            
            life_exp_total = base_le + trend + noise
            life_exp_female = life_exp_total + np.random.normal(3, 0.5)
            life_exp_male = life_exp_total - np.random.normal(3, 0.5)
            
            data.append({
                'Country': country,
                'Year': year,
                'Life_Expectancy_Total': round(life_exp_total, 1),
                'Life_Expectancy_Female': round(life_exp_female, 1),
                'Life_Expectancy_Male': round(life_exp_male, 1),
                'Gender_Gap': round(life_exp_female - life_exp_male, 1),
                'Region': 'Europe & Central Asia'
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def predictor():
    """Fixture providing a LifeExpectancyPredictor instance."""
    return LifeExpectancyPredictor()


class TestLifeExpectancyPredictor:
    """Test cases for LifeExpectancyPredictor class."""
    
    def test_init(self, predictor):
        """Test predictor initialization."""
        assert isinstance(predictor, LifeExpectancyPredictor)
        assert len(predictor.available_models) > 0
        assert 'xgboost' in predictor.available_models
        assert 'lightgbm' in predictor.available_models
        assert 'prophet' in predictor.available_models
        
        assert isinstance(predictor.models, dict)
        assert isinstance(predictor.scalers, dict)
        assert isinstance(predictor.encoders, dict)
    
    def test_prepare_features(self, predictor, sample_data):
        """Test feature preparation."""
        prepared_data = predictor.prepare_features(sample_data)
        
        assert isinstance(prepared_data, pd.DataFrame)
        assert len(prepared_data) == len(sample_data)
        
        # Check that new features were created
        expected_features = [
            'Region_Encoded', 'Country_Encoded', 'Year_Normalized', 
            'Year_Squared', 'Life_Expectancy_Lag1'
        ]
        
        for feature in expected_features:
            assert feature in prepared_data.columns, f"Missing feature: {feature}"
        
        # Check encoders were created
        assert 'region_encoder' in predictor.encoders
        assert 'country_encoder' in predictor.encoders
    
    def test_prepare_features_data_types(self, predictor, sample_data):
        """Test that prepared features have correct data types."""
        prepared_data = predictor.prepare_features(sample_data)
        
        # Numeric features should be numeric
        numeric_features = ['Year_Normalized', 'Year_Squared', 'Region_Encoded', 'Country_Encoded']
        for feature in numeric_features:
            if feature in prepared_data.columns:
                assert pd.api.types.is_numeric_dtype(prepared_data[feature])
    
    def test_train_models_basic(self, predictor, sample_data):
        """Test basic model training without optimization."""
        # Use a subset for faster testing
        subset_data = sample_data.head(50)
        
        # Train without hyperparameter optimization for speed
        results = predictor.train_all_models(
            subset_data, 
            optimize_hp=False,
            test_size=0.3
        )
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that at least some models were trained
        expected_models = ['xgboost', 'lightgbm', 'random_forest']
        trained_models = list(results.keys())
        
        assert any(model in trained_models for model in expected_models)
        
        # Check performance metrics
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'mae' in metrics:
                assert 'mae' in metrics
                assert 'rmse' in metrics
                assert 'r2' in metrics
                assert metrics['mae'] >= 0
                assert metrics['rmse'] >= 0
                assert metrics['r2'] <= 1
    
    def test_evaluate_model(self, predictor):
        """Test model evaluation function."""
        # Create mock model and data
        class MockModel:
            def predict(self, X):
                return np.array([80.0, 81.0, 82.0])
        
        mock_model = MockModel()
        X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y_test = np.array([79.5, 81.2, 82.1])
        
        metrics = predictor.evaluate_model(mock_model, X_test, y_test, 'test_model')
        
        assert isinstance(metrics, dict)
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert metrics['model_name'] == 'test_model'
        
        # Check that metrics are reasonable
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mape'] >= 0
    
    def test_get_model_comparison_empty(self, predictor):
        """Test model comparison with no trained models."""
        with pytest.raises(ValueError, match="No models trained yet"):
            predictor.get_model_comparison()
    
    def test_get_feature_importance_not_available(self, predictor):
        """Test feature importance when not available."""
        with pytest.raises(ValueError, match="Feature importance not available"):
            predictor.get_feature_importance('nonexistent_model')
    
    def test_predict_future_no_model(self, predictor):
        """Test future prediction with untrained model."""
        with pytest.raises(ValueError, match="Model .* not trained yet"):
            predictor.predict_future('Spain', years_ahead=5, model_name='xgboost')
    
    @patch('src.predictive_models.Prophet')
    def test_train_prophet_model(self, mock_prophet_class, predictor, sample_data):
        """Test Prophet model training."""
        # Mock Prophet
        mock_prophet = MagicMock()
        mock_prophet_class.return_value = mock_prophet
        
        # Train Prophet model for a country
        country_data = sample_data[sample_data['Country'] == 'Spain']
        prophet_model = predictor.train_prophet_model(sample_data, 'Spain')
        
        # Verify Prophet was instantiated and fit was called
        mock_prophet_class.assert_called_once()
        mock_prophet.fit.assert_called_once()
        
        assert prophet_model == mock_prophet


class TestModelPerformance:
    """Test model performance and validation."""
    
    def test_model_accuracy_threshold(self, sample_data):
        """Test that models meet minimum accuracy thresholds."""
        predictor = LifeExpectancyPredictor()
        
        # Train models with a reasonable subset
        subset_data = sample_data.head(80)
        results = predictor.train_all_models(subset_data, optimize_hp=False)
        
        # Check that models meet basic accuracy requirements
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'mae' in metrics:
                # MAE should be less than 5 years for life expectancy prediction
                assert metrics['mae'] < 5.0, f"{model_name} MAE too high: {metrics['mae']}"
                
                # R² should be positive (model better than baseline)
                if 'r2' in metrics:
                    assert metrics['r2'] > 0, f"{model_name} R² too low: {metrics['r2']}"
    
    def test_prediction_consistency(self, sample_data):
        """Test that predictions are consistent and reasonable."""
        predictor = LifeExpectancyPredictor()
        
        # Train a simple model
        subset_data = sample_data.head(60)
        predictor.train_all_models(subset_data, optimize_hp=False)
        
        # Test predictions if models were trained
        if 'xgboost' in predictor.models:
            # Get features for prediction
            prepared_data = predictor.prepare_features(subset_data)
            feature_cols = [
                'Year', 'Year_Normalized', 'Year_Squared', 'Region_Encoded', 'Country_Encoded',
                'Life_Expectancy_Lag1', 'Life_Expectancy_Lag5', 'Life_Expectancy_MA3', 
                'Life_Expectancy_MA10', 'Historical_Trend', 'Trend_R_Squared',
                'Regional_Mean', 'Regional_Std', 'Country_vs_Regional'
            ]
            
            # Filter features that exist
            available_features = [col for col in feature_cols if col in prepared_data.columns]
            
            if len(available_features) > 0:
                X_sample = prepared_data[available_features].dropna().head(5)
                
                if len(X_sample) > 0:
                    predictions = predictor.models['xgboost'].predict(X_sample.values)
                    
                    # Predictions should be reasonable life expectancy values
                    assert all(50 < pred < 100 for pred in predictions), \
                        f"Unrealistic predictions: {predictions}"


class TestConvenienceFunctions:
    """Test convenience functions and integration."""
    
    @patch('src.predictive_models.LifeExpectancyPredictor')
    def test_train_predictive_models(self, mock_predictor_class, sample_data):
        """Test train_predictive_models convenience function."""
        # Mock the predictor
        mock_predictor = MagicMock()
        mock_predictor_class.return_value = mock_predictor
        
        # Call function
        result = train_predictive_models(sample_data)
        
        # Verify
        mock_predictor_class.assert_called_once()
        mock_predictor.train_all_models.assert_called_once()
        assert result == mock_predictor


class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_empty_data_handling(self, predictor):
        """Test handling of empty datasets."""
        empty_data = pd.DataFrame()
        
        with pytest.raises((ValueError, IndexError, KeyError)):
            predictor.train_all_models(empty_data)
    
    def test_missing_columns_handling(self, predictor):
        """Test handling of data with missing required columns."""
        incomplete_data = pd.DataFrame({
            'Country': ['Spain'],
            'Year': [2020]
            # Missing Life_Expectancy_Total
        })
        
        with pytest.raises((ValueError, KeyError)):
            predictor.train_all_models(incomplete_data)
    
    def test_invalid_data_types(self, predictor):
        """Test handling of invalid data types."""
        invalid_data = pd.DataFrame({
            'Country': ['Spain'],
            'Year': ['not_a_year'],  # Invalid year
            'Life_Expectancy_Total': ['not_a_number'],  # Invalid life expectancy
            'Region': ['Europe']
        })
        
        # This should either handle gracefully or raise an appropriate error
        with pytest.raises((ValueError, TypeError)):
            predictor.prepare_features(invalid_data)


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""
    
    def test_memory_efficiency(self, sample_data):
        """Test that models don't consume excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Train models
        predictor = LifeExpectancyPredictor()
        predictor.train_all_models(sample_data.head(50), optimize_hp=False)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500 * 1024 * 1024, \
            f"Excessive memory usage: {memory_increase / 1024 / 1024:.1f}MB"
    
    def test_training_time(self, sample_data):
        """Test that model training completes in reasonable time."""
        import time
        
        start_time = time.time()
        
        predictor = LifeExpectancyPredictor()
        predictor.train_all_models(sample_data.head(50), optimize_hp=False)
        
        training_time = time.time() - start_time
        
        # Training should complete in less than 60 seconds for small dataset
        assert training_time < 60, f"Training too slow: {training_time:.1f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
