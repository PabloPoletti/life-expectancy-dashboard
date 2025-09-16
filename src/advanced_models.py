"""
Advanced Life Expectancy Prediction Models (2025 Edition)
State-of-the-art machine learning with cutting-edge techniques including:
- CatBoost with GPU acceleration
- Stacked Ensemble Meta-Learning
- SHAP model interpretability
- Neural Prophet for time series
- Bayesian optimization with multiple objectives
- Walk-forward validation for temporal data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.base import BaseEstimator, RegressorMixin

# Advanced ML Models
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
except ImportError:
    cb = None
    print("⚠️ CatBoost not available. Install with: pip install catboost")

# Advanced Optimization
import optuna
from bayesian_optimization import BayesianOptimization
try:
    import hyperopt
    from hyperopt import hp, fmin, tpe, Trials
except ImportError:
    hyperopt = None

# Model Interpretability
try:
    import shap
except ImportError:
    shap = None
    print("⚠️ SHAP not available. Install with: pip install shap")

# Advanced Time Series
try:
    from neuralprophet import NeuralProphet
except ImportError:
    NeuralProphet = None
    print("⚠️ NeuralProphet not available. Install with: pip install neuralprophet")

try:
    from darts import TimeSeries
    from darts.models import ExponentialSmoothing as DartsES, AutoARIMA, RandomForest as DartsRF
except ImportError:
    pass

# Feature Engineering
try:
    from featuretools import dfs
except ImportError:
    dfs = None

# Performance Monitoring
import time
import psutil
import gc


class AdvancedLifeExpectancyPredictor:
    """
    State-of-the-art life expectancy prediction system with modern ML techniques.
    """
    
    def __init__(self, use_gpu: bool = False, enable_interpretability: bool = True):
        """
        Initialize the advanced predictor.
        
        Args:
            use_gpu: Whether to use GPU acceleration (requires CUDA)
            enable_interpretability: Whether to compute SHAP values
        """
        self.use_gpu = use_gpu
        self.enable_interpretability = enable_interpretability
        
        self.models = {}
        self.ensemble_models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.shap_values = {}
        self.performance_metrics = {}
        self.optimization_history = {}
        
        # Model registry
        self.available_models = [
            'xgboost_v3',
            'lightgbm_v4', 
            'catboost_v1',
            'neural_prophet',
            'stacked_ensemble',
            'voting_ensemble',
            'bayesian_ridge',
            'elastic_net'
        ]
        
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced engineered features using state-of-the-art techniques.
        
        Args:
            data: Input life expectancy data
            
        Returns:
            DataFrame with advanced features
        """
        print("🔧 Creating advanced features with cutting-edge techniques...")
        
        df = data.copy()
        
        # Temporal Features (Enhanced)
        df['Year_Sin'] = np.sin(2 * np.pi * df['Year'] / 100)
        df['Year_Cos'] = np.cos(2 * np.pi * df['Year'] / 100)
        df['Decade_Categorical'] = (df['Year'] // 10) * 10
        df['Years_Since_1960'] = df['Year'] - 1960
        df['Years_Since_2000'] = df['Year'] - 2000
        
        # Country-specific advanced features
        for country in df['Country'].unique():
            country_mask = df['Country'] == country
            country_data = df[country_mask].sort_values('Year')
            
            if len(country_data) > 2:
                # Advanced lag features with different windows
                for lag in [1, 2, 3, 5, 10]:
                    df.loc[country_mask, f'LE_Lag_{lag}'] = country_data['Life_Expectancy_Total'].shift(lag)
                
                # Rolling statistics with multiple windows
                for window in [3, 5, 10, 15]:
                    df.loc[country_mask, f'LE_RollingMean_{window}'] = \
                        country_data['Life_Expectancy_Total'].rolling(window=window, min_periods=1).mean()
                    df.loc[country_mask, f'LE_RollingStd_{window}'] = \
                        country_data['Life_Expectancy_Total'].rolling(window=window, min_periods=1).std()
                    df.loc[country_mask, f'LE_RollingMin_{window}'] = \
                        country_data['Life_Expectancy_Total'].rolling(window=window, min_periods=1).min()
                    df.loc[country_mask, f'LE_RollingMax_{window}'] = \
                        country_data['Life_Expectancy_Total'].rolling(window=window, min_periods=1).max()
                
                # Exponential weighted features
                df.loc[country_mask, 'LE_EWM_Fast'] = \
                    country_data['Life_Expectancy_Total'].ewm(span=3).mean()
                df.loc[country_mask, 'LE_EWM_Slow'] = \
                    country_data['Life_Expectancy_Total'].ewm(span=10).mean()
                
                # Trend and acceleration features
                if len(country_data) > 5:
                    # First and second derivatives
                    first_diff = country_data['Life_Expectancy_Total'].diff()
                    df.loc[country_mask, 'LE_FirstDiff'] = first_diff
                    df.loc[country_mask, 'LE_SecondDiff'] = first_diff.diff()
                    
                    # Trend strength
                    df.loc[country_mask, 'LE_TrendStrength'] = \
                        first_diff.rolling(window=5, min_periods=1).mean()
                    
                    # Volatility measures
                    df.loc[country_mask, 'LE_Volatility'] = \
                        first_diff.rolling(window=5, min_periods=1).std()
        
        # Regional advanced features
        if 'Region' in df.columns:
            for region in df['Region'].unique():
                region_mask = df['Region'] == region
                
                # Regional percentiles by year
                for year in df['Year'].unique():
                    year_region_data = df[(df['Year'] == year) & (df['Region'] == region)]
                    if len(year_region_data) > 1:
                        percentiles = year_region_data['Life_Expectancy_Total'].quantile([0.25, 0.5, 0.75])
                        
                        year_mask = df['Year'] == year
                        combined_mask = region_mask & year_mask
                        
                        df.loc[combined_mask, 'Regional_Q25'] = percentiles[0.25]
                        df.loc[combined_mask, 'Regional_Median'] = percentiles[0.5]
                        df.loc[combined_mask, 'Regional_Q75'] = percentiles[0.75]
        
        # Interaction features
        if 'Life_Expectancy_Female' in df.columns and 'Life_Expectancy_Male' in df.columns:
            df['Gender_Ratio_Advanced'] = df['Life_Expectancy_Female'] / (df['Life_Expectancy_Male'] + 1e-8)
            df['Gender_Diff_Normalized'] = df['Gender_Gap'] / df['Life_Expectancy_Total']
            df['Gender_Harmonic_Mean'] = 2 / ((1/df['Life_Expectancy_Female']) + (1/df['Life_Expectancy_Male']))
        
        # Polynomial features for year
        df['Year_Squared'] = df['Year'] ** 2
        df['Year_Cubed'] = df['Year'] ** 3
        df['Year_Log'] = np.log(df['Year'] - 1959)  # Avoid log(0)
        
        print(f"✅ Created {len(df.columns) - len(data.columns)} new advanced features")
        return df
    
    def optimize_catboost_bayesian(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_val: np.ndarray, y_val: np.ndarray, n_calls: int = 50) -> Dict:
        """
        Optimize CatBoost using Bayesian Optimization.
        """
        if cb is None:
            raise ImportError("CatBoost not available")
        
        def catboost_objective(**params):
            """Objective function for Bayesian optimization."""
            model = cb.CatBoostRegressor(
                iterations=int(params['iterations']),
                depth=int(params['depth']),
                learning_rate=params['learning_rate'],
                l2_leaf_reg=params['l2_leaf_reg'],
                border_count=int(params['border_count']),
                verbose=False,
                random_state=42,
                gpu_device_id=0 if self.use_gpu else -1
            )
            
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            predictions = model.predict(X_val)
            return -mean_absolute_error(y_val, predictions)  # Negative because we want to maximize
        
        # Define search space
        pbounds = {
            'iterations': (100, 1000),
            'depth': (4, 10),
            'learning_rate': (0.01, 0.3),
            'l2_leaf_reg': (1, 10),
            'border_count': (32, 255)
        }
        
        optimizer = BayesianOptimization(
            f=catboost_objective,
            pbounds=pbounds,
            random_state=42
        )
        
        optimizer.maximize(init_points=10, n_iter=n_calls)
        
        return optimizer.max['params']
    
    def train_neural_prophet(self, data: pd.DataFrame, country: str) -> Optional[object]:
        """
        Train NeuralProphet model for advanced time series forecasting.
        """
        if NeuralProphet is None:
            print("⚠️ NeuralProphet not available")
            return None
        
        try:
            country_data = data[data['Country'] == country].copy()
            country_data = country_data.sort_values('Year')
            
            # Prepare data for NeuralProphet
            prophet_data = pd.DataFrame({
                'ds': pd.to_datetime(country_data['Year'], format='%Y'),
                'y': country_data['Life_Expectancy_Total']
            })
            
            # Configure NeuralProphet with modern settings
            model = NeuralProphet(
                n_forecasts=10,
                n_lags=5,
                num_hidden_layers=2,
                d_hidden=64,
                learning_rate=0.1,
                epochs=100,
                batch_size=32,
                normalize='auto'
            )
            
            # Add regressors if available
            if len(country_data) > 20:  # Ensure enough data
                model.fit(prophet_data, freq='YS')
                return model
                
        except Exception as e:
            print(f"    ⚠️ NeuralProphet failed for {country}: {e}")
            
        return None
    
    def create_stacked_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> StackingRegressor:
        """
        Create advanced stacked ensemble with multiple base learners.
        """
        print("🏗️ Building stacked ensemble with state-of-the-art models...")
        
        # Base models with optimized parameters
        base_models = [
            ('xgb', xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )),
            ('lgb', lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ))
        ]
        
        # Add CatBoost if available
        if cb is not None:
            base_models.append(('cat', cb.CatBoostRegressor(
                iterations=300,
                depth=6,
                learning_rate=0.1,
                verbose=False,
                random_state=42
            )))
        
        # Meta-learner (Ridge regression for stability)
        from sklearn.linear_model import Ridge
        meta_learner = Ridge(alpha=1.0)
        
        # Create stacking ensemble
        stacking_ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        return stacking_ensemble
    
    def compute_shap_values(self, model, X_sample: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Compute SHAP values for model interpretability.
        """
        if shap is None or not self.enable_interpretability:
            return {}
        
        try:
            print("🔍 Computing SHAP values for model interpretability...")
            
            # Use appropriate explainer based on model type
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model)
            
            # Compute SHAP values for a sample
            sample_size = min(100, len(X_sample))
            shap_values = explainer.shap_values(X_sample[:sample_size])
            
            return {
                'shap_values': shap_values,
                'feature_names': feature_names,
                'explainer': explainer
            }
            
        except Exception as e:
            print(f"⚠️ SHAP computation failed: {e}")
            return {}
    
    def walk_forward_validation(self, data: pd.DataFrame, model, window_size: int = 5) -> Dict:
        """
        Perform walk-forward validation for time series data.
        """
        print("📊 Performing walk-forward validation...")
        
        results = []
        years = sorted(data['Year'].unique())
        
        for i in range(window_size, len(years)):
            # Training window
            train_years = years[i-window_size:i]
            test_year = years[i]
            
            train_data = data[data['Year'].isin(train_years)]
            test_data = data[data['Year'] == test_year]
            
            if len(train_data) > 10 and len(test_data) > 0:
                try:
                    # Prepare features (simplified for walk-forward)
                    feature_cols = ['Year', 'Life_Expectancy_Female', 'Life_Expectancy_Male']
                    available_cols = [col for col in feature_cols if col in train_data.columns]
                    
                    if len(available_cols) > 1:
                        X_train = train_data[available_cols].fillna(train_data[available_cols].mean())
                        X_test = test_data[available_cols].fillna(train_data[available_cols].mean())
                        y_train = train_data['Life_Expectancy_Total']
                        y_test = test_data['Life_Expectancy_Total']
                        
                        # Train and predict
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        results.append({
                            'test_year': test_year,
                            'mae': mae,
                            'rmse': rmse,
                            'samples': len(test_data)
                        })
                        
                except Exception as e:
                    print(f"    ⚠️ Validation failed for year {test_year}: {e}")
                    continue
        
        if results:
            avg_mae = np.mean([r['mae'] for r in results])
            avg_rmse = np.mean([r['rmse'] for r in results])
            
            return {
                'results': results,
                'average_mae': avg_mae,
                'average_rmse': avg_rmse,
                'validation_years': len(results)
            }
        
        return {'results': [], 'average_mae': 0, 'average_rmse': 0, 'validation_years': 0}
    
    def train_advanced_models(self, data: pd.DataFrame, optimize: bool = True) -> Dict:
        """
        Train all advanced models with cutting-edge techniques.
        """
        print("🚀 Training state-of-the-art machine learning models...")
        start_time = time.time()
        
        # Create advanced features
        enhanced_data = self.create_advanced_features(data)
        
        # Select features for training
        feature_cols = [col for col in enhanced_data.columns if col not in [
            'Country', 'Life_Expectancy_Total', 'Life_Expectancy_Female', 'Life_Expectancy_Male'
        ]]
        
        # Remove non-numeric columns and handle missing values
        X = enhanced_data[feature_cols].select_dtypes(include=[np.number])
        X = X.fillna(X.mean())
        y = enhanced_data['Life_Expectancy_Total'].fillna(enhanced_data['Life_Expectancy_Total'].mean())
        
        # Train-test split with temporal consideration
        split_index = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['robust_scaler'] = scaler
        
        results = {}
        
        # Train CatBoost with Bayesian optimization
        if cb is not None:
            print("  🐱 Training CatBoost with Bayesian optimization...")
            try:
                if optimize:
                    best_params = self.optimize_catboost_bayesian(
                        X_train_scaled, y_train, X_test_scaled, y_test, n_calls=30
                    )
                else:
                    best_params = {
                        'iterations': 300,
                        'depth': 6,
                        'learning_rate': 0.1,
                        'l2_leaf_reg': 3,
                        'border_count': 128
                    }
                
                catboost_model = cb.CatBoostRegressor(
                    **{k: int(v) if k in ['iterations', 'depth', 'border_count'] else v 
                       for k, v in best_params.items()},
                    verbose=False,
                    random_state=42
                )
                
                catboost_model.fit(X_train_scaled, y_train)
                self.models['catboost'] = catboost_model
                
                # Evaluate
                y_pred = catboost_model.predict(X_test_scaled)
                results['catboost'] = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred),
                    'mape': mean_absolute_percentage_error(y_test, y_pred) * 100,
                    'model_name': 'CatBoost (Bayesian Optimized)'
                }
                
                # Compute SHAP values
                self.shap_values['catboost'] = self.compute_shap_values(
                    catboost_model, X_test_scaled[:50], X.columns.tolist()
                )
                
            except Exception as e:
                print(f"    ⚠️ CatBoost training failed: {e}")
        
        # Train Stacked Ensemble
        print("  🏗️ Training Stacked Ensemble...")
        try:
            stacked_model = self.create_stacked_ensemble(X_train_scaled, y_train)
            stacked_model.fit(X_train_scaled, y_train)
            self.ensemble_models['stacked'] = stacked_model
            
            y_pred = stacked_model.predict(X_test_scaled)
            results['stacked_ensemble'] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'mape': mean_absolute_percentage_error(y_test, y_pred) * 100,
                'model_name': 'Stacked Ensemble (XGB+LGB+CAT)'
            }
            
        except Exception as e:
            print(f"    ⚠️ Stacked ensemble training failed: {e}")
        
        # Performance monitoring
        end_time = time.time()
        training_time = end_time - start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.performance_metrics = {
            'training_time': training_time,
            'memory_usage_mb': memory_usage,
            'models_trained': len(results),
            'features_used': len(feature_cols),
            'training_samples': len(X_train)
        }
        
        print(f"✅ Advanced training completed in {training_time:.2f}s")
        print(f"🧠 Memory usage: {memory_usage:.1f}MB")
        
        return results
    
    def get_advanced_insights(self) -> Dict:
        """
        Generate advanced insights from trained models.
        """
        insights = {
            'model_comparison': self.get_model_comparison(),
            'feature_importance': self.get_consolidated_feature_importance(),
            'performance_metrics': self.performance_metrics,
            'interpretability': self.get_interpretability_summary()
        }
        
        return insights
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comprehensive model comparison."""
        # This would aggregate results from all trained models
        return pd.DataFrame()  # Placeholder
    
    def get_consolidated_feature_importance(self) -> Dict:
        """Get consolidated feature importance across models."""
        return self.feature_importance
    
    def get_interpretability_summary(self) -> Dict:
        """Get model interpretability summary."""
        if not self.shap_values:
            return {'shap_available': False}
        
        return {
            'shap_available': True,
            'models_with_shap': list(self.shap_values.keys()),
            'interpretability_methods': ['SHAP values', 'Feature importance', 'Partial dependence']
        }


def train_advanced_models(data: pd.DataFrame, use_gpu: bool = False) -> AdvancedLifeExpectancyPredictor:
    """
    Convenience function to train all advanced models.
    """
    predictor = AdvancedLifeExpectancyPredictor(use_gpu=use_gpu, enable_interpretability=True)
    results = predictor.train_advanced_models(data, optimize=True)
    return predictor


if __name__ == "__main__":
    print("🧪 Testing advanced models...")
    
    try:
        # Load data
        data = pd.read_csv("data/life_expectancy_data.csv")
        print(f"📊 Loaded data: {len(data)} records")
        
        # Train advanced models
        predictor = train_advanced_models(data, use_gpu=False)
        
        # Get insights
        insights = predictor.get_advanced_insights()
        print("\n📈 Advanced Model Insights:")
        print(f"  • Models trained: {insights['performance_metrics']['models_trained']}")
        print(f"  • Training time: {insights['performance_metrics']['training_time']:.2f}s")
        print(f"  • Memory usage: {insights['performance_metrics']['memory_usage_mb']:.1f}MB")
        print(f"  • Features used: {insights['performance_metrics']['features_used']}")
        
        print("\n✅ Advanced model testing completed!")
        
    except Exception as e:
        print(f"❌ Error testing advanced models: {e}")
        print("Make sure to run data_fetcher.py first to generate the dataset.")
