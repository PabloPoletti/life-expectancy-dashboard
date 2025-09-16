# 🌍 Life Expectancy Dashboard

> **Advanced Analytics and Predictions for Global Life Expectancy Data**

A modern, interactive dashboard built with Streamlit that provides comprehensive insights into global life expectancy trends, featuring state-of-the-art machine learning models for predictive analytics.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20Prophet-green?style=for-the-badge)

## ✨ Features

### 📊 **Interactive Dashboard**
- **Global Overview**: Comprehensive trends and statistics for worldwide life expectancy
- **Country Analysis**: Detailed analysis for individual countries with historical trends
- **Regional Comparison**: Compare life expectancy across different world regions
- **Gender Analysis**: Explore gender gaps in life expectancy

### 🤖 **Advanced Machine Learning**
- **XGBoost**: Gradient boosting for high-accuracy predictions
- **LightGBM**: Fast and efficient gradient boosting
- **Random Forest**: Ensemble learning for robust predictions  
- **Prophet**: Facebook's time series forecasting for trend analysis
- **Automated Hyperparameter Optimization**: Using Optuna for optimal model performance

### 📈 **Predictive Analytics**
- **Future Predictions**: Forecast life expectancy up to 20 years ahead
- **Confidence Intervals**: Statistical uncertainty quantification
- **Model Comparison**: Performance metrics for all ML models
- **Feature Importance**: Understanding what drives life expectancy changes

### 🔧 **Modern Technology Stack**
- **Data Source**: World Bank API for real-time data
- **Caching**: Intelligent data caching to minimize API calls
- **Responsive Design**: Works on desktop and mobile devices
- **Export Capabilities**: Download filtered data and visualizations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/life-expectancy-dashboard.git
   cd life-expectancy-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the dashboard**
   ```bash
   python run_dashboard.py
   ```

   Or alternatively:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   The dashboard will automatically open at `http://localhost:8501`

## 📖 Usage Guide

### 🏠 Overview Page
- View global life expectancy trends
- See key metrics and statistics
- Compare regions and top/bottom performing countries

### 🌍 Country Analysis
- Select up to 5 countries for comparison
- Analyze historical trends and patterns
- View detailed gender-specific data

### 🔮 Predictions
- Choose any country for future predictions
- Set prediction horizon (1-20 years)
- View confidence intervals and model accuracy

### 🤖 Model Performance
- Compare different ML algorithms
- View feature importance rankings
- Understand model accuracy metrics

### 📊 Data Explorer
- Filter and explore raw data
- Download customized datasets
- View statistical summaries

## 🛠️ Project Structure

```
life-expectancy-dashboard/
├── 📱 app.py                 # Main Streamlit application
├── 🚀 run_dashboard.py       # Convenient launcher script
├── 📋 requirements.txt       # Python dependencies
├── 📊 data/                  # Data storage directory
│   ├── life_expectancy_data.csv
│   └── data_metadata.json
├── 🔧 src/                   # Source code modules
│   ├── data_fetcher.py       # Data acquisition and processing
│   └── predictive_models.py  # ML models and training
├── 🛠️ utils/                 # Utility functions
│   └── data_analysis.py      # Analysis and statistics
├── 🧪 tests/                 # Unit tests
├── ⚙️ .streamlit/            # Streamlit configuration
│   ├── config.toml
│   └── secrets.toml
└── 📚 README.md              # This file
```

## 🔬 Machine Learning Models

### XGBoost Regressor
- **Use Case**: High-accuracy predictions with complex feature interactions
- **Optimization**: Automated hyperparameter tuning with Optuna
- **Features**: Supports early stopping and cross-validation

### LightGBM Regressor  
- **Use Case**: Fast training with excellent performance
- **Advantages**: Memory efficient, handles large datasets well
- **Optimization**: Bayesian optimization for best parameters

### Random Forest
- **Use Case**: Robust ensemble predictions
- **Benefits**: Good baseline model, feature importance insights
- **Configuration**: 200 trees with optimized depth

### Prophet
- **Use Case**: Time series forecasting with trend analysis
- **Strengths**: Handles seasonality and trend changes
- **Features**: Confidence intervals and anomaly detection

## 📊 Data Sources

### Primary Source: World Bank API
- **Indicators**: Life expectancy at birth (total, male, female)
- **Coverage**: Global data from 1960 to present
- **Update Frequency**: Annual updates
- **Reliability**: Official government statistics

### Data Processing
- **Automatic Updates**: Smart caching with configurable refresh intervals
- **Data Validation**: Quality checks and outlier detection
- **Feature Engineering**: Creates derived metrics and trend indicators
- **Missing Data**: Intelligent handling of gaps in historical data

## ⚙️ Configuration

### Environment Variables
```bash
# Optional: Customize data update frequency
STREAMLIT_DATA_UPDATE_HOURS=24

# Optional: Set cache TTL
STREAMLIT_CACHE_TTL=3600
```

### Streamlit Configuration
Edit `.streamlit/config.toml` to customize:
- **Theme colors**
- **Server settings** 
- **Performance options**

## 🔧 Development

### Adding New Models
1. Implement your model in `src/predictive_models.py`
2. Add to the `LifeExpectancyPredictor` class
3. Update the training pipeline
4. Add performance evaluation

### Custom Data Sources
1. Extend the `LifeExpectancyDataFetcher` class
2. Implement new data acquisition methods
3. Ensure data format compatibility

### Testing
```bash
# Run data fetcher tests
python src/data_fetcher.py

# Run predictive models tests  
python src/predictive_models.py

# Run full application
python run_dashboard.py
```

## 📈 Performance Metrics

The dashboard tracks several key performance indicators:

- **Mean Absolute Error (MAE)**: Average prediction error
- **Root Mean Square Error (RMSE)**: Penalizes large errors
- **R² Score**: Variance explained by the model
- **Mean Absolute Percentage Error (MAPE)**: Relative error percentage

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **World Bank** for providing comprehensive global health data
- **Streamlit** team for the excellent web app framework
- **Prophet** team at Facebook for time series forecasting
- **XGBoost** and **LightGBM** communities for powerful ML libraries
- **Plotly** for interactive visualization capabilities

## 📞 Support

- **Documentation**: Check the inline help and tooltips
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join community discussions
- **Email**: Contact the development team

---

<div align="center">
  <strong>Built with ❤️ for global health analytics</strong>
  <br>
  <em>Making life expectancy data accessible and actionable</em>
</div>
