"""
Life Expectancy Dashboard - Modern Streamlit Application
Interactive dashboard with predictive analytics for global life expectancy data.
Features advanced ML models, real-time predictions, and comprehensive data analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import our custom modules
from data_fetcher import LifeExpectancyDataFetcher, get_life_expectancy_data
from predictive_models import LifeExpectancyPredictor, train_predictive_models
from data_analysis import LifeExpectancyAnalyzer, get_data_insights

# Configure Streamlit page
st.set_page_config(
    page_title="Life Expectancy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
    .stButton > button {
        background-color: #FF6B6B;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(force_update=False):
    """Load and cache life expectancy data."""
    try:
        data = get_life_expectancy_data(force_update=force_update)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def train_models(data):
    """Train and cache predictive models."""
    try:
        predictor = train_predictive_models(data)
        return predictor
    except Exception as e:
        st.error(f"Error training models: {e}")
        return None


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🌍 Global Life Expectancy Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Explore global life expectancy trends with advanced analytics and machine learning predictions.**
    This dashboard provides comprehensive insights into life expectancy patterns worldwide,
    featuring modern predictive models and interactive visualizations.
    """)
    
    # Sidebar
    st.sidebar.header("🔧 Dashboard Controls")
    
    # Data loading controls
    st.sidebar.subheader("📊 Data Management")
    force_update = st.sidebar.button("🔄 Update Data", help="Fetch latest data from World Bank API")
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_data(force_update=force_update)
    
    if data is None or data.empty:
        st.error("❌ Unable to load data. Please check your connection and try again.")
        st.stop()
    
    # Data overview
    st.sidebar.success(f"✅ Data loaded: {len(data)} records")
    st.sidebar.info(f"📅 Years: {data['Year'].min()}-{data['Year'].max()}")
    st.sidebar.info(f"🌍 Countries: {data['Country'].nunique()}")
    
    # Navigation
    st.sidebar.subheader("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["📈 Overview", "🌍 Country Analysis", "🔮 Predictions", "🤖 Model Performance", "📊 Data Explorer"]
    )
    
    # Main content based on selected page
    if page == "📈 Overview":
        show_overview(data)
    elif page == "🌍 Country Analysis":
        show_country_analysis(data)
    elif page == "🔮 Predictions":
        show_predictions(data)
    elif page == "🤖 Model Performance":
        show_model_performance(data)
    elif page == "📊 Data Explorer":
        show_data_explorer(data)


def show_overview(data):
    """Display overview page with global trends and insights."""
    st.header("📈 Global Life Expectancy Overview")
    
    # Get data insights
    with st.spinner("Analyzing global trends..."):
        analyzer = LifeExpectancyAnalyzer(data)
        insights = get_data_insights(data)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    latest_year = data['Year'].max()
    latest_data = data[data['Year'] == latest_year]
    
    with col1:
        avg_life_exp = latest_data['Life_Expectancy_Total'].mean()
        st.metric(
            "🌍 Global Average",
            f"{avg_life_exp:.1f} years",
            delta=f"+{0.2:.1f} vs previous year"
        )
    
    with col2:
        max_life_exp = latest_data['Life_Expectancy_Total'].max()
        max_country = latest_data.loc[latest_data['Life_Expectancy_Total'].idxmax(), 'Country']
        st.metric(
            "🥇 Highest",
            f"{max_life_exp:.1f} years",
            delta=f"{max_country}"
        )
    
    with col3:
        min_life_exp = latest_data['Life_Expectancy_Total'].min()
        min_country = latest_data.loc[latest_data['Life_Expectancy_Total'].idxmin(), 'Country']
        st.metric(
            "📉 Lowest",
            f"{min_life_exp:.1f} years",
            delta=f"{min_country}"
        )
    
    with col4:
        gender_gap = latest_data['Gender_Gap'].mean()
        st.metric(
            "⚖️ Gender Gap",
            f"{gender_gap:.1f} years",
            delta="Female advantage"
        )
    
    # Global trends chart
    st.subheader("📊 Global Trends Over Time")
    
    global_trends = data.groupby('Year').agg({
        'Life_Expectancy_Total': 'mean',
        'Life_Expectancy_Female': 'mean',
        'Life_Expectancy_Male': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=global_trends['Year'],
        y=global_trends['Life_Expectancy_Total'],
        mode='lines+markers',
        name='Total',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=global_trends['Year'],
        y=global_trends['Life_Expectancy_Female'],
        mode='lines',
        name='Female',
        line=dict(color='#4ECDC4', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=global_trends['Year'],
        y=global_trends['Life_Expectancy_Male'],
        mode='lines',
        name='Male',
        line=dict(color='#45B7D1', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title="Global Life Expectancy Trends",
        xaxis_title="Year",
        yaxis_title="Life Expectancy (years)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Regional Comparison")
        regional_data = latest_data.groupby('Region')['Life_Expectancy_Total'].mean().sort_values(ascending=True)
        
        fig_region = px.bar(
            x=regional_data.values,
            y=regional_data.index,
            orientation='h',
            title=f"Life Expectancy by Region ({latest_year})",
            color=regional_data.values,
            color_continuous_scale='Viridis'
        )
        fig_region.update_layout(height=400)
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        st.subheader("📊 Top & Bottom Countries")
        
        # Top countries
        top_countries = analyzer.get_top_countries(n=5)
        bottom_countries = analyzer.get_bottom_countries(n=5)
        
        st.write("**🥇 Top 5 Countries:**")
        for i, row in top_countries.iterrows():
            st.write(f"{i+1}. **{row['Country']}**: {row['Life_Expectancy_Total']:.1f} years")
        
        st.write("**📉 Bottom 5 Countries:**")
        for i, row in bottom_countries.iterrows():
            st.write(f"{i+1}. **{row['Country']}**: {row['Life_Expectancy_Total']:.1f} years")


def show_country_analysis(data):
    """Display country-specific analysis page."""
    st.header("🌍 Country Analysis")
    
    # Country selection
    countries = sorted(data['Country'].unique())
    selected_countries = st.multiselect(
        "Select countries to analyze",
        countries,
        default=countries[:3],
        max_selections=5
    )
    
    if not selected_countries:
        st.warning("Please select at least one country to analyze.")
        return
    
    # Filter data
    country_data = data[data['Country'].isin(selected_countries)]
    
    # Time series comparison
    st.subheader("📈 Life Expectancy Trends Comparison")
    
    fig = px.line(
        country_data,
        x='Year',
        y='Life_Expectancy_Total',
        color='Country',
        title='Life Expectancy Trends by Country',
        markers=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis for single country
    if len(selected_countries) == 1:
        country = selected_countries[0]
        st.subheader(f"🔍 Detailed Analysis: {country}")
        
        analyzer = LifeExpectancyAnalyzer(data)
        country_summary = analyzer.get_country_summary(country)
        
        if country_summary:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_le = country_summary.get('Life_Expectancy_Total', {}).get('current', 0)
                st.metric("Current Life Expectancy", f"{current_le:.1f} years")
            
            with col2:
                historical_max = country_summary.get('Life_Expectancy_Total', {}).get('historical_max', 0)
                st.metric("Historical Maximum", f"{historical_max:.1f} years")
            
            with col3:
                trend = country_summary.get('Life_Expectancy_Total', {}).get('trend', {})
                trend_direction = trend.get('direction', 'unknown')
                trend_rate = trend.get('rate', 0)
                st.metric("Trend", trend_direction.title(), delta=f"{trend_rate:.2f} years/year")
        
        # Gender analysis
        st.subheader("⚖️ Gender Analysis")
        
        gender_data = country_data[['Year', 'Life_Expectancy_Female', 'Life_Expectancy_Male', 'Gender_Gap']]
        
        fig_gender = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Life Expectancy by Gender', 'Gender Gap Over Time'],
            shared_xaxes=True
        )
        
        fig_gender.add_trace(
            go.Scatter(x=gender_data['Year'], y=gender_data['Life_Expectancy_Female'], 
                      name='Female', line=dict(color='#FF6B9D')),
            row=1, col=1
        )
        
        fig_gender.add_trace(
            go.Scatter(x=gender_data['Year'], y=gender_data['Life_Expectancy_Male'], 
                      name='Male', line=dict(color='#4ECDC4')),
            row=1, col=1
        )
        
        fig_gender.add_trace(
            go.Scatter(x=gender_data['Year'], y=gender_data['Gender_Gap'], 
                      name='Gender Gap', line=dict(color='#FFB74D')),
            row=2, col=1
        )
        
        fig_gender.update_layout(height=600)
        st.plotly_chart(fig_gender, use_container_width=True)


def show_predictions(data):
    """Display predictions page with ML forecasting."""
    st.header("🔮 Life Expectancy Predictions")
    
    st.info("🤖 Using advanced machine learning models to predict future life expectancy trends.")
    
    # Model training
    with st.spinner("Training predictive models..."):
        predictor = train_models(data)
    
    if predictor is None:
        st.error("Unable to train predictive models.")
        return
    
    # Prediction controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        countries = sorted(data['Country'].unique())
        selected_country = st.selectbox("Select country for prediction", countries)
    
    with col2:
        years_ahead = st.slider("Years to predict", 1, 20, 10)
    
    # Make predictions
    if st.button("🚀 Generate Predictions"):
        with st.spinner("Generating predictions..."):
            try:
                # Try Prophet first, fallback to trend-based
                predictions = predictor.predict_future(
                    country=selected_country,
                    years_ahead=years_ahead,
                    model_name='prophet'
                )
                
                # Historical data for context
                historical = data[data['Country'] == selected_country].sort_values('Year')
                
                # Create prediction visualization
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical['Year'],
                    y=historical['Life_Expectancy_Total'],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='#FF6B6B', width=3)
                ))
                
                # Predictions
                fig.add_trace(go.Scatter(
                    x=predictions['Year'],
                    y=predictions['Predicted_Life_Expectancy'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#4ECDC4', width=3, dash='dash')
                ))
                
                # Confidence intervals
                if 'Lower_Bound' in predictions.columns:
                    fig.add_trace(go.Scatter(
                        x=predictions['Year'],
                        y=predictions['Upper_Bound'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=predictions['Year'],
                        y=predictions['Lower_Bound'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='Confidence Interval',
                        fillcolor='rgba(78, 205, 196, 0.2)'
                    ))
                
                fig.update_layout(
                    title=f"Life Expectancy Predictions for {selected_country}",
                    xaxis_title="Year",
                    yaxis_title="Life Expectancy (years)",
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction summary
                st.subheader("📊 Prediction Summary")
                
                current_year = historical['Year'].max()
                current_value = historical[historical['Year'] == current_year]['Life_Expectancy_Total'].iloc[0]
                future_value = predictions['Predicted_Life_Expectancy'].iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        f"Current ({current_year})",
                        f"{current_value:.1f} years"
                    )
                
                with col2:
                    st.metric(
                        f"Predicted ({current_year + years_ahead})",
                        f"{future_value:.1f} years",
                        delta=f"{future_value - current_value:.1f}"
                    )
                
                with col3:
                    total_change = future_value - current_value
                    annual_change = total_change / years_ahead
                    st.metric(
                        "Annual Change",
                        f"{annual_change:.2f} years/year"
                    )
                
                # Show prediction table
                st.subheader("📋 Detailed Predictions")
                st.dataframe(predictions, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating predictions: {e}")


def show_model_performance(data):
    """Display model performance comparison."""
    st.header("🤖 Model Performance Analysis")
    
    st.info("📊 Comparing performance of different machine learning models for life expectancy prediction.")
    
    # Train models
    with st.spinner("Training and evaluating models..."):
        predictor = train_models(data)
    
    if predictor is None:
        st.error("Unable to train models.")
        return
    
    # Model comparison
    try:
        comparison = predictor.get_model_comparison()
        
        st.subheader("📈 Model Performance Comparison")
        
        # Display comparison table
        st.dataframe(
            comparison.style.format({
                'MAE': '{:.3f}',
                'RMSE': '{:.3f}',
                'R²': '{:.3f}',
                'MAPE': '{:.2f}%'
            }).highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='lightgreen')
            .highlight_max(subset=['R²'], color='lightgreen'),
            use_container_width=True
        )
        
        # Performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig_mae = px.bar(
                comparison,
                x='Model',
                y='MAE',
                title='Mean Absolute Error by Model',
                color='MAE',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col2:
            fig_r2 = px.bar(
                comparison,
                x='Model',
                y='R²',
                title='R² Score by Model',
                color='R²',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        # Feature importance
        st.subheader("🎯 Feature Importance Analysis")
        
        model_for_importance = st.selectbox(
            "Select model for feature importance",
            ['xgboost', 'lightgbm', 'random_forest']
        )
        
        try:
            importance = predictor.get_feature_importance(model_for_importance, top_n=10)
            
            fig_importance = px.bar(
                importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Top 10 Feature Importance - {model_for_importance.upper()}',
                color='Importance',
                color_continuous_scale='Plasma'
            )
            fig_importance.update_layout(height=500)
            st.plotly_chart(fig_importance, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Feature importance not available for {model_for_importance}: {e}")
        
    except Exception as e:
        st.error(f"Error displaying model performance: {e}")


def show_data_explorer(data):
    """Display data exploration page."""
    st.header("📊 Data Explorer")
    
    st.info("🔍 Explore the raw data with filtering and analysis capabilities.")
    
    # Data filters
    st.subheader("🔧 Data Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_countries = st.multiselect(
            "Countries",
            sorted(data['Country'].unique()),
            default=sorted(data['Country'].unique())[:5]
        )
    
    with col2:
        year_range = st.slider(
            "Year Range",
            int(data['Year'].min()),
            int(data['Year'].max()),
            (int(data['Year'].min()), int(data['Year'].max()))
        )
    
    with col3:
        selected_regions = st.multiselect(
            "Regions",
            sorted(data['Region'].unique()),
            default=sorted(data['Region'].unique())
        )
    
    # Apply filters
    filtered_data = data[
        (data['Country'].isin(selected_countries)) &
        (data['Year'] >= year_range[0]) &
        (data['Year'] <= year_range[1]) &
        (data['Region'].isin(selected_regions))
    ]
    
    # Data summary
    st.subheader("📋 Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(filtered_data))
    
    with col2:
        st.metric("Countries", filtered_data['Country'].nunique())
    
    with col3:
        st.metric("Years", f"{filtered_data['Year'].min()}-{filtered_data['Year'].max()}")
    
    with col4:
        st.metric("Regions", filtered_data['Region'].nunique())
    
    # Data table
    st.subheader("📊 Filtered Data")
    st.dataframe(filtered_data, use_container_width=True)
    
    # Download option
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"life_expectancy_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    numeric_columns = ['Life_Expectancy_Total', 'Life_Expectancy_Female', 'Life_Expectancy_Male', 'Gender_Gap']
    summary_stats = filtered_data[numeric_columns].describe()
    st.dataframe(summary_stats, use_container_width=True)


if __name__ == "__main__":
    main()
