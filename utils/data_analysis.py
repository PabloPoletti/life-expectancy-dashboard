"""
Utilidades para análisis de datos de esperanza de vida.
Funciones para calcular estadísticas, tendencias y comparaciones.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class LifeExpectancyAnalyzer:
    """
    Clase para análisis avanzado de datos de esperanza de vida.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el analizador con datos de esperanza de vida.
        
        Args:
            data (pd.DataFrame): DataFrame con datos de esperanza de vida
        """
        self.data = data.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepara y limpia los datos para análisis."""
        # Asegurar que Year es numérico
        self.data['Year'] = pd.to_numeric(self.data['Year'])
        
        # Filtrar valores válidos
        numeric_columns = ['Life_Expectancy_Total', 'Life_Expectancy_Female', 'Life_Expectancy_Male']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data = self.data[self.data[col].notna()]
    
    def get_country_summary(self, country: str) -> Dict:
        """
        Obtiene resumen estadístico para un país específico.
        
        Args:
            country (str): Nombre del país
            
        Returns:
            Dict: Diccionario con estadísticas del país
        """
        country_data = self.data[self.data['Country'] == country]
        
        if country_data.empty:
            return {}
        
        summary = {
            'country': country,
            'years_available': len(country_data),
            'year_range': {
                'min': int(country_data['Year'].min()),
                'max': int(country_data['Year'].max())
            }
        }
        
        # Estadísticas por indicador
        indicators = ['Life_Expectancy_Total', 'Life_Expectancy_Female', 'Life_Expectancy_Male']
        
        for indicator in indicators:
            if indicator in country_data.columns:
                values = country_data[indicator].dropna()
                if not values.empty:
                    summary[indicator] = {
                        'current': float(values.iloc[-1]),
                        'historical_max': float(values.max()),
                        'historical_min': float(values.min()),
                        'average': float(values.mean()),
                        'trend': self._calculate_trend(country_data, indicator)
                    }
        
        return summary
    
    def _calculate_trend(self, data: pd.DataFrame, column: str, years: int = 10) -> Dict:
        """
        Calcula la tendencia de un indicador en los últimos años.
        
        Args:
            data (pd.DataFrame): Datos del país
            column (str): Columna a analizar
            years (int): Número de años para calcular la tendencia
            
        Returns:
            Dict: Información de tendencia
        """
        recent_data = data.nlargest(years, 'Year')
        
        if len(recent_data) < 2 or column not in recent_data.columns:
            return {'direction': 'unknown', 'rate': 0}
        
        # Calcular tendencia lineal
        x = recent_data['Year'].values
        y = recent_data[column].dropna().values
        
        if len(y) < 2:
            return {'direction': 'unknown', 'rate': 0}
        
        # Regresión lineal simple
        slope = np.polyfit(x[-len(y):], y, 1)[0]
        
        direction = 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable'
        
        return {
            'direction': direction,
            'rate': float(slope),
            'years_analyzed': len(y)
        }
    
    def get_regional_comparison(self, year: int = None) -> pd.DataFrame:
        """
        Compara esperanza de vida por regiones.
        
        Args:
            year (int, optional): Año específico. Si es None, usa el más reciente.
            
        Returns:
            pd.DataFrame: Comparación regional
        """
        if year is None:
            year = self.data['Year'].max()
        
        year_data = self.data[self.data['Year'] == year]
        
        if 'Region' in year_data.columns:
            regional_stats = year_data.groupby('Region').agg({
                'Life_Expectancy_Total': ['mean', 'median', 'std', 'count'],
                'Life_Expectancy_Female': ['mean'],
                'Life_Expectancy_Male': ['mean']
            }).round(2)
            
            # Aplanar columnas multi-nivel
            regional_stats.columns = ['_'.join(col).strip() for col in regional_stats.columns.values]
            return regional_stats.reset_index()
        
        return pd.DataFrame()
    
    def get_top_countries(self, n: int = 10, indicator: str = 'Life_Expectancy_Total', 
                         year: int = None) -> pd.DataFrame:
        """
        Obtiene los países con mayor esperanza de vida.
        
        Args:
            n (int): Número de países a retornar
            indicator (str): Indicador a usar para el ranking
            year (int, optional): Año específico. Si es None, usa el más reciente.
            
        Returns:
            pd.DataFrame: Top países
        """
        if year is None:
            year = self.data['Year'].max()
        
        year_data = self.data[
            (self.data['Year'] == year) & 
            (self.data[indicator].notna())
        ]
        
        return year_data.nlargest(n, indicator)[['Country', 'Region', indicator]].reset_index(drop=True)
    
    def get_bottom_countries(self, n: int = 10, indicator: str = 'Life_Expectancy_Total', 
                            year: int = None) -> pd.DataFrame:
        """
        Obtiene los países con menor esperanza de vida.
        
        Args:
            n (int): Número de países a retornar
            indicator (str): Indicador a usar para el ranking
            year (int, optional): Año específico. Si es None, usa el más reciente.
            
        Returns:
            pd.DataFrame: Países con menor esperanza de vida
        """
        if year is None:
            year = self.data['Year'].max()
        
        year_data = self.data[
            (self.data['Year'] == year) & 
            (self.data[indicator].notna())
        ]
        
        return year_data.nsmallest(n, indicator)[['Country', 'Region', indicator]].reset_index(drop=True)
    
    def calculate_global_trends(self) -> Dict:
        """
        Calcula tendencias globales de esperanza de vida.
        
        Returns:
            Dict: Tendencias globales
        """
        # Calcular promedios anuales globales
        global_trends = self.data.groupby('Year').agg({
            'Life_Expectancy_Total': 'mean',
            'Life_Expectancy_Female': 'mean',
            'Life_Expectancy_Male': 'mean'
        }).reset_index()
        
        trends = {}
        
        for indicator in ['Life_Expectancy_Total', 'Life_Expectancy_Female', 'Life_Expectancy_Male']:
            if indicator in global_trends.columns:
                values = global_trends[indicator].dropna()
                if len(values) >= 2:
                    # Tendencia general
                    years = global_trends['Year'].iloc[-len(values):]
                    slope = np.polyfit(years, values, 1)[0]
                    
                    # Crecimiento en las últimas décadas
                    recent_10y = values.iloc[-10:] if len(values) >= 10 else values
                    growth_10y = (recent_10y.iloc[-1] - recent_10y.iloc[0]) if len(recent_10y) >= 2 else 0
                    
                    trends[indicator] = {
                        'current_global_average': float(values.iloc[-1]),
                        'historical_trend_rate': float(slope),
                        'growth_last_10_years': float(growth_10y),
                        'total_improvement': float(values.iloc[-1] - values.iloc[0]) if len(values) >= 2 else 0
                    }
        
        return trends
    
    def get_gender_gap_analysis(self, year: int = None) -> Dict:
        """
        Analiza la brecha de género en esperanza de vida.
        
        Args:
            year (int, optional): Año específico. Si es None, usa el más reciente.
            
        Returns:
            Dict: Análisis de brecha de género
        """
        if year is None:
            year = self.data['Year'].max()
        
        year_data = self.data[self.data['Year'] == year]
        
        if 'Gender_Gap' in year_data.columns:
            gap_data = year_data['Gender_Gap'].dropna()
            
            if not gap_data.empty:
                return {
                    'global_average_gap': float(gap_data.mean()),
                    'median_gap': float(gap_data.median()),
                    'max_gap': float(gap_data.max()),
                    'min_gap': float(gap_data.min()),
                    'countries_analyzed': len(gap_data),
                    'countries_with_larger_female_advantage': len(gap_data[gap_data > 0]),
                    'year': year
                }
        
        return {}
    
    def compare_countries(self, countries: List[str], 
                         indicator: str = 'Life_Expectancy_Total') -> pd.DataFrame:
        """
        Compara múltiples países en un indicador específico.
        
        Args:
            countries (List[str]): Lista de países a comparar
            indicator (str): Indicador a comparar
            
        Returns:
            pd.DataFrame: Comparación de países
        """
        comparison_data = self.data[
            (self.data['Country'].isin(countries)) &
            (self.data[indicator].notna())
        ][['Country', 'Year', indicator]]
        
        # Pivot para tener países como columnas
        comparison_pivot = comparison_data.pivot(
            index='Year', 
            columns='Country', 
            values=indicator
        )
        
        return comparison_pivot.reset_index()


def get_data_insights(data: pd.DataFrame) -> Dict:
    """
    Función de conveniencia para obtener insights generales de los datos.
    
    Args:
        data (pd.DataFrame): DataFrame con datos de esperanza de vida
        
    Returns:
        Dict: Insights y estadísticas generales
    """
    analyzer = LifeExpectancyAnalyzer(data)
    
    insights = {
        'data_overview': {
            'total_records': len(data),
            'countries': data['Country'].nunique(),
            'years_range': {
                'min': int(data['Year'].min()),
                'max': int(data['Year'].max())
            },
            'regions': data['Region'].nunique() if 'Region' in data.columns else 0
        },
        'global_trends': analyzer.calculate_global_trends(),
        'regional_comparison': analyzer.get_regional_comparison(),
        'top_countries': analyzer.get_top_countries(),
        'bottom_countries': analyzer.get_bottom_countries(),
        'gender_gap': analyzer.get_gender_gap_analysis()
    }
    
    return insights
