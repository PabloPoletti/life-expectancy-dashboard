"""
Módulo para obtener datos de esperanza de vida desde fuentes públicas gratuitas.
Utiliza la API del Banco Mundial para obtener datos actualizados automáticamente.
"""

import pandas as pd
import requests
import wbdata
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


class LifeExpectancyDataFetcher:
    """
    Clase para obtener y procesar datos de esperanza de vida desde el Banco Mundial.
    """
    
    def __init__(self, cache_dir: str = "data"):
        """
        Inicializa el fetcher de datos.
        
        Args:
            cache_dir (str): Directorio donde se almacenarán los datos en cache
        """
        self.cache_dir = cache_dir
        self.data_file = os.path.join(cache_dir, "life_expectancy_data.csv")
        self.metadata_file = os.path.join(cache_dir, "data_metadata.json")
        
        # Indicadores del Banco Mundial para esperanza de vida
        self.indicators = {
            'SP.DYN.LE00.IN': 'Life expectancy at birth, total (years)',
            'SP.DYN.LE00.FE.IN': 'Life expectancy at birth, female (years)',
            'SP.DYN.LE00.MA.IN': 'Life expectancy at birth, male (years)'
        }
        
        # Asegurar que el directorio de datos existe
        os.makedirs(cache_dir, exist_ok=True)
    
    def should_update_data(self, max_age_hours: int = 24) -> bool:
        """
        Determina si los datos necesitan ser actualizados.
        
        Args:
            max_age_hours (int): Máximo de horas antes de considerar los datos obsoletos
            
        Returns:
            bool: True si los datos necesitan actualización
        """
        if not os.path.exists(self.metadata_file):
            return True
            
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            last_update = datetime.fromisoformat(metadata['last_update'])
            age = datetime.now() - last_update
            
            return age > timedelta(hours=max_age_hours)
        except:
            return True
    
    def fetch_world_bank_data(self, start_year: int = 1960, end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Obtiene datos del Banco Mundial usando wbdata de forma simplificada.
        
        Args:
            start_year (int): Año de inicio para la consulta
            end_year (int, optional): Año final para la consulta. Si es None, usa el año actual.
            
        Returns:
            pd.DataFrame: DataFrame con los datos de esperanza de vida
        """
        if end_year is None:
            end_year = datetime.now().year
        
        print(f"📊 Obteniendo datos del Banco Mundial ({start_year}-{end_year})...")
        
        try:
            # Obtener datos para todos los indicadores usando la API más simple
            all_data = []
            
            for indicator_code, indicator_name in self.indicators.items():
                print(f"  • Descargando: {indicator_name}")
                
                try:
                    # Obtener datos usando wbdata de forma básica
                    data_dict = wbdata.get_data(indicator_code)
                    
                    # Convertir a lista de diccionarios para procesar
                    for entry in data_dict:
                        if entry['value'] is not None:
                            # Convertir fecha
                            try:
                                year = int(entry['date'])
                                if start_year <= year <= end_year:
                                    all_data.append({
                                        'Country': entry['country']['value'],
                                        'Country_Code': entry['country']['id'],
                                        'Year': year,
                                        'Value': float(entry['value']),
                                        'indicator': indicator_name,
                                        'indicator_code': indicator_code
                                    })
                            except (ValueError, TypeError):
                                continue
                
                except Exception as e:
                    print(f"    ⚠️ Error con {indicator_name}: {e}")
                    continue
            
            if not all_data:
                # Si falla wbdata, usar datos de muestra
                print("⚠️ Generando datos de muestra...")
                return self._generate_sample_data(start_year, end_year)
            
            # Convertir a DataFrame
            combined_data = pd.DataFrame(all_data)
            
            # Filtrar años válidos
            combined_data = combined_data[
                (combined_data['Year'] >= start_year) & 
                (combined_data['Year'] <= end_year)
            ]
            
            print(f"✅ Datos obtenidos: {len(combined_data)} registros")
            return combined_data
            
        except Exception as e:
            print(f"❌ Error obteniendo datos del Banco Mundial: {e}")
            print("🔄 Generando datos de muestra...")
            return self._generate_sample_data(start_year, end_year)
    
    def _generate_sample_data(self, start_year: int = 2010, end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Genera datos de muestra para pruebas cuando la API no funciona.
        
        Args:
            start_year (int): Año de inicio
            end_year (int, optional): Año final
            
        Returns:
            pd.DataFrame: DataFrame con datos de muestra
        """
        if end_year is None:
            end_year = datetime.now().year
        
        print("📝 Generando datos de muestra...")
        
        # Países de muestra con datos realistas
        sample_countries = {
            'Spain': {'base': 82, 'trend': 0.1},
            'United States': {'base': 78, 'trend': 0.05},
            'Japan': {'base': 84, 'trend': 0.08},
            'Germany': {'base': 81, 'trend': 0.07},
            'France': {'base': 82, 'trend': 0.06},
            'United Kingdom': {'base': 81, 'trend': 0.04},
            'China': {'base': 76, 'trend': 0.3},
            'Brazil': {'base': 75, 'trend': 0.2},
            'India': {'base': 69, 'trend': 0.4},
            'Nigeria': {'base': 54, 'trend': 0.3},
            'Argentina': {'base': 76, 'trend': 0.15}
        }
        
        sample_data = []
        years = list(range(start_year, end_year + 1))
        
        for country, stats in sample_countries.items():
            base_life_exp = stats['base']
            trend = stats['trend']
            
            for i, year in enumerate(years):
                # Simular tendencia creciente con algo de variabilidad
                life_exp_total = base_life_exp + (trend * i) + np.random.normal(0, 0.3)
                life_exp_female = life_exp_total + np.random.normal(3, 0.5)  # Las mujeres viven más
                life_exp_male = life_exp_total - np.random.normal(3, 0.5)
                
                # Datos para total
                sample_data.append({
                    'Country': country,
                    'Country_Code': country[:3].upper(),
                    'Year': year,
                    'Value': round(life_exp_total, 1),
                    'indicator': 'Life expectancy at birth, total (years)',
                    'indicator_code': 'SP.DYN.LE00.IN'
                })
                
                # Datos para mujeres
                sample_data.append({
                    'Country': country,
                    'Country_Code': country[:3].upper(),
                    'Year': year,
                    'Value': round(life_exp_female, 1),
                    'indicator': 'Life expectancy at birth, female (years)',
                    'indicator_code': 'SP.DYN.LE00.FE.IN'
                })
                
                # Datos para hombres
                sample_data.append({
                    'Country': country,
                    'Country_Code': country[:3].upper(),
                    'Year': year,
                    'Value': round(life_exp_male, 1),
                    'indicator': 'Life expectancy at birth, male (years)',
                    'indicator_code': 'SP.DYN.LE00.MA.IN'
                })
        
        return pd.DataFrame(sample_data)
    
    def get_country_regions(self) -> Dict[str, str]:
        """
        Obtiene información de regiones para los países.
        
        Returns:
            Dict[str, str]: Diccionario con país -> región
        """
        try:
            countries = wbdata.get_country()
            regions = {}
            
            for country in countries:
                if country['region']['value'] != 'Aggregates':
                    regions[country['name']] = country['region']['value']
            
            return regions
        except:
            # Regiones básicas de respaldo
            return {
                'Spain': 'Europe & Central Asia',
                'United States': 'North America',
                'China': 'East Asia & Pacific',
                'Germany': 'Europe & Central Asia',
                'France': 'Europe & Central Asia',
                'Japan': 'East Asia & Pacific',
                'Brazil': 'Latin America & Caribbean',
                'India': 'South Asia',
                'United Kingdom': 'Europe & Central Asia',
                'Nigeria': 'Sub-Saharan Africa',
                'Argentina': 'Latin America & Caribbean'
            }
    
    def process_and_save_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa y guarda los datos obtenidos.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos crudos
            
        Returns:
            pd.DataFrame: DataFrame procesado
        """
        print("🔄 Procesando datos...")
        
        # Obtener información de regiones
        regions = self.get_country_regions()
        df['Region'] = df['Country'].map(regions)
        
        # Crear pivot table para tener indicadores como columnas
        processed_df = df.pivot_table(
            index=['Country', 'Year', 'Region'],
            columns='indicator',
            values='Value',
            aggfunc='first'
        ).reset_index()
        
        # Limpiar nombres de columnas
        processed_df.columns.name = None
        
        # Renombrar columnas para mayor claridad
        column_mapping = {
            'Life expectancy at birth, total (years)': 'Life_Expectancy_Total',
            'Life expectancy at birth, female (years)': 'Life_Expectancy_Female', 
            'Life expectancy at birth, male (years)': 'Life_Expectancy_Male'
        }
        
        processed_df = processed_df.rename(columns=column_mapping)
        
        # Calcular diferencia de género si ambos datos están disponibles
        if 'Life_Expectancy_Female' in processed_df.columns and 'Life_Expectancy_Male' in processed_df.columns:
            processed_df['Gender_Gap'] = (
                processed_df['Life_Expectancy_Female'] - processed_df['Life_Expectancy_Male']
            )
        
        # Ordenar por país y año
        processed_df = processed_df.sort_values(['Country', 'Year'])
        
        # Guardar datos
        processed_df.to_csv(self.data_file, index=False)
        
        # Guardar metadata
        metadata = {
            'last_update': datetime.now().isoformat(),
            'total_records': len(processed_df),
            'countries': sorted(processed_df['Country'].unique().tolist()),
            'year_range': {
                'min': int(processed_df['Year'].min()),
                'max': int(processed_df['Year'].max())
            },
            'indicators': list(column_mapping.values())
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Datos procesados y guardados: {len(processed_df)} registros")
        print(f"📁 Archivo: {self.data_file}")
        
        return processed_df
    
    def load_cached_data(self) -> Optional[pd.DataFrame]:
        """
        Carga datos desde el cache si existen.
        
        Returns:
            pd.DataFrame or None: DataFrame con los datos o None si no existen
        """
        try:
            if os.path.exists(self.data_file):
                return pd.read_csv(self.data_file)
        except:
            pass
        return None
    
    def get_data(self, force_update: bool = False, start_year: int = 1960) -> pd.DataFrame:
        """
        Método principal para obtener datos de esperanza de vida.
        
        Args:
            force_update (bool): Forzar actualización de datos
            start_year (int): Año de inicio para la consulta
            
        Returns:
            pd.DataFrame: DataFrame con los datos de esperanza de vida
        """
        # Verificar si necesitamos actualizar
        if not force_update and not self.should_update_data():
            print("📂 Cargando datos desde cache...")
            cached_data = self.load_cached_data()
            if cached_data is not None:
                print(f"✅ Datos cargados: {len(cached_data)} registros")
                return cached_data
        
        # Obtener datos frescos
        print("🌐 Obteniendo datos actualizados...")
        raw_data = self.fetch_world_bank_data(start_year=start_year)
        processed_data = self.process_and_save_data(raw_data)
        
        return processed_data


# Función de conveniencia para uso directo
def get_life_expectancy_data(force_update: bool = False) -> pd.DataFrame:
    """
    Función de conveniencia para obtener datos de esperanza de vida.
    
    Args:
        force_update (bool): Forzar actualización de datos
        
    Returns:
        pd.DataFrame: DataFrame con los datos de esperanza de vida
    """
    fetcher = LifeExpectancyDataFetcher()
    return fetcher.get_data(force_update=force_update)


if __name__ == "__main__":
    # Test del módulo
    print("🧪 Probando obtención de datos...")
    data = get_life_expectancy_data(force_update=True)
    print(f"\n📊 Resumen de datos:")
    print(f"  • Total registros: {len(data)}")
    print(f"  • Países: {data['Country'].nunique()}")
    print(f"  • Años: {data['Year'].min()}-{data['Year'].max()}")
    print(f"  • Columnas: {list(data.columns)}")
    print(f"\n🔍 Primeras 5 filas:")
    print(data.head())
