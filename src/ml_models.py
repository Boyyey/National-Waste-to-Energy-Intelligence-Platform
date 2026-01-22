import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class WasteForecastingModel:
    """Advanced ML models for waste generation forecasting"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.emirate_models = {}
        
    def prepare_features(self, df):
        """Engineer features for waste prediction"""
        df = df.copy()
        
        # Time-based features
        df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        df['decade'] = df['year'] // 10
        
        # Economic features
        df['gdp_total'] = df['population'] * df['gdp_per_capita']
        df['gdp_growth_rate'] = df.groupby('emirate')['gdp_total'].pct_change()
        
        # Urbanization features
        df['population_density_sqkm'] = df['population'] / df['area_km2']
        df['urbanization_index'] = df['population_density_sqkm'] / df['population_density_sqkm'].max()
        
        # Interaction features
        df['pop_gdp_interaction'] = df['population'] * df['gdp_per_capita'] / 1e6
        df['density_gdp_interaction'] = df['population_density_sqkm'] * df['gdp_per_capita'] / 1e3
        
        # Lag features (for time series)
        df = df.sort_values(['emirate', 'year'])
        df['waste_lag_1'] = df.groupby('emirate')['total_waste_tons_per_day'].shift(1)
        df['waste_lag_3'] = df.groupby('emirate')['total_waste_tons_per_day'].shift(3)
        
        # Rolling averages
        df['waste_ma_3'] = df.groupby('emirate')['total_waste_tons_per_day'].rolling(3).mean().reset_index(0, drop=True)
        df['waste_ma_5'] = df.groupby('emirate')['total_waste_tons_per_day'].rolling(5).mean().reset_index(0, drop=True)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def train_emirate_models(self, population_df, waste_df):
        """Train separate models for each emirate"""
        # Merge datasets
        waste_total = waste_df.groupby(['year', 'emirate'])['tons_per_day'].sum().reset_index()
        waste_total.columns = ['year', 'emirate', 'total_waste_tons_per_day']
        
        # Add geographic data
        geo_data = pd.read_json('data/uae_geographic_data.json')
        emirate_info = []
        for emirate, info in geo_data['emirates'].items():
            emirate_info.append({
                'emirate': emirate,
                'area_km2': info['area_km2']
            })
        area_df = pd.DataFrame(emirate_info)
        
        # Merge all data
        full_df = population_df.merge(waste_total, on=['year', 'emirate'])
        full_df = full_df.merge(area_df, on='emirate')
        
        # Prepare features
        full_df = self.prepare_features(full_df)
        
        # Define feature columns
        self.feature_columns = [
            'year_norm', 'population', 'gdp_per_capita', 'gdp_total', 'gdp_growth_rate',
            'population_density_sqkm', 'urbanization_index', 'pop_gdp_interaction',
            'density_gdp_interaction', 'waste_lag_1', 'waste_lag_3', 'waste_ma_3', 'waste_ma_5'
        ]
        
        # Train models for each emirate
        for emirate in full_df['emirate'].unique():
            emirate_data = full_df[full_df['emirate'] == emirate].copy()
            
            # Skip if not enough data
            if len(emirate_data) < 10:
                continue
            
            X = emirate_data[self.feature_columns].fillna(0)
            y = emirate_data['total_waste_tons_per_day']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train multiple models
            models = {
                'xgboost': xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42
                )
            }
            
            best_model = None
            best_score = float('inf')
            best_model_name = None
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                
                if mae < best_score:
                    best_score = mae
                    best_model = model
                    best_model_name = name
            
            # Store best model
            self.emirate_models[emirate] = {
                'model': best_model,
                'model_name': best_model_name,
                'mae': best_score,
                'feature_columns': self.feature_columns
            }
            
            print(f"✅ {emirate}: {best_model_name} (MAE: {best_score:.2f} tons/day)")
    
    def predict_waste_generation(self, population_df, years_ahead=10):
        """Predict waste generation for future years"""
        predictions = []
        
        # Add geographic data
        geo_data = pd.read_json('data/uae_geographic_data.json')
        emirate_info = []
        for emirate, info in geo_data['emirates'].items():
            emirate_info.append({
                'emirate': emirate,
                'area_km2': info['area_km2']
            })
        area_df = pd.DataFrame(emirate_info)
        
        # Generate future years
        last_year = population_df['year'].max()
        future_years = range(last_year + 1, last_year + years_ahead + 1)
        
        for year in future_years:
            for emirate in self.emirate_models.keys():
                # Get population data for this year
                pop_data = population_df[population_df['emirate'] == emirate].iloc[-1].copy()
                pop_data['year'] = year
                
                # Update population with growth
                growth_rate = 0.025 if emirate == 'Abu Dhabi' else 0.03
                years_diff = year - pop_data['year']
                pop_data['population'] = pop_data['population'] * (1 + growth_rate) ** years_diff
                pop_data['gdp_per_capita'] = pop_data['gdp_per_capita'] * (1 + 0.03) ** years_diff
                
                # Merge with area data
                area_data = area_df[area_df['emirate'] == emirate].iloc[0]
                for col, val in area_data.items():
                    if col != 'emirate':
                        pop_data[col] = val
                
                # Create DataFrame and prepare features
                pop_df = pd.DataFrame([pop_data])
                
                # Add dummy waste columns for feature preparation
                pop_df['total_waste_tons_per_day'] = 0  # Will be predicted
                pop_df['gdp_total'] = pop_df['population'] * pop_df['gdp_per_capita']
                pop_df['gdp_growth_rate'] = 0.03  # Assume 3% growth
                pop_df['population_density_sqkm'] = pop_df['population'] / pop_df['area_km2']
                pop_df['urbanization_index'] = 0.5  # Normalized
                pop_df['pop_gdp_interaction'] = pop_df['population'] * pop_df['gdp_per_capita'] / 1e6
                pop_df['density_gdp_interaction'] = pop_df['population_density_sqkm'] * pop_df['gdp_per_capita'] / 1e3
                
                # Add lag features (use zeros for future predictions)
                pop_df['waste_lag_1'] = 0
                pop_df['waste_lag_3'] = 0
                pop_df['waste_ma_3'] = 0
                pop_df['waste_ma_5'] = 0
                
                # Add year normalization
                min_year = 2024
                max_year = 2049
                pop_df['year_norm'] = (year - min_year) / (max_year - min_year)
                pop_df['decade'] = year // 10
                
                # Make prediction
                model_info = self.emirate_models[emirate]
                X = pop_df[model_info['feature_columns']].fillna(0)
                predicted_waste = model_info['model'].predict(X)[0]
                
                predictions.append({
                    'year': year,
                    'emirate': emirate,
                    'predicted_waste_tons_per_day': max(0, predicted_waste),
                    'predicted_waste_tons_per_year': max(0, predicted_waste) * 365,
                    'model_used': model_info['model_name'],
                    'model_mae': model_info['mae']
                })
        
        return pd.DataFrame(predictions)
    
    def save_models(self):
        """Save trained models"""
        for emirate, model_info in self.emirate_models.items():
            filename = f"models/waste_model_{emirate.replace(' ', '_').lower()}.pkl"
            joblib.dump(model_info, filename)
        print("✅ All models saved successfully!")
    
    def load_models(self):
        """Load pre-trained models"""
        import glob
        model_files = glob.glob("models/waste_model_*.pkl")
        
        for file in model_files:
            model_info = joblib.load(file)
            emirate = file.split('_')[-1].replace('.pkl', '').replace('_', ' ').title()
            self.emirate_models[emirate] = model_info
        
        print(f"✅ Loaded {len(model_files)} models")

class EnergyYieldModel:
    """Predict energy yield from waste composition"""
    
    def __init__(self):
        self.models = {}
        self.technology_types = ['incineration', 'anaerobic_digestion', 'gasification']
        
    def train_energy_models(self, waste_df):
        """Train energy prediction models"""
        # Load energy parameters
        import json
        with open('data/energy_parameters.json', 'r') as f:
            self.energy_params = json.load(f)
        
        # Create training data
        training_data = []
        
        for _, waste_row in waste_df.iterrows():
            waste_type = waste_row['waste_type']
            tons_per_day = waste_row['tons_per_day']
            
            for tech in self.technology_types:
                energy_yield = self.energy_params['energy_yields'][tech].get(waste_type, 0)
                emissions = self.energy_params['emission_factors'][tech].get(waste_type, 0)
                
                training_data.append({
                    'waste_type': waste_type,
                    'tons_per_day': tons_per_day,
                    'technology': tech,
                    'energy_kwh_per_day': tons_per_day * energy_yield,
                    'emissions_kg_co2_per_day': tons_per_day * emissions,
                    'base_energy_yield': energy_yield,
                    'base_emissions': emissions
                })
        
        self.energy_training_df = pd.DataFrame(training_data)
        
        # Simple rule-based models for now (can be enhanced with ML)
        print("✅ Energy yield models trained successfully!")
    
    def predict_energy_output(self, waste_composition_df, technology='incineration'):
        """Predict energy output for given waste composition"""
        results = []
        
        for _, row in waste_composition_df.iterrows():
            waste_type = row['waste_type']
            tons_per_day = row['tons_per_day']
            
            # Get base parameters
            energy_yield = self.energy_params['energy_yields'][technology].get(waste_type, 0)
            emissions = self.energy_params['emission_factors'][technology].get(waste_type, 0)
            
            # Calculate outputs
            energy_output = tons_per_day * energy_yield
            emissions_output = tons_per_day * emissions
            
            # Economic calculations
            electricity_revenue = energy_output * self.energy_params['energy_prices']['electricity_sell_price']
            
            results.append({
                'waste_type': waste_type,
                'tons_per_day': tons_per_day,
                'technology': technology,
                'energy_kwh_per_day': energy_output,
                'energy_mw_per_day': energy_output / 24 / 1000,  # Convert to MW
                'emissions_kg_co2_per_day': emissions_output,
                'emissions_tons_co2_per_year': emissions_output * 365 / 1000,
                'electricity_revenue_aed_per_day': electricity_revenue,
                'electricity_revenue_aed_per_year': electricity_revenue * 365
            })
        
        return pd.DataFrame(results)

def train_all_models():
    """Train all ML models for the WTE-UAE platform"""
    
    print("🤖 Training ML Models for WTE-UAE Platform...")
    
    # Load data
    population_df = pd.read_csv('data/uae_population_forecast.csv')
    waste_df = pd.read_csv('data/uae_waste_generation.csv')
    
    # Train waste forecasting models
    print("\n📊 Training Waste Generation Forecasting Models...")
    waste_model = WasteForecastingModel()
    waste_model.train_emirate_models(population_df, waste_df)
    waste_model.save_models()
    
    # Train energy prediction models
    print("\n⚡ Training Energy Yield Prediction Models...")
    energy_model = EnergyYieldModel()
    energy_model.train_energy_models(waste_df)
    
    # Generate sample predictions
    print("\n🔮 Generating Sample Predictions...")
    future_predictions = waste_model.predict_waste_generation(population_df, years_ahead=10)
    future_predictions.to_csv('data/waste_predictions_2025_2034.csv', index=False)
    
    print(f"✅ Generated {len(future_predictions)} future predictions")
    
    return waste_model, energy_model

if __name__ == "__main__":
    train_all_models()
