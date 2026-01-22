import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class UAEGeographicData:
    """Generate UAE-specific geographic and demographic data"""
    
    def __init__(self):
        self.emirates = {
            'Abu Dhabi': {'population_2024': 3789000, 'area_km2': 97200, 'gdp_per_capita': 117000, 'coordinates': [24.4667, 54.3667]},
            'Dubai': {'population_2024': 3660000, 'area_km2': 4110, 'gdp_per_capita': 44000, 'coordinates': [25.2769, 55.2962]},
            'Sharjah': {'population_2024': 1800000, 'area_km2': 2590, 'gdp_per_capita': 32000, 'coordinates': [25.3714, 55.4064]},
            'Ajman': {'population_2024': 504000, 'area_km2': 460, 'gdp_per_capita': 28000, 'coordinates': [25.4181, 55.4444]},
            'Umm Al Quwain': {'population_2024': 290000, 'area_km2': 720, 'gdp_per_capita': 26000, 'coordinates': [25.5647, 55.5552]},
            'Ras Al Khaimah': {'population_2024': 345000, 'area_km2': 1684, 'gdp_per_capita': 29000, 'coordinates': [25.7895, 55.9432]},
            'Fujairah': {'population_2024': 285000, 'area_km2': 1450, 'gdp_per_capita': 31000, 'coordinates': [25.1164, 56.3215]}
        }
        
        self.cities = {
            'Abu Dhabi': ['Abu Dhabi City', 'Al Ain', 'Madinat Zayed', 'Ghayathi'],
            'Dubai': ['Dubai', 'Jebel Ali', 'Al Quoz', 'Deira', 'Bur Dubai'],
            'Sharjah': ['Sharjah City', 'Al Dhaid', 'Khor Fakkan', 'Kalba'],
            'Ajman': ['Ajman City', 'Manama'],
            'Umm Al Quwain': ['Umm Al Quwain City', 'Al Salamah'],
            'Ras Al Khaimah': ['RAK City', 'Al Jazirah Al Hamra', 'Dhayd'],
            'Fujairah': ['Fujairah City', 'Dibba', 'Masafi']
        }

    def generate_population_forecast(self, years=25):
        """Generate population forecasts with realistic growth rates"""
        data = []
        base_year = 2024
        
        for year in range(base_year, base_year + years):
            for emirate, info in self.emirates.items():
                # Different growth rates per emirate
                growth_rates = {
                    'Abu Dhabi': 0.025, 'Dubai': 0.035, 'Sharjah': 0.028,
                    'Ajman': 0.032, 'Umm Al Quwain': 0.030,
                    'Ras Al Khaimah': 0.031, 'Fujairah': 0.029
                }
                
                population = info['population_2024'] * (1 + growth_rates[emirate]) ** (year - base_year)
                
                # Add some randomness for realism
                population *= np.random.normal(1.0, 0.02)
                
                data.append({
                    'year': year,
                    'emirate': emirate,
                    'population': int(population),
                    'urban_density': population / info['area_km2'],
                    'gdp_per_capita': info['gdp_per_capita'] * (1 + 0.03) ** (year - base_year)
                })
        
        return pd.DataFrame(data)

class UAEWasteData:
    """Generate UAE-specific waste composition and generation data"""
    
    def __init__(self):
        self.waste_types = ['organic', 'plastic', 'paper', 'metal', 'glass', 'textile', 'construction', 'hazardous']
        
        # UAE-specific waste composition percentages (based on regional studies)
        self.base_composition = {
            'organic': 0.35,      # Food waste, garden waste
            'plastic': 0.18,       # High due to packaging
            'paper': 0.15,         # Office and household
            'metal': 0.08,         # Cans, construction
            'glass': 0.07,         # Bottles, containers
            'textile': 0.06,       # Clothing, fabrics
            'construction': 0.08,  # Building materials
            'hazardous': 0.03      # Medical, chemicals
        }
        
        # Per capita waste generation (kg/day) by emirate
        self.per_capita_waste = {
            'Abu Dhabi': 1.8,
            'Dubai': 2.1,      # Higher due to tourism and commerce
            'Sharjah': 1.7,
            'Ajman': 1.6,
            'Umm Al Quwain': 1.5,
            'Ras Al Khaimah': 1.6,
            'Fujairah': 1.5
        }

    def generate_waste_data(self, population_df):
        """Generate waste composition and generation data"""
        waste_data = []
        
        for _, row in population_df.iterrows():
            emirate = row['emirate']
            population = row['population']
            year = row['year']
            
            # Base waste generation
            total_waste_tons_per_day = population * self.per_capita_waste[emirate] / 1000
            
            # Adjust for economic factors
            economic_factor = row['gdp_per_capita'] / 50000  # Normalized
            total_waste_tons_per_day *= (0.8 + 0.4 * economic_factor)
            
            # Seasonal variation (higher in summer due to tourism)
            seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (year % 4) / 4)
            total_waste_tons_per_day *= seasonal_factor
            
            # Generate composition with some variation
            composition = {}
            remaining = 1.0
            
            for waste_type in self.waste_types[:-1]:  # Exclude last type
                base_pct = self.base_composition[waste_type]
                # Add some variation
                pct = base_pct * np.random.normal(1.0, 0.1)
                pct = max(0.01, min(pct, remaining - 0.01))
                composition[waste_type] = pct
                remaining -= pct
            
            composition[self.waste_types[-1]] = remaining  # Last type gets remaining
            
            # Calculate tons per waste type
            for waste_type, pct in composition.items():
                waste_data.append({
                    'year': year,
                    'emirate': emirate,
                    'waste_type': waste_type,
                    'percentage': pct,
                    'tons_per_day': total_waste_tons_per_day * pct,
                    'tons_per_year': total_waste_tons_per_day * pct * 365
                })
        
        return pd.DataFrame(waste_data)

class UAEnergyParameters:
    """Energy conversion parameters for different waste types"""
    
    def __init__(self):
        # Energy yields (kWh per ton) for different conversion technologies
        self.energy_yields = {
            'incineration': {
                'organic': 350, 'plastic': 800, 'paper': 450, 'metal': 50,
                'glass': 30, 'textile': 400, 'construction': 200, 'hazardous': 600
            },
            'anaerobic_digestion': {
                'organic': 150, 'plastic': 0, 'paper': 50, 'metal': 0,
                'glass': 0, 'textile': 30, 'construction': 0, 'hazardous': 0
            },
            'gasification': {
                'organic': 400, 'plastic': 900, 'paper': 500, 'metal': 100,
                'glass': 50, 'textile': 450, 'construction': 300, 'hazardous': 700
            }
        }
        
        # Emission factors (kg CO2 per ton waste)
        self.emission_factors = {
            'landfill': {
                'organic': 500, 'plastic': 200, 'paper': 300, 'metal': 50,
                'glass': 30, 'textile': 250, 'construction': 100, 'hazardous': 400
            },
            'incineration': {
                'organic': 300, 'plastic': 800, 'paper': 400, 'metal': 100,
                'glass': 50, 'textile': 350, 'construction': 200, 'hazardous': 600
            },
            'anaerobic_digestion': {
                'organic': 100, 'plastic': 0, 'paper': 100, 'metal': 20,
                'glass': 10, 'textile': 80, 'construction': 50, 'hazardous': 200
            },
            'gasification': {
                'organic': 250, 'plastic': 700, 'paper': 350, 'metal': 80,
                'glass': 40, 'textile': 300, 'construction': 150, 'hazardous': 500
            }
        }
        
        # Economic parameters
        self.plant_costs = {
            'incineration': {'capex_per_ton': 750000, 'opex_per_ton': 85},
            'anaerobic_digestion': {'capex_per_ton': 500000, 'opex_per_ton': 60},
            'gasification': {'capex_per_ton': 900000, 'opex_per_ton': 95}
        }
        
        self.energy_prices = {
            'electricity_sell_price': 0.45,  # AED per kWh
            'carbon_credit_price': 150,      # AED per ton CO2 avoided
            'transport_cost_per_km': 3.5     # AED per ton per km
        }

def generate_all_datasets():
    """Generate all synthetic datasets for the WTE-UAE platform"""
    
    # Initialize generators
    geo_gen = UAEGeographicData()
    waste_gen = UAEWasteData()
    energy_params = UAEnergyParameters()
    
    # Generate datasets
    population_df = geo_gen.generate_population_forecast(25)  # 2024-2049
    waste_df = waste_gen.generate_waste_data(population_df)
    
    # Save datasets
    population_df.to_csv('data/uae_population_forecast.csv', index=False)
    waste_df.to_csv('data/uae_waste_generation.csv', index=False)
    
    # Save parameters
    with open('data/energy_parameters.json', 'w') as f:
        json.dump({
            'energy_yields': energy_params.energy_yields,
            'emission_factors': energy_params.emission_factors,
            'plant_costs': energy_params.plant_costs,
            'energy_prices': energy_params.energy_prices
        }, f, indent=2)
    
    # Save geographic data
    with open('data/uae_geographic_data.json', 'w') as f:
        json.dump({
            'emirates': geo_gen.emirates,
            'cities': geo_gen.cities
        }, f, indent=2)
    
    print("✅ All UAE datasets generated successfully!")
    print(f"📊 Population data: {len(population_df)} records")
    print(f"🗑️  Waste data: {len(waste_df)} records")
    
    return population_df, waste_df, energy_params

if __name__ == "__main__":
    generate_all_datasets()
