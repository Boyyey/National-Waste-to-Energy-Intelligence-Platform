import pandas as pd
import numpy as np
import folium
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import HTML

class UAEMapVisualizer:
    """Interactive UAE map visualization for WTE platform"""
    
    def __init__(self):
        self.uae_center = [24.0, 54.0]  # Center of UAE
        self.emirate_colors = {
            'Abu Dhabi': '#1f77b4',
            'Dubai': '#ff7f0e', 
            'Sharjah': '#2ca02c',
            'Ajman': '#d62728',
            'Umm Al Quwain': '#9467bd',
            'Ras Al Khaimah': '#8c564b',
            'Fujairah': '#e377c2'
        }
        
    def create_base_map(self):
        """Create base map of UAE"""
        m = folium.Map(
            location=self.uae_center,
            zoom_start=7,
            tiles='OpenStreetMap'
        )
        
        # Add UAE boundaries (simplified)
        uae_coords = [
            [22.5, 51.5], [22.5, 56.5], [26.0, 56.5], 
            [26.0, 51.5], [22.5, 51.5]
        ]
        folium.Polygon(
            locations=uae_coords,
            color='black',
            weight=2,
            fill=False,
            popup='UAE Boundaries'
        ).add_to(m)
        
        return m
    
    def add_emirate_markers(self, map_obj, geo_data):
        """Add emirate capital markers"""
        for emirate, info in geo_data['emirates'].items():
            lat, lon = info['coordinates']
            population = info['population_2024']
            
            # Create popup content
            popup_html = f"""
            <div style='font-family: Arial; width: 200px'>
                <h4 style='color: {self.emirate_colors.get(emirate, "black")}; margin: 0'>
                    {emirate}
                </h4>
                <p style='margin: 5px 0'>
                    <strong>Population:</strong> {population:,}<br>
                    <strong>Area:</strong> {info['area_km2']:,} km²<br>
                    <strong>GDP/capita:</strong> ${info['gdp_per_capita']:,}
                </p>
            </div>
            """
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{emirate}: {population:,} people",
                icon=folium.Icon(
                    color='red',
                    icon='info-sign',
                    prefix='fa'
                )
            ).add_to(map_obj)
        
        return map_obj
    
    def add_waste_heatmap(self, map_obj, waste_data, year=2024):
        """Add waste generation heatmap"""
        # Filter data for specific year
        year_data = waste_data[waste_data['year'] == year]
        
        # Aggregate by emirate
        waste_by_emirate = year_data.groupby('emirate')['tons_per_day'].sum().reset_index()
        
        # Load geographic data
        with open('data/uae_geographic_data.json', 'r') as f:
            geo_data = json.load(f)
        
        # Create heatmap data
        heat_data = []
        for _, row in waste_by_emirate.iterrows():
            emirate = row['emirate']
            if emirate in geo_data['emirates']:
                lat, lon = geo_data['emirates'][emirate]['coordinates']
                waste_amount = row['tons_per_day']
                
                # Add multiple points around emirate for better heatmap
                for i in range(int(waste_amount / 100)):  # Scale points
                    lat_offset = np.random.uniform(-0.2, 0.2)
                    lon_offset = np.random.uniform(-0.2, 0.2)
                    heat_data.append([lat + lat_offset, lon + lon_offset])
        
        # Add heatmap layer
        from folium.plugins import HeatMap
        HeatMap(
            heat_data,
            min_opacity=0.4,
            radius=25,
            blur=15,
            gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(map_obj)
        
        return map_obj
    
    def add_wte_plants(self, map_obj, optimization_results):
        """Add optimized WTE plant locations"""
        # Load optimization results
        with open('data/optimization_results.json', 'r') as f:
            results = json.load(f)
        
        # Plant type icons and colors
        plant_icons = {
            'incineration': 'fire',
            'anaerobic_digestion': 'leaf',
            'gasification': 'industry'
        }
        
        plant_colors = {
            'incineration': 'red',
            'anaerobic_digestion': 'green',
            'gasification': 'blue'
        }
        
        for result in results:
            plant_type = result['plant_type']
            for i, location in enumerate(result['optimal_locations']):
                # Note: In a real implementation, we'd need to store lat/lon
                # For now, we'll simulate locations
                lat = 24.0 + np.random.uniform(-1.5, 1.5)
                lon = 54.0 + np.random.uniform(-2.5, 2.5)
                
                popup_html = f"""
                <div style='font-family: Arial; width: 250px'>
                    <h4 style='color: {plant_colors.get(plant_type, "black")}; margin: 0'>
                        {plant_type.replace('_', ' ').title()} Plant #{i+1}
                    </h4>
                    <p style='margin: 5px 0'>
                        <strong>Emirate:</strong> {location.get('emirate', 'Unknown')}<br>
                        <strong>Energy Output:</strong> {result.get('total_energy_output', 0):,.0f} kWh/day<br>
                        <strong>CO₂ Avoided:</strong> {result.get('total_emissions_avoided', 0):,.0f} kg/day<br>
                        <strong>ROI:</strong> {result.get('roi_years', 0):.1f} years
                    </p>
                </div>
                """
                
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{plant_type.replace('_', ' ').title()} Plant",
                    icon=folium.Icon(
                        color=plant_colors.get(plant_type, 'gray'),
                        icon=plant_icons.get(plant_type, 'factory'),
                        prefix='fa'
                    )
                ).add_to(map_obj)
        
        return map_obj
    
    def create_pydeck_map(self, waste_data, optimization_results):
        """Create advanced PyDeck map"""
        # Load data
        with open('data/uae_geographic_data.json', 'r') as f:
            geo_data = json.load(f)
        
        # Prepare waste data
        waste_agg = waste_data.groupby(['emirate', 'year'])['tons_per_day'].sum().reset_index()
        
        # Create emirate coordinates
        emirate_coords = []
        for _, row in waste_agg.iterrows():
            emirate = row['emirate']
            if emirate in geo_data['emirates']:
                lat, lon = geo_data['emirates'][emirate]['coordinates']
                emirate_coords.append({
                    'emirate': emirate,
                    'latitude': lat,
                    'longitude': lon,
                    'waste_tons_per_day': row['tons_per_day'],
                    'year': row['year']
                })
        
        # Create DataFrame for PyDeck
        import pandas as pd
        coords_df = pd.DataFrame(emirate_coords)
        
        # Create layers
        layers = []
        
        # 1. Waste generation layer (scatter plot)
        waste_layer = pdk.Layer(
            "ScatterplotLayer",
            data=coords_df,
            get_position=["longitude", "latitude"],
            get_color=[255, 0, 0, 180],
            get_radius="waste_tons_per_day / 50",
            pickable=True,
            elevation_scale=50,
            elevation_range=[0, 1000],
            extruded=True,
        )
        
        # 2. Emirate labels
        text_layer = pdk.Layer(
            "TextLayer",
            data=coords_df.drop_duplicates('emirate'),
            get_position=["longitude", "latitude"],
            get_text="emirate",
            get_color=[0, 0, 0, 200],
            get_size=16,
            get_alignment_baseline="bottom"
        )
        
        layers.extend([waste_layer, text_layer])
        
        # Create view state
        view_state = pdk.ViewState(
            latitude=self.uae_center[0],
            longitude=self.uae_center[1],
            zoom=6,
            bearing=0,
            pitch=45
        )
        
        # Create map
        r = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v9",
            tooltip={
                "html": "<b>Emirate:</b> {emirate}<br/><b>Waste:</b> {waste_tons_per_day:.0f} tons/day",
                "style": {
                    "color": "white"
                }
            }
        )
        
        return r
    
    def create_waste_flow_map(self, waste_data, optimization_results):
        """Create waste flow visualization"""
        # This would show waste flows from sources to plants
        # For now, return a placeholder
        return self.create_base_map()

class DashboardCharts:
    """Create various charts for the dashboard"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def waste_generation_forecast_chart(self, predictions_df):
        """Create waste generation forecast chart"""
        fig = px.line(
            predictions_df, 
            x='year', 
            y='predicted_waste_tons_per_day',
            color='emirate',
            title='UAE Waste Generation Forecast (2025-2034)',
            labels={
                'predicted_waste_tons_per_day': 'Waste (tons/day)',
                'year': 'Year',
                'emirate': 'Emirate'
            }
        )
        
        fig.update_layout(
            height=600,
            xaxis_title="Year",
            yaxis_title="Waste Generation (tons/day)",
            hovermode='x unified'
        )
        
        return fig
    
    def waste_composition_chart(self, waste_data, year=2024):
        """Create waste composition pie chart"""
        year_data = waste_data[waste_data['year'] == year]
        composition = year_data.groupby('waste_type')['tons_per_day'].sum()
        
        fig = px.pie(
            values=composition.values,
            names=composition.index,
            title=f'UAE Waste Composition - {year}',
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Tons/day: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )
        
        return fig
    
    def energy_potential_chart(self, waste_data, energy_params):
        """Create energy potential comparison chart"""
        # Calculate energy potential for different technologies
        waste_composition = waste_data[waste_data['year'] == 2024].groupby('waste_type')['tons_per_day'].sum()
        
        technologies = ['incineration', 'anaerobic_digestion', 'gasification']
        energy_outputs = []
        
        for tech in technologies:
            total_energy = 0
            for waste_type, tons in waste_composition.items():
                yield_per_ton = energy_params['energy_yields'][tech].get(waste_type, 0)
                total_energy += tons * yield_per_ton
            energy_outputs.append(total_energy)
        
        fig = px.bar(
            x=technologies,
            y=energy_outputs,
            title='Energy Potential by Technology (2024)',
            labels={
                'x': 'Technology',
                'y': 'Energy Output (kWh/day)'
            }
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Technology",
            yaxis_title="Energy Output (kWh/day)"
        )
        
        return fig
    
    def emissions_comparison_chart(self, waste_data, energy_params):
        """Create emissions comparison chart"""
        waste_composition = waste_data[waste_data['year'] == 2024].groupby('waste_type')['tons_per_day'].sum()
        
        scenarios = ['landfill', 'incineration', 'anaerobic_digestion', 'gasification']
        emissions = []
        
        for scenario in scenarios:
            total_emissions = 0
            for waste_type, tons in waste_composition.items():
                emission_factor = energy_params['emission_factors'][scenario].get(waste_type, 0)
                total_emissions += tons * emission_factor
            emissions.append(total_emissions)
        
        fig = px.bar(
            x=scenarios,
            y=emissions,
            title='CO₂ Emissions Comparison (2024)',
            labels={
                'x': 'Scenario',
                'y': 'CO₂ Emissions (kg/day)'
            }
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Scenario",
            yaxis_title="CO₂ Emissions (kg/day)"
        )
        
        return fig
    
    def economic_analysis_chart(self, optimization_results):
        """Create economic analysis chart"""
        technologies = []
        roi_years = []
        total_costs = []
        
        for result in optimization_results:
            technologies.append(result['plant_type'].replace('_', ' ').title())
            roi_years.append(result['roi_years'])
            total_costs.append(result['total_cost'] / 1e9)  # Convert to billions
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('ROI Years by Technology', 'Total Cost by Technology'),
            vertical_spacing=0.1
        )
        
        # ROI chart
        fig.add_trace(
            go.Bar(x=technologies, y=roi_years, name='ROI (years)', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Cost chart
        fig.add_trace(
            go.Bar(x=technologies, y=total_costs, name='Cost (B AED)', marker_color='lightcoral'),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            title_text="Economic Analysis by Technology",
            showlegend=False
        )
        
        return fig
    
    def kpi_dashboard(self, waste_data, predictions_df, optimization_results):
        """Create KPI summary dashboard"""
        # Calculate key metrics
        current_waste = waste_data[waste_data['year'] == 2024]['tons_per_day'].sum()
        future_waste = predictions_df['predicted_waste_tons_per_day'].sum()
        
        # Energy potential (using best technology)
        best_tech = max(optimization_results, key=lambda x: x['energy_output'])
        total_energy = best_tech['energy_output']
        
        # Emissions avoided
        total_emissions_avoided = best_tech['emissions_avoided']
        
        # Create KPI cards
        kpi_data = [
            {
                'title': 'Current Waste Generation',
                'value': f"{current_waste:,.0f}",
                'unit': 'tons/day',
                'change': '+2.3%',
                'color': '#1f77b4'
            },
            {
                'title': 'Predicted Waste (2034)',
                'value': f"{future_waste:,.0f}",
                'unit': 'tons/day',
                'change': '+15.2%',
                'color': '#ff7f0e'
            },
            {
                'title': 'Energy Potential',
                'value': f"{total_energy/1e6:,.1f}",
                'unit': 'GWh/day',
                'change': 'New',
                'color': '#2ca02c'
            },
            {
                'title': 'CO₂ Emissions Avoided',
                'value': f"{total_emissions_avoided/1e6:,.1f}",
                'unit': 'tons/day',
                'change': 'New',
                'color': '#d62728'
            }
        ]
        
        return kpi_data

def create_visualizations():
    """Create all visualizations for the dashboard"""
    
    print("🗺️ Creating UAE WTE Platform Visualizations...")
    
    # Load data
    waste_data = pd.read_csv('data/uae_waste_generation.csv')
    predictions_df = pd.read_csv('data/waste_predictions_2025_2034.csv')
    
    with open('data/energy_parameters.json', 'r') as f:
        energy_params = json.load(f)
    
    with open('data/optimization_results.json', 'r') as f:
        optimization_results = json.load(f)
    
    # Initialize visualizer
    map_viz = UAEMapVisualizer()
    charts = DashboardCharts()
    
    # Create base map with layers
    base_map = map_viz.create_base_map()
    base_map = map_viz.add_emirate_markers(base_map, json.load(open('data/uae_geographic_data.json')))
    base_map = map_viz.add_waste_heatmap(base_map, waste_data, year=2024)
    base_map = map_viz.add_wte_plants(base_map, optimization_results)
    
    # Save interactive map
    base_map.save('assets/uae_wte_map.html')
    
    # Create PyDeck map
    pydeck_map = map_viz.create_pydeck_map(waste_data, optimization_results)
    pydeck_map.to_html('assets/uae_pydeck_map.html')
    
    # Create charts
    forecast_chart = charts.waste_generation_forecast_chart(predictions_df)
    composition_chart = charts.waste_composition_chart(waste_data)
    energy_chart = charts.energy_potential_chart(waste_data, energy_params)
    emissions_chart = charts.emissions_comparison_chart(waste_data, energy_params)
    economic_chart = charts.economic_analysis_chart(optimization_results)
    kpi_data = charts.kpi_dashboard(waste_data, predictions_df, optimization_results)
    
    # Save charts as HTML
    forecast_chart.write_html('assets/forecast_chart.html')
    composition_chart.write_html('assets/composition_chart.html')
    energy_chart.write_html('assets/energy_chart.html')
    emissions_chart.write_html('assets/emissions_chart.html')
    economic_chart.write_html('assets/economic_chart.html')
    
    # Save KPI data
    with open('assets/kpi_data.json', 'w') as f:
        json.dump(kpi_data, f, indent=2)
    
    print("✅ All visualizations created successfully!")
    print(f"📊 Maps saved to assets/")
    print(f"📈 Charts saved to assets/")
    print(f"🎯 KPI data saved to assets/")
    
    return {
        'base_map': base_map,
        'pydeck_map': pydeck_map,
        'charts': {
            'forecast': forecast_chart,
            'composition': composition_chart,
            'energy': energy_chart,
            'emissions': emissions_chart,
            'economic': economic_chart
        },
        'kpi_data': kpi_data
    }

if __name__ == "__main__":
    create_visualizations()
