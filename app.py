import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import json
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="UAE Waste-to-Energy Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load all necessary data"""
    population_df = pd.read_csv('data/uae_population_realistic.csv')
    waste_df = pd.read_csv('data/uae_waste_realistic.csv')
    predictions_df = pd.read_csv('data/waste_predictions_2025_2034.csv')
    infrastructure_df = pd.read_csv('data/uae_infrastructure.csv')
    
    with open('data/energy_parameters_realistic.json', 'r') as f:
        energy_params = json.load(f)
    
    with open('data/uae_geographic_realistic.json', 'r') as f:
        geo_data = json.load(f)
    
    with open('data/optimization_results.json', 'r') as f:
        optimization_results = json.load(f)
    
    with open('data/ai_advisor_report.json', 'r') as f:
        ai_advisor_report = json.load(f)
    
    with open('data/policy_scenarios.json', 'r') as f:
        policy_scenarios = json.load(f)
    
    return population_df, waste_df, predictions_df, infrastructure_df, energy_params, geo_data, optimization_results, ai_advisor_report, policy_scenarios

# Load ML models
@st.cache_resource
def load_models():
    """Load trained ML models"""
    try:
        from src.ml_models import WasteForecastingModel, EnergyYieldModel
        waste_model = WasteForecastingModel()
        waste_model.load_models()
        energy_model = EnergyYieldModel()
        return waste_model, energy_model
    except:
        return None, None

# Main dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🇦🇪 UAE National Waste-to-Energy Intelligence Platform</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    population_df, waste_df, predictions_df, infrastructure_df, energy_params, geo_data, optimization_results, ai_advisor_report, policy_scenarios = load_data()
    waste_model, energy_model = load_models()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Year selector
    available_years = sorted(waste_df['year'].unique())
    selected_year = st.sidebar.selectbox("Select Year", available_years, index=len(available_years)-1)
    
    # Emirate selector
    emirates = sorted(waste_df['emirate'].unique())
    selected_emirates = st.sidebar.multiselect("Select Emirates", emirates, default=emirates)
    
    # Technology selector
    technologies = ['incineration', 'anaerobic_digestion', 'gasification']
    selected_tech = st.sidebar.selectbox("WTE Technology", technologies)
    
    # View selector
    view_options = ["🗺️ Interactive Map", "📊 Analytics Dashboard", "🔮 Forecasts", "⚡ Energy Analysis", "🌍 Carbon Impact", "🤖 AI Advisor"]
    selected_view = st.sidebar.selectbox("Select View", view_options)
    
    # Filter data based on selections
    filtered_waste = waste_df[
        (waste_df['year'] == selected_year) & 
        (waste_df['emirate'].isin(selected_emirates))
    ]
    
    filtered_predictions = predictions_df[
        (predictions_df['emirate'].isin(selected_emirates))
    ]
    
    # Main content based on selected view
    if selected_view == "🗺️ Interactive Map":
        st.markdown("## 🗺️ Interactive UAE Waste-to-Energy Map")
        st.markdown("*Navigate freely - zoom, pan, and explore waste flows and energy potential across the UAE*")
        
        # Create map data
        map_data = create_map_data(filtered_waste, geo_data, optimization_results, selected_tech)
        
        # Simple 2D Plotly Map that actually works
        st.markdown("## 🗺️ Interactive UAE Waste-to-Energy Map")
        st.markdown("*Real-time visualization of waste generation and WTE facilities across the UAE*")
        
        # Create map data
        map_data = []
        
        # Add emirate data
        if not filtered_waste.empty:
            emirate_waste = filtered_waste.groupby('emirate')['tons_per_day'].sum().reset_index()
            
            for _, row in emirate_waste.iterrows():
                emirate = row['emirate']
                if emirate in geo_data['emirates']:
                    coords = geo_data['emirates'][emirate]['coordinates']
                    
                    # Handle coordinates that might be strings or lists
                    if isinstance(coords, str):
                        coords = coords.strip('[]').split(',')
                        coords = [float(coord.strip()) for coord in coords]
                    
                    lat, lon = float(coords[0]), float(coords[1])
                    
                    map_data.append({
                        'emirate': emirate,
                        'latitude': lat,
                        'longitude': lon,
                        'waste_tons_per_day': row['tons_per_day'],
                        'type': 'Emirate',
                        'size': row['tons_per_day'] / 100
                    })
        
        # Add WTE plant data
        existing_plants = infrastructure_df[infrastructure_df['type'] == 'wte_plant']
        
        for _, plant in existing_plants.iterrows():
            coords = plant['coordinates']
            
            if isinstance(coords, str):
                coords = coords.strip('[]').split(',')
                coords = [float(coord.strip()) for coord in coords]
            
            map_data.append({
                'emirate': plant['emirate'],
                'latitude': float(coords[0]),
                'longitude': float(coords[1]),
                'waste_tons_per_day': plant['capacity_mw'] * 10,  # Scale for visibility
                'type': f"WTE Plant ({plant['capacity_mw']} MW)",
                'size': 20
            })
        
        # Create DataFrame
        map_df = pd.DataFrame(map_data)
        
        if not map_df.empty:
            # Create scatter map
            fig = px.scatter_mapbox(
                map_df,
                lat="latitude",
                lon="longitude",
                color="type",
                size="size",
                hover_name="emirate",
                hover_data=["waste_tons_per_day"],
                color_discrete_map={
                    "Emirate": "red",
                    "WTE Plant (100 MW)": "green",
                    "WTE Plant (45 MW)": "blue",
                    "WTE Plant (30 MW)": "orange"
                },
                zoom=6,
                height=600,
                mapbox_style="open-street-map"
            )
            
            fig.update_layout(
                title="UAE Waste Generation and WTE Facilities",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_waste = filtered_waste['tons_per_day'].sum()
                st.metric("🗑️ Total Waste", f"{total_waste:,.0f}", "tons/day")
            
            with col2:
                total_plants = len(existing_plants)
                st.metric("⚡ WTE Plants", f"{total_plants}", "facilities")
            
            with col3:
                total_capacity = existing_plants['capacity_mw'].sum()
                st.metric("🔋 Total Capacity", f"{total_capacity:,.0f}", "MW")
        
        else:
            st.error("No data available for the selected filters")
        
    elif selected_view == "📊 Analytics Dashboard":
        st.markdown("## 📊 Waste Analytics Dashboard")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_waste = filtered_waste['tons_per_day'].sum()
            st.metric("🗑️ Total Waste", f"{total_waste:,.0f}", "tons/day")
        
        with col2:
            if not filtered_predictions.empty:
                future_waste = filtered_predictions['predicted_waste_tons_per_day'].sum()
                growth = ((future_waste - total_waste) / total_waste * 100) if total_waste > 0 else 0
                st.metric("📈 2034 Prediction", f"{future_waste:,.0f}", f"{growth:+.1f}%")
        
        with col3:
            # Calculate energy potential
            energy_potential = calculate_energy_potential(filtered_waste, energy_params, selected_tech)
            st.metric("⚡ Energy Potential", f"{energy_potential/1e6:,.1f}", "GWh/day")
        
        with col4:
            # Calculate emissions avoided
            emissions_avoided = calculate_emissions_avoided(filtered_waste, energy_params, selected_tech)
            st.metric("🌍 CO₂ Avoided", f"{emissions_avoided/1e6:,.1f}", "tons/day")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Waste Composition")
            waste_composition = filtered_waste.groupby('waste_type')['tons_per_day'].sum()
            fig_pie = px.pie(
                values=waste_composition.values,
                names=waste_composition.index,
                title=f"Waste Composition - {selected_year}"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("### Waste by Emirate")
            waste_by_emirate = filtered_waste.groupby('emirate')['tons_per_day'].sum().sort_values(ascending=False)
            fig_bar = px.bar(
                x=waste_by_emirate.values,
                y=waste_by_emirate.index,
                orientation='h',
                title="Waste Generation by Emirate"
            )
            fig_bar.update_layout(xaxis_title="Tons per day")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Time series
        st.markdown("### Historical Trends")
        historical_data = waste_df[waste_df['emirate'].isin(selected_emirates)]
        time_series = historical_data.groupby(['year', 'emirate'])['tons_per_day'].sum().reset_index()
        
        fig_line = px.line(
            time_series,
            x='year',
            y='tons_per_day',
            color='emirate',
            title="Waste Generation Trends (2024-2049)"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
    elif selected_view == "🔮 Forecasts":
        st.markdown("## 🔮 Waste Generation Forecasts")
        
        if not filtered_predictions.empty:
            # Forecast chart
            fig_forecast = px.line(
                filtered_predictions,
                x='year',
                y='predicted_waste_tons_per_day',
                color='emirate',
                title="Waste Generation Forecast (2025-2034)",
                labels={
                    'predicted_waste_tons_per_day': 'Predicted Waste (tons/day)',
                    'year': 'Year'
                }
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast summary table
            st.markdown("### Forecast Summary")
            summary_data = []
            for emirate in selected_emirates:
                emirate_predictions = filtered_predictions[filtered_predictions['emirate'] == emirate]
                if not emirate_predictions.empty:
                    if 2025 in emirate_predictions['year'].values:
                        current = emirate_predictions[emirate_predictions['year'] == 2025]['predicted_waste_tons_per_day'].iloc[0]
                    else:
                        current = emirate_predictions['predicted_waste_tons_per_day'].iloc[0]
                    
                    if 2034 in emirate_predictions['year'].values:
                        future = emirate_predictions[emirate_predictions['year'] == 2034]['predicted_waste_tons_per_day'].iloc[0]
                    else:
                        future = emirate_predictions['predicted_waste_tons_per_day'].iloc[-1]
                    
                    growth = ((future - current) / current * 100) if current > 0 else 0
                    
                    summary_data.append({
                        'Emirate': emirate,
                        '2025 (tons/day)': f"{current:,.0f}",
                        '2034 (tons/day)': f"{future:,.0f}",
                        'Growth %': f"{growth:+.1f}%"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        
    elif selected_view == "⚡ Energy Analysis":
        st.markdown("## ⚡ Energy Potential Analysis")
        
        # Energy comparison by technology
        energy_comparison = []
        for tech in technologies:
            energy_output = calculate_energy_potential(filtered_waste, energy_params, tech)
            energy_comparison.append({
                'Technology': tech.replace('_', ' ').title(),
                'Energy Output (GWh/day)': energy_output / 1e6,
                'Revenue (M AED/day)': (energy_output * energy_params['economic_params']['electricity_sell_price']) / 1e6,
            })
        
        energy_df = pd.DataFrame(energy_comparison)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_energy = px.bar(
                energy_df,
                x='Technology',
                y='Energy Output (GWh/day)',
                title="Energy Output by Technology"
            )
            st.plotly_chart(fig_energy, use_container_width=True)
        
        with col2:
            fig_revenue = px.bar(
                energy_df,
                x='Technology',
                y='Revenue (M AED/day)',
                title="Daily Revenue by Technology"
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Technology comparison table
        st.markdown("### Technology Comparison")
        st.dataframe(energy_df, use_container_width=True)
        
    elif selected_view == "🌍 Carbon Impact":
        st.markdown("## 🌍 Environmental Impact Analysis")
        
        # Emissions comparison
        emissions_comparison = []
        scenarios = ['landfill', 'incineration', 'anaerobic_digestion', 'gasification']
        
        for scenario in scenarios:
            emissions = calculate_scenario_emissions(filtered_waste, energy_params, scenario)
            emissions_comparison.append({
                'Scenario': scenario.replace('_', ' ').title(),
                'CO₂ Emissions (tons/day)': emissions / 1000,
                'Carbon Value (k AED/day)': (emissions * energy_params['economic_params']['carbon_credit_price']) / 1e6
            })
        
        emissions_df = pd.DataFrame(emissions_comparison)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_emissions = px.bar(
                emissions_df,
                x='Scenario',
                y='CO₂ Emissions (tons/day)',
                title="CO₂ Emissions by Scenario"
            )
            st.plotly_chart(fig_emissions, use_container_width=True)
        
        with col2:
            fig_carbon = px.bar(
                emissions_df,
                x='Scenario',
                y='Carbon Value (k AED/day)',
                title="Carbon Credit Value"
            )
            st.plotly_chart(fig_carbon, use_container_width=True)
        
        # Environmental benefits
        st.markdown("### Environmental Benefits Summary")
        landfill_emissions = calculate_scenario_emissions(filtered_waste, energy_params, 'landfill')
        wte_emissions = calculate_scenario_emissions(filtered_waste, energy_params, selected_tech)
        emissions_avoided = (landfill_emissions - wte_emissions) / 1000
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🌍 CO₂ Avoided", f"{emissions_avoided:,.0f}", "tons/day")
        
        with col2:
            carbon_value = emissions_avoided * energy_params['economic_params']['carbon_credit_price']
            st.metric("💰 Carbon Value", f"{carbon_value/1e6:,.1f}", "M AED/day")
        
        with col3:
            trees_equivalent = emissions_avoided * 0.027  # ~27 trees per ton CO2
            st.metric("🌳 Trees Equivalent", f"{trees_equivalent:,.0f}", "trees/day")
    
    elif selected_view == "🤖 AI Advisor":
        st.markdown("## 🤖 AI-Powered Strategic Advisor")
        st.markdown("*Get intelligent recommendations for UAE waste-to-energy strategy*")
        
        # Scenario selector
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_scenario = st.selectbox("Select Policy Scenario", list(policy_scenarios.keys()), index=0)
            st.markdown(f"**{policy_scenarios[selected_scenario]['description']}**")
        
        with col2:
            st.metric("🎯 Target Recycling", f"{policy_scenarios[selected_scenario]['recycling_target_2030']:.0%}", "by 2030")
            st.metric("⚡ WTE Capacity", f"{policy_scenarios[selected_scenario]['wte_capacity_target_2030']:,}", "MW by 2030")
        
        # Get AI recommendations
        scenario_recommendations = ai_advisor_report['scenario_recommendations'][selected_scenario]
        
        # Executive Summary
        st.markdown("### 📋 Executive Summary")
        st.markdown(scenario_recommendations['summary'])
        
        # Current Situation Analysis
        st.markdown("### 📊 Current Situation Analysis")
        current_situation = scenario_recommendations['current_situation']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🗑️ Total Waste", f"{current_situation['total_waste_tons_per_day']:,.0f}", "tons/day")
        
        with col2:
            st.metric("⚡ WTE Capacity", f"{current_situation['total_wte_capacity_mw']:,}", "MW")
        
        with col3:
            st.metric("🏭 Landfill Capacity", f"{current_situation['landfill_capacity_tons_per_day']:,.0f}", "tons/day")
        
        with col4:
            avg_recycling = np.mean(list(current_situation['recycling_rates'].values()))
            st.metric("♻️ Avg Recycling Rate", f"{avg_recycling:.1%}", "current")
        
        # Strategic Recommendations
        st.markdown("### 🎯 Strategic Recommendations")
        
        recommendations = scenario_recommendations['recommendations']
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec['category']} - {rec['priority']} Priority"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Action:** {rec['action']}")
                    st.markdown(f"**Timeline:** {rec['timeline']}")
                    
                    if 'expected_impact' in rec:
                        st.markdown(f"**Expected Impact:** {rec['expected_impact']}")
                    
                    if 'opportunity' in rec:
                        st.markdown(f"**Opportunity:** {rec['opportunity']}")
                    
                    if 'policy_levers' in rec:
                        st.markdown("**Policy Levers:**")
                        for lever in rec['policy_levers']:
                            st.markdown(f"• {lever}")
                    
                    if 'funding_options' in rec:
                        st.markdown("**Funding Options:**")
                        for option in rec['funding_options']:
                            st.markdown(f"• {option}")
                
                with col2:
                    if 'estimated_cost' in rec:
                        st.markdown(f"**Cost:** {rec['estimated_cost']}")
                    
                    if 'roi_period' in rec:
                        st.markdown(f"**ROI:** {rec['roi_period']}")
                    
                    if 'revenue_potential' in rec:
                        st.markdown(f"**Revenue:** {rec['revenue_potential']}")
                    
                    if 'co2_reduction' in rec:
                        st.markdown(f"**CO₂ Reduction:** {rec['co2_reduction']}")
                    
                    if 'carbon_value' in rec:
                        st.markdown(f"**Carbon Value:** {rec['carbon_value']}")
        
        # Implementation Roadmap
        st.markdown("### 🗺️ Implementation Roadmap")
        roadmap = scenario_recommendations['implementation_roadmap']
        
        for phase, details in roadmap.items():
            with st.expander(f"📅 {phase}: {details['phase']}"):
                st.markdown("**Activities:**")
                for activity in details['activities']:
                    st.markdown(f"• {activity}")
                
                st.markdown("**Deliverables:**")
                for deliverable in details['deliverables']:
                    st.markdown(f"• {deliverable}")
        
        # Risk Assessment
        st.markdown("### ⚠️ Risk Assessment")
        risk_assessment = ai_advisor_report['risk_assessment']
        
        # Top risks
        top_risks = risk_assessment['risk_matrix'][:5]
        
        for risk in top_risks:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{risk['risk']}** ({risk['category']})")
                st.markdown(risk['description'])
            
            with col2:
                st.markdown(f"**Probability:** {risk['probability']:.1%}")
                st.markdown(f"**Impact:** {risk['impact']:.1%}")
            
            with col3:
                risk_score = risk['risk_score']
                if risk_score > 0.6:
                    st.markdown("🔴 **High Risk**")
                elif risk_score > 0.4:
                    st.markdown("🟡 **Medium Risk**")
                else:
                    st.markdown("🟢 **Low Risk**")
        
        # Mitigation Strategies
        st.markdown("### 🛡️ Mitigation Strategies")
        
        for category, strategies in risk_assessment['mitigation_strategies'].items():
            with st.expander(f"📋 {category.title()} Risks"):
                for strategy in strategies:
                    st.markdown(f"• {strategy}")
        
        # Knowledge Base Insights
        st.markdown("### 🧠 AI Knowledge Base")
        
        tab1, tab2, tab3 = st.tabs(["UAE Context", "Technology Insights", "Best Practices"])
        
        with tab1:
            for key, value in ai_advisor_report['knowledge_base']['uae_context'].items():
                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        
        with tab2:
            for tech, insights in ai_advisor_report['knowledge_base']['technology_insights'].items():
                st.markdown(f"### {tech.title()}")
                st.markdown(f"**UAE Suitability:** {insights['uae_suitability']}")
                st.markdown("**Pros:**")
                for pro in insights['pros']:
                    st.markdown(f"• {pro}")
                st.markdown("**Cons:**")
                for con in insights['cons']:
                    st.markdown(f"• {con}")
        
        with tab3:
            for country, practice in ai_advisor_report['knowledge_base']['waste_management_best_practices'].items():
                st.markdown(f"**{country.title()}:** {practice}")

# Helper functions
def create_map_data(waste_df, geo_data, optimization_results, selected_tech):
    """Create data for map visualization"""
    return {
        'waste_data': waste_df,
        'geo_data': geo_data,
        'optimization_results': optimization_results,
        'selected_tech': selected_tech
    }

def calculate_energy_potential(waste_df, energy_params, technology):
    """Calculate total energy potential for given technology"""
    total_energy = 0
    waste_composition = waste_df.groupby('waste_type')['tons_per_day'].sum()
    
    for waste_type, tons in waste_composition.items():
        yield_per_ton = energy_params['energy_yields'][technology].get(waste_type, 0)
        total_energy += tons * yield_per_ton
    
    return total_energy

def calculate_emissions_avoided(waste_df, energy_params, technology):
    """Calculate emissions avoided vs landfill"""
    landfill_emissions = calculate_scenario_emissions(waste_df, energy_params, 'landfill')
    wte_emissions = calculate_scenario_emissions(waste_df, energy_params, technology)
    return max(0, landfill_emissions - wte_emissions)

def calculate_scenario_emissions(waste_df, energy_params, scenario):
    """Calculate emissions for a given scenario"""
    total_emissions = 0
    waste_composition = waste_df.groupby('waste_type')['tons_per_day'].sum()
    
    for waste_type, tons in waste_composition.items():
        emission_factor = energy_params['emission_factors'][scenario].get(waste_type, 0)
        total_emissions += tons * emission_factor
    
    return total_emissions

if __name__ == "__main__":
    main()
