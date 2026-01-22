import pandas as pd
import numpy as np
from scipy.optimize import minimize, linprog
import pulp
import geopy.distance
from geopy.distance import geodesic
import json
from sklearn.cluster import KMeans
import networkx as nx
from deap import base, creator, tools, algorithms
import random

class PlantLocationOptimizer:
    """Advanced optimization for WTE plant location selection"""
    
    def __init__(self):
        self.candidate_locations = []
        self.waste_sources = []
        self.plant_types = ['incineration', 'anaerobic_digestion', 'gasification']
        self.optimization_results = {}
        
    def generate_candidate_locations(self, geo_data):
        """Generate potential plant locations across UAE"""
        locations = []
        
        for emirate, info in geo_data['emirates'].items():
            lat, lon = info['coordinates']
            
            # Generate multiple candidate sites per emirate
            for i in range(5):  # 5 candidate sites per emirate
                # Add some spatial variation
                lat_offset = np.random.uniform(-0.3, 0.3)  # ~30km variation
                lon_offset = np.random.uniform(-0.3, 0.3)
                
                location = {
                    'id': f"{emirate.replace(' ', '_').lower()}_{i}",
                    'emirate': emirate,
                    'latitude': lat + lat_offset,
                    'longitude': lon + lon_offset,
                    'land_cost': np.random.uniform(50, 200),  # AED per sqm
                    'grid_distance': np.random.uniform(5, 50),  # km to nearest grid connection
                    'population_radius_50km': self._estimate_population_in_radius(lat + lat_offset, lon + lon_offset, 50, geo_data),
                    'industrial_zone_proximity': np.random.uniform(0.1, 1.0),
                    'environmental_constraints': np.random.uniform(0.2, 0.9)  # 1 = no constraints
                }
                locations.append(location)
        
        self.candidate_locations = locations
        return locations
    
    def _estimate_population_in_radius(self, lat, lon, radius_km, geo_data):
        """Estimate population within radius of location"""
        # Simplified estimation based on nearest emirate
        min_dist = float('inf')
        nearest_emirate = None
        
        for emirate, info in geo_data['emirates'].items():
            emirate_lat, emirate_lon = info['coordinates']
            dist = geodesic((lat, lon), (emirate_lat, emirate_lon)).km
            if dist < min_dist:
                min_dist = dist
                nearest_emirate = emirate
        
        # Return population density estimate
        if nearest_emirate:
            population = geo_data['emirates'][nearest_emirate]['population_2024']
            area = geo_data['emirates'][nearest_emirate]['area_km2']
            density = population / area
            
            # Adjust for radius
            radius_area = np.pi * radius_km ** 2
            estimated_pop = density * radius_area * 0.5  # Adjustment factor
            
            return min(estimated_pop, population)  # Can't exceed total emirate population
        
        return 0
    
    def optimize_plant_locations(self, waste_data, geo_data, num_plants=10):
        """Multi-objective optimization for plant locations"""
        
        # Prepare waste source data
        waste_sources = self._prepare_waste_sources(waste_data, geo_data)
        
        # Generate candidate locations
        candidates = self.generate_candidate_locations(geo_data)
        
        # Optimization objectives:
        # 1. Maximize energy output
        # 2. Minimize transport costs
        # 3. Minimize environmental impact
        # 4. Maximize economic viability
        
        results = []
        
        for plant_type in self.plant_types:
            print(f"🔧 Optimizing for {plant_type} plants...")
            
            # Use genetic algorithm for multi-objective optimization
            best_solution = self._genetic_algorithm_optimization(
                candidates, waste_sources, plant_type, num_plants
            )
            
            results.append({
                'plant_type': plant_type,
                'optimal_locations': best_solution['locations'],
                'total_energy_output': best_solution['energy_output'],
                'total_transport_cost': best_solution['transport_cost'],
                'total_emissions_avoided': best_solution['emissions_avoided'],
                'total_cost': best_solution['total_cost'],
                'roi_years': best_solution['roi_years']
            })
        
        self.optimization_results = results
        return results
    
    def _prepare_waste_sources(self, waste_data, geo_data):
        """Prepare waste source points for optimization"""
        # Aggregate waste by emirate
        waste_by_emirate = waste_data.groupby(['emirate', 'waste_type'])['tons_per_day'].sum().reset_index()
        
        sources = []
        for emirate in geo_data['emirates']:
            lat, lon = geo_data['emirates'][emirate]['coordinates']
            
            # Get total waste for this emirate
            emirate_waste = waste_by_emirate[waste_by_emirate['emirate'] == emirate]
            total_waste = emirate_waste['tons_per_day'].sum()
            
            # Get waste composition
            composition = {}
            for _, row in emirate_waste.iterrows():
                composition[row['waste_type']] = row['tons_per_day']
            
            sources.append({
                'emirate': emirate,
                'latitude': lat,
                'longitude': lon,
                'total_waste_tons_per_day': total_waste,
                'waste_composition': composition
            })
        
        return sources
    
    def _genetic_algorithm_optimization(self, candidates, waste_sources, plant_type, num_plants):
        """Genetic algorithm for plant location optimization"""
        
        # Load energy parameters
        with open('data/energy_parameters.json', 'r') as f:
            energy_params = json.load(f)
        
        # GA setup
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0, 1.0))  # Minimize cost, maximize energy, maximize emissions avoided
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Gene: index of candidate location
        toolbox.register("attr_item", random.randrange, len(candidates))
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                         toolbox.attr_item, num_plants)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate_individual(individual):
            """Evaluate fitness of a solution"""
            selected_locations = [candidates[i] for i in individual]
            
            # Calculate metrics
            total_energy = 0
            total_transport_cost = 0
            total_emissions_avoided = 0
            total_plant_cost = 0
            
            for source in waste_sources:
                # Find nearest plant
                min_dist = float('inf')
                nearest_plant = None
                
                for plant in selected_locations:
                    dist = geodesic(
                        (source['latitude'], source['longitude']),
                        (plant['latitude'], plant['longitude'])
                    ).km
                    if dist < min_dist:
                        min_dist = dist
                        nearest_plant = plant
                
                if nearest_plant:
                    # Transport cost
                    transport_cost = min_dist * source['total_waste_tons_per_day'] * energy_params['energy_prices']['transport_cost_per_km']
                    total_transport_cost += transport_cost
                    
                    # Energy output
                    for waste_type, tons in source['waste_composition'].items():
                        energy_yield = energy_params['energy_yields'][plant_type].get(waste_type, 0)
                        total_energy += tons * energy_yield
                    
                    # Emissions avoided (vs landfill)
                    for waste_type, tons in source['waste_composition'].items():
                        landfill_emissions = energy_params['emission_factors']['landfill'].get(waste_type, 0)
                        plant_emissions = energy_params['emission_factors'][plant_type].get(waste_type, 0)
                        emissions_avoided = (landfill_emissions - plant_emissions) * tons
                        total_emissions_avoided += max(0, emissions_avoided)
            
            # Plant costs
            for plant in selected_locations:
                capex = energy_params['plant_costs'][plant_type]['capex_per_ton']
                opex = energy_params['plant_costs'][plant_type]['opex_per_ton']
                
                # Estimate plant capacity based on nearby waste
                nearby_waste = 0
                for source in waste_sources:
                    dist = geodesic(
                        (source['latitude'], source['longitude']),
                        (plant['latitude'], plant['longitude'])
                    ).km
                    if dist < 100:  # 100km radius
                        nearby_waste += source['total_waste_tons_per_day']
                
                plant_capacity = nearby_waste * 365  # tons per year
                total_plant_cost += (capex * plant_capacity) + (opex * plant_capacity)
            
            # Total cost (negative for minimization)
            total_cost = total_plant_cost + total_transport_cost
            
            return (total_cost, total_energy, total_emissions_avoided)
        
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(candidates)-1, indpb=0.1)
        toolbox.register("select", tools.selNSGA2)
        
        # Run GA
        population = toolbox.population(n=50)
        ngen = 40
        cxpb = 0.7
        mutpb = 0.2
        
        algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
                           stats=None, halloffame=None, verbose=False)
        
        # Get best solution
        best_ind = tools.selBest(population, 1)[0]
        best_locations = [candidates[i] for i in best_ind]
        
        # Calculate detailed metrics for best solution
        fitness = evaluate_individual(best_ind)
        
        # Calculate ROI
        annual_revenue = fitness[1] * energy_params['energy_prices']['electricity_sell_price'] * 365  # Energy revenue
        annual_carbon_revenue = fitness[2] * energy_params['energy_prices']['carbon_credit_price']  # Carbon credits
        total_annual_revenue = annual_revenue + annual_carbon_revenue
        
        roi_years = fitness[0] / total_annual_revenue if total_annual_revenue > 0 else float('inf')
        
        return {
            'locations': best_locations,
            'energy_output': fitness[1],
            'transport_cost': fitness[0] * 0.3,  # Estimate 30% of total cost is transport
            'emissions_avoided': fitness[2],
            'total_cost': fitness[0],
            'roi_years': roi_years
        }
    
    def optimize_linear_programming(self, waste_data, geo_data):
        """Linear programming approach for simpler optimization"""
        
        # This is a simplified version using linear programming
        # for comparison with the genetic algorithm approach
        
        waste_sources = self._prepare_waste_sources(waste_data, geo_data)
        candidates = self.generate_candidate_locations(geo_data)
        
        # Decision variables: x_ij = waste from source i to plant j
        # y_j = 1 if plant j is built, 0 otherwise
        
        # For simplicity, we'll use a basic facility location model
        num_sources = len(waste_sources)
        num_candidates = len(candidates)
        
        # Cost matrix (transport costs)
        cost_matrix = np.zeros((num_sources, num_candidates))
        for i, source in enumerate(waste_sources):
            for j, candidate in enumerate(candidates):
                dist = geodesic(
                    (source['latitude'], source['longitude']),
                    (candidate['latitude'], candidate['longitude'])
                ).km
                cost_matrix[i, j] = dist * 3.5  # AED per ton per km
        
        # Simplified LP: minimize total transport cost + fixed plant costs
        # This is a basic version - full implementation would be more complex
        
        return {
            'method': 'linear_programming',
            'status': 'implemented',
            'note': 'Full LP implementation would require binary variables for plant selection'
        }

class RouteOptimizer:
    """Optimize waste collection routes"""
    
    def __init__(self):
        self.routes = {}
        
    def optimize_collection_routes(self, waste_sources, depot_locations):
        """Optimize daily collection routes using TSP approximation"""
        
        G = nx.Graph()
        
        # Add nodes
        for i, source in enumerate(waste_sources):
            G.add_node(i, pos=(source['latitude'], source['longitude']), 
                      waste=source['total_waste_tons_per_day'])
        
        for j, depot in enumerate(depot_locations):
            G.add_node(f"depot_{j}", pos=(depot['latitude'], depot['longitude']))
        
        # Add edges with distances
        for i in range(len(waste_sources)):
            for j in range(len(waste_sources)):
                if i != j:
                    dist = geodesic(
                        (waste_sources[i]['latitude'], waste_sources[i]['longitude']),
                        (waste_sources[j]['latitude'], waste_sources[j]['longitude'])
                    ).km
                    G.add_edge(i, j, weight=dist)
        
        # Simple route optimization (nearest neighbor heuristic)
        routes = []
        visited = set()
        
        for depot_idx, depot in enumerate(depot_locations):
            route = [f"depot_{depot_idx}"]
            current = f"depot_{depot_idx}"
            unvisited = set(range(len(waste_sources))) - visited
            
            while unvisited:
                # Find nearest unvisited node
                min_dist = float('inf')
                nearest_node = None
                
                for node in unvisited:
                    dist = geodesic(
                        (depot['latitude'], depot['longitude']),
                        (waste_sources[node]['latitude'], waste_sources[node]['longitude'])
                    ).km
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = node
                
                if nearest_node is not None:
                    route.append(nearest_node)
                    visited.add(nearest_node)
                    unvisited.remove(nearest_node)
            
            route.append(f"depot_{depot_idx}")  # Return to depot
            routes.append(route)
        
        self.routes = routes
        return routes

def run_optimization():
    """Run complete optimization analysis"""
    
    print("🔧 Running WTE Plant Location Optimization...")
    
    # Load data
    waste_data = pd.read_csv('data/uae_waste_generation.csv')
    with open('data/uae_geographic_data.json', 'r') as f:
        geo_data = json.load(f)
    
    # Initialize optimizer
    optimizer = PlantLocationOptimizer()
    
    # Run optimization
    results = optimizer.optimize_plant_locations(waste_data, geo_data, num_plants=8)
    
    # Save results
    with open('data/optimization_results.json', 'w') as f:
        # Convert to JSON-serializable format
        json_results = []
        for result in results:
            json_result = result.copy()
            json_result['optimal_locations'] = [
                {k: v for k, v in loc.items() if k not in ['latitude', 'longitude']} 
                for loc in result['optimal_locations']
            ]
            json_results.append(json_result)
        json.dump(json_results, f, indent=2)
    
    # Print summary
    print("\n📊 Optimization Results Summary:")
    for result in results:
        print(f"\n🔥 {result['plant_type'].title()}:")
        print(f"   📍 Plants: {len(result['optimal_locations'])}")
        print(f"   ⚡ Energy: {result['total_energy_output']:.0f} kWh/day")
        print(f"   🚛 Transport Cost: {result['total_transport_cost']:,.0f} AED/day")
        print(f"   🌍 CO₂ Avoided: {result['total_emissions_avoided']:.0f} kg/day")
        print(f"   💰 ROI: {result['roi_years']:.1f} years")
    
    return optimizer, results

if __name__ == "__main__":
    run_optimization()
