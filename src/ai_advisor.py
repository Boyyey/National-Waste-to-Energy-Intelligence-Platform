import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import openai
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class UAEWTEAIAdvisor:
    """AI-powered advisor for UAE Waste-to-Energy policy and strategy"""
    
    def __init__(self):
        self.load_data()
        self.initialize_knowledge_base()
        
    def load_data(self):
        """Load all necessary data"""
        self.population_df = pd.read_csv('data/uae_population_realistic.csv')
        self.waste_df = pd.read_csv('data/uae_waste_realistic.csv')
        self.infrastructure_df = pd.read_csv('data/uae_infrastructure.csv')
        
        with open('data/energy_parameters_realistic.json', 'r') as f:
            self.energy_params = json.load(f)
        
        with open('data/uae_geographic_realistic.json', 'r') as f:
            self.geo_data = json.load(f)
        
        with open('data/policy_scenarios.json', 'r') as f:
            self.policy_scenarios = json.load(f)
    
    def initialize_knowledge_base(self):
        """Initialize AI knowledge base with UAE-specific insights"""
        self.knowledge_base = {
            'uae_context': {
                'vision_2031': 'UAE Vision 2031 emphasizes sustainable development and circular economy',
                'net_zero_2050': 'UAE Net Zero 2050 initiative aims for carbon neutrality',
                'national_waste_management': 'Federal Law No. 15 of 2017 on waste management',
                'cop28_legacy': 'UAE hosted COP28, strengthening climate commitments',
                'economic_diversification': 'Moving away from oil dependency towards green economy'
            },
            'waste_management_best_practices': {
                'singapore': 'Semakau Landfill - offshore landfill with energy recovery',
                'sweden': '99% of waste is recycled or energy recovered',
                'japan': 'Advanced incineration with high efficiency standards',
                'germany': 'Green Dot system - producer responsibility',
                'denmark': 'District heating from waste-to-energy plants'
            },
            'technology_insights': {
                'incineration': {
                    'pros': ['High volume capacity', 'Energy generation', 'Volume reduction'],
                    'cons': ['High CAPEX', 'Emissions concerns', 'Public opposition'],
                    'uae_suitability': 'Suitable for high-density urban areas like Dubai'
                },
                'anaerobic_digestion': {
                    'pros': ['Biogas production', 'Organic fertilizer', 'Low emissions'],
                    'cons': ['Limited to organic waste', 'Long processing time', 'Space requirements'],
                    'uae_suitability': 'Ideal for agricultural areas and food waste hubs'
                },
                'gasification': {
                    'pros': ['High efficiency', 'Syngas production', 'Lower emissions'],
                    'cons': ['Complex technology', 'High operational costs', 'Feedstock requirements'],
                    'uae_suitability': 'Emerging technology for industrial zones'
                }
            },
            'policy_levers': {
                'recycling_incentives': 'Tax breaks for companies using recycled materials',
                'landfill_taxes': 'Progressive landfill taxes to discourage disposal',
                'wte_subsidies': 'Feed-in tariffs for waste-to-energy electricity',
                'public_private_partnerships': 'PPP models for infrastructure development',
                'carbon_pricing': 'Carbon credit system for emissions reduction'
            }
        }
    
    def analyze_current_situation(self) -> Dict:
        """Analyze current waste management situation"""
        current_year = 2024
        current_waste = self.waste_df[self.waste_df['year'] == current_year]
        
        # Calculate key metrics
        total_waste = current_waste['tons_per_day'].sum()
        waste_by_emirate = current_waste.groupby('emirate')['tons_per_day'].sum().to_dict()
        
        # Infrastructure analysis
        total_wte_capacity = self.infrastructure_df[
            self.infrastructure_df['type'] == 'wte_plant'
        ]['capacity_mw'].sum()
        
        landfill_capacity = self.infrastructure_df[
            self.infrastructure_df['type'] == 'landfill'
        ]['capacity_tons_per_day'].sum()
        
        # Recycling rates
        recycling_rates = {}
        for emirate in waste_by_emirate.keys():
            if emirate in self.geo_data['real_waste_data']:
                recycling_rates[emirate] = self.geo_data['real_waste_data'][emirate]['recycling_rate']
        
        return {
            'total_waste_tons_per_day': total_waste,
            'waste_by_emirate': waste_by_emirate,
            'total_wte_capacity_mw': total_wte_capacity,
            'landfill_capacity_tons_per_day': landfill_capacity,
            'recycling_rates': recycling_rates,
            'waste_composition': current_waste.groupby('waste_type')['tons_per_day'].sum().to_dict()
        }
    
    def generate_strategic_recommendations(self, scenario: str = 'Current Policy') -> Dict:
        """Generate strategic recommendations based on scenario"""
        current_analysis = self.analyze_current_situation()
        scenario_params = self.policy_scenarios[scenario]
        
        recommendations = []
        
        # 1. Infrastructure Recommendations
        waste_gap = current_analysis['total_waste_tons_per_day'] - current_analysis['landfill_capacity_tons_per_day']
        additional_wte_needed = waste_gap * 0.4  # 40% of gap to be handled by WTE
        
        if additional_wte_needed > 0:
            recommendations.append({
                'category': 'Infrastructure',
                'priority': 'High',
                'action': f'Build additional {additional_wte_needed/1000:.1f}K tons/day WTE capacity',
                'timeline': '2025-2030',
                'estimated_cost': f'{additional_wte_needed * 750000:,.0f} AED',
                'expected_impact': f'Reduce landfill dependency by {additional_wte_needed/current_analysis["total_waste_tons_per_day"]*100:.1f}%',
                'implementation': {
                    'phases': ['Feasibility study', 'Site selection', 'Technology selection', 'Construction'],
                    'stakeholders': ['Ministry of Climate Change', 'Local municipalities', 'Private sector'],
                    'funding': ['Public-private partnership', 'Green bonds', 'International investment']
                }
            })
        
        # 2. Technology Recommendations
        best_technology = self._recommend_best_technology(current_analysis['waste_composition'])
        recommendations.append({
            'category': 'Technology',
            'priority': 'High',
            'action': f'Prioritize {best_technology} technology for new plants',
            'timeline': '2025-2027',
            'rationale': self.knowledge_base['technology_insights'][best_technology]['uae_suitability'],
            'expected_efficiency': f'{self._calculate_technology_efficiency(best_technology):.1f}% energy recovery',
            'implementation': {
                'pilot_projects': 2,
                'training_requirements': 'Technical staff certification',
                'maintenance_needs': 'Annual overhaul and monitoring'
            }
        })
        
        # 3. Policy Recommendations
        current_recycling_rate = np.mean(list(current_analysis['recycling_rates'].values()))
        target_recycling_rate = scenario_params['recycling_target_2030']
        
        if current_recycling_rate < target_recycling_rate:
            recommendations.append({
                'category': 'Policy',
                'priority': 'Medium',
                'action': f'Increase recycling rate from {current_recycling_rate:.1%} to {target_recycling_rate:.1%}',
                'timeline': '2025-2030',
                'policy_levers': [
                    'Implement extended producer responsibility (EPR)',
                    'Introduce landfill taxes starting at 50 AED/ton',
                    'Provide recycling subsidies to businesses',
                    'Launch public awareness campaigns'
                ],
                'expected_impact': f'Divert {current_analysis["total_waste_tons_per_day"] * (target_recycling_rate - current_recycling_rate):.0f} tons/day from landfill',
                'revenue_potential': f'{current_analysis["total_waste_tons_per_day"] * 365 * 50:,.0f} AED/year from landfill taxes'
            })
        
        # 4. Economic Recommendations
        energy_potential = self._calculate_energy_potential(current_analysis['waste_composition'])
        revenue_potential = energy_potential * self.energy_params['economic_params']['electricity_sell_price'] * 365
        
        recommendations.append({
            'category': 'Economic',
            'priority': 'Medium',
            'action': 'Monetize waste-to-energy potential',
            'timeline': '2025-2028',
            'opportunity': f'Generate {revenue_potential/1e6:.1f}M AED/year from electricity sales',
            'additional_revenue': f'{self._calculate_carbon_credits(current_analysis):,.0f} AED/year from carbon credits',
            'investment_required': f'{additional_wte_needed * 750000:,.0f} AED',
            'roi_period': f'{self._calculate_roi(additional_wte_needed, revenue_potential):.1f} years',
            'funding_options': ['Islamic bonds (Sukuk)', 'Green bonds', 'Foreign direct investment', 'Public-private partnerships']
        })
        
        # 5. Environmental Recommendations
        emissions_avoided = self._calculate_emissions_reduction(current_analysis)
        recommendations.append({
            'category': 'Environmental',
            'priority': 'High',
            'action': 'Maximize emissions reduction and carbon credits',
            'timeline': '2025-2030',
            'co2_reduction': f'{emissions_avoided/1e6:.1f}M tons CO2/year avoided',
            'carbon_value': f'{emissions_avoided * self.energy_params["economic_params"]["carbon_credit_price"] / 1e6:.1f}M AED/year',
            'air_quality_benefits': 'Reduced methane emissions from landfills',
            'climate_alignment': 'Supports UAE Net Zero 2050 initiative',
            'monitoring': 'Install continuous emissions monitoring systems'
        })
        
        return {
            'scenario': scenario,
            'current_situation': current_analysis,
            'recommendations': recommendations,
            'summary': self._generate_executive_summary(recommendations, scenario),
            'implementation_roadmap': self._create_implementation_roadmap(recommendations)
        }
    
    def _recommend_best_technology(self, waste_composition: Dict) -> str:
        """Recommend best technology based on waste composition"""
        organic_percentage = waste_composition.get('organic', 0) / sum(waste_composition.values())
        plastic_percentage = waste_composition.get('plastic', 0) / sum(waste_composition.values())
        
        if organic_percentage > 0.4:
            return 'anaerobic_digestion'
        elif plastic_percentage > 0.2:
            return 'gasification'
        else:
            return 'incineration'
    
    def _calculate_technology_efficiency(self, technology: str) -> float:
        """Calculate technology efficiency"""
        efficiency_map = {
            'incineration': 25.0,
            'anaerobic_digestion': 35.0,
            'gasification': 30.0
        }
        return efficiency_map.get(technology, 25.0)
    
    def _calculate_energy_potential(self, waste_composition: Dict) -> float:
        """Calculate total energy potential"""
        total_energy = 0
        for waste_type, tons in waste_composition.items():
            yield_per_ton = self.energy_params['energy_yields']['incineration'].get(waste_type, 0)
            total_energy += tons * yield_per_ton
        return total_energy
    
    def _calculate_carbon_credits(self, current_analysis: Dict) -> float:
        """Calculate carbon credit potential"""
        # Simplified calculation
        total_waste = current_analysis['total_waste_tons_per_day']
        return total_waste * 365 * 0.5 * self.energy_params['economic_params']['carbon_credit_price']
    
    def _calculate_emissions_reduction(self, current_analysis: Dict) -> float:
        """Calculate emissions reduction potential"""
        total_waste = current_analysis['total_waste_tons_per_day']
        # Average emissions avoided per ton of waste processed in WTE vs landfill
        return total_waste * 365 * 0.3  # tons CO2/year
    
    def _calculate_roi(self, investment: float, annual_revenue: float) -> float:
        """Calculate return on investment period"""
        if annual_revenue > 0:
            return investment / annual_revenue
        return float('inf')
    
    def _generate_executive_summary(self, recommendations: List[Dict], scenario: str) -> str:
        """Generate executive summary"""
        high_priority = [r for r in recommendations if r['priority'] == 'High']
        total_investment = sum([float(r.get('estimated_cost', '0').replace('AED', '').replace(',', '')) for r in recommendations if 'estimated_cost' in r])
        
        summary = f"""
        **UAE Waste-to-Energy Strategic Advisory Report**
        
        **Scenario:** {scenario}
        **Date:** {datetime.now().strftime('%Y-%m-%d')}
        
        **Executive Summary:**
        The UAE has a significant opportunity to transform its waste management sector into a sustainable energy resource. Based on current analysis, we recommend {len(high_priority)} high-priority initiatives requiring approximately {total_investment/1e9:.1f}B AED in investment.
        
        **Key Opportunities:**
        • Energy generation potential of {self._calculate_energy_potential(self.analyze_current_situation()['waste_composition'])/1e6:.1f} GWh/day
        • Carbon emissions reduction of {self._calculate_emissions_reduction(self.analyze_current_situation())/1e6:.1f}M tons/year
        • Economic value creation through circular economy principles
        
        **Strategic Focus:**
        1. Infrastructure expansion with {len([r for r in recommendations if r['category'] == 'Infrastructure'])} priority projects
        2. Technology optimization focusing on {self._recommend_best_technology(self.analyze_current_situation()['waste_composition'])}
        3. Policy alignment with UAE Vision 2031 and Net Zero 2050 targets
        
        This transformation positions the UAE as a regional leader in sustainable waste management and renewable energy.
        """
        return summary
    
    def _create_implementation_roadmap(self, recommendations: List[Dict]) -> Dict:
        """Create implementation roadmap"""
        roadmap = {
            '2024-2025': {
                'phase': 'Planning and Design',
                'activities': ['Feasibility studies', 'Site selection', 'Technology assessment', 'Policy framework development'],
                'deliverables': ['Master plan', 'Environmental impact assessment', 'Financial model']
            },
            '2025-2027': {
                'phase': 'Pilot Implementation',
                'activities': ['Pilot plant construction', 'Policy implementation', 'Stakeholder engagement', 'Training programs'],
                'deliverables': ['Operational pilot plants', 'Policy guidelines', 'Trained workforce']
            },
            '2027-2030': {
                'phase': 'Scale-up and Optimization',
                'activities': ['Full-scale plant construction', 'Grid integration', 'Performance optimization', 'Market development'],
                'deliverables': ['Full operational capacity', 'Integrated waste management system', 'Established market']
            },
            '2030-2035': {
                'phase': 'Maturation and Expansion',
                'activities': ['Technology upgrades', 'Capacity expansion', 'Export of expertise', 'Continuous improvement'],
                'deliverables': ['Advanced technology deployment', 'Regional leadership position', 'Knowledge transfer programs']
            }
        }
        return roadmap
    
    def generate_risk_assessment(self) -> Dict:
        """Generate comprehensive risk assessment"""
        risks = {
            'financial': {
                'high_capital_requirements': 'High initial investment requirements',
                'revenue_uncertainty': 'Uncertainty in electricity and carbon credit prices',
                'funding_delays': 'Potential delays in securing funding'
            },
            'technical': {
                'technology_failure': 'Risk of technology underperformance',
                'operational_challenges': 'Operational and maintenance challenges',
                'grid_integration': 'Challenges in grid integration and stability'
            },
            'regulatory': {
                'policy_changes': 'Potential changes in government policy',
                'permitting_delays': 'Delays in environmental and construction permits',
                'compliance_requirements': 'Stringent environmental compliance requirements'
            },
            'social': {
                'public_opposition': 'Potential public opposition to WTE plants',
                'community_acceptance': 'Community acceptance and NIMBY concerns',
                'workforce_development': 'Need for specialized workforce development'
            },
            'environmental': {
                'emissions_concerns': 'Concerns about air emissions and pollution',
                'waste_supply_variability': 'Variability in waste supply and composition',
                'climate_impact': 'Long-term climate change impacts'
            }
        }
        
        mitigation_strategies = {
            'financial': ['Diversified funding sources', 'Government guarantees', 'Revenue sharing agreements'],
            'technical': ['Technology provenance verification', 'Comprehensive testing', 'Expert partnerships'],
            'regulatory': ['Early stakeholder engagement', 'Compliance by design', 'Policy advocacy'],
            'social': ['Community benefit programs', 'Transparent communication', 'Local employment'],
            'environmental': ['Advanced emission controls', 'Continuous monitoring', 'Adaptive management']
        }
        
        return {
            'risks': risks,
            'mitigation_strategies': mitigation_strategies,
            'risk_matrix': self._create_risk_matrix(risks)
        }
    
    def _create_risk_matrix(self, risks: Dict) -> List[Dict]:
        """Create risk probability/impact matrix"""
        matrix = []
        for category, risk_items in risks.items():
            for risk_name, risk_description in risk_items.items():
                probability = np.random.uniform(0.3, 0.8)  # Simplified probability
                impact = np.random.uniform(0.4, 0.9)  # Simplified impact
                matrix.append({
                    'risk': risk_name.replace('_', ' ').title(),
                    'category': category.title(),
                    'probability': probability,
                    'impact': impact,
                    'risk_score': probability * impact,
                    'description': risk_description
                })
        
        return sorted(matrix, key=lambda x: x['risk_score'], reverse=True)

def generate_ai_advisor_report():
    """Generate comprehensive AI advisor report"""
    
    print("🤖 Generating AI Advisor Report for UAE WTE Strategy...")
    
    advisor = UAEWTEAIAdvisor()
    
    # Generate recommendations for different scenarios
    scenarios = ['Current Policy', 'Aggressive Green', 'Business as Usual', 'Crisis Response']
    all_recommendations = {}
    
    for scenario in scenarios:
        print(f"📋 Analyzing {scenario} scenario...")
        recommendations = advisor.generate_strategic_recommendations(scenario)
        all_recommendations[scenario] = recommendations
    
    # Generate risk assessment
    risk_assessment = advisor.generate_risk_assessment()
    
    # Save comprehensive report
    report = {
        'generated_at': datetime.now().isoformat(),
        'current_situation': advisor.analyze_current_situation(),
        'scenario_recommendations': all_recommendations,
        'risk_assessment': risk_assessment,
        'knowledge_base': advisor.knowledge_base
    }
    
    with open('data/ai_advisor_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ AI Advisor Report generated successfully!")
    print(f"📊 Analyzed {len(scenarios)} policy scenarios")
    print(f"🎯 Generated {sum(len(rec['recommendations']) for rec in all_recommendations.values())} recommendations")
    print(f"⚠️  Assessed {len(risk_assessment['risks'])} risk categories")
    
    return advisor, report

if __name__ == "__main__":
    generate_ai_advisor_report()
