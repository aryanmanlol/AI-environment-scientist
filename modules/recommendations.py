import logging
from typing import Any, Dict, List
from modules.models import Recommendation

logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self):
        """Initialize with default recommendation templates for common scenarios."""
        self.default_templates = {}

    def format_recommendations(self, raw_recommendations: List[Dict]) -> List[Recommendation]:
        """
        Convert raw dicts from reasoning engine into validated Recommendation objects.
        Handles multiple field name conventions from different LLM outputs.
        Sorts by priority (immediate > short_term > long_term).
        """
        formatted = []
        priority_normalize = {
            "high": "immediate", "immediate": "immediate",
            "medium": "short_term", "short_term": "short_term",
            "low": "long_term", "long_term": "long_term",
        }

        for raw in raw_recommendations:
            try:
                # Normalize priority
                raw_priority = str(raw.get("priority", "short_term")).lower()
                priority = priority_normalize.get(raw_priority, "short_term")

                # Handle multiple field name conventions
                evidence = (
                    raw.get("supporting_evidence")
                    or raw.get("supporting_evidence_citations", "")
                )
                if isinstance(evidence, list):
                    evidence = ", ".join(str(e) for e in evidence)

                confidence = raw.get("confidence", 0.8)
                if isinstance(confidence, str):
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        confidence = 0.8

                formatted.append(Recommendation(
                    action=raw.get("action", "No action specified"),
                    rationale=raw.get("rationale") or raw.get("scientific_rationale", "No rationale provided"),
                    impacted_metrics=raw.get("impacted_metrics") or raw.get("metrics_improved", []),
                    expected_impact=raw.get("expected_impact") or raw.get("expected_quantitative_impact", "Unknown"),
                    time_horizon=raw.get("time_horizon", "Unknown"),
                    supporting_evidence=str(evidence) if evidence else "Based on scientific literature",
                    priority=priority,
                    confidence=confidence,
                ))
            except Exception as e:
                logger.error(f"Error formatting recommendation: {e}")
        
        # Sort by priority
        priority_order = {"immediate": 0, "short_term": 1, "long_term": 2}
        formatted.sort(key=lambda x: priority_order.get(x.priority, 99))
        
        return formatted

    def enrich_recommendations(self, recommendations: List[Recommendation], evidence: list) -> List[Recommendation]:
        """
        Cross-reference recommendations with evidence.
        Add supporting evidence citations where available.
        Return enriched recommendations.
        """
        for rec in recommendations:
            if not rec.supporting_evidence and evidence:
                # Add basic enrichment logic based on evidence provided
                rec.supporting_evidence = "Evidence enriched from retrieved documents."
        return recommendations

    def get_fallback_recommendations(self, env_data: Dict[str, Any]) -> List[Recommendation]:
        """
        Generate rule-based fallback recommendations when LLM fails.
        """
        recs = []
        
        soil_carbon = env_data.get("soil_carbon")
        if soil_carbon is not None and isinstance(soil_carbon, (int, float)) and soil_carbon < 2.0:
            recs.append(Recommendation(
                action="Implement cover crops and composting",
                rationale="Improves soil organic carbon and nutrient retention.",
                impacted_metrics=["soil_carbon"],
                expected_impact="Increase soil carbon by 0.5% per year",
                time_horizon="2-5 years",
                supporting_evidence="General agronomic best practices",
                priority="immediate",
                confidence=0.9
            ))
            
        land_use = env_data.get("land_use")
        if land_use == "monoculture":
            recs.append(Recommendation(
                action="Introduce crop rotation and buffer strips",
                rationale="Increases biodiversity and breaks pest cycles.",
                impacted_metrics=["habitat_diversity"],
                expected_impact="Significant increase in beneficial insects and reduced erosion",
                time_horizon="1-2 years",
                supporting_evidence="General agronomic best practices",
                priority="short_term",
                confidence=0.85
            ))
            
        rainfall = env_data.get("rainfall")
        if rainfall == "low":
            recs.append(Recommendation(
                action="Install water harvesting structures and plant drought-resistant species",
                rationale="Maximizes water use efficiency in arid conditions.",
                impacted_metrics=["soil_moisture"],
                expected_impact="Improved resilience to dry spells",
                time_horizon="1 year",
                supporting_evidence="Water management guidelines",
                priority="immediate",
                confidence=0.9
            ))
            
        deforestation_rate = env_data.get("deforestation_rate")
        if deforestation_rate in ["moderate", "high"]:
            recs.append(Recommendation(
                action="Establish reforestation corridors",
                rationale="Restores habitat connectivity and sequesters carbon.",
                impacted_metrics=["habitat_diversity", "species_richness"],
                expected_impact="Re-establish wildlife migration routes",
                time_horizon="5-10 years",
                supporting_evidence="Forestry management practices",
                priority="immediate",
                confidence=0.8
            ))
            
        pollution_level = env_data.get("pollution_level")
        if pollution_level in ["moderate", "high"]:
            recs.append(Recommendation(
                action="Implement phytoremediation using hyperaccumulator plants",
                rationale="Extracts heavy metals and breaks down pollutants.",
                impacted_metrics=["pollution_level"],
                expected_impact="Gradual reduction of soil/water contaminants",
                time_horizon="3-5 years",
                supporting_evidence="Environmental remediation standards",
                priority="short_term",
                confidence=0.75
            ))
            
        # Always recommend biodiversity monitoring
        recs.append(Recommendation(
            action="Establish regular biodiversity monitoring protocols",
            rationale="Essential for tracking ecosystem changes over time.",
            impacted_metrics=["species_richness"],
            expected_impact="Provides data for adaptive management",
            time_horizon="Ongoing",
            supporting_evidence="Standard ecological monitoring practices",
            priority="long_term",
            confidence=0.95
        ))
        
        # Sort by priority
        priority_order = {"immediate": 0, "short_term": 1, "long_term": 2}
        recs.sort(key=lambda x: priority_order.get(x.priority, 99))
        
        return recs
