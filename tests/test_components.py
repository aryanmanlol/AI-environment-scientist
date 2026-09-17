"""
Unit tests for EcoIntel AI components.
"""

import unittest
from datetime import datetime, timezone
import uuid

from modules.models import (
    EnvironmentalInput,
    Recommendation,
    RiskLevel,
    EnvironmentalMetric,
    RootCause,
    EcologicalImpact,
    AssessmentReport,
)
from modules.recommendations import RecommendationEngine
from modules.report_generator import AssessmentReportGenerator


class TestEnvironmentalModels(unittest.TestCase):
    """Test Pydantic data model validation and defaults."""

    def test_environmental_input_validation(self):
        input_data = EnvironmentalInput(
            soil_carbon=1.5,
            soil_moisture=25.0,
            soil_ph=6.5,
            rainfall="low",
            temperature=30.0,
            land_use="monoculture",
        )
        self.assertEqual(input_data.soil_carbon, 1.5)
        self.assertEqual(input_data.land_use, "monoculture")

    def test_invalid_soil_carbon_validation(self):
        with self.assertRaises(ValueError):
            EnvironmentalInput(soil_carbon=150.0)  # Exceeds max 100%

    def test_invalid_soil_ph_validation(self):
        with self.assertRaises(ValueError):
            EnvironmentalInput(soil_ph=16.0)  # Exceeds max pH 14


class TestRecommendationEngine(unittest.TestCase):
    """Test recommendation formatting and fallback logic."""

    def setUp(self):
        self.engine = RecommendationEngine()

    def test_fallback_recommendations(self):
        env_data = {
            "soil_carbon": 0.5,
            "land_use": "monoculture",
            "rainfall": "low",
            "deforestation_rate": "high",
        }
        recs = self.engine.get_fallback_recommendations(env_data)
        self.assertGreater(len(recs), 0)
        
        actions = [r.action for r in recs]
        self.assertTrue(any("cover crop" in a.lower() for a in actions))
        self.assertTrue(any("monoculture" in r.rationale.lower() or "crop" in a.lower() for r, a in zip(recs, actions)))

    def test_format_recommendations(self):
        raw = [
            {
                "action": "Implement Agroforestry",
                "scientific_rationale": "Trees enhance SOC and microclimate.",
                "metrics_improved": ["soil_carbon", "habitat_diversity"],
                "expected_quantitative_impact": "+1.2% SOC over 3 years",
                "time_horizon": "2-3 years",
                "supporting_evidence": "ICRAF study 2021",
                "priority": "High",
                "confidence": 0.9,
            }
        ]
        formatted = self.engine.format_recommendations(raw)
        self.assertEqual(len(formatted), 1)
        rec = formatted[0]
        self.assertIsInstance(rec, Recommendation)
        self.assertEqual(rec.action, "Implement Agroforestry")
        self.assertEqual(rec.priority, "immediate")


class TestReportGenerator(unittest.TestCase):
    """Test assessment report text formatting."""

    def setUp(self):
        self.generator = AssessmentReportGenerator()

    def test_format_report_as_text(self):
        report = AssessmentReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            ecosystem_health_status="Degraded semi-arid ecosystem requiring immediate intervention.",
            risk_level=RiskLevel.HIGH,
            environmental_metrics=[
                EnvironmentalMetric(
                    metric_name="Soil Carbon",
                    current_value="0.4%",
                    status="critical",
                    trend="declining",
                )
            ],
            root_cause_analysis=[
                RootCause(
                    cause="Monoculture cultivation without organic inputs",
                    contributing_factors=["Low organic return", "Tillage"],
                    affected_metrics=["Soil Carbon", "Soil Moisture"],
                    severity="high",
                )
            ],
            ecological_impacts=[
                EcologicalImpact(
                    impact_description="Decline in soil microbial biomass and crop yields",
                    affected_species_groups=["Soil microbiome", "Pollinators"],
                    ecosystem_services_affected=["Primary production", "Nutrient cycling"],
                    reversibility="partially_reversible",
                )
            ],
            scientific_interpretation="The ecosystem exhibits advanced land degradation symptoms.",
            recommendations=[
                Recommendation(
                    action="Plant leguminous cover crops",
                    rationale="Fixes atmospheric nitrogen and increases SOC.",
                    impacted_metrics=["Soil Carbon", "Soil Nitrogen"],
                    expected_impact="+0.5% SOC in 2 years",
                    time_horizon="1-2 years",
                    supporting_evidence="FAO Soil Report 2020",
                    priority="immediate",
                    confidence=0.85,
                )
            ],
            impacted_metrics_summary=["Soil Carbon"],
            time_horizon="1-3 years",
            confidence_score=0.85,
            scientific_evidence=[],
            references=["FAO Soil Health Report 2020"],
            input_data={"soil_carbon": 0.4},
            methodology_notes="Generated using EcoIntel AI multi-metric engine.",
        )

        text = self.generator.format_report_as_text(report)
        self.assertIn("ECOINTEL AI - BIODIVERSITY ASSESSMENT REPORT", text)
        self.assertIn("1. ECOSYSTEM HEALTH STATUS", text)
        self.assertIn("HIGH", text)
        self.assertIn("Plant leguminous cover crops", text)
        self.assertIn("10. REFERENCES", text)


if __name__ == "__main__":
    unittest.main()
