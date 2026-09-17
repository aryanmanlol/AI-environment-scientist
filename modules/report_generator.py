"""
EcoIntel AI - Assessment Report Generator

Synthesizes all pipeline outputs into a structured scientific assessment report
with 10 required sections: Ecosystem Health, Risk Level, Root Cause Analysis,
Scientific Interpretation, Recommendations, Impacted Metrics, Time Horizon,
Confidence Score, Scientific Evidence, and References.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from modules.models import (
    AssessmentReport,
    EnvironmentalMetric,
    RiskLevel,
    RootCause,
    EcologicalImpact,
    Recommendation,
    ScientificEvidence,
)
from prompts.templates import REPORT_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class AssessmentReportGenerator:
    """Generates structured scientific assessment reports from pipeline outputs."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initialize with LLM for narrative synthesis."""
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)

    def generate_report(
        self,
        input_data: dict,
        env_analysis: str,
        reasoning_output: dict,
        recommendations: List[Recommendation],
        evidence: List[dict],
    ) -> AssessmentReport:
        """
        Generate a complete structured assessment report.

        Args:
            input_data: Parsed environmental input variables.
            env_analysis: Multi-variable environmental analysis text.
            reasoning_output: Scientific reasoning results dict.
            recommendations: List of validated Recommendation objects.
            evidence: List of evidence dicts with page_content and metadata.

        Returns:
            Complete AssessmentReport object.
        """
        logger.info("Generating assessment report...")

        # Generate narrative sections via LLM
        narrative = self._generate_narrative(
            env_analysis, reasoning_output, recommendations, evidence
        )

        # Map reasoning output to structured RootCause objects
        root_causes = []
        for rc in reasoning_output.get("root_causes", []):
            if isinstance(rc, dict):
                root_causes.append(
                    RootCause(
                        cause=rc.get("cause", "Unknown"),
                        contributing_factors=rc.get("contributing_factors", []),
                        affected_metrics=rc.get("affected_metrics", []),
                        severity=rc.get("severity", "moderate"),
                    )
                )
            elif isinstance(rc, str):
                root_causes.append(
                    RootCause(
                        cause=rc,
                        contributing_factors=[],
                        affected_metrics=[],
                        severity="moderate",
                    )
                )

        # Map ecological impacts
        ecological_impacts = []
        for imp in reasoning_output.get("ecological_impacts", []):
            if isinstance(imp, dict):
                ecological_impacts.append(
                    EcologicalImpact(
                        impact_description=imp.get("impact_description", "Unknown"),
                        affected_species_groups=imp.get("affected_species_groups", []),
                        ecosystem_services_affected=imp.get(
                            "ecosystem_services_affected", []
                        ),
                        reversibility=imp.get("reversibility", "partially_reversible"),
                    )
                )
            elif isinstance(imp, str):
                ecological_impacts.append(
                    EcologicalImpact(
                        impact_description=imp,
                        affected_species_groups=[],
                        ecosystem_services_affected=[],
                        reversibility="partially_reversible",
                    )
                )

        # Determine risk level
        risk_level_str = reasoning_output.get("risk_level", "MEDIUM")
        try:
            risk_level = RiskLevel(risk_level_str.upper())
        except (ValueError, AttributeError):
            risk_level = RiskLevel.MEDIUM

        # Extract environmental metrics from input data
        metrics = self._extract_metrics(env_analysis, input_data)

        # Format scientific evidence from retrieved documents
        formatted_evidence = []
        for ev in evidence:
            formatted_evidence.append(
                ScientificEvidence(
                    source=ev.get("metadata", {}).get("source", "Unknown"),
                    relevant_text=ev.get("page_content", "")[:500],
                    relevance_score=0.8,
                    topic=ev.get("metadata", {}).get("topic", "General"),
                )
            )

        # Determine confidence score
        confidence = reasoning_output.get("confidence", 0.7)
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 0.7

        # Determine overall time horizon
        time_horizons = set()
        for rec in recommendations:
            if rec.time_horizon:
                time_horizons.add(rec.time_horizon)
        overall_time_horizon = ", ".join(time_horizons) if time_horizons else "1-5 years"

        # Build the report
        report = AssessmentReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            ecosystem_health_status=narrative.get(
                "ecosystem_health_status",
                "Assessment based on available environmental data.",
            ),
            risk_level=risk_level,
            environmental_metrics=metrics,
            root_cause_analysis=root_causes,
            ecological_impacts=ecological_impacts,
            scientific_interpretation=narrative.get(
                "scientific_interpretation",
                reasoning_output.get("scientific_interpretation", ""),
            ),
            recommendations=recommendations,
            impacted_metrics_summary=[m.metric_name for m in metrics if m.status != "healthy"],
            time_horizon=overall_time_horizon,
            confidence_score=confidence,
            scientific_evidence=formatted_evidence,
            references=self._format_references(evidence),
            input_data=input_data,
            methodology_notes=narrative.get(
                "methodology_notes",
                "Analysis performed using multi-variable environmental reasoning "
                "with RAG-based scientific evidence retrieval. Assessment generated "
                "by EcoIntel AI Biodiversity Assessment Engine.",
            ),
        )

        logger.info(f"Report generated: {report.report_id}")
        return report

    def _generate_narrative(
        self,
        env_analysis: str,
        reasoning_output: dict,
        recommendations: List[Recommendation],
        evidence: List[dict],
    ) -> dict:
        """Generate narrative sections via LLM."""
        try:
            evidence_text = "\n".join(
                [e.get("page_content", "")[:300] for e in evidence[:5]]
            )
            rec_text = "\n".join(
                [f"- {r.action}: {r.rationale}" for r in recommendations[:5]]
            )

            prompt = f"""You are an environmental scientist writing an assessment report.

Based on the following analysis, generate three narrative sections as JSON:

ENVIRONMENTAL ANALYSIS:
{env_analysis[:2000]}

SCIENTIFIC REASONING:
{json.dumps(reasoning_output, indent=2)[:2000]}

RECOMMENDATIONS:
{rec_text}

SCIENTIFIC EVIDENCE:
{evidence_text[:1500]}

Return a JSON object with exactly these keys:
- "ecosystem_health_status": A 2-3 sentence overview of the ecosystem's current health state.
- "scientific_interpretation": A detailed scientific interpretation (3-5 sentences) connecting the observed conditions to known ecological principles.
- "methodology_notes": A brief note on the methodology used for this assessment.
"""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            return {}

    def _extract_metrics(
        self, env_analysis: str, input_data: dict
    ) -> List[EnvironmentalMetric]:
        """
        Extract individual metric assessments from input data.

        Determines status based on known thresholds and categories.
        """
        metrics = []
        metric_status_rules = {
            "soil_carbon": lambda v: "critical" if isinstance(v, (int, float)) and v < 1.0
            else "degraded" if isinstance(v, (int, float)) and v < 2.0
            else "healthy",
            "soil_moisture": lambda v: "degraded" if isinstance(v, (int, float)) and v < 20
            else "healthy",
            "soil_ph": lambda v: "degraded" if isinstance(v, (int, float)) and (v < 5.5 or v > 8.5)
            else "healthy",
            "rainfall": lambda v: "degraded" if v in ("low",) or (isinstance(v, (int, float)) and v < 400)
            else "healthy",
            "temperature": lambda v: "degraded" if isinstance(v, (int, float)) and v > 35
            else "healthy",
            "species_richness": lambda v: "degraded" if v in ("low",)
            else "healthy",
            "habitat_diversity": lambda v: "degraded" if v in ("low",)
            else "healthy",
            "land_use": lambda v: "degraded" if v in ("monoculture", "urban")
            else "healthy",
            "pollution_level": lambda v: "critical" if v == "high"
            else "degraded" if v == "moderate"
            else "healthy",
            "deforestation_rate": lambda v: "critical" if v == "high"
            else "degraded" if v == "moderate"
            else "healthy",
        }

        for field, value in input_data.items():
            if value is None or field in ("natural_language_input",):
                continue

            status_fn = metric_status_rules.get(field)
            status = status_fn(value) if status_fn else "healthy"

            metrics.append(
                EnvironmentalMetric(
                    metric_name=field.replace("_", " ").title(),
                    current_value=str(value),
                    status=status,
                    trend="stable",  # Default; could be enhanced with time-series data
                )
            )

        return metrics

    def _format_references(self, evidence: List[dict]) -> List[str]:
        """Extract and format unique source references from evidence."""
        refs = set()
        for ev in evidence:
            source = ev.get("metadata", {}).get("source", "")
            topic = ev.get("metadata", {}).get("topic", "")
            year = ev.get("metadata", {}).get("year", "")
            if source:
                ref = source
                if topic:
                    ref += f" - {topic}"
                if year:
                    ref += f" ({year})"
                refs.add(ref)
        return sorted(refs)

    def format_report_as_text(self, report: AssessmentReport) -> str:
        """
        Convert the structured report to a human-readable text format
        with all 10 required sections.
        """
        lines = []

        lines.append("=" * 70)
        lines.append("ECOINTEL AI - BIODIVERSITY ASSESSMENT REPORT")
        lines.append("=" * 70)
        lines.append(f"Report ID: {report.report_id}")
        lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")

        # 1. Ecosystem Health Status
        lines.append("─" * 50)
        lines.append("1. ECOSYSTEM HEALTH STATUS")
        lines.append("─" * 50)
        lines.append(report.ecosystem_health_status)
        lines.append("")

        # 2. Risk Level
        lines.append("─" * 50)
        lines.append("2. RISK LEVEL")
        lines.append("─" * 50)
        lines.append(f"  ⚠ {report.risk_level.value}")
        lines.append("")

        # 3. Root Cause Analysis
        lines.append("─" * 50)
        lines.append("3. ROOT CAUSE ANALYSIS")
        lines.append("─" * 50)
        for i, rc in enumerate(report.root_cause_analysis, 1):
            lines.append(f"  {i}. {rc.cause}")
            lines.append(f"     Severity: {rc.severity}")
            if rc.contributing_factors:
                lines.append(f"     Contributing Factors: {', '.join(rc.contributing_factors)}")
            if rc.affected_metrics:
                lines.append(f"     Affected Metrics: {', '.join(rc.affected_metrics)}")
            lines.append("")

        # 4. Scientific Interpretation
        lines.append("─" * 50)
        lines.append("4. SCIENTIFIC INTERPRETATION")
        lines.append("─" * 50)
        lines.append(report.scientific_interpretation)
        lines.append("")

        # 5. Recommendations
        lines.append("─" * 50)
        lines.append("5. RECOMMENDATIONS")
        lines.append("─" * 50)
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"  [{rec.priority.upper()}] Recommendation {i}: {rec.action}")
            lines.append(f"    Why: {rec.rationale}")
            lines.append(f"    Impacted Metrics: {', '.join(rec.impacted_metrics)}")
            lines.append(f"    Expected Impact: {rec.expected_impact}")
            lines.append(f"    Time Horizon: {rec.time_horizon}")
            lines.append(f"    Evidence: {rec.supporting_evidence}")
            lines.append(f"    Confidence: {rec.confidence:.0%}")
            lines.append("")

        # 6. Impacted Metrics
        lines.append("─" * 50)
        lines.append("6. IMPACTED METRICS")
        lines.append("─" * 50)
        for m in report.environmental_metrics:
            status_icon = "✓" if m.status == "healthy" else "⚠" if m.status == "degraded" else "✗"
            lines.append(f"  {status_icon} {m.metric_name}: {m.current_value} [{m.status}]")
        lines.append("")

        # 7. Time Horizon
        lines.append("─" * 50)
        lines.append("7. TIME HORIZON")
        lines.append("─" * 50)
        lines.append(f"  {report.time_horizon}")
        lines.append("")

        # 8. Confidence Score
        lines.append("─" * 50)
        lines.append("8. CONFIDENCE SCORE")
        lines.append("─" * 50)
        lines.append(f"  {report.confidence_score:.0%}")
        lines.append("")

        # 9. Scientific Evidence
        lines.append("─" * 50)
        lines.append("9. SCIENTIFIC EVIDENCE")
        lines.append("─" * 50)
        for ev in report.scientific_evidence:
            lines.append(f"  Source: {ev.source} | Topic: {ev.topic}")
            lines.append(f"  {ev.relevant_text[:200]}...")
            lines.append("")

        # 10. References
        lines.append("─" * 50)
        lines.append("10. REFERENCES")
        lines.append("─" * 50)
        for ref in report.references:
            lines.append(f"  • {ref}")
        lines.append("")

        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)
