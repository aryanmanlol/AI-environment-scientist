from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskLevel(str, Enum):
    """Enumeration for risk levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EnvironmentalInput(BaseModel):
    """Structured input for environmental data."""
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True)

    soil_carbon: Optional[float] = Field(None, description="Soil carbon percentage (0-100)", ge=0, le=100)
    soil_moisture: Optional[float] = Field(None, description="Soil moisture percentage (0-100)", ge=0, le=100)
    soil_ph: Optional[float] = Field(None, description="Soil pH level (0-14)", ge=0, le=14)
    rainfall: Optional[Union[str, float]] = Field(None, description="Rainfall as 'low/moderate/high' or float in mm")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    species_richness: Optional[Union[str, int]] = Field(None, description="Species richness 'low/moderate/high' or count")
    habitat_diversity: Optional[str] = Field(None, description="Habitat diversity 'low/moderate/high'")
    land_use: Optional[str] = Field(None, description="Land use type (monoculture, mixed_agriculture, agroforestry, forest, urban, grassland)")
    region: Optional[str] = Field(None, description="Region type (semi-arid, tropical, temperate, arid, boreal, mediterranean)")
    pollution_level: Optional[str] = Field(None, description="Pollution level (none/low/moderate/high)")
    deforestation_rate: Optional[str] = Field(None, description="Deforestation rate (none/low/moderate/high)")
    natural_language_input: Optional[str] = Field(None, description="Free-text query input")


class MissingDataRequest(BaseModel):
    """Request for follow-up questions when data is missing."""
    model_config = ConfigDict(populate_by_name=True)

    missing_fields: List[str] = Field(default_factory=list, description="List of fields that are missing but required")
    questions: List[str] = Field(default_factory=list, description="Follow-up questions to ask the user")
    context: str = Field(..., description="Context for why the data is needed")


class EnvironmentalMetric(BaseModel):
    """Individual metric assessment."""
    model_config = ConfigDict(populate_by_name=True)

    metric_name: str = Field(..., description="Name of the assessed metric")
    current_value: Optional[str] = Field(None, description="Current value of the metric")
    status: str = Field(..., description="Status (healthy/degraded/critical)")
    trend: str = Field(..., description="Trend (improving/stable/declining)")


class RootCause(BaseModel):
    """Analysis result for root causes."""
    model_config = ConfigDict(populate_by_name=True)

    cause: str = Field(..., description="The main root cause")
    contributing_factors: List[str] = Field(default_factory=list, description="List of contributing factors")
    affected_metrics: List[str] = Field(default_factory=list, description="Metrics affected by this cause")
    severity: str = Field(..., description="Severity of the root cause")


class EcologicalImpact(BaseModel):
    """Description of ecological impacts."""
    model_config = ConfigDict(populate_by_name=True)

    impact_description: str = Field(..., description="Description of the ecological impact")
    affected_species_groups: List[str] = Field(default_factory=list, description="List of affected species groups")
    ecosystem_services_affected: List[str] = Field(default_factory=list, description="List of affected ecosystem services")
    reversibility: str = Field(..., description="Reversibility (reversible/partially_reversible/irreversible)")


class Recommendation(BaseModel):
    """Actionable recommendation based on the assessment."""
    model_config = ConfigDict(populate_by_name=True)

    action: str = Field(..., description="Action to take")
    rationale: str = Field(..., description="Rationale for why it works")
    impacted_metrics: List[str] = Field(default_factory=list, description="Metrics impacted by this action")
    expected_impact: str = Field(..., description="Expected impact of the recommendation")
    time_horizon: str = Field(..., description="Time horizon for the recommendation")
    supporting_evidence: str = Field(..., description="Supporting evidence for the action")
    priority: str = Field(..., description="Priority (immediate/short_term/long_term)")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)


class ScientificEvidence(BaseModel):
    """Scientific evidence supporting the analysis or recommendations."""
    model_config = ConfigDict(populate_by_name=True)

    source: str = Field(..., description="Source of the evidence")
    relevant_text: str = Field(..., description="Relevant text excerpt")
    relevance_score: float = Field(..., description="Relevance score of the evidence")
    topic: str = Field(..., description="Topic of the evidence")


class AssessmentReport(BaseModel):
    """The full output report of the biodiversity assessment."""
    model_config = ConfigDict(populate_by_name=True)

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique report ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the report creation")
    ecosystem_health_status: str = Field(..., description="Overall ecosystem health status")
    risk_level: RiskLevel = Field(..., description="Assessed risk level")
    environmental_metrics: List[EnvironmentalMetric] = Field(default_factory=list, description="Individual metric assessments")
    root_cause_analysis: List[RootCause] = Field(default_factory=list, description="Root cause analysis results")
    ecological_impacts: List[EcologicalImpact] = Field(default_factory=list, description="Identified ecological impacts")
    scientific_interpretation: str = Field(..., description="Scientific interpretation of the data")
    recommendations: List[Recommendation] = Field(default_factory=list, description="Actionable recommendations")
    impacted_metrics_summary: List[str] = Field(default_factory=list, description="Summary of impacted metrics")
    time_horizon: str = Field(..., description="Time horizon for the overall assessment/recommendations")
    confidence_score: float = Field(..., description="Overall confidence score", ge=0, le=1)
    scientific_evidence: List[ScientificEvidence] = Field(default_factory=list, description="Supporting scientific evidence")
    references: List[str] = Field(default_factory=list, description="List of references")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Original input data used for assessment")
    methodology_notes: str = Field(..., description="Notes on the methodology used")


class AnalysisRequest(BaseModel):
    """API request model for triggering an analysis."""
    model_config = ConfigDict(populate_by_name=True)

    input_data: Optional[EnvironmentalInput] = Field(None, description="Structured environmental input data")
    natural_language: Optional[str] = Field(None, description="Free-text natural language query")
    session_id: Optional[str] = Field(None, description="Session ID for tracking conversations")


class GraphState(TypedDict):
    """State for the LangGraph workflow."""
    raw_input: str
    parsed_input: Optional[dict]
    missing_fields: list
    follow_up_questions: list
    environmental_analysis: Optional[str]
    retrieved_evidence: list
    scientific_reasoning: Optional[str]
    recommendations: list
    assessment_report: Optional[dict]
    error: Optional[str]
    current_node: str
    iteration_count: int
