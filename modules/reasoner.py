import json
import logging
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from modules.models import (
    RootCause, EcologicalImpact, Recommendation, 
    ScientificEvidence, RiskLevel
)
from prompts.templates import (
    SCIENTIFIC_REASONING_PROMPT, RECOMMENDATION_PROMPT, 
    RAG_QUERY_GENERATION_PROMPT
)

logger = logging.getLogger(__name__)


class MultiMetricReasoningEngine:
    """
    Multi-Metric Reasoning Engine for cross-variable scientific reasoning.
    """

    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        """
        Initializes the reasoning engine with a low-temperature LLM for scientific accuracy.
        """
        api_key = os.getenv("GOOGLE_API_KEY", "dummy_key_for_testing")
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.1,
                google_api_key=api_key,
                response_mime_type="application/json",
            )
        except Exception as e:
            logger.warning(f"Could not initialize ChatGoogleGenerativeAI in reasoner: {e}")
            self.llm = None

    def generate_rag_queries(self, environmental_analysis: str) -> List[str]:
        """
        Generates 3-5 targeted search queries for RAG based on the environmental analysis.
        """
        try:
            prompt = RAG_QUERY_GENERATION_PROMPT.format(analysis=environmental_analysis)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            data = json.loads(response.content)
            return data.get("queries", [])
        except Exception as e:
            logger.error(f"Error generating RAG queries: {e}")
            return []

    def perform_scientific_reasoning(self, environmental_analysis: str, evidence: List[Document]) -> Dict[str, Any]:
        """
        Combines the environmental analysis with retrieved scientific evidence to produce structured scientific reasoning.
        """
        try:
            formatted_evidence = self._format_evidence(evidence)
            # Override original prompt formatting to meet exact structured schema requirements for output dict
            format_instruction = (
                "\n\nCRITICAL INSTRUCTION - IGNORE PREVIOUS OUTPUT FORMAT AND USE THIS INSTEAD:\n"
                "Return a JSON object with exactly these keys and structure:\n"
                "- root_causes: list of objects with keys (cause, contributing_factors, affected_metrics, severity)\n"
                "- risk_drivers: list of strings representing key factors driving biodiversity risk\n"
                "- ecological_impacts: list of objects with keys (impact_description, affected_species_groups, ecosystem_services_affected, reversibility)\n"
                "- risk_level: string (LOW/MEDIUM/HIGH/CRITICAL)\n"
                "- confidence: float (0.0 to 1.0)\n"
                "- scientific_interpretation: string explaining the rationale"
            )
            prompt = SCIENTIFIC_REASONING_PROMPT.format(
                analysis=environmental_analysis,
                evidence=formatted_evidence
            ) + format_instruction
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            data = json.loads(response.content)
            
            # Defensive extraction in case LLM partially hallucinates format
            return {
                "root_causes": data.get("root_causes", []),
                "risk_drivers": data.get("risk_drivers", []),
                "ecological_impacts": data.get("ecological_impacts", []),
                "risk_level": data.get("risk_level", "MEDIUM"),
                "confidence": data.get("confidence", 0.5),
                "scientific_interpretation": data.get("scientific_interpretation", "")
            }
        except Exception as e:
            logger.error(f"Error in scientific reasoning: {e}")
            return {
                "root_causes": [],
                "risk_drivers": [],
                "ecological_impacts": [],
                "risk_level": "MEDIUM",
                "confidence": 0.0,
                "scientific_interpretation": f"Error during analysis: {e}"
            }

    def generate_recommendations(self, env_analysis: str, reasoning_output: Dict, evidence: List[Document]) -> List[Dict]:
        """
        Synthesizes the environmental analysis, reasoning, and evidence to generate prioritized recommendations.
        """
        try:
            formatted_evidence = self._format_evidence(evidence)
            format_instruction = (
                "\n\nCRITICAL INSTRUCTION - IGNORE PREVIOUS OUTPUT FORMAT AND USE THIS INSTEAD:\n"
                "Return a JSON object with a single key 'recommendations' containing a list of objects.\n"
                "Each recommendation MUST contain the following keys exactly:\n"
                "- action: what to do (string)\n"
                "- rationale: why it works based on science (string)\n"
                "- impacted_metrics: list of strings (metrics that improve)\n"
                "- expected_impact: quantitative where possible (string)\n"
                "- time_horizon: e.g. '1-2 years' (string)\n"
                "- supporting_evidence: from retrieved docs (string)\n"
                "- priority: immediate/short_term/long_term (string)\n"
                "- confidence: float between 0.0 and 1.0"
            )
            combined_analysis = f"Analysis:\n{env_analysis}\n\nReasoning Output:\n{json.dumps(reasoning_output)}"
            prompt = RECOMMENDATION_PROMPT.format(
                analysis=combined_analysis,
                evidence=formatted_evidence
            ) + format_instruction
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            data = json.loads(response.content)
            return data.get("recommendations", [])
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    def assess_risk_level(self, reasoning_output: Dict) -> RiskLevel:
        """
        Extracts the risk level from the reasoning output and maps it to the RiskLevel enum.
        Defaults to MEDIUM if unclear.
        """
        level_str = reasoning_output.get("risk_level", "MEDIUM")
        if isinstance(level_str, str):
            level_str = level_str.upper()
        
        try:
            return RiskLevel(level_str)
        except ValueError:
            logger.warning(f"Unrecognized risk level '{level_str}'. Defaulting to MEDIUM.")
            return RiskLevel.MEDIUM

    def _format_evidence(self, evidence: List[Document]) -> str:
        """
        Formats a list of retrieved documents into a readable string for prompt insertion.
        """
        formatted = []
        for doc in evidence:
            source = doc.metadata.get("source", "Unknown Source")
            content = doc.page_content
            formatted.append(f"Source: {source}, Content: {content}")
        
        return "\n\n".join(formatted)
