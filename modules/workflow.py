"""
EcoIntel AI - LangGraph Workflow Orchestrator

Orchestrates the 7-node environmental assessment pipeline:
Input Processing → Missing Data Detection → Environmental Analysis →
Knowledge Retrieval → Scientific Reasoning → Recommendation Generation →
Report Creation
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from modules.models import GraphState, RiskLevel
from modules.analyzer import EnvironmentalAnalyzer
from modules.reasoner import MultiMetricReasoningEngine
from modules.retriever import ScientificKnowledgeBase
from modules.recommendations import RecommendationEngine
from modules.report_generator import AssessmentReportGenerator

logger = logging.getLogger(__name__)


class EcoIntelWorkflow:
    """
    LangGraph-based workflow that orchestrates the full EcoIntel AI
    biodiversity assessment pipeline.
    """

    def __init__(self, data_dir: str = "data", vectordb_dir: str = "vectordb"):
        """
        Initialize all components and build the LangGraph workflow.

        Args:
            data_dir: Path to directory containing PDF documents for RAG.
            vectordb_dir: Path to ChromaDB persistence directory.
        """
        logger.info("Initializing EcoIntel AI Workflow components...")

        # Initialize all pipeline components
        self.analyzer = EnvironmentalAnalyzer()
        self.reasoner = MultiMetricReasoningEngine()
        self.knowledge_base = ScientificKnowledgeBase(data_dir, vectordb_dir)
        self.recommendation_engine = RecommendationEngine()
        self.report_generator = AssessmentReportGenerator()

        # Index knowledge base documents
        num_chunks = self.knowledge_base.load_and_index_documents()
        logger.info(f"Knowledge base indexed: {num_chunks} chunks")

        # Build and compile the LangGraph workflow
        self.graph = self._build_graph()
        logger.info("EcoIntel AI Workflow initialized successfully.")

    def _build_graph(self):
        """Build and compile the LangGraph state graph with all nodes and edges."""
        workflow = StateGraph(GraphState)

        # Add all processing nodes
        workflow.add_node("input_processing", self._input_processing_node)
        workflow.add_node("missing_data_detection", self._missing_data_detection_node)
        workflow.add_node("environmental_analysis", self._environmental_analysis_node)
        workflow.add_node("knowledge_retrieval", self._knowledge_retrieval_node)
        workflow.add_node("scientific_reasoning", self._scientific_reasoning_node)
        workflow.add_node("recommendation_generation", self._recommendation_generation_node)
        workflow.add_node("report_creation", self._report_creation_node)

        # Define edges
        workflow.add_edge(START, "input_processing")
        workflow.add_edge("input_processing", "missing_data_detection")

        # Conditional routing: if too many critical fields missing, return follow-up questions
        def route_missing_data(state: GraphState) -> str:
            missing = state.get("missing_fields", [])
            iteration = state.get("iteration_count", 0)
            # If >2 critical fields missing AND this is the first pass, ask for more data
            if len(missing) > 2 and iteration == 0:
                return END
            return "environmental_analysis"

        workflow.add_conditional_edges(
            "missing_data_detection",
            route_missing_data,
            {END: END, "environmental_analysis": "environmental_analysis"},
        )

        workflow.add_edge("environmental_analysis", "knowledge_retrieval")
        workflow.add_edge("knowledge_retrieval", "scientific_reasoning")
        workflow.add_edge("scientific_reasoning", "recommendation_generation")
        workflow.add_edge("recommendation_generation", "report_creation")
        workflow.add_edge("report_creation", END)

        return workflow.compile()

    # -------------------------------------------------------------------------
    # Node implementations
    # -------------------------------------------------------------------------

    def _input_processing_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 1: Parse raw input (try JSON first, fall back to NL parsing)."""
        raw_input = state.get("raw_input", "")
        logger.info(f"[Node 1] Processing input: {raw_input[:100]}...")

        try:
            parsed = json.loads(raw_input)
            logger.info("[Node 1] Input parsed as structured JSON.")
        except (json.JSONDecodeError, TypeError):
            logger.info("[Node 1] Input is natural language. Parsing with LLM...")
            parsed = self.analyzer.parse_natural_language(raw_input)

        return {"parsed_input": parsed, "current_node": "input_processing"}

    def _missing_data_detection_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 2: Check for missing critical environmental data."""
        parsed_input = state.get("parsed_input", {})
        logger.info("[Node 2] Detecting missing data...")

        result = self.analyzer.detect_missing_data(parsed_input)

        if result is not None:
            # MissingDataRequest object returned
            return {
                "missing_fields": result.missing_fields,
                "follow_up_questions": result.questions,
                "current_node": "missing_data_detection",
            }

        return {
            "missing_fields": [],
            "follow_up_questions": [],
            "current_node": "missing_data_detection",
        }

    def _environmental_analysis_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 3: Perform multi-variable environmental analysis."""
        parsed_input = state.get("parsed_input", {})
        logger.info("[Node 3] Running environmental analysis...")

        analysis = self.analyzer.analyze_environment(parsed_input)

        return {
            "environmental_analysis": analysis,
            "current_node": "environmental_analysis",
        }

    def _knowledge_retrieval_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 4: Generate RAG queries and retrieve scientific evidence."""
        analysis = state.get("environmental_analysis", "")
        logger.info("[Node 4] Retrieving scientific knowledge...")

        # Use the reasoner to generate targeted RAG queries
        queries = self.reasoner.generate_rag_queries(analysis)

        if queries:
            # Multi-query retrieval for comprehensive evidence
            evidence_docs = self.knowledge_base.multi_query_retrieve(queries, k=3)
        else:
            # Fallback: use the raw analysis as a single query
            evidence_docs = self.knowledge_base.retrieve(analysis[:500], k=5)

        # Serialize Document objects for state storage (LangGraph state must be serializable)
        serialized_evidence = []
        for doc in evidence_docs:
            serialized_evidence.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            })

        logger.info(f"[Node 4] Retrieved {len(serialized_evidence)} evidence documents.")

        return {
            "retrieved_evidence": serialized_evidence,
            "current_node": "knowledge_retrieval",
        }

    def _scientific_reasoning_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 5: Cross-reference analysis with scientific evidence for reasoning."""
        analysis = state.get("environmental_analysis", "")
        evidence_dicts = state.get("retrieved_evidence", [])
        logger.info("[Node 5] Performing scientific reasoning...")

        # Reconstruct Document objects from serialized state
        evidence_docs = [
            Document(
                page_content=e.get("page_content", ""),
                metadata=e.get("metadata", {}),
            )
            for e in evidence_dicts
        ]

        # Perform scientific reasoning
        reasoning_output = self.reasoner.perform_scientific_reasoning(
            analysis, evidence_docs
        )

        return {
            "scientific_reasoning": json.dumps(reasoning_output),
            "current_node": "scientific_reasoning",
        }

    def _recommendation_generation_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 6: Generate evidence-backed biodiversity recommendations."""
        analysis = state.get("environmental_analysis", "")
        evidence_dicts = state.get("retrieved_evidence", [])
        reasoning_str = state.get("scientific_reasoning", "{}")
        parsed_input = state.get("parsed_input", {})
        logger.info("[Node 6] Generating recommendations...")

        # Parse reasoning output
        try:
            reasoning_output = json.loads(reasoning_str)
        except (json.JSONDecodeError, TypeError):
            reasoning_output = {}

        # Reconstruct Document objects
        evidence_docs = [
            Document(
                page_content=e.get("page_content", ""),
                metadata=e.get("metadata", {}),
            )
            for e in evidence_dicts
        ]

        # Generate LLM-based recommendations via reasoner
        raw_recommendations = self.reasoner.generate_recommendations(
            analysis, reasoning_output, evidence_docs
        )

        if raw_recommendations:
            # Format and validate through recommendation engine
            recommendations = self.recommendation_engine.format_recommendations(
                raw_recommendations
            )
        else:
            # Fallback to rule-based recommendations
            logger.warning("[Node 6] LLM recommendations failed. Using fallback rules.")
            recommendations = self.recommendation_engine.get_fallback_recommendations(
                parsed_input
            )

        # Enrich with evidence
        recommendations = self.recommendation_engine.enrich_recommendations(
            recommendations, evidence_dicts
        )

        # Serialize recommendations for state
        serialized_recs = [r.model_dump() for r in recommendations]

        return {
            "recommendations": serialized_recs,
            "current_node": "recommendation_generation",
        }

    def _report_creation_node(self, state: GraphState) -> Dict[str, Any]:
        """Node 7: Create the final structured assessment report."""
        logger.info("[Node 7] Creating assessment report...")

        parsed_input = state.get("parsed_input", {})
        analysis = state.get("environmental_analysis", "")
        reasoning_str = state.get("scientific_reasoning", "{}")
        rec_dicts = state.get("recommendations", [])
        evidence_dicts = state.get("retrieved_evidence", [])

        # Parse reasoning
        try:
            reasoning_output = json.loads(reasoning_str)
        except (json.JSONDecodeError, TypeError):
            reasoning_output = {}

        # Reconstruct Recommendation objects
        from modules.models import Recommendation

        recommendations = []
        for rd in rec_dicts:
            try:
                recommendations.append(Recommendation(**rd))
            except Exception as e:
                logger.warning(f"Failed to reconstruct recommendation: {e}")

        # Generate the report
        report = self.report_generator.generate_report(
            input_data=parsed_input,
            env_analysis=analysis,
            reasoning_output=reasoning_output,
            recommendations=recommendations,
            evidence=evidence_dicts,
        )

        # Convert report to serializable dict
        report_dict = report.model_dump()
        # Also generate text version
        report_text = self.report_generator.format_report_as_text(report)
        report_dict["formatted_text"] = report_text

        return {
            "assessment_report": report_dict,
            "current_node": "report_creation",
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, raw_input: str) -> Dict[str, Any]:
        """
        Execute the full assessment pipeline.

        Args:
            raw_input: Either a JSON string of environmental data or
                       a natural language description.

        Returns:
            Final state dict containing assessment_report or follow_up_questions.
        """
        initial_state: GraphState = {
            "raw_input": raw_input,
            "parsed_input": {},
            "missing_fields": [],
            "follow_up_questions": [],
            "environmental_analysis": None,
            "retrieved_evidence": [],
            "scientific_reasoning": None,
            "recommendations": [],
            "assessment_report": None,
            "error": None,
            "current_node": "start",
            "iteration_count": 0,
        }

        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return {**initial_state, "error": str(e)}

    def get_knowledge_base_stats(self) -> dict:
        """Return knowledge base statistics."""
        return self.knowledge_base.get_collection_stats()
