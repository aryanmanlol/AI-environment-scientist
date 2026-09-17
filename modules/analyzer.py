import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from modules.models import EnvironmentalInput, MissingDataRequest
from prompts.templates import INPUT_PARSER_PROMPT, MISSING_DATA_PROMPT, ENVIRONMENTAL_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

class EnvironmentalAnalyzer:
    """Handles environmental data parsing, validation, and missing data detection."""
    
    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        """
        Initialize the EnvironmentalAnalyzer.
        
        Args:
            model_name (str): The name of the Gemini model to use.
        """
        self.model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)

    def parse_natural_language(self, text: str) -> Dict[str, Any]:
        """
        Parses natural language text into structured environmental variables.
        
        Args:
            text (str): Natural language description.
            
        Returns:
            Dict[str, Any]: A dictionary of parsed environmental variables.
        """
        logger.info("Parsing natural language input.")
        try:
            messages = [
                SystemMessage(content=INPUT_PARSER_PROMPT),
                HumanMessage(content=text)
            ]
            response = self.model.invoke(messages)
            
            content = response.content.strip()
            # Clean up markdown JSON block if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for LLM output: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error during natural language parsing: {e}")
            return {}

    def detect_missing_data(self, parsed_input: Dict[str, Any]) -> Optional[MissingDataRequest]:
        """
        Detects missing critical environmental variables and generates follow-up questions.
        
        Args:
            parsed_input (Dict[str, Any]): The parsed environmental variables.
            
        Returns:
            Optional[MissingDataRequest]: A request for missing data or None if sufficient data exists.
        """
        logger.info("Checking for missing critical data.")
        critical_fields = ['soil_carbon', 'rainfall', 'temperature', 'land_use', 'region']
        
        missing_fields = [field for field in critical_fields if parsed_input.get(field) is None]
        
        if len(missing_fields) > 2:
            logger.info(f"Found {len(missing_fields)} missing critical fields. Generating follow-up questions.")
            try:
                prompt = MISSING_DATA_PROMPT.format(parsed_data=json.dumps(parsed_input, indent=2))
                messages = [HumanMessage(content=prompt)]
                response = self.model.invoke(messages)
                
                content = response.content.strip()
                # Clean up markdown JSON block if present
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                result = json.loads(content)
                return MissingDataRequest(
                    missing_fields=missing_fields,
                    questions=result.get("questions", []),
                    context="We need more details about the environment to provide an accurate and comprehensive analysis."
                )
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for missing data questions: {e}")
                return None
            except Exception as e:
                logger.error(f"Error during missing data detection: {e}")
                return None
                
        return None

    def analyze_environment(self, env_data: Dict[str, Any]) -> str:
        """
        Performs multi-variable analysis on the environmental variables.
        
        Args:
            env_data (Dict[str, Any]): The environmental variables.
            
        Returns:
            str: JSON-formatted string of the comprehensive environmental analysis.
        """
        logger.info("Performing comprehensive environmental analysis.")
        try:
            prompt = ENVIRONMENTAL_ANALYSIS_PROMPT.format(variables=json.dumps(env_data, indent=2))
            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages)
            
            return response.content
        except Exception as e:
            logger.error(f"Error during environmental analysis: {e}")
            return "{}"

    def process_input(self, raw_input: str) -> Tuple[Dict[str, Any], Optional[MissingDataRequest], str]:
        """
        Orchestrator method to process the raw input, check for missing data, and run analysis.
        
        Args:
            raw_input (str): The raw input (JSON or natural language).
            
        Returns:
            Tuple[Dict[str, Any], Optional[MissingDataRequest], str]: 
            (parsed_data, missing_data_request, analysis_result)
        """
        logger.info("Processing raw user input.")
        parsed_data = {}
        
        # Try to parse as JSON first
        try:
            parsed_data = json.loads(raw_input)
            logger.info("Successfully parsed input as JSON.")
        except json.JSONDecodeError:
            # Fallback to natural language parsing
            logger.info("Input is not JSON. Falling back to natural language parsing.")
            parsed_data = self.parse_natural_language(raw_input)
            
        missing_data_request = self.detect_missing_data(parsed_data)
        analysis = self.analyze_environment(parsed_data)
        
        return parsed_data, missing_data_request, analysis
