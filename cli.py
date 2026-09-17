#!/usr/bin/env python3
"""
EcoIntel AI - Command Line Interface (CLI)

Allows executing ecosystem biodiversity assessments directly from terminal.

Usage:
    python cli.py --file data/sample_inputs/degraded_agricultural_land.json
    python cli.py --text "My soil carbon is low, rainfall is sparse, and land is monoculture wheat."
"""

import argparse
import json
import os
import sys
import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ecointel-cli")


def main():
    parser = argparse.ArgumentParser(
        description="EcoIntel AI - Biodiversity Assessment & Decision Support CLI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f", "--file",
        help="Path to JSON file containing environmental variables"
    )
    group.add_argument(
        "-t", "--text",
        help="Natural language description of ecosystem conditions"
    )

    parser.add_argument(
        "-o", "--output",
        help="Optional path to save output report (JSON format)"
    )
    parser.add_argument(
        "--html",
        help="Optional path to export HTML assessment report"
    )

    args = parser.parse_args()

    # Verify Google API Key is present
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error(
            "GOOGLE_API_KEY environment variable not set. "
            "Please set it in your environment or in a .env file."
        )
        sys.exit(1)

    # Determine input string
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"Input file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r") as f:
            raw_input = f.read()
    else:
        raw_input = args.text

    print("\n" + "=" * 60)
    print("[EcoIntel AI] Starting Ecosystem Assessment Pipeline...")
    print("=" * 60 + "\n")

    from modules.workflow import EcoIntelWorkflow

    workflow = EcoIntelWorkflow()
    result = workflow.run(raw_input)

    if result.get("error"):
        print(f"\n[Error] Error during assessment: {result['error']}")
        sys.exit(1)

    if result.get("assessment_report"):
        report = result["assessment_report"]
        print("\n" + report.get("formatted_text", "No formatted text available."))

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n[Saved] JSON Report saved to {args.output}")

        if args.html:
            from modules.models import AssessmentReport
            from modules.pdf_exporter import ReportExporter
            
            try:
                report_obj = AssessmentReport(**report)
                html_path = ReportExporter.export_to_html(report_obj, args.html)
                print(f"[Exported] HTML Report exported to: {html_path}")
            except Exception as e:
                logger.error(f"Failed to export HTML report: {e}")

    elif result.get("follow_up_questions"):
        print("\n[Data Insufficient] Additional data needed for assessment:")
        print("Missing critical fields:", ", ".join(result.get("missing_fields", [])))
        print("\nPlease provide answers to the following questions:")
        for q in result.get("follow_up_questions", []):
            print(f"  * {q}")
    else:
        print("\n[Notice] Assessment completed with state:", result)


if __name__ == "__main__":
    main()
