"""
EcoIntel AI - Report Exporter Module

Exports AssessmentReport objects into standalone, self-contained HTML documents
ready for viewing, printing, or converting to PDF.
"""

import html
import os
import logging
from typing import Dict, Any

from modules.models import AssessmentReport

logger = logging.getLogger(__name__)


class ReportExporter:
    """Exports structured assessment reports into HTML / PDF formats."""

    @staticmethod
    def export_to_html(report: AssessmentReport, output_file: str) -> str:
        """
        Generate a self-contained, publication-ready HTML file from AssessmentReport.

        Args:
            report: The AssessmentReport object.
            output_file: Target file path to write HTML.

        Returns:
            Absolute path of written HTML file.
        """
        risk_color = {
            "CRITICAL": "#dc3545",
            "HIGH": "#fd7e14",
            "MEDIUM": "#ffc107",
            "LOW": "#198754",
        }.get(report.risk_level.value, "#6c757d")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EcoIntel AI Assessment Report - {html.escape(report.report_id)}</title>
    <style>
        body {{ font-family: 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 20px; }}
        .header {{ border-bottom: 2px solid #1b4332; padding-bottom: 15px; margin-bottom: 25px; }}
        .header h1 {{ color: #1b4332; margin: 0; font-size: 26px; }}
        .meta {{ color: #666; font-size: 13px; margin-top: 5px; }}
        .risk-badge {{ display: inline-block; background-color: {risk_color}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 14px; text-transform: uppercase; }}
        .section {{ margin-bottom: 30px; background: #fafafa; border-left: 4px solid #1b4332; padding: 15px 20px; border-radius: 0 8px 8px 0; }}
        .section h2 {{ color: #1b4332; margin-top: 0; font-size: 18px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
        .rec-card {{ background: white; border: 1px solid #e0e0e0; border-radius: 6px; padding: 15px; margin-bottom: 15px; }}
        .rec-priority {{ font-size: 11px; text-transform: uppercase; font-weight: bold; padding: 3px 8px; border-radius: 4px; background: #e9ecef; color: #495057; display: inline-block; margin-bottom: 8px; }}
        .metric-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .metric-table th, .metric-table td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; font-size: 14px; }}
        .metric-table th {{ background: #f1f3f5; }}
        .status-healthy {{ color: #198754; font-weight: bold; }}
        .status-degraded {{ color: #fd7e14; font-weight: bold; }}
        .status-critical {{ color: #dc3545; font-weight: bold; }}
        .footer {{ font-size: 12px; color: #888; text-align: center; margin-top: 40px; border-top: 1px solid #eee; padding-top: 15px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌿 EcoIntel AI Biodiversity Assessment Report</h1>
        <div class="meta">
            Report ID: {html.escape(report.report_id)} | Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
    </div>

    <div class="section">
        <h2>1. Ecosystem Health Status</h2>
        <p>{html.escape(report.ecosystem_health_status)}</p>
    </div>

    <div class="section">
        <h2>2. Risk Level</h2>
        <div><span class="risk-badge">Risk Level: {html.escape(report.risk_level.value)}</span></div>
    </div>

    <div class="section">
        <h2>3. Root Cause Analysis</h2>
        <ul>
        {"".join(f"<li><strong>{html.escape(rc.cause)}</strong> (Severity: {html.escape(rc.severity)})</li>" for rc in report.root_cause_analysis)}
        </ul>
    </div>

    <div class="section">
        <h2>4. Scientific Interpretation</h2>
        <p>{html.escape(report.scientific_interpretation)}</p>
    </div>

    <div class="section">
        <h2>5. Actionable Recommendations</h2>
        {"".join(f'''
        <div class="rec-card">
            <span class="rec-priority">Priority: {html.escape(rec.priority)}</span>
            <h4 style="margin: 5px 0;">{html.escape(rec.action)}</h4>
            <p><strong>Rationale:</strong> {html.escape(rec.rationale)}</p>
            <p><strong>Expected Impact:</strong> {html.escape(rec.expected_impact)} | <strong>Time Horizon:</strong> {html.escape(rec.time_horizon)}</p>
            <p style="font-size:12px; color:#555;"><strong>Evidence:</strong> {html.escape(rec.supporting_evidence)}</p>
        </div>
        ''' for rec in report.recommendations)}
    </div>

    <div class="section">
        <h2>6. Impacted Metrics</h2>
        <table class="metric-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Current Value</th>
                    <th>Status</th>
                    <th>Trend</th>
                </tr>
            </thead>
            <tbody>
            {"".join(f'''
                <tr>
                    <td>{html.escape(m.metric_name)}</td>
                    <td>{html.escape(m.current_value or 'N/A')}</td>
                    <td class="status-{html.escape(m.status.lower())}">{html.escape(m.status.upper())}</td>
                    <td>{html.escape(m.trend)}</td>
                </tr>
            ''' for m in report.environmental_metrics)}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>7. Time Horizon & Confidence</h2>
        <p><strong>Time Horizon:</strong> {html.escape(report.time_horizon)}</p>
        <p><strong>Confidence Score:</strong> {report.confidence_score:.0%}</p>
    </div>

    <div class="section">
        <h2>8. References & Citations</h2>
        <ul>
        {"".join(f"<li>{html.escape(ref)}</li>" for ref in report.references)}
        </ul>
    </div>

    <div class="footer">
        Generated by EcoIntel AI — AI Biodiversity Assessment Engine.
    </div>
</body>
</html>
"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML assessment report exported to: {output_file}")
        return os.path.abspath(output_file)
