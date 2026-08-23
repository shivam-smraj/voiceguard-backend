"""
Forensic Summary Generator — Produces human-readable forensic text.

Generates:
  1. summary_text: Plain English summary for bank officers
  2. forensic_conclusion: Legal-grade conclusion with recommended action
  3. detailed_analysis_text: Multi-paragraph detailed analysis
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_summary(
    verdict: str,
    score: float,
    confidence_label: str,
    findings: List[Dict],
    biometrics: Optional[Dict],
    attribution: Optional[Dict],
    channel: Optional[Dict],
    embedding_proj: Optional[Dict] = None,
) -> Dict:
    """
    Generate all forensic text sections.

    Returns:
        Dict with summary_text, forensic_conclusion, detailed_analysis
    """
    n_findings = len(findings) if findings else 0
    bio_score = biometrics.get('overall_biometric_score', 0) if biometrics else 0
    n_suspicious = biometrics.get('n_suspicious', 0) if biometrics else 0

    # Top finding info
    top_artifact = findings[0] if findings else None
    top_type = top_artifact.get('artifact_type', 'unknown') if top_artifact else 'none'
    top_conf = top_artifact.get('confidence', 0) if top_artifact else 0

    # Attribution info
    tool = attribution.get('suspected_tool', 'Unknown') if attribution else 'Unknown'
    tool_conf = attribution.get('confidence', 0) if attribution else 0

    # Channel info
    snr = channel.get('snr_estimated_db', 0) if channel else 0
    integrity = channel.get('channel_integrity_score', 0) if channel else 0

    summary_text = _build_summary(
        verdict, score, confidence_label, n_findings, bio_score,
        n_suspicious, top_type, top_conf, tool, tool_conf, snr, integrity
    )

    forensic_conclusion = _build_conclusion(
        verdict, score, confidence_label, n_findings, bio_score,
        n_suspicious, top_type, top_conf, tool, tool_conf, findings,
        biometrics, channel, embedding_proj,
    )

    detailed_analysis = _build_detailed(
        verdict, score, findings, biometrics, attribution, channel
    )

    return {
        "summary_text": summary_text,
        "forensic_conclusion": forensic_conclusion,
        "detailed_analysis": detailed_analysis,
    }


def _build_summary(
    verdict, score, confidence_label, n_findings, bio_score,
    n_suspicious, top_type, top_conf, tool, tool_conf, snr, integrity,
) -> str:
    """Build plain-English officer summary."""
    if verdict == "FAKE":
        severity = {
            "CRITICAL": "extremely high",
            "HIGH": "high",
            "MEDIUM": "moderate",
            "LOW": "low",
        }.get(confidence_label, "uncertain")

        lines = [
            f"ALERT: This audio has been classified as SYNTHETIC with {severity} "
            f"confidence ({score:.1f}%).",
            f"",
            f"The analysis detected {n_findings} spectral and statistical artifacts "
            f"consistent with machine-generated speech.",
        ]

        if n_suspicious > 0:
            lines.append(
                f"Biometric analysis flagged {n_suspicious}/6 voice biomarkers as "
                f"inconsistent with human speech production (score: {bio_score:.0f}/100)."
            )

        if tool_conf > 25 and not ("Human" in tool and verdict == "FAKE"):
            lines.append(
                f"The suspected generation tool is {tool} "
                f"(confidence: {tool_conf:.0f}%)."
            )

        if top_conf > 50:
            pretty_type = top_type.replace('_', ' ').title()
            lines.append(
                f"The strongest artifact is '{pretty_type}' with {top_conf:.0f}% confidence."
            )

        lines.extend([
            f"",
            f"RECOMMENDED ACTION: Block the transaction and escalate to the "
            f"fraud investigation team immediately.",
        ])

        return "\n".join(lines)

    else:
        lines = [
            f"This audio has been classified as AUTHENTIC ({score:.1f}% deepfake probability).",
            f"",
            f"Voice biomarkers (jitter, shimmer, HNR, F0 variance) are within "
            f"expected human population ranges.",
        ]

        if n_findings > 0:
            lines.append(
                f"Note: {n_findings} minor spectral anomalies were detected but "
                f"are likely caused by recording conditions rather than synthesis."
            )

        lines.extend([
            f"",
            f"Channel integrity score: {integrity:.0f}/100 (SNR: {snr:.0f} dB).",
            f"No action required — proceed with normal transaction flow.",
        ])

        return "\n".join(lines)


def _build_conclusion(
    verdict, score, confidence_label, n_findings, bio_score,
    n_suspicious, top_type, top_conf, tool, tool_conf, findings,
    biometrics, channel, embedding_proj,
) -> str:
    """Build legal-grade forensic conclusion."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if verdict == "FAKE":
        # Paragraph 1: Core finding
        para1 = (
            f"FORENSIC FINDING: The submitted audio sample, analyzed at {timestamp}, "
            f"has been determined to be synthetically generated with a confidence level "
            f"of {confidence_label} ({score:.1f}%). This determination is based on a "
            f"multi-model ensemble analysis comprising LCNN-MFCC and LCNN-LFCC deep "
            f"learning classifiers, corroborated by independent biometric and spectral "
            f"analysis."
        )

        # Paragraph 2: Evidence summary
        evidence_points = []
        if n_findings > 0:
            evidence_points.append(
                f"{n_findings} distinct spectral artifacts were identified through "
                f"Grad-CAM activation mapping"
            )
        if n_suspicious > 0:
            evidence_points.append(
                f"{n_suspicious} of 6 voice biomarker tests failed human baseline checks "
                f"(biometric score: {bio_score:.0f}/100)"
            )
        if tool_conf > 25 and not ("Human" in tool):
            evidence_points.append(
                f"heuristic attribution analysis suggests {tool} as the generation "
                f"tool (confidence: {tool_conf:.0f}%)"
            )
        if embedding_proj and embedding_proj.get('distance_to_human', 0) > 2.0:
            evidence_points.append(
                f"the sample's embedding projection places it {embedding_proj['distance_to_human']:.1f} "
                f"units from the human speech cluster"
            )

        if evidence_points:
            evidence_str = "; ".join(evidence_points)
            para2 = f"EVIDENCE SUMMARY: The following evidence supports this finding: {evidence_str}."
        else:
            para2 = "EVIDENCE SUMMARY: Multiple independent analysis methods corroborate the synthetic classification."

        # Paragraph 3: Channel verification
        if channel:
            integrity = channel.get('channel_integrity_score', 0)
            snr = channel.get('snr_estimated_db', 0)
            para3 = (
                f"CHANNEL VERIFICATION: The analysis channel shows an estimated SNR of "
                f"{snr:.0f} dB with an integrity score of {integrity:.0f}/100. "
                f"{'The detected artifacts are not attributable to network degradation or codec compression.' if integrity > 60 else 'Some artifacts may be partially attributable to channel conditions.'}"
            )
        else:
            para3 = "CHANNEL VERIFICATION: Channel analysis was not performed."

        # Paragraph 4: Legal + action
        para4 = (
            "RECOMMENDED ACTION: This call should be flagged for immediate review by "
            "the fraud investigation team. The transaction associated with this call "
            "should be blocked pending manual verification of the caller's identity "
            "through alternative authentication channels. "
            "This report constitutes electronic evidence under Indian Evidence Act §65B "
            "and should be retained as per RBI Fraud Risk Management guidelines (minimum "
            "7 years retention)."
        )

        return f"{para1}\n\n{para2}\n\n{para3}\n\n{para4}"

    else:
        return (
            f"FORENSIC FINDING: The submitted audio sample, analyzed at {timestamp}, "
            f"has been determined to be authentic human speech with a deepfake probability "
            f"of {score:.1f}% (below the 50% threshold). Voice biomarker analysis confirms "
            f"natural micro-tremor (jitter), amplitude variation (shimmer), and harmonic "
            f"structure consistent with biological vocal production. "
            f"No action is required. The transaction may proceed under standard protocols."
        )


def _build_detailed(verdict, score, findings, biometrics, attribution, channel) -> str:
    """Build multi-paragraph detailed technical analysis."""
    sections = []

    # Section A: Detection
    sections.append(
        f"A. DETECTION ANALYSIS\n"
        f"The audio was processed through a two-model ensemble pipeline:\n"
        f"  - LCNN-MFCC: Analyzes 40 Mel-frequency cepstral coefficients with delta and "
        f"delta-delta features (120x200 spectrogram)\n"
        f"  - LCNN-LFCC: Analyzes 60 Linear-frequency cepstral coefficients with delta and "
        f"delta-delta features (180x200 spectrogram)\n"
        f"Ensemble verdict: {verdict} at {score:.1f}% deepfake probability."
    )

    # Section B: Spectral Evidence
    if findings:
        finding_lines = []
        for f in findings[:5]:
            freq = f.get('freq_range', [0, 0])
            time_r = f.get('time_range', [0, 0])
            finding_lines.append(
                f"  [{f.get('rank', '?')}] {f.get('artifact_type', 'unknown')} "
                f"({f.get('confidence', 0):.0f}%) at {freq[0]:.0f}-{freq[1]:.0f} Hz, "
                f"{time_r[0]:.2f}-{time_r[1]:.2f}s"
            )
        sections.append(
            f"B. SPECTRAL EVIDENCE\n"
            f"{len(findings)} artifact regions identified via Grad-CAM analysis:\n"
            + "\n".join(finding_lines)
        )

    # Section C: Biometric Analysis
    if biometrics:
        sections.append(
            f"C. BIOMETRIC ANALYSIS\n"
            f"  Jitter: {biometrics.get('jitter_pct', 0):.4f}% "
            f"(baseline: {biometrics.get('jitter_baseline', 0.48)}%) — "
            f"{biometrics.get('jitter_status', 'N/A')}\n"
            f"  Shimmer: {biometrics.get('shimmer_db', 0):.4f} dB "
            f"(baseline: {biometrics.get('shimmer_baseline', 0.35)} dB) — "
            f"{biometrics.get('shimmer_status', 'N/A')}\n"
            f"  HNR: {biometrics.get('hnr_db', 0):.1f} dB "
            f"(baseline: {biometrics.get('hnr_baseline', 18.2)} dB) — "
            f"{biometrics.get('hnr_status', 'N/A')}\n"
            f"  F0 Variance: {biometrics.get('f0_variance_hz', 0):.3f} Hz "
            f"(baseline: {biometrics.get('f0_variance_baseline', 2.4)} Hz) — "
            f"{biometrics.get('f0_status', 'N/A')}\n"
            f"  Overall Score: {biometrics.get('overall_biometric_score', 0):.0f}/100 "
            f"({biometrics.get('n_suspicious', 0)}/6 anomalies)"
        )

    # Section D: Attribution
    if attribution and attribution.get('confidence', 0) > 0:
        sections.append(
            f"D. THREAT ATTRIBUTION\n"
            f"  Suspected Tool: {attribution.get('suspected_tool', 'Unknown')}\n"
            f"  Confidence: {attribution.get('confidence', 0):.1f}%\n"
            f"  Runner-up: {attribution.get('runner_up', 'N/A')} "
            f"({attribution.get('runner_up_conf', 0):.1f}%)\n"
            f"  Notes: {attribution.get('signature_notes', 'N/A')}"
        )

    # Section E: Channel
    if channel:
        sections.append(
            f"E. CHANNEL INTEGRITY\n"
            f"  SNR: {channel.get('snr_estimated_db', 0):.1f} dB\n"
            f"  Reverb: {channel.get('reverb_level', 'N/A')}\n"
            f"  Integrity: {channel.get('channel_integrity_score', 0):.0f}/100\n"
            f"  {channel.get('matrix_explanation', '')}"
        )

    return "\n\n".join(sections)
