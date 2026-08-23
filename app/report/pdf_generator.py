"""
Forensic PDF Generator — Creates the authoritative multi-page forensic report.
Uses ReportLab for server-side PDF generation.

Sections (18 total):
  1. Secure Header Banner
  2. Report Metadata
  3. Verdict Banner (red/green)
  4. Ensemble Score Breakdown
  5. Biometric Biomarker Drift Table
  6. Pause Energy Analysis
  7. Formant F2 Velocity Analysis
  8. Biometric Radar Chart
  9. Audio Waveform
  10. Mel Spectrogram
  11. F0 Contour
  12. Grad-CAM Heatmaps (MFCC + LFCC)
  13. Artifact Findings Table
  14. Channel Integrity Matrix
  15. Threat Attribution + Source Probabilities
  16. Embedding Projection Scatter Plot
  17. Forensic Conclusion + Summary
  18. Technical Metadata + Certification
"""

import io
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether
)

logger = logging.getLogger(__name__)

# ── Color Palette ─────────────────────────────────────────────────────
DARK_BG       = colors.HexColor("#1a1a2e")
HEADER_BG     = colors.HexColor("#16213e")
FAKE_RED      = colors.HexColor("#e74c3c")
REAL_GREEN    = colors.HexColor("#27ae60")
ACCENT_BLUE   = colors.HexColor("#3498db")
WARNING_AMBER = colors.HexColor("#f39c12")
TEXT_LIGHT    = colors.HexColor("#ecf0f1")
TEXT_DARK     = colors.HexColor("#2c3e50")
TABLE_HEADER  = colors.HexColor("#2c3e50")
TABLE_ALT     = colors.HexColor("#f8f9fa")
BORDER_COLOR  = colors.HexColor("#bdc3c7")
SECTION_BG    = colors.HexColor("#eaf2f8")
CRITICAL_BG   = colors.HexColor("#fdedec")
SAFE_BG       = colors.HexColor("#eafaf1")


def generate_forensic_pdf(analysis_result: Dict, output_path: str = None) -> bytes:
    """Generate a complete multi-page forensic PDF report."""
    buf = io.BytesIO()

    page_info = {"page_num": 0}

    def _on_page(canvas, doc):
        """Add page number footer to every page."""
        page_info["page_num"] += 1
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.gray)
        canvas.drawCentredString(
            A4[0] / 2, 8 * mm,
            f"VoiceGuard AI Forensic Report — Page {page_info['page_num']} — {analysis_result.get('audit_id', 'N/A')}"
        )
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=18*mm, leftMargin=18*mm,
        topMargin=12*mm, bottomMargin=18*mm,
        title=f"VoiceGuard AI Forensic Report - {analysis_result.get('audit_id', 'N/A')}",
        author="VoiceGuard AI v2.0",
    )

    styles = _build_styles()
    elements = []
    xai = analysis_result.get('xai') or {}

    # ━━━━━━━━━━ PAGE 1: VERDICT + SCORES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # §1 Secure Header
    elements.extend(_section_header(analysis_result, styles))

    # Issue 9: Processing Mode Notice
    processing_note = xai.get('processing_note', '')
    if processing_note:
        elements.append(Spacer(1, 2*mm))
        note_style = ParagraphStyle(
            'ProcessingNote', parent=styles['Normal'],
            fontSize=7.5, textColor=colors.HexColor('#e67e22'),
            leftIndent=8, rightIndent=8,
        )
        elements.append(Paragraph(
            f"\u26a0\ufe0f <i>{processing_note}</i>", note_style
        ))
        elements.append(Spacer(1, 2*mm))

    # Issue 4: Analysis Window Note
    window_s = xai.get('analysis_window_s', 2.0)
    full_dur = xai.get('full_duration_s', 0)
    if full_dur > 3.0:
        elements.append(Paragraph(
            f"<i>Artifact analysis performed on {window_s}s analysis window "
            f"(most suspicious chunk of {full_dur:.1f}s total audio). "
            f"Visual plots display full audio duration for context.</i>",
            ParagraphStyle('WindowNote', parent=styles['Normal'],
                          fontSize=7, textColor=colors.gray, leftIndent=8),
        ))
        elements.append(Spacer(1, 2*mm))

    # §2 Report Metadata
    elements.extend(_section_metadata(analysis_result, styles))
    elements.append(Spacer(1, 5*mm))

    # §3 Verdict Banner
    elements.extend(_section_verdict(analysis_result, styles))
    elements.append(Spacer(1, 5*mm))

    # Officer Summary (NEW)
    summary_text = xai.get('summary_text', '')
    if summary_text:
        elements.extend(_section_summary_box(summary_text, analysis_result, styles))
        elements.append(Spacer(1, 5*mm))

    # §4 Ensemble Breakdown
    elements.extend(_section_ensemble(analysis_result, styles))
    elements.append(Spacer(1, 5*mm))

    # ━━━━━━━━━━ PAGE 2: BIOMETRIC ANALYSIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elements.append(PageBreak())

    # §5 Biometric Drift Table
    bio = xai.get('biometric_drift')
    if bio:
        elements.extend(_section_biometrics(bio, styles))
        elements.append(Spacer(1, 4*mm))

    # §6 Pause Energy Analysis (NEW)
    if bio and bio.get('pause_energy'):
        elements.extend(_section_pause_energy(bio, styles))
        elements.append(Spacer(1, 4*mm))

    # §7 Formant F2 Velocity (NEW)
    if bio and bio.get('formant_velocity'):
        elements.extend(_section_formant_velocity(bio, styles))
        elements.append(Spacer(1, 4*mm))

    # §8 Biometric Radar Chart (NEW)
    radar_b64 = xai.get('biometric_radar_b64')
    if not radar_b64 and bio:
        # Generate radar inline if not provided by engine
        try:
            from app.xai.spectrogram import render_biometric_radar
            radar_b64 = render_biometric_radar(bio)
        except Exception:
            pass

    if radar_b64:
        elements.extend(_section_image(
            radar_b64, "BIOMETRIC RADAR CHART",
            "Measured biomarkers vs human baseline (green ring = 1.0x baseline)",
            styles, width=100*mm, height=100*mm,
        ))
        elements.append(Spacer(1, 4*mm))

    # ━━━━━━━━━━ PAGE 3: VISUAL EVIDENCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elements.append(PageBreak())

    # §9 Audio Waveform (NEW)
    waveform_b64 = xai.get('waveform_b64')
    if waveform_b64:
        elements.extend(_section_image(
            waveform_b64, "AUDIO WAVEFORM",
            "Time-domain amplitude view with RMS envelope (red)",
            styles, width=170*mm, height=45*mm,
        ))
        elements.append(Spacer(1, 4*mm))

    # §10 Mel Spectrogram (NEW)
    spec_b64 = xai.get('spectrogram_b64')
    if spec_b64:
        elements.extend(_section_image(
            spec_b64, "MEL SPECTROGRAM",
            "Frequency × Time representation — reveals spectral energy distribution",
            styles, width=170*mm, height=60*mm,
        ))
        elements.append(Spacer(1, 4*mm))

    # §11 F0 Contour (NEW)
    f0_b64 = xai.get('f0_contour_b64')
    if f0_b64:
        elements.extend(_section_image(
            f0_b64, "FUNDAMENTAL FREQUENCY (F0) CONTOUR",
            "Pitch track showing voiced/unvoiced segments — flat F0 indicates synthesis",
            styles, width=170*mm, height=50*mm,
        ))
        elements.append(Spacer(1, 4*mm))

    # ━━━━━━━━━━ PAGE 4: GRAD-CAM EVIDENCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elements.append(PageBreak())

    # §12 Grad-CAM Heatmaps
    heatmap_elements = _section_heatmaps(xai, styles)
    if heatmap_elements:
        elements.extend(heatmap_elements)
        elements.append(Spacer(1, 4*mm))

    # §13 Artifact Findings Table
    if xai.get('findings'):
        elements.extend(_section_findings(xai['findings'], styles))
        elements.append(Spacer(1, 4*mm))

    # §19 GradientSHAP (NEW)
    shap_b64 = xai.get('shap_b64')
    if shap_b64:
        elements.append(PageBreak())
        elements.extend(_section_shap(shap_b64, styles))
        elements.append(Spacer(1, 4*mm))

        # §20 Phoneme Timeline (NEW)
        transcript = xai.get('transcript', '')
        timeline = xai.get('phoneme_timeline', [])
        if transcript or timeline:
            elements.extend(_section_phoneme_timeline(transcript, timeline, styles))
            elements.append(Spacer(1, 4*mm))

    # ━━━━━━━━━━ PAGE 5: ATTRIBUTION + EMBEDDING ━━━━━━━━━━━━━━━━━━━━━━
    elements.append(PageBreak())

    # §14 Channel Integrity
    if xai.get('channel_matrix'):
        elements.extend(_section_channel(xai['channel_matrix'], styles))
        elements.append(Spacer(1, 4*mm))

    # §15 Threat Attribution + Source Probabilities (ENHANCED)
    attr = xai.get('threat_attribution')
    if attr:
        elements.extend(_section_attribution_enhanced(attr, styles))
        elements.append(Spacer(1, 4*mm))

    # §16 Embedding Projection (NEW)
    emb_proj = xai.get('embedding_projection')
    if emb_proj:
        elements.extend(_section_embedding(emb_proj, styles))
        elements.append(Spacer(1, 4*mm))

    # §21 Voice Enrollment Match (NEW)
    speaker_match = xai.get('speaker_match')
    if speaker_match:
        elements.extend(_section_voice_enrollment(speaker_match, styles))
        elements.append(Spacer(1, 4*mm))

    # Embedding scatter plot (NEW)
    emb_plot_b64 = xai.get('embedding_plot_b64')
    if not emb_plot_b64 and emb_proj:
        try:
            from app.xai.embedding import render_embedding_plot
            emb_plot_b64 = render_embedding_plot(emb_proj)
        except Exception:
            pass

    if emb_plot_b64:
        elements.extend(_section_image(
            emb_plot_b64, "EMBEDDING SPACE PROJECTION",
            "PCA 2D projection of 256-dim LCNN embedding vs reference clusters",
            styles, width=140*mm, height=100*mm,
        ))
        elements.append(Spacer(1, 4*mm))

    # ━━━━━━━━━━ PAGE 6: CONCLUSION + CERTIFICATION ━━━━━━━━━━━━━━━━━━━
    elements.append(PageBreak())

    # §17 Forensic Conclusion (ENHANCED)
    elements.extend(_section_conclusion_enhanced(analysis_result, xai, styles))
    elements.append(Spacer(1, 4*mm))

    # Detailed Analysis Text (NEW)
    detailed = xai.get('detailed_analysis', '')
    if detailed:
        elements.extend(_section_detailed_analysis(detailed, styles))
        elements.append(Spacer(1, 4*mm))

    # §18 Technical Metadata
    elements.extend(_section_technical(analysis_result, styles))
    elements.append(Spacer(1, 4*mm))

    # Certification Footer
    elements.extend(_section_certification(analysis_result, styles))

    # Build PDF with page number callback
    doc.build(elements, onFirstPage=_on_page, onLaterPages=_on_page)
    pdf_bytes = buf.getvalue()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
        pages = pdf_bytes.count(b'/Type /Page') - 1
        logger.info(f"PDF saved: {output_path} ({len(pdf_bytes)/1024:.0f} KB, ~{pages} pages)")

    return pdf_bytes


# ══════════════════════════════════════════════════════════════════════
#  Section Builders
# ══════════════════════════════════════════════════════════════════════

def _section_header(result, styles):
    """§1: Secure header banner."""
    data = [[
        Paragraph("<b>VOICEGUARD AI</b>", styles['header_title']),
        Paragraph("FORENSIC ANALYSIS REPORT", styles['header_sub']),
    ]]
    t = Table(data, colWidths=[80*mm, 94*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), TABLE_HEADER),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('ROUNDEDCORNERS', [4, 4, 0, 0]),
    ]))
    return [t, Spacer(1, 2*mm)]


def _section_metadata(result, styles):
    """§2: Report metadata box."""
    audit_id = result.get('audit_id', 'N/A')
    call_id = result.get('call_id', 'N/A')
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    audio_info = result.get('audio_info', {})
    duration = audio_info.get('duration_s', 0)
    sr = audio_info.get('sample_rate', 0)

    data = [
        ['Audit ID', audit_id, 'Call ID', call_id],
        ['Timestamp', timestamp, 'Duration', f'{duration:.1f}s'],
        ['Original SR', f'{sr}Hz', 'Processing SR', f"{audio_info.get('processing_sr', 16000)}Hz"],
        ['Audio Hash', audio_info.get('audio_hash', 'N/A')[:24] + '...', 'File Size', f"{audio_info.get('file_size_mb', 0):.2f} MB"],
    ]
    t = Table(data, colWidths=[28*mm, 58*mm, 28*mm, 60*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), TABLE_ALT),
        ('BACKGROUND', (2,0), (2,-1), TABLE_ALT),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    return [
        Paragraph("<b>REPORT INFORMATION</b>", styles['section_title']),
        t,
    ]


def _section_verdict(result, styles):
    """§3: Large verdict banner."""
    verdict = result.get('verdict', 'UNKNOWN')
    score = result.get('ensemble_score', 0)
    confidence = result.get('confidence_label', 'N/A')

    bg = FAKE_RED if verdict == 'FAKE' else REAL_GREEN
    label = "⚠ DEEPFAKE DETECTED" if verdict == 'FAKE' else "✓ AUTHENTIC VOICE"

    data = [[
        Paragraph(f"<b>{label}</b>", styles['verdict_text']),
        Paragraph(f"<b>{score:.1f}%</b>  |  {confidence}", styles['verdict_score']),
    ]]
    t = Table(data, colWidths=[100*mm, 74*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ]))
    return [t]


def _section_summary_box(summary_text, result, styles):
    """Officer summary box with colored background."""
    verdict = result.get('verdict', 'UNKNOWN')
    bg = CRITICAL_BG if verdict == 'FAKE' else SAFE_BG

    # Format summary as paragraphs
    lines = summary_text.split('\n')
    paras = []
    paras.append(Paragraph("<b>OFFICER SUMMARY</b>", styles['section_title']))
    for line in lines:
        line = line.strip()
        if not line:
            paras.append(Spacer(1, 2*mm))
        elif line.startswith('ALERT:') or line.startswith('RECOMMENDED'):
            paras.append(Paragraph(f"<b>{line}</b>", styles['body_bold']))
        else:
            paras.append(Paragraph(line, styles['body']))

    # Wrap in a colored box
    inner_data = [[paras]]
    inner_t = Table(inner_data, colWidths=[170*mm])
    inner_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('ROUNDEDCORNERS', [4, 4, 4, 4]),
        ('BOX', (0,0), (-1,-1), 1, FAKE_RED if verdict == 'FAKE' else REAL_GREEN),
    ]))
    return [inner_t]


def _section_ensemble(result, styles):
    """§4: Per-model score breakdown."""
    per_model = result.get('per_model', {})
    if not per_model:
        return []

    rows = [['Model', 'P(Spoof)', 'Weight', 'EER', 'Contribution']]
    for name, info in per_model.items():
        prob = info.get('spoof_probability', 0)
        weight = info.get('weight', 0)
        eer = info.get('eer', 0)
        contribution = prob * weight
        rows.append([name, f'{prob*100:.1f}%', f'{weight:.2f}', f'{eer}%', f'{contribution*100:.1f}%'])

    t = Table(rows, colWidths=[42*mm, 32*mm, 28*mm, 28*mm, 34*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), TABLE_HEADER),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, TABLE_ALT]),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    return [
        Paragraph("<b>ENSEMBLE MODEL SCORES</b>", styles['section_title']),
        t,
    ]


def _section_biometrics(bio, styles):
    """§5: Biometric biomarker drift table — all 6 tests."""
    rows = [['Biomarker', 'Measured', 'Baseline', 'Status']]
    rows.append(['F0 Variance', f"{bio.get('f0_variance_hz', 0):.3f} Hz",
                 f"{bio.get('f0_variance_baseline', 0)} Hz", bio.get('f0_status', 'N/A')])
    rows.append(['Jitter', f"{bio.get('jitter_pct', 0):.4f}%",
                 f"{bio.get('jitter_baseline', 0)}%", bio.get('jitter_status', 'N/A')])
    rows.append(['Shimmer', f"{bio.get('shimmer_db', 0):.4f} dB",
                 f"{bio.get('shimmer_baseline', 0)} dB", bio.get('shimmer_status', 'N/A')])
    rows.append(['HNR', f"{bio.get('hnr_db', 0):.1f} dB",
                 f"{bio.get('hnr_baseline', 0)} dB", bio.get('hnr_status', 'N/A')])

    # Fix L16: Add pause energy and formant rows (6-test coverage)
    pe = bio.get('pause_energy', {})
    rows.append(['Pause Energy', f"min RMS: {pe.get('min_rms', 0):.6f}",
                 f"min: {pe.get('baseline_min', 0.0008)}", bio.get('pause_status', 'N/A')])

    fv = bio.get('formant_velocity', {})
    rows.append(['Formant F2', f"max: {fv.get('max_hz_per_frame', 0):.1f} Hz/f",
                 f"limit: {fv.get('limit', 50):.0f} Hz/f", bio.get('formant_status', 'N/A')])

    rows.append([
        Paragraph(f"<b>Overall Biometric Score: {bio.get('overall_biometric_score', 0):.0f}/100</b>",
                  styles['cell_bold']),
        '', '', f"{bio.get('n_suspicious', 0)}/6 anomalies"
    ])

    t = Table(rows, colWidths=[40*mm, 40*mm, 40*mm, 40*mm])
    style_cmds = [
        ('BACKGROUND', (0,0), (-1,0), TABLE_HEADER),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, TABLE_ALT]),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('SPAN', (0,-1), (2,-1)),
    ]

    for i, row in enumerate(rows[1:-1], start=1):
        status = row[3]
        if status not in ('NORMAL', 'N/A', 'UNKNOWN'):
            style_cmds.append(('TEXTCOLOR', (3,i), (3,i), FAKE_RED))
            style_cmds.append(('FONTNAME', (3,i), (3,i), 'Helvetica-Bold'))

    t.setStyle(TableStyle(style_cmds))

    elems = [
        Paragraph("<b>BIOMETRIC BIOMARKER ANALYSIS (6-test)</b>", styles['section_title']),
        t,
    ]

    # Issue 5: Add note when biometric values are EXCESS (too high, not too low)
    excess_metrics = []
    if bio.get('jitter_status') == 'EXCESS':
        excess_metrics.append(f"Jitter ({bio.get('jitter_pct', 0):.2f}%)")
    if bio.get('shimmer_status') == 'EXCESS':
        excess_metrics.append(f"Shimmer ({bio.get('shimmer_db', 0):.2f} dB)")
    if bio.get('f0_status') == 'TOO_ERRATIC':
        excess_metrics.append(f"F0 Variance ({bio.get('f0_variance_hz', 0):.1f} Hz)")

    if excess_metrics:
        from reportlab.platypus import Spacer as _Sp
        elems.append(_Sp(1, 2*mm))
        elems.append(Paragraph(
            f"<i>Note: Elevated {', '.join(excess_metrics)} may result from "
            f"background noise or varied speech context. "
            f"These values indicate atypical recording conditions rather than "
            f"confirmed TTS characteristics. TTS systems typically produce "
            f"values below human baseline, not above.</i>",
            ParagraphStyle('BioNote', parent=styles['Normal'],
                          fontSize=7, textColor=colors.HexColor('#e67e22'), leftIndent=6),
        ))

    # Issue 6: HNR computation warning
    hnr_warning = bio.get('hnr_warning')
    if hnr_warning:
        from reportlab.platypus import Spacer as _Sp
        elems.append(_Sp(1, 2*mm))
        elems.append(Paragraph(
            f"<i>\u26a0\ufe0f {hnr_warning}</i>",
            ParagraphStyle('HNRWarn', parent=styles['Normal'],
                          fontSize=7, textColor=FAKE_RED, leftIndent=6),
        ))

    return elems


def _section_pause_energy(bio, styles):
    """§6: Pause energy analysis."""
    pe = bio.get('pause_energy', {})
    status = bio.get('pause_status', 'NORMAL')

    data = [
        ['Minimum RMS', f"{pe.get('min_rms', 0):.8f}"],
        ['Mean Pause RMS', f"{pe.get('mean_pause_rms', 0):.8f}"],
        ['Baseline Min', f"{pe.get('baseline_min', 0.0008)}"],
        ['Silent Frames', f"{pe.get('n_silent_frames', 0)} ({pe.get('pct_silent', 0):.1f}%)"],
        ['Status', status],
    ]
    t = Table(data, colWidths=[50*mm, 120*mm])
    style_cmds = [
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0,0), (0,-1), TABLE_ALT),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]
    if status != 'NORMAL':
        style_cmds.append(('TEXTCOLOR', (1,4), (1,4), FAKE_RED))
        style_cmds.append(('FONTNAME', (1,4), (1,4), 'Helvetica-Bold'))
    t.setStyle(TableStyle(style_cmds))

    explanation = (
        "Digital TTS systems produce perfect silence (RMS = 0) during pauses, "
        "while human recordings always have background noise. "
        f"{'⚠ Digital silence detected — likely synthetic.' if status != 'NORMAL' else '✓ Pause energy is within natural recording range.'}"
    )
    return [
        Paragraph("<b>PAUSE ENERGY ANALYSIS</b>", styles['section_title']),
        t,
        Spacer(1, 2*mm),
        Paragraph(f"<i>{explanation}</i>", styles['body']),
    ]


def _section_formant_velocity(bio, styles):
    """§7: Formant F2 velocity analysis."""
    fv = bio.get('formant_velocity', {})
    status = bio.get('formant_status', 'NORMAL')

    data = [
        ['Max Velocity', f"{fv.get('max_hz_per_frame', 0):.1f} Hz/frame"],
        ['Mean Velocity', f"{fv.get('mean_velocity', 0):.1f} Hz/frame"],
        ['Human Limit', f"{fv.get('limit', 50):.0f} Hz/frame (per 10ms)"],
        ['Violations (>2× limit)', f"{fv.get('n_violations', 0)} frames"],
        ['Status', status],
    ]
    t = Table(data, colWidths=[50*mm, 120*mm])
    style_cmds = [
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0,0), (0,-1), TABLE_ALT),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]
    if status != 'NORMAL':
        style_cmds.append(('TEXTCOLOR', (1,4), (1,4), FAKE_RED))
        style_cmds.append(('FONTNAME', (1,4), (1,4), 'Helvetica-Bold'))
    t.setStyle(TableStyle(style_cmds))

    explanation = (
        "Formant F2 tracks the second resonance frequency of the vocal tract. "
        "Human articulators (tongue, jaw) have physical speed limits. TTS systems "
        "can produce impossibly fast formant transitions. "
        f"{'⚠ Formant transitions exceed human articulation limits.' if status != 'NORMAL' else '✓ Formant transitions are within human speed limits.'}"
    )
    return [
        Paragraph("<b>FORMANT F2 VELOCITY ANALYSIS</b>", styles['section_title']),
        t,
        Spacer(1, 2*mm),
        Paragraph(f"<i>{explanation}</i>", styles['body']),
    ]


def _section_image(b64_data, title, caption, styles, width=170*mm, height=60*mm):
    """Generic image section from base64 data."""
    elements = [
        Paragraph(f"<b>{title}</b>", styles['section_title']),
    ]
    try:
        img_data = base64.b64decode(b64_data)
        img_stream = io.BytesIO(img_data)
        img = Image(img_stream, width=width, height=height)
        elements.append(img)
        elements.append(Spacer(1, 2*mm))
        elements.append(Paragraph(f"<i>{caption}</i>", styles['caption']))
    except Exception as e:
        logger.error(f"Failed to embed image for '{title}': {e}")
    return elements


def _section_shap(shap_b64, styles):
    """§19: GradientSHAP Attribution Map."""
    if not shap_b64:
        return []
    
    elems = [
        Paragraph("<b>GRADIENTSHAP FEATURE ATTRIBUTION MAP (MFCC)</b>", styles['section_title']),
        Paragraph(
            "<i>GradientSHAP shows the pixel-level contribution of each input spectrogram region "
            "towards the FAKE verdict, relative to a G.711 µ-law codec degraded baseline. "
            "Red regions indicate features that strongly contributed to the FAKE classification; "
            "Blue regions indicate features that align with the codec degraded real baseline.</i>",
            styles['body']
        ),
        Spacer(1, 2*mm)
    ]
    
    try:
        from reportlab.platypus import Image as _Img
        img_data = base64.b64decode(shap_b64)
        img_stream = io.BytesIO(img_data)
        img = _Img(img_stream, width=170*mm, height=75*mm)
        elems.append(img)
        elems.append(Spacer(1, 2*mm))
        elems.append(Paragraph("<i>Red = Contributes to FAKE, Blue = Contributes to REAL (codec baseline)</i>", styles['caption']))
    except Exception as e:
        logger.error(f"Failed to embed SHAP image: {e}")
        
    return elems


def _section_phoneme_timeline(transcript, timeline, styles):
    """§20: Phoneme Timeline and Transcript."""
    if not timeline and not transcript:
        return []
        
    elems = [
        Paragraph("<b>PHONETIC SEGMENTATION & ALIGNMENT</b>", styles['section_title']),
    ]
    
    if transcript:
        elems.append(Paragraph(f"<b>Transcript:</b> \"{transcript}\"", styles['body']))
        elems.append(Spacer(1, 2*mm))
        
    if timeline:
        # Build table representing word timeline with start, end, and phonemes
        rows = [['Word', 'Start', 'End', 'Confidence', 'ARPAbet Phonemes']]
        for w in timeline[:15]:  # limit to top 15 words to fit page nicely
            phonemes_str = "-".join(w.get("phonemes", []))
            rows.append([
                w.get("word", ""),
                f"{w.get('start', 0.0):.2f}s",
                f"{w.get('end', 0.0):.2f}s",
                f"{w.get('prob', 0.0)*100:.0f}%",
                phonemes_str if phonemes_str else "N/A"
            ])
            
        t = Table(rows, colWidths=[30*mm, 20*mm, 20*mm, 25*mm, 75*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fafafa')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 7.5),
        ]))
        elems.append(t)
        
        if len(timeline) > 15:
            elems.append(Spacer(1, 1*mm))
            elems.append(Paragraph(f"<i>... showing top 15 of {len(timeline)} transcribed words</i>", styles['caption']))
            
    return elems


def _section_voice_enrollment(speaker_match, styles):
    """§21: Voice Enrollment Match."""
    if not speaker_match:
        return []
        
    match_found = speaker_match.get("match_found", False)
    speaker_name = speaker_match.get("speaker_name", "No Match Found")
    sim = speaker_match.get("similarity", 0.0)
    comparisons = speaker_match.get("all_comparisons", [])
    
    elems = [
        Paragraph("<b>BIOMETRIC VOICE ENROLLMENT COMPARISON</b>", styles['section_title']),
    ]
    
    if match_found:
        status_text = f"<font color='green'><b>MATCH DETECTED:</b> Speaker identity verified as '{speaker_name}' (Similarity: {sim*100:.1f}%)</font>"
        expl = (
            f"The voice print matches enrolled speaker '{speaker_name}' with a cosine similarity "
            f"of {sim:.4f} (threshold: 0.75). This indicates a positive identity match."
        )
    else:
        status_text = f"<font color='red'><b>NO BIOMETRIC MATCH DETECTED</b> (Best match: '{speaker_name}' at {sim*100:.1f}%)</font>"
        expl = (
            f"The voice print does not match any enrolled speaker above the verification threshold of 0.75. "
            f"The sample is classified as an unknown speaker or potentially a voice clone targeting an enrolled customer."
        )
        
    elems.append(Paragraph(status_text, styles['body_bold']))
    elems.append(Spacer(1, 1*mm))
    elems.append(Paragraph(expl, styles['body']))
    elems.append(Spacer(1, 2*mm))
    
    if comparisons:
        # Build comparison table
        rows = [['Enrolled Speaker Name', 'Cosine Similarity', 'Verification Verdict']]
        for c in comparisons[:5]:
            score = c.get("similarity", 0.0)
            verdict = "<font color='green'><b>MATCH</b></font>" if score >= 0.75 else "<font color='red'>NO MATCH</font>"
            rows.append([
                c.get("speaker_name", ""),
                f"{score:.4f}",
                Paragraph(verdict, styles['cell_small'])
            ])
            
        t = Table(rows, colWidths=[70*mm, 50*mm, 50*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fafafa')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 7.5),
        ]))
        elems.append(t)
        
    return elems


def _section_heatmaps(xai, styles):
    """§12: Grad-CAM heatmap images."""
    elements = [
        Paragraph("<b>SPECTRAL EVIDENCE (Grad-CAM)</b>", styles['section_title']),
        Paragraph(
            "<i>Grad-CAM highlights regions where the neural network focuses to distinguish "
            "real from fake speech. Red/yellow regions indicate suspicious areas.</i>",
            styles['caption']
        ),
        Spacer(1, 2*mm),
    ]

    for key, label in [('heatmap_mfcc_b64', 'MFCC'), ('heatmap_lfcc_b64', 'LFCC')]:
        b64 = xai.get(key)
        if b64:
            try:
                img_data = base64.b64decode(b64)
                img_stream = io.BytesIO(img_data)
                img = Image(img_stream, width=170*mm, height=48*mm)
                elements.append(Paragraph(f"<i>{label} Grad-CAM Analysis</i>", styles['caption']))
                elements.append(img)
                elements.append(Spacer(1, 3*mm))
            except Exception as e:
                logger.error(f"Failed to embed {label} heatmap: {e}")

    return elements if len(elements) > 3 else []


def _section_findings(findings, styles):
    """§13: Artifact findings table."""
    rows = [['#', 'Type', 'Freq Range', 'Time Range', 'Conf.', 'Description']]
    # Issue 8: Show ALL findings, not just top 12
    for f in findings:
        # Fix L15/L17: Show full description (up to 200 chars) — truncation at 80 lost context
        reason = f.get('reason', '')
        desc = reason[:200] + '...' if len(reason) > 200 else reason
        freq = f.get('freq_range', [0, 0])
        time_r = f.get('time_range', [0, 0])
        rows.append([
            str(f.get('rank', '')),
            f.get('artifact_type', 'unknown')[:18],
            f'{freq[0]:.0f}-{freq[1]:.0f} Hz',
            f'{time_r[0]:.2f}-{time_r[1]:.2f}s',
            f"{f.get('confidence', 0):.0f}%",
            Paragraph(desc, styles['cell_small']),
        ])

    col_widths = [8*mm, 25*mm, 24*mm, 20*mm, 12*mm, 81*mm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), TABLE_HEADER),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, TABLE_ALT]),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    return [
        Paragraph("<b>ARTIFACT FINDINGS</b>", styles['section_title']),
        Paragraph(f"<i>{len(findings)} artifacts detected (all shown)</i>",
                  styles['caption']),
        t,
    ]


def _section_channel(channel, styles):
    """§14: Channel integrity matrix."""
    data = [
        ['SNR', f"{channel.get('snr_estimated_db', 0)} dB"],
        ['Noise Floor', f"{channel.get('noise_floor_db', 0)} dB"],
        ['Reverb Level', channel.get('reverb_level', 'N/A')],
        ['Codec Detected', channel.get('codec_detected', 'unknown')],
        ['HF Energy Ratio', f"{channel.get('hf_energy_ratio', 0):.4f}"],
        ['Integrity Score', f"{channel.get('channel_integrity_score', 0)}/100"],
    ]
    t = Table(data, colWidths=[50*mm, 120*mm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0,0), (0,-1), TABLE_ALT),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    explanation = channel.get('matrix_explanation', '')
    return [
        Paragraph("<b>CHANNEL INTEGRITY</b>", styles['section_title']),
        t,
        Spacer(1, 2*mm),
        Paragraph(f"<i>{explanation}</i>", styles['body']),
    ]


def _section_attribution_enhanced(attr, styles):
    """§15: Enhanced threat attribution with source probabilities."""
    data = [
        ['Suspected Tool', attr.get('suspected_tool', 'Unknown')],
        ['Confidence', f"{attr.get('confidence', 0):.1f}%"],
        ['Runner-up', f"{attr.get('runner_up', 'N/A')} ({attr.get('runner_up_conf', 0):.1f}%)"],
        ['Cluster Label', attr.get('cluster_label', 'N/A')],
        ['Notes', Paragraph(attr.get('signature_notes', 'N/A')[:200], styles['cell_small'])],
    ]
    t = Table(data, colWidths=[40*mm, 130*mm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0,0), (0,-1), TABLE_ALT),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))

    elements = [
        Paragraph("<b>THREAT ATTRIBUTION</b>", styles['section_title']),
        t,
    ]

    # Source probability table
    probs = attr.get('source_probs', {})
    if probs:
        elements.append(Spacer(1, 3*mm))
        elements.append(Paragraph("<b>Source Probability Distribution</b>", styles['subsection_title']))

        prob_rows = [['Source', 'Probability', 'Visual']]
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for name, prob in sorted_probs:
            bar_len = int(prob / 2)  # Scale to ~50 chars max
            bar = '█' * bar_len + '░' * max(0, 50 - bar_len)
            prob_rows.append([name, f"{prob:.1f}%", bar[:40]])

        pt = Table(prob_rows, colWidths=[50*mm, 25*mm, 95*mm])
        pt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), TABLE_HEADER),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('FONTNAME', (2,1), (2,-1), 'Courier'),
            ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, TABLE_ALT]),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        elements.append(pt)

    return elements


def _section_embedding(emb_proj, styles):
    """§16: Embedding projection analysis."""
    data = [
        ['PC1 (x)', f"{emb_proj.get('x', 0):.3f}"],
        ['PC2 (y)', f"{emb_proj.get('y', 0):.3f}"],
        ['Nearest Cluster', emb_proj.get('nearest_cluster', 'N/A')],
        ['Distance to Human', f"{emb_proj.get('distance_to_human', 0):.2f} units"],
    ]

    # Add cluster distances
    distances = emb_proj.get('cluster_distances', {})
    if distances:
        sorted_d = sorted(distances.items(), key=lambda x: x[1])
        for name, dist in sorted_d:
            data.append([f"  → {name}", f"{dist:.2f}"])

    pca_var = emb_proj.get('pca_explained_variance', [])
    if pca_var:
        data.append(['PCA Variance', f"PC1: {pca_var[0]*100:.1f}%, PC2: {pca_var[1]*100:.1f}%"])

    t = Table(data, colWidths=[50*mm, 120*mm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0,0), (0,-1), TABLE_ALT),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    elems = [
        Paragraph("<b>EMBEDDING SPACE ANALYSIS</b>", styles['section_title']),
        Paragraph(
            "<i>The audio sample's 256-dimensional embedding is projected to 2D and compared "
            "against reference clusters for known voice types.</i>",
            styles['caption'],
        ),
        t,
    ]

    # Issue 10: PCA variance limitation note
    pca_note = emb_proj.get('pca_note', '')
    if pca_note:
        elems.append(Spacer(1, 2*mm))
        elems.append(Paragraph(
            f"<i>Note: {pca_note}</i>",
            ParagraphStyle('PCANote', parent=styles['Normal'],
                          fontSize=7, textColor=colors.gray, leftIndent=6),
        ))

    return elems


def _section_conclusion_enhanced(result, xai, styles):
    """§17: Enhanced forensic conclusion with legal text."""
    verdict = result.get('verdict', 'UNKNOWN')
    score = result.get('ensemble_score', 0)
    conf = result.get('confidence_label', 'N/A')
    n_findings = xai.get('n_findings', 0)
    bio_score = (xai.get('biometric_drift') or {}).get('overall_biometric_score', 0)

    # Use the pre-generated forensic conclusion if available
    conclusion_text = xai.get('forensic_conclusion', '')

    elements = [
        Paragraph("<b>FORENSIC CONCLUSION</b>", styles['section_title']),
    ]

    if conclusion_text:
        # Split conclusion into paragraphs and render each
        for para_text in conclusion_text.split('\n\n'):
            para_text = para_text.strip()
            if not para_text:
                continue
            if para_text.startswith('FORENSIC FINDING:'):
                elements.append(Paragraph(f"<b>{para_text}</b>", styles['body']))
            elif para_text.startswith('RECOMMENDED ACTION:'):
                elements.append(Spacer(1, 2*mm))
                elements.append(Paragraph(f"<b>{para_text}</b>", styles['body_bold']))
            else:
                elements.append(Paragraph(para_text, styles['body']))
            elements.append(Spacer(1, 2*mm))
    else:
        # Fallback conclusion
        if verdict == 'FAKE':
            text = (
                f"Based on multi-model ensemble analysis (score: {score:.1f}%, "
                f"confidence: {conf}), this audio sample has been classified as "
                f"<b>SYNTHETIC/DEEPFAKE</b>. {n_findings} spectral and statistical "
                f"artifacts were identified. Biometric score: {bio_score:.0f}/100. "
                f"This finding should be treated as HIGH PRIORITY."
            )
        else:
            text = (
                f"Based on multi-model ensemble analysis (score: {score:.1f}%, "
                f"confidence: {conf}), this audio sample has been classified as "
                f"<b>AUTHENTIC</b>. No significant artifacts detected."
            )
        elements.append(Paragraph(text, styles['body']))

    # Legal disclaimer
    elements.append(Spacer(1, 3*mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "<i>This analysis constitutes electronic evidence under Indian Evidence Act §65B. "
        "The report should be retained as per RBI Fraud Risk Management guidelines "
        "(minimum 7 years). Any disputes should be resolved through appropriate "
        "judicial channels with original audio evidence preserved.</i>",
        styles['footer'],
    ))

    return elements


def _section_detailed_analysis(detailed_text, styles):
    """Detailed technical analysis section."""
    elements = [
        Paragraph("<b>DETAILED TECHNICAL ANALYSIS</b>", styles['section_title']),
    ]

    # Split into lettered sections
    sections = detailed_text.split('\n\n')
    for section in sections:
        lines = section.strip().split('\n')
        if not lines:
            continue

        # First line is section header
        header = lines[0].strip()
        if header and header[0].isalpha() and header[1] == '.':
            elements.append(Paragraph(f"<b>{header}</b>", styles['body_bold']))
        else:
            elements.append(Paragraph(header, styles['body']))

        # Remaining lines
        for line in lines[1:]:
            line = line.strip()
            if line.startswith('  [') or line.startswith('  -'):
                elements.append(Paragraph(f"    {line}", styles['cell_small']))
            else:
                elements.append(Paragraph(line, styles['body']))

        elements.append(Spacer(1, 2*mm))

    return elements


def _section_technical(result, styles):
    """§18a: Technical metadata."""
    perf = result.get('performance', {})
    data = [
        ['Engine Version', 'VoiceGuard AI v2.0'],
        ['Ensemble', '2-model (LCNN-MFCC + LCNN-LFCC)'],
        ['XAI Modules', 'Grad-CAM, Biometrics (6-test: jitter, shimmer, HNR, F0, pause, formant), Attribution, Embedding, Visual Evidence'],
        ['Detection Time', f"{perf.get('detection_ms', 0)} ms"],
        ['XAI Time', f"{perf.get('xai_ms', 0)} ms"],
        ['Total Time', f"{perf.get('total_ms', 0)} ms"],
        ['PDF Generator', 'ReportLab (18-section forensic report)'],
    ]
    t = Table(data, colWidths=[40*mm, 130*mm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0,0), (0,-1), TABLE_ALT),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    return [
        Paragraph("<b>TECHNICAL METADATA</b>", styles['section_title']),
        t,
    ]


def _section_certification(result, styles):
    """§18b: Certification footer."""
    audit_id = result.get('audit_id', 'N/A')
    text = (
        f"This report was automatically generated by VoiceGuard AI v2.0 "
        f"(18-section forensic analysis). "
        f"Audit ID: {audit_id}. "
        f"Report timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}. "
        f"This document is intended for authorized UCO Bank personnel only. "
        f"Unauthorized distribution is prohibited. "
        f"Document integrity can be verified using the audit ID."
    )
    return [
        HRFlowable(width="100%", thickness=1, color=BORDER_COLOR),
        Spacer(1, 2*mm),
        Paragraph(text, styles['footer']),
    ]


# ══════════════════════════════════════════════════════════════════════
#  Style Builder
# ══════════════════════════════════════════════════════════════════════

def _build_styles():
    """Build custom paragraph styles for the report."""
    base = getSampleStyleSheet()

    styles = {
        'Normal': base['Normal'],
        'header_title': ParagraphStyle(
            'header_title', parent=base['Normal'],
            fontSize=16, fontName='Helvetica-Bold',
            textColor=colors.white, alignment=TA_LEFT,
        ),
        'header_sub': ParagraphStyle(
            'header_sub', parent=base['Normal'],
            fontSize=10, fontName='Helvetica',
            textColor=colors.white, alignment=TA_CENTER,
        ),
        'section_title': ParagraphStyle(
            'section_title', parent=base['Normal'],
            fontSize=11, fontName='Helvetica-Bold',
            textColor=TABLE_HEADER, spaceAfter=3*mm, spaceBefore=2*mm,
        ),
        'subsection_title': ParagraphStyle(
            'subsection_title', parent=base['Normal'],
            fontSize=9, fontName='Helvetica-Bold',
            textColor=ACCENT_BLUE, spaceAfter=2*mm, spaceBefore=1*mm,
        ),
        'verdict_text': ParagraphStyle(
            'verdict_text', parent=base['Normal'],
            fontSize=18, fontName='Helvetica-Bold',
            textColor=colors.white, alignment=TA_LEFT,
        ),
        'verdict_score': ParagraphStyle(
            'verdict_score', parent=base['Normal'],
            fontSize=14, fontName='Helvetica-Bold',
            textColor=colors.white, alignment=TA_CENTER,
        ),
        'body': ParagraphStyle(
            'body', parent=base['Normal'],
            fontSize=9, fontName='Helvetica',
            textColor=TEXT_DARK, alignment=TA_JUSTIFY, leading=14,
        ),
        'body_bold': ParagraphStyle(
            'body_bold', parent=base['Normal'],
            fontSize=9, fontName='Helvetica-Bold',
            textColor=TEXT_DARK, alignment=TA_JUSTIFY, leading=14,
        ),
        'caption': ParagraphStyle(
            'caption', parent=base['Normal'],
            fontSize=8, fontName='Helvetica-Oblique',
            textColor=colors.gray, alignment=TA_CENTER, spaceAfter=2*mm,
        ),
        'cell_bold': ParagraphStyle(
            'cell_bold', parent=base['Normal'],
            fontSize=9, fontName='Helvetica-Bold', textColor=TEXT_DARK,
        ),
        'cell_small': ParagraphStyle(
            'cell_small', parent=base['Normal'],
            fontSize=7, fontName='Helvetica', textColor=TEXT_DARK, leading=10,
        ),
        'footer': ParagraphStyle(
            'footer', parent=base['Normal'],
            fontSize=7, fontName='Helvetica',
            textColor=colors.gray, alignment=TA_CENTER,
        ),
    }
    return styles
