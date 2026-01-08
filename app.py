import io
import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# -----------------------------
# Design system (matches your spec)
# -----------------------------
C = {
    "text_primary": "#1F2933",
    "text_secondary": "#4B5563",
    "text_muted": "#9CA3AF",
    "bg": "#FFFFFF",
    "divider": "#E5E7EB",
    "accent": "#1D4ED8",
    "positive": "#16A34A",
    "amber": "#D97706",
    "critical": "#DC2626",
    "neutral": "#64748B",
    "table_header_bg": "#F3F4F6",
    "row_alt": "#F9FAFB",
    "kpi_pos": "#DCFCE7",
    "kpi_risk": "#FEF3C7",
    "kpi_crit": "#FEE2E2",
}

MILESTONES = {
    "Arrival at Origin": {
        "actual": "Origin actual arrival time",
        "planned_end": "Origin latest planned arrival end time",
        "estimated": "Origin estimated arrival time",
    },
    "Departure at Origin": {
        "actual": "Origin actual departure time",
        "planned_end": "Origin latest planned departure end time",
        "estimated": "Origin estimated departure time",
    },
    "Arrival at Destination": {
        "actual": "Destination actual arrival time",
        "planned_end": "Destination latest planned arrival end time",
        "estimated": "Destination estimated arrival time",
    },
}

SHIPMENT_COLS_REQUIRED = [
    "Shipment ID",
    "Shipment type",
    "Created time",
    "Current state",
    "Current state reason",
    "Exceptions",
    "Current carrier",
    "Origin",
    "Origin city",
    "Origin state",
    "Origin country",
    "Origin estimated arrival time",
    "Origin actual arrival time",
    "Origin latest planned arrival time",
    "Origin latest planned arrival end time",
    "Origin estimated departure time",
    "Origin actual departure time",
    "Origin latest planned departure time",
    "Origin latest planned departure end time",
    "Destination",
    "Destination city",
    "Destination state",
    "Destination country",
    "Destination estimated arrival time",
    "Destination actual arrival time",
    "Destination latest planned arrival time",
    "Destination latest planned arrival end time",
]

RCA_KEEP_COLS = [
    "Root Cause Error",
    "Root Cause Error Details",
    "Bill Of Lading",
    "Order Number",
    "Carrier",
    "Carrier Identifier",
    "Type Carrier Identifier",
    "Value Equipment Id",
    "Equipment Id",
    "Origin Location Name",
    "Origin",
    "Country of Origin",
    "Region of Origin",
    "Destination",
    "Location Name",
    "Destination",
    "Country of Destination",
    "Region of Destination",
    "Creation Date",
    "Tracking End Date",
    "Planned Pickup Planned Arrival",
    "API Request",
    "API Response",
]

# -----------------------------
# Helpers: parsing & normalization
# -----------------------------
def _to_dt(s: pd.Series) -> pd.Series:
    # Treat blanks as missing; coerce errors; keep timezone-naive
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def load_shipments(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()

    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded)
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel for shipments.")

    # Ensure required columns exist (some files may have extras)
    missing = [c for c in SHIPMENT_COLS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Shipment file is missing columns: {missing}")

    # Parse dates
    date_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    for c in date_cols:
        df[c] = _to_dt(df[c])

    # Clean strings
    for c in ["Shipment ID", "Current state", "Current state reason", "Exceptions", "Current carrier"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace("nan", "").replace("NaT", "").fillna("")

    return df

def load_rca(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()

    name = uploaded.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded)
    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        raise ValueError("Unsupported RCA file type. Please upload Excel or CSV.")

    # Keep only known cols that exist
    keep = [c for c in RCA_KEEP_COLS if c in df.columns]
    df = df[keep].copy()

    # Normalize match key (Shipment ID) using BOL/Order
    for c in ["Bill Of Lading", "Order Number"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace("nan", "").fillna("")

    df["match_shipment_id"] = ""
    if "Bill Of Lading" in df.columns and "Order Number" in df.columns:
        df["match_shipment_id"] = df["Bill Of Lading"].where(df["Bill Of Lading"] != "", df["Order Number"])
    elif "Bill Of Lading" in df.columns:
        df["match_shipment_id"] = df["Bill Of Lading"]
    elif "Order Number" in df.columns:
        df["match_shipment_id"] = df["Order Number"]

    # Dates
    for c in ["Creation Date", "Tracking End Date"]:
        if c in df.columns:
            df[c] = _to_dt(df[c])

    for c in ["Root Cause Error", "Carrier"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace("nan", "").fillna("")

    return df

def lane_from_row(r: pd.Series) -> str:
    o = ", ".join([x for x in [r.get("Origin city", ""), r.get("Origin state", ""), r.get("Origin country", "")] if str(x).strip()])
    d = ", ".join([x for x in [r.get("Destination city", ""), r.get("Destination state", ""), r.get("Destination country", "")] if str(x).strip()])
    o = o if o else str(r.get("Origin", "")).strip()
    d = d if d else str(r.get("Destination", "")).strip()
    return f"{o} → {d}".strip()

def parse_exceptions(val: str) -> List[str]:
    if val is None:
        return []
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return []
    # Split on common separators
    parts = []
    for token in s.replace("|", ";").replace(",", ";").split(";"):
        t = token.strip()
        if t:
            parts.append(t)
    return parts

def milestone_reported(df: pd.DataFrame, milestone_name: str) -> pd.Series:
    col = MILESTONES[milestone_name]["actual"]
    return df[col].notna()

def on_time(df: pd.DataFrame, milestone_name: str, latency_minutes: int) -> pd.Series:
    actual = df[MILESTONES[milestone_name]["actual"]]
    planned_end = df[MILESTONES[milestone_name]["planned_end"]]
    latency = pd.to_timedelta(latency_minutes, unit="m")
    # on-time only where both exist
    ok = actual.notna() & planned_end.notna()
    out = pd.Series(False, index=df.index)
    out[ok] = actual[ok] <= (planned_end[ok] + latency)
    return out

def estimated_accuracy_destination(df: pd.DataFrame, latency_minutes: int) -> pd.Series:
    est = df["Destination estimated arrival time"]
    act = df["Destination actual arrival time"]
    latency = pd.to_timedelta(latency_minutes, unit="m")
    ok = est.notna() & act.notna()
    out = pd.Series(False, index=df.index)
    out[ok] = (act[ok] - est[ok]).abs() <= latency
    return out

def derive_completion_bucket(df: pd.DataFrame, selected_milestones: List[str]) -> pd.Series:
    """
    If Current state is COMPLETED but state reason TRACKING_TIMED_OUT:
      - treat as Timed out unless ALL selected milestones have actual timestamps
      - if all actuals exist, treat as Completed
    """
    state = df["Current state"].astype(str)
    reason = df["Current state reason"].astype(str)
    is_completed = state.str.upper().eq("COMPLETED")
    is_timedout_reason = reason.str.upper().eq("TRACKING_TIMED_OUT")

    all_selected_reported = pd.Series(True, index=df.index)
    for m in selected_milestones:
        all_selected_reported &= milestone_reported(df, m)

    bucket = pd.Series("In Progress / Other", index=df.index)
    bucket[is_completed & ~is_timedout_reason] = "Completed"
    bucket[is_completed & is_timedout_reason & ~all_selected_reported] = "Timed Out"
    bucket[is_completed & is_timedout_reason & all_selected_reported] = "Completed"
    # keep other states as-is for extra visibility
    other_mask = ~is_completed
    bucket[other_mask] = state[other_mask].replace("", "Unknown")
    return bucket


# -----------------------------
# UI helpers (cards)
# -----------------------------
def kpi_card(title: str, value: str, subtitle: str = "", tone: str = "neutral"):
    tone_map = {
        "neutral": C["divider"],
        "good": C["positive"],
        "warn": C["amber"],
        "bad": C["critical"],
        "accent": C["accent"],
    }
    border = tone_map.get(tone, C["divider"])
    st.markdown(
        f"""
        <div style="
            border: 1px solid {border};
            border-radius: 14px;
            padding: 14px 14px;
            background: {C['bg']};
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            height: 92px;
        ">
            <div style="font-size: 12.5px; color: {C['text_secondary']}; margin-bottom: 8px;">{title}</div>
            <div style="font-size: 22px; font-weight: 700; color: {C['text_primary']}; line-height: 1;">{value}</div>
            <div style="font-size: 11.5px; color: {C['text_muted']}; margin-top: 8px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# PDF generator (ReportLab)
# -----------------------------
@dataclass
class PdfConfig:
    customer_name: str
    date_range_label: str
    logo_bytes: Optional[bytes]
    latency_minutes: int
    selected_milestones: List[str]

def hex_to_rl(h: str):
    h = h.lstrip("#")
    r = int(h[0:2], 16) / 255
    g = int(h[2:4], 16) / 255
    b = int(h[4:6], 16) / 255
    return colors.Color(r, g, b)

def _draw_footer(cnv: canvas.Canvas, cfg: PdfConfig, page_num: int, page_total: int, margin_bottom: float, margin_left: float, margin_right: float):
    cnv.setFont("Helvetica", 8.5)
    cnv.setFillColor(hex_to_rl(C["text_muted"]))
    y = margin_bottom - 8
    w, h = landscape(A4)

    cnv.drawString(margin_left, y, cfg.customer_name)
    cnv.drawCentredString(w / 2, y, "Confidential – Customer Use Only")
    cnv.drawRightString(w - margin_right, y, f"Page {page_num} of {page_total}")

def _draw_header(cnv: canvas.Canvas, cfg: PdfConfig, title: str, margin_top: float, margin_left: float, margin_right: float):
    w, h = landscape(A4)
    y = h - margin_top + 6

    # Logo left
    if cfg.logo_bytes:
        try:
            img = ImageReader(io.BytesIO(cfg.logo_bytes))
            cnv.drawImage(img, margin_left, y - 18, width=60, height=18, mask="auto", preserveAspectRatio=True, anchor="sw")
        except Exception:
            pass

    # Section title right-ish
    cnv.setFont("Helvetica-Bold", 14)
    cnv.setFillColor(hex_to_rl(C["text_primary"]))
    cnv.drawRightString(w - margin_right, y - 6, title)

    # Divider line
    cnv.setStrokeColor(hex_to_rl(C["divider"]))
    cnv.setLineWidth(1)
    cnv.line(margin_left, y - 24, w - margin_right, y - 24)

def _draw_cover(cnv: canvas.Canvas, cfg: PdfConfig):
    w, h = landscape(A4)
    margin_left = 32 * mm
    margin_right = 32 * mm
    margin_top = 28 * mm
    margin_bottom = 28 * mm

    cnv.setFillColor(hex_to_rl(C["bg"]))
    cnv.rect(0, 0, w, h, fill=1, stroke=0)

    # Logo
    if cfg.logo_bytes:
        try:
            img = ImageReader(io.BytesIO(cfg.logo_bytes))
            cnv.drawImage(img, margin_left, h - margin_top - 22, width=110, height=32, mask="auto", preserveAspectRatio=True)
        except Exception:
            pass

    cnv.setFillColor(hex_to_rl(C["text_primary"]))
    cnv.setFont("Helvetica-Bold", 36)
    cnv.drawString(margin_left, h - margin_top - 110, "Monthly Business Review Report")

    cnv.setFont("Helvetica", 20)
    cnv.setFillColor(hex_to_rl(C["text_secondary"]))
    cnv.drawString(margin_left, h - margin_top - 145, f"{cfg.customer_name} / {cfg.date_range_label}")

    cnv.setFont("Helvetica", 14)
    cnv.setFillColor(hex_to_rl(C["text_muted"]))
    cnv.drawString(margin_left, h - margin_top - 175, f"Latency allowance: {cfg.latency_minutes} minutes")
    cnv.drawString(margin_left, h - margin_top - 195, f"Milestones: {', '.join(cfg.selected_milestones)}")

def _draw_section_intro(cnv: canvas.Canvas, cfg: PdfConfig, section_title: str, section_desc: str):
    w, h = landscape(A4)
    margin_left = 32 * mm
    margin_right = 32 * mm
    margin_top = 28 * mm
    margin_bottom = 28 * mm

    cnv.setFillColor(hex_to_rl(C["bg"]))
    cnv.rect(0, 0, w, h, fill=1, stroke=0)

    if cfg.logo_bytes:
        try:
            img = ImageReader(io.BytesIO(cfg.logo_bytes))
            cnv.drawImage(img, margin_left, h - margin_top - 22, width=90, height=26, mask="auto", preserveAspectRatio=True)
        except Exception:
            pass

    cnv.setFont("Helvetica-Bold", 28)
    cnv.setFillColor(hex_to_rl(C["text_primary"]))
    cnv.drawString(margin_left, h - margin_top - 110, section_title)

    cnv.setFont("Helvetica", 14)
    cnv.setFillColor(hex_to_rl(C["text_secondary"]))
    text_obj = cnv.beginText(margin_left, h - margin_top - 145)
    text_obj.setLeading(18)
    for line in section_desc.split("\n"):
        text_obj.textLine(line)
    cnv.drawText(text_obj)

def _draw_kpi_row(cnv: canvas.Canvas, x: float, y: float, w: float, h: float, items: List[Tuple[str, str, str, str]]):
    """
    items: [(title, value, subtitle, tone)]
    """
    gap = 8
    card_w = (w - gap * (len(items) - 1)) / len(items)
    for i, (title, value, subtitle, tone) in enumerate(items):
        cx = x + i * (card_w + gap)
        border = {
            "neutral": C["divider"],
            "good": C["positive"],
            "warn": C["amber"],
            "bad": C["critical"],
            "accent": C["accent"],
        }.get(tone, C["divider"])

        cnv.setFillColor(hex_to_rl(C["bg"]))
        cnv.setStrokeColor(hex_to_rl(border))
        cnv.setLineWidth(1)
        cnv.roundRect(cx, y, card_w, h, 10, fill=1, stroke=1)

        cnv.setFont("Helvetica", 10.5)
        cnv.setFillColor(hex_to_rl(C["text_secondary"]))
        cnv.drawString(cx + 10, y + h - 18, title)

        cnv.setFont("Helvetica-Bold", 18)
        cnv.setFillColor(hex_to_rl(C["text_primary"]))
        cnv.drawString(cx + 10, y + h - 42, value)

        cnv.setFont("Helvetica", 9.5)
        cnv.setFillColor(hex_to_rl(C["text_muted"]))
        cnv.drawString(cx + 10, y + 10, subtitle)

def _draw_table(cnv: canvas.Canvas, x: float, y: float, w: float, row_h: float, df: pd.DataFrame, max_rows: int):
    """
    Draw a simple paginated-style table block (caller handles pagination).
    y is bottom of table block.
    """
    # Header
    cnv.setFillColor(hex_to_rl(C["table_header_bg"]))
    cnv.setStrokeColor(hex_to_rl(C["divider"]))
    cnv.setLineWidth(0.5)
    cnv.rect(x, y + row_h * max_rows, w, row_h, fill=1, stroke=1)

    cols = list(df.columns)
    col_w = w / len(cols)

    cnv.setFont("Helvetica-Bold", 10.5)
    cnv.setFillColor(hex_to_rl(C["text_primary"]))
    for j, col in enumerate(cols):
        cnv.drawString(x + j * col_w + 6, y + row_h * max_rows + 6, str(col)[:38])

    # Rows
    cnv.setFont("Helvetica", 10)
    for i in range(min(max_rows, len(df))):
        row_y = y + row_h * (max_rows - 1 - i)
        fill = C["bg"] if i % 2 == 0 else C["row_alt"]
        cnv.setFillColor(hex_to_rl(fill))
        cnv.rect(x, row_y, w, row_h, fill=1, stroke=1)

        cnv.setFillColor(hex_to_rl(C["text_secondary"]))
        for j, col in enumerate(cols):
            val = df.iloc[i, j]
            s = "" if pd.isna(val) else str(val)
            cnv.drawString(x + j * col_w + 6, row_y + 6, s[:40])

def make_chart_bar(values: pd.Series, title: str) -> bytes:
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_xlabel("")
    values.plot(kind="bar", ax=ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    return buf.getvalue()

def build_pdf(
    cfg: PdfConfig,
    kpis: Dict[str, str],
    contract: Dict[str, float],
    milestone_reporting: pd.DataFrame,
    ontime_summary: pd.DataFrame,
    carrier_table: pd.DataFrame,
    lane_table: pd.DataFrame,
    rca_blocks: Dict[str, pd.DataFrame],
) -> bytes:
    w, h = landscape(A4)

    # Margins
    margin_left = 24 * mm
    margin_right = 20 * mm
    margin_top = 18 * mm
    margin_bottom = 18 * mm

    # Assemble pages list (draw functions)
    pages = []

    def add_page(draw_fn):
        pages.append(draw_fn)

    # Cover
    add_page(lambda cnv, pn, pt: _draw_cover(cnv, cfg))

    # Executive Summary intro
    add_page(lambda cnv, pn, pt: _draw_section_intro(
        cnv, cfg,
        "Executive Summary",
        "High-level performance highlights for the selected period.\nIncludes milestone reporting and on-time performance."
    ))

    # Executive Summary content page
    def exec_content(cnv, pn, pt):
        cnv.setFillColor(hex_to_rl(C["bg"]))
        cnv.rect(0, 0, w, h, fill=1, stroke=0)
        _draw_header(cnv, cfg, "Executive Summary", margin_top, margin_left, margin_right)

        # KPI row
        x = margin_left
        y = h - margin_top - 120
        block_w = w - margin_left - margin_right
        _draw_kpi_row(
            cnv, x, y, block_w, 64,
            [
                ("# Shipments", kpis["shipments"], "Filtered shipments", "accent"),
                ("# Carriers", kpis["carriers"], "Unique carriers", "neutral"),
                ("Reporting %", kpis["reporting_pct"], "All selected milestones", "good" if float(kpis["reporting_pct"].strip("%")) >= 80 else "warn"),
                ("On-time %", kpis["ontime_pct"], f"Latency: {cfg.latency_minutes} min", "good" if float(kpis["ontime_pct"].strip("%")) >= 80 else "warn"),
            ]
        )

        # Contract consumption block
        cnv.setFont("Helvetica-Bold", 14)
        cnv.setFillColor(hex_to_rl(C["text_primary"]))
        cnv.drawString(margin_left, y - 26, "Contract Consumption")

        contracted = contract.get("contracted", 0.0)
        billed = contract.get("billed", 0.0)
        pct = (billed / contracted * 100) if contracted else 0.0
        tone = C["positive"] if billed <= contracted else C["critical"]

        cnv.setFont("Helvetica", 11)
        cnv.setFillColor(hex_to_rl(C["text_secondary"]))
        cnv.drawString(margin_left, y - 46, f"Contracted volume: {contracted:,.0f}")
        cnv.drawString(margin_left, y - 62, f"Billed volume: {billed:,.0f}")
        cnv.setFillColor(hex_to_rl(tone))
        cnv.drawString(margin_left, y - 78, f"Consumption: {pct:.1f}%")

        # Charts
        # Milestone reporting chart
        mr = milestone_reporting.set_index("Milestone")["Reporting %"].str.rstrip("%").astype(float)
        chart1 = make_chart_bar(mr, "Milestone Reporting %")
        img1 = ImageReader(io.BytesIO(chart1))
        cnv.drawImage(img1, margin_left, 80, width=(block_w/2)-6, height=190, mask="auto", preserveAspectRatio=True)

        # On-time chart
        ot = ontime_summary.set_index("Milestone")["On-time %"].str.rstrip("%").astype(float)
        chart2 = make_chart_bar(ot, "On-time % (Planned End vs Actual)")
        img2 = ImageReader(io.BytesIO(chart2))
        cnv.drawImage(img2, margin_left + (block_w/2)+6, 80, width=(block_w/2)-6, height=190, mask="auto", preserveAspectRatio=True)

        _draw_footer(cnv, cfg, pn, pt, margin_bottom, margin_left, margin_right)

    add_page(exec_content)

    # Carrier Performance intro
    add_page(lambda cnv, pn, pt: _draw_section_intro(
        cnv, cfg, "Carrier Performance",
        "Carrier-wise shipment volume, milestone reporting, and on-time performance."
    ))

    # Carrier table pages (paginate to <= 14 rows/page)
    carrier_df = carrier_table.copy()
    rows_per_page = 14
    total_carrier_pages = max(1, math.ceil(len(carrier_df) / rows_per_page))

    for pi in range(total_carrier_pages):
        def carrier_page_factory(start=pi*rows_per_page):
            def _p(cnv, pn, pt):
                cnv.setFillColor(hex_to_rl(C["bg"]))
                cnv.rect(0, 0, w, h, fill=1, stroke=0)
                _draw_header(cnv, cfg, "Carrier Performance", margin_top, margin_left, margin_right)

                cnv.setFont("Helvetica-Bold", 14)
                cnv.setFillColor(hex_to_rl(C["text_primary"]))
                cnv.drawString(margin_left, h - margin_top - 56, "Carrier Summary")

                block_w = w - margin_left - margin_right
                table_x = margin_left
                table_y = 70
                table_h_rows = rows_per_page
                table_row_h = 18

                slice_df = carrier_df.iloc[start:start+rows_per_page].copy()
                _draw_table(cnv, table_x, table_y, block_w, table_row_h, slice_df, table_h_rows)
                _draw_footer(cnv, cfg, pn, pt, margin_bottom, margin_left, margin_right)
            return _p
        add_page(carrier_page_factory())

    # Lane Insights intro
    add_page(lambda cnv, pn, pt: _draw_section_intro(
        cnv, cfg, "Lane Insights",
        "Lane-wise milestone reporting and on-time performance (Origin → Destination).\nIncludes Carrier × Lane rollups."
    ))

    # Lane table pages
    lane_df = lane_table.copy()
    total_lane_pages = max(1, math.ceil(len(lane_df) / rows_per_page))
    for pi in range(total_lane_pages):
        def lane_page_factory(start=pi*rows_per_page):
            def _p(cnv, pn, pt):
                cnv.setFillColor(hex_to_rl(C["bg"]))
                cnv.rect(0, 0, w, h, fill=1, stroke=0)
                _draw_header(cnv, cfg, "Lane Insights", margin_top, margin_left, margin_right)

                cnv.setFont("Helvetica-Bold", 14)
                cnv.setFillColor(hex_to_rl(C["text_primary"]))
                cnv.drawString(margin_left, h - margin_top - 56, "Lane Summary")

                block_w = w - margin_left - margin_right
                table_x = margin_left
                table_y = 70
                table_row_h = 18
                slice_df = lane_df.iloc[start:start+rows_per_page].copy()
                _draw_table(cnv, table_x, table_y, block_w, table_row_h, slice_df, rows_per_page)
                _draw_footer(cnv, cfg, pn, pt, margin_bottom, margin_left, margin_right)
            return _p
        add_page(lane_page_factory())

    # RCA intro
    add_page(lambda cnv, pn, pt: _draw_section_intro(
        cnv, cfg, "Root Cause Analysis",
        "Carrier-wise affected shipments and top root causes for Creation, Tracking, and Milestone issues."
    ))

    # RCA pages (one per type)
    for rca_title, rca_df in rca_blocks.items():
        def rca_page_factory(title=rca_title, df_block=rca_df):
            def _p(cnv, pn, pt):
                cnv.setFillColor(hex_to_rl(C["bg"]))
                cnv.rect(0, 0, w, h, fill=1, stroke=0)
                _draw_header(cnv, cfg, "Root Cause Analysis", margin_top, margin_left, margin_right)

                cnv.setFont("Helvetica-Bold", 14)
                cnv.setFillColor(hex_to_rl(C["text_primary"]))
                cnv.drawString(margin_left, h - margin_top - 56, title)

                block_w = w - margin_left - margin_right
                table_x = margin_left
                table_y = 70
                table_row_h = 18

                # Keep table to <=14 rows
                df_slice = df_block.head(rows_per_page).copy()
                _draw_table(cnv, table_x, table_y, block_w, table_row_h, df_slice, rows_per_page)

                _draw_footer(cnv, cfg, pn, pt, margin_bottom, margin_left, margin_right)
            return _p
        add_page(rca_page_factory())

    # Render
    buf = io.BytesIO()
    cnv = canvas.Canvas(buf, pagesize=landscape(A4))
    total_pages = len(pages)

    for idx, draw_fn in enumerate(pages, start=1):
        draw_fn(cnv, idx, total_pages)
        cnv.showPage()

    cnv.save()
    return buf.getvalue()


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Monthly Business Review Report", layout="wide")

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
      h1, h2, h3, h4, h5 {{ color: {C['text_primary']}; }}
      .stCaption, .stMarkdown p {{ color: {C['text_secondary']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Monthly Business Review Report Builder")

with st.sidebar:
    st.header("1) Upload files")
    shipments_file = st.file_uploader("Shipment details (CSV or Excel)", type=["csv", "xlsx", "xls"])
    rca_creation_file = st.file_uploader("RCA – Creation issues (Excel/CSV)", type=["xlsx", "xls", "csv"])
    rca_tracking_file = st.file_uploader("RCA – Tracking issues (Excel/CSV)", type=["xlsx", "xls", "csv"])
    rca_milestone_file = st.file_uploader("RCA – Milestone issues (Excel/CSV)", type=["xlsx", "xls", "csv"])
    logo_file = st.file_uploader("Project44 logo (PNG)", type=["png"])

    st.header("2) Controls")
    customer_name = st.text_input("Customer name", value="Customer Name")
    contracted_volume = st.number_input("Contracted volume", min_value=0.0, value=0.0, step=1.0)
    billed_volume = st.number_input("Billed volume", min_value=0.0, value=0.0, step=1.0)

    latency_minutes = st.selectbox("Latency allowed (minutes)", options=[5, 10, 15, 30, 60], index=0)

    st.divider()
    st.subheader("Filters")
    date_basis = st.radio(
        "Date filter based on",
        ["Destination latest planned arrival end time", "Created time"],
        index=0,
    )

    selected_milestones = st.multiselect(
        "Milestones / events",
        list(MILESTONES.keys()),
        default=["Arrival at Destination", "Departure at Origin", "Arrival at Origin"],
    )

# Load data
shipments_df = pd.DataFrame()
rca_creation_df = pd.DataFrame()
rca_tracking_df = pd.DataFrame()
rca_milestone_df = pd.DataFrame()

errors = []
if shipments_file:
    try:
        shipments_df = load_shipments(shipments_file)
    except Exception as e:
        errors.append(str(e))

if rca_creation_file:
    try:
        rca_creation_df = load_rca(rca_creation_file)
    except Exception as e:
        errors.append(f"Creation RCA: {e}")

if rca_tracking_file:
    try:
        rca_tracking_df = load_rca(rca_tracking_file)
    except Exception as e:
        errors.append(f"Tracking RCA: {e}")

if rca_milestone_file:
    try:
        rca_milestone_df = load_rca(rca_milestone_file)
    except Exception as e:
        errors.append(f"Milestone RCA: {e}")

if errors:
    st.error("Upload issues:\n- " + "\n- ".join(errors))
    st.stop()

if shipments_df.empty:
    st.info("Upload your shipment details file to start.")
    st.stop()

# Apply filters
df = shipments_df.copy()
df["Lane"] = df.apply(lane_from_row, axis=1)

date_col = "Destination latest planned arrival end time" if date_basis.startswith("Destination") else "Created time"
min_d = df[date_col].min()
max_d = df[date_col].max()

if pd.isna(min_d) or pd.isna(max_d):
    st.warning(f"Selected date basis column '{date_col}' has no parseable dates. Please check the file.")
    st.stop()

colA, colB, colC = st.columns([1.2, 1.2, 1.6])
with colA:
    date_from = st.date_input("From", value=min_d.date())
with colB:
    date_to = st.date_input("To", value=max_d.date())

mask_range = (df[date_col] >= pd.Timestamp(date_from)) & (df[date_col] < (pd.Timestamp(date_to) + pd.Timedelta(days=1)))
df = df[mask_range].copy()

# Include/exclude specific dates
all_dates = sorted([d.date() for d in df[date_col].dropna().dt.to_pydatetime()])
all_dates_unique = sorted(list(dict.fromkeys(all_dates)))
with colC:
    mode_dates = st.radio("Specific dates", ["Include all", "Exclude selected"], horizontal=True, index=0)
    dates_selected = st.multiselect("Pick dates", options=all_dates_unique, default=[])

if mode_dates == "Exclude selected" and dates_selected:
    df = df[~df[date_col].dt.date.isin(dates_selected)].copy()

# Carrier filter include/exclude
carriers = sorted([c for c in df["Current carrier"].dropna().unique() if str(c).strip() and str(c).lower() != "nan"])
col1, col2 = st.columns([1, 1])
with col1:
    carrier_mode = st.radio("Carriers filter", ["Include selected", "Exclude selected"], horizontal=True, index=0)
with col2:
    carrier_pick = st.multiselect("Carriers", options=carriers, default=[])

if carrier_pick:
    if carrier_mode == "Include selected":
        df = df[df["Current carrier"].isin(carrier_pick)].copy()
    else:
        df = df[~df["Current carrier"].isin(carrier_pick)].copy()

if df.empty:
    st.warning("No shipments match the selected filters.")
    st.stop()

# Derived fields
df["Completion bucket"] = derive_completion_bucket(df, selected_milestones)

# Milestone reporting flags
for m in selected_milestones:
    df[f"Reported: {m}"] = milestone_reported(df, m)
    df[f"On-time: {m}"] = on_time(df, m, latency_minutes)

df["All selected milestones reported"] = True
for m in selected_milestones:
    df["All selected milestones reported"] &= df[f"Reported: {m}"]

# On-time overall = average of event on-time among those that reported? We'll use strict: must be on-time for all selected milestones where planned+actual exist.
df["All selected milestones on-time"] = True
for m in selected_milestones:
    # If milestone not reported, treat as False for overall on-time (executive strictness)
    df["All selected milestones on-time"] &= df[f"On-time: {m}"]

df["Destination ETA accurate"] = estimated_accuracy_destination(df, latency_minutes)

# Exception parsing
df["Exception list"] = df["Exceptions"].apply(parse_exceptions)
df["Has exceptions"] = df["Exception list"].apply(lambda x: len(x) > 0)

# KPI computations
num_shipments = df["Shipment ID"].nunique()
num_carriers = df["Current carrier"].nunique()
reporting_all = int(df["All selected milestones reported"].sum())
reporting_all_pct = (reporting_all / len(df) * 100) if len(df) else 0
ontime_all = int(df["All selected milestones on-time"].sum())
ontime_all_pct = (ontime_all / len(df) * 100) if len(df) else 0

completed_count = int((df["Completion bucket"] == "Completed").sum())
timedout_count = int((df["Completion bucket"] == "Timed Out").sum())

# Contract consumption tone
consumption_pct = (billed_volume / contracted_volume * 100) if contracted_volume else 0.0
consumption_tone = "good" if billed_volume <= contracted_volume else "bad"
consumption_label = "On-track" if billed_volume <= contracted_volume else "Overconsumption"

# Tabs
tab_overview, tab_carrier, tab_ontime, tab_lanes, tab_rca, tab_pdf = st.tabs(
    ["Overview", "Carrier view", "On-time performance", "Lane insights", "RCA analysis", "Export PDF"]
)

with tab_overview:
    st.subheader("Summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("# Shipments", f"{num_shipments:,}", "Filtered shipments", "accent")
    with c2:
        kpi_card("# Carriers", f"{num_carriers:,}", "Unique carriers", "neutral")
    with c3:
        kpi_card("Reporting (all milestones)", f"{reporting_all_pct:.1f}%", f"{reporting_all:,} shipments", "good" if reporting_all_pct >= 80 else "warn")
    with c4:
        kpi_card("On-time (all milestones)", f"{ontime_all_pct:.1f}%", f"Latency: {latency_minutes} min", "good" if ontime_all_pct >= 80 else "warn")

    st.markdown("### Contract consumption")
    cA, cB, cC, cD = st.columns([1.2, 1.2, 1.2, 1.2])
    with cA:
        kpi_card("Contracted volume", f"{contracted_volume:,.0f}", "", "neutral")
    with cB:
        kpi_card("Billed volume", f"{billed_volume:,.0f}", "", "neutral")
    with cC:
        kpi_card("Consumption", f"{consumption_pct:.1f}%", consumption_label, consumption_tone)
    with cD:
        kpi_card("Delta", f"{(billed_volume - contracted_volume):,.0f}", "Billed - Contracted", "bad" if billed_volume > contracted_volume else "good")

    st.markdown("### Shipment state summary")
    state_counts = df["Completion bucket"].value_counts(dropna=False).reset_index()
    state_counts.columns = ["State / Bucket", "Shipments"]
    st.dataframe(state_counts, use_container_width=True)

    st.markdown("### Milestone reporting")
    rows = []
    for m in selected_milestones:
        cnt = int(df[f"Reported: {m}"].sum())
        pct = (cnt / len(df) * 100) if len(df) else 0
        rows.append([m, cnt, f"{pct:.1f}%"])
    milestone_reporting_df = pd.DataFrame(rows, columns=["Milestone", "Shipments reported", "Reporting %"])
    st.dataframe(milestone_reporting_df, use_container_width=True)

    st.markdown("### Exceptions summary")
    ex_none = int((~df["Has exceptions"]).sum())
    ex_yes = int((df["Has exceptions"]).sum())
    st.write(f"- Shipments without exceptions: **{ex_none:,}**")
    st.write(f"- Shipments with exceptions: **{ex_yes:,}**")

    # Exception-wise breakdown
    ex_map = {}
    for lst in df["Exception list"]:
        for e in lst:
            ex_map[e] = ex_map.get(e, 0) + 1
    ex_df = pd.DataFrame(sorted(ex_map.items(), key=lambda x: x[1], reverse=True), columns=["Exception", "Shipments"])
    if not ex_df.empty:
        st.dataframe(ex_df, use_container_width=True)
    else:
        st.caption("No exception values found in the filtered set.")

with tab_carrier:
    st.subheader("Carrier-wise view")

    # Build carrier table
    group = df.groupby("Current carrier", dropna=False)
    carrier_rows = []
    for carrier, g in group:
        carrier = str(carrier).strip() if str(carrier).strip() else "Unknown"
        shipments = g["Shipment ID"].nunique()
        completed = int((g["Completion bucket"] == "Completed").sum())
        timedout = int((g["Completion bucket"] == "Timed Out").sum())
        rep_all = int(g["All selected milestones reported"].sum())
        rep_all_pct = (rep_all / len(g) * 100) if len(g) else 0
        ot_all = int(g["All selected milestones on-time"].sum())
        ot_all_pct = (ot_all / len(g) * 100) if len(g) else 0
        ex_pct = (g["Has exceptions"].mean() * 100) if len(g) else 0

        carrier_rows.append([
            carrier, shipments, completed, timedout,
            f"{rep_all_pct:.1f}%", f"{ot_all_pct:.1f}%", f"{ex_pct:.1f}%"
        ])

    carrier_table = pd.DataFrame(
        carrier_rows,
        columns=["Carrier", "# Shipments", "# Completed", "# Timed out", "Reporting %", "On-time %", "Exceptions %"]
    ).sort_values("# Shipments", ascending=False)

    st.dataframe(carrier_table, use_container_width=True)

with tab_ontime:
    st.subheader("On-time performance (event-wise)")

    ot_rows = []
    for m in selected_milestones:
        cnt_reported = int(df[f"Reported: {m}"].sum())
        cnt_ontime = int(df[f"On-time: {m}"].sum())
        pct_ontime = (cnt_ontime / len(df) * 100) if len(df) else 0
        ot_rows.append([m, cnt_reported, cnt_ontime, f"{pct_ontime:.1f}%"])

    ontime_df = pd.DataFrame(ot_rows, columns=["Milestone", "Reported shipments", "On-time shipments", "On-time %"])
    st.dataframe(ontime_df, use_container_width=True)

    st.markdown("### Destination ETA accuracy vs Actual (within latency)")
    eta_acc = int(df["Destination ETA accurate"].sum())
    eta_pct = (eta_acc / len(df) * 100) if len(df) else 0
    kpi_card("Destination ETA accuracy", f"{eta_pct:.1f}%", f"{eta_acc:,} shipments", "good" if eta_pct >= 80 else "warn")

with tab_lanes:
    st.subheader("Lane insights (Origin → Destination)")

    lane_group = df.groupby("Lane", dropna=False)
    lane_rows = []
    for lane, g in lane_group:
        shipments = g["Shipment ID"].nunique()
        rep_all_pct = (g["All selected milestones reported"].mean() * 100) if len(g) else 0
        ot_all_pct = (g["All selected milestones on-time"].mean() * 100) if len(g) else 0
        lane_rows.append([lane, shipments, f"{rep_all_pct:.1f}%", f"{ot_all_pct:.1f}%"])

    lane_table = pd.DataFrame(lane_rows, columns=["Lane", "# Shipments", "Reporting %", "On-time %"])\
        .sort_values("# Shipments", ascending=False)

    st.dataframe(lane_table, use_container_width=True)

    st.markdown("### Carrier × Lane rollup (top 20 lanes)")
    top_lanes = lane_table.head(20)["Lane"].tolist()
    cl = df[df["Lane"].isin(top_lanes)].copy()
    cl_group = cl.groupby(["Current carrier", "Lane"], dropna=False).agg(
        shipments=("Shipment ID", "nunique"),
        reporting=("All selected milestones reported", "mean"),
        ontime=("All selected milestones on-time", "mean"),
    ).reset_index()
    cl_group["Reporting %"] = (cl_group["reporting"] * 100).round(1).astype(str) + "%"
    cl_group["On-time %"] = (cl_group["ontime"] * 100).round(1).astype(str) + "%"
    cl_group = cl_group.drop(columns=["reporting", "ontime"]).sort_values("shipments", ascending=False)

    st.dataframe(cl_group, use_container_width=True)

with tab_rca:
    st.subheader("Root Cause Analysis (carrier-wise affected shipments)")

    def rca_block(df_rca: pd.DataFrame, title: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        returns:
          carrier summary: Carrier, affected shipments
          top root causes: Root Cause Error, count
        """
        if df_rca.empty:
            return pd.DataFrame(columns=["Carrier", "Affected shipments"]), pd.DataFrame(columns=["Root Cause Error", "Count"])

        # Match to shipment IDs (except creation can still be shown even if unmatched)
        df2 = df_rca.copy()
        df2["match_shipment_id"] = df2["match_shipment_id"].astype(str).replace("nan", "").fillna("")
        df2["Carrier"] = df2.get("Carrier", "").astype(str).replace("nan", "").fillna("Unknown")

        # "Other than creation should match shipment details sheet"
        if title.lower().startswith("tracking") or title.lower().startswith("milestone"):
            df2 = df2[df2["match_shipment_id"].isin(df["Shipment ID"].astype(str))].copy()

        carrier_sum = df2.groupby("Carrier", dropna=False)["match_shipment_id"].nunique().reset_index()
        carrier_sum.columns = ["Carrier", "Affected shipments"]
        carrier_sum = carrier_sum.sort_values("Affected shipments", ascending=False)

        top_rc = df2.groupby("Root Cause Error", dropna=False).size().reset_index(name="Count") \
            .sort_values("Count", ascending=False)

        return carrier_sum, top_rc

    csum, topc = rca_block(rca_creation_df, "Creation")
    tsum, topt = rca_block(rca_tracking_df, "Tracking")
    msum, topm = rca_block(rca_milestone_df, "Milestone")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Creation issues")
        st.dataframe(csum, use_container_width=True, height=240)
        st.caption("Top root causes")
        st.dataframe(topc.head(10), use_container_width=True, height=240)
    with c2:
        st.markdown("#### Tracking issues")
        st.dataframe(tsum, use_container_width=True, height=240)
        st.caption("Top root causes")
        st.dataframe(topt.head(10), use_container_width=True, height=240)
    with c3:
        st.markdown("#### Milestone issues")
        st.dataframe(msum, use_container_width=True, height=240)
        st.caption("Top root causes")
        st.dataframe(topm.head(10), use_container_width=True, height=240)

with tab_pdf:
    st.subheader("Export a designed PDF (A4 Landscape)")

    # Prepare PDF inputs
    logo_bytes = logo_file.getvalue() if logo_file else None
    date_range_label = f"{date_from.isoformat()} to {date_to.isoformat()}"

    # Tables for PDF should be compact
    pdf_carrier = carrier_table.copy()
    pdf_lane = lane_table.copy()

    # Milestone reporting summary for PDF
    mr_rows = []
    for m in selected_milestones:
        cnt = int(df[f"Reported: {m}"].sum())
        pct = (cnt / len(df) * 100) if len(df) else 0
        mr_rows.append([m, cnt, f"{pct:.1f}%"])
    mr_pdf = pd.DataFrame(mr_rows, columns=["Milestone", "Shipments", "Reporting %"])

    ot_rows = []
    for m in selected_milestones:
        cnt_ot = int(df[f"On-time: {m}"].sum())
        pct = (cnt_ot / len(df) * 100) if len(df) else 0
        ot_rows.append([m, cnt_ot, f"{pct:.1f}%"])
    ot_pdf = pd.DataFrame(ot_rows, columns=["Milestone", "On-time shipments", "On-time %"])

    # RCA blocks for PDF
    def rca_for_pdf(df_rca: pd.DataFrame, title: str) -> pd.DataFrame:
        if df_rca.empty:
            return pd.DataFrame({"Carrier": [], "Affected shipments": []})
        df2 = df_rca.copy()
        df2["match_shipment_id"] = df2["match_shipment_id"].astype(str).replace("nan", "").fillna("")
        df2["Carrier"] = df2.get("Carrier", "").astype(str).replace("nan", "").fillna("Unknown")
        if title.lower() in ["tracking", "milestone"]:
            df2 = df2[df2["match_shipment_id"].isin(df["Shipment ID"].astype(str))].copy()
        out = df2.groupby("Carrier", dropna=False)["match_shipment_id"].nunique().reset_index()
        out.columns = ["Carrier", "Affected shipments"]
        return out.sort_values("Affected shipments", ascending=False).head(20)

    rca_blocks = {
        "Creation issues – affected shipments by carrier": rca_for_pdf(rca_creation_df, "creation"),
        "Tracking issues – affected shipments by carrier": rca_for_pdf(rca_tracking_df, "tracking"),
        "Milestone issues – affected shipments by carrier": rca_for_pdf(rca_milestone_df, "milestone"),
    }

    # KPI strings
    kpis = {
        "shipments": f"{num_shipments:,}",
        "carriers": f"{num_carriers:,}",
        "reporting_pct": f"{reporting_all_pct:.1f}%",
        "ontime_pct": f"{ontime_all_pct:.1f}%",
    }

    cfg = PdfConfig(
        customer_name=customer_name,
        date_range_label=date_range_label,
        logo_bytes=logo_bytes,
        latency_minutes=int(latency_minutes),
        selected_milestones=selected_milestones,
    )

    if st.button("Generate PDF"):
        pdf_bytes = build_pdf(
            cfg=cfg,
            kpis=kpis,
            contract={"contracted": float(contracted_volume), "billed": float(billed_volume)},
            milestone_reporting=mr_pdf,
            ontime_summary=ot_pdf,
            carrier_table=pdf_carrier,
            lane_table=pdf_lane,
            rca_blocks=rca_blocks,
        )

        filename = f"Monthly Business Review Report - {customer_name} - {date_range_label}.pdf"
        st.success("PDF generated.")
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
        )

st.caption("Tip: If you want the PDF to use Inter fonts exactly, we can add Inter TTF files to the repo and register them in ReportLab.")
