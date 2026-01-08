import io
import math
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# PDF (no charts)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# -----------------------------
# Theme (close to your spec)
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
    "table_header_bg": "#F3F4F6",
    "row_alt": "#F9FAFB",
}

MILESTONES = {
    "Arrival at Origin": {
        "actual": "Origin actual arrival time",
        "planned_end": "Origin latest planned arrival end time",
    },
    "Departure at Origin": {
        "actual": "Origin actual departure time",
        "planned_end": "Origin latest planned departure end time",
    },
    "Arrival at Destination": {
        "actual": "Destination actual arrival time",
        "planned_end": "Destination latest planned arrival end time",
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
    "Creation Date",
    "Tracking End Date",
]


# -----------------------------
# Helpers
# -----------------------------
def hex_to_rl(h: str):
    h = h.lstrip("#")
    r = int(h[0:2], 16) / 255
    g = int(h[2:4], 16) / 255
    b = int(h[4:6], 16) / 255
    return colors.Color(r, g, b)


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


@st.cache_data(show_spinner=False)
def load_shipments_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(bio)
    else:
        df = pd.read_excel(bio)

    missing = [c for c in SHIPMENT_COLS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Shipment file missing columns: {missing}")

    # Parse date/time columns (only the ones we actually use + milestone columns)
    needed_dt = ["Created time"]
    for m in MILESTONES.values():
        needed_dt += [m["actual"], m["planned_end"]]
    needed_dt += ["Destination estimated arrival time", "Destination actual arrival time", "Destination latest planned arrival end time"]
    needed_dt = [c for c in dict.fromkeys(needed_dt) if c in df.columns]

    for c in needed_dt:
        df[c] = _to_dt(df[c])

    # Normalize key fields
    df["Shipment ID"] = df["Shipment ID"].astype(str).replace("nan", "").fillna("")
    df["Current carrier"] = df["Current carrier"].astype(str).replace("nan", "").fillna("")
    df["Current state"] = df["Current state"].astype(str).replace("nan", "").fillna("")
    df["Current state reason"] = df["Current state reason"].astype(str).replace("nan", "").fillna("")
    df["Exceptions"] = df["Exceptions"].astype(str).replace("nan", "").fillna("")

    return df


@st.cache_data(show_spinner=False)
def load_rca_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(bio)
    else:
        df = pd.read_excel(bio)

    keep = [c for c in RCA_KEEP_COLS if c in df.columns]
    df = df[keep].copy()

    for c in ["Bill Of Lading", "Order Number", "Root Cause Error", "Carrier"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace("nan", "").fillna("")

    # match key = Shipment ID
    df["match_shipment_id"] = ""
    if "Bill Of Lading" in df.columns and "Order Number" in df.columns:
        df["match_shipment_id"] = df["Bill Of Lading"].where(df["Bill Of Lading"] != "", df["Order Number"])
    elif "Bill Of Lading" in df.columns:
        df["match_shipment_id"] = df["Bill Of Lading"]
    elif "Order Number" in df.columns:
        df["match_shipment_id"] = df["Order Number"]

    for c in ["Creation Date", "Tracking End Date"]:
        if c in df.columns:
            df[c] = _to_dt(df[c])

    return df


def parse_exceptions(val: str) -> List[str]:
    s = "" if val is None else str(val).strip()
    if s == "" or s.lower() == "nan":
        return []
    parts = []
    for token in s.replace("|", ";").replace(",", ";").split(";"):
        t = token.strip()
        if t:
            parts.append(t)
    return parts


def build_lane_cols(df: pd.DataFrame) -> pd.Series:
    # Vectorized lane: prefer city/state/country; fall back to Origin/Destination
    o_city = df["Origin city"].fillna("").astype(str).str.strip()
    o_state = df["Origin state"].fillna("").astype(str).str.strip()
    o_country = df["Origin country"].fillna("").astype(str).str.strip()
    d_city = df["Destination city"].fillna("").astype(str).str.strip()
    d_state = df["Destination state"].fillna("").astype(str).str.strip()
    d_country = df["Destination country"].fillna("").astype(str).str.strip()

    def join3(a, b, c):
        out = a
        out = np.where((out != "") & (b != ""), out + ", " + b, np.where(out == "", b, out))
        out = np.where((out != "") & (c != ""), out + ", " + c, np.where(out == "", c, out))
        return pd.Series(out)

    o = join3(o_city, o_state, o_country)
    d = join3(d_city, d_state, d_country)

    o_fallback = df["Origin"].fillna("").astype(str).str.strip()
    d_fallback = df["Destination"].fillna("").astype(str).str.strip()

    o = np.where(o.astype(str).str.strip() == "", o_fallback, o)
    d = np.where(d.astype(str).str.strip() == "", d_fallback, d)

    return pd.Series(o).astype(str) + " → " + pd.Series(d).astype(str)


def milestone_reported(df: pd.DataFrame, milestone_name: str) -> pd.Series:
    return df[MILESTONES[milestone_name]["actual"]].notna()


def on_time(df: pd.DataFrame, milestone_name: str, latency_minutes: int) -> pd.Series:
    actual = df[MILESTONES[milestone_name]["actual"]]
    planned_end = df[MILESTONES[milestone_name]["planned_end"]]
    latency = pd.to_timedelta(latency_minutes, unit="m")
    ok = actual.notna() & planned_end.notna()
    out = pd.Series(False, index=df.index)
    out[ok] = actual[ok] <= (planned_end[ok] + latency)
    return out


def destination_eta_accurate(df: pd.DataFrame, latency_minutes: int) -> pd.Series:
    est = df["Destination estimated arrival time"]
    act = df["Destination actual arrival time"]
    latency = pd.to_timedelta(latency_minutes, unit="m")
    ok = est.notna() & act.notna()
    out = pd.Series(False, index=df.index)
    out[ok] = (act[ok] - est[ok]).abs() <= latency
    return out


def completion_bucket(df: pd.DataFrame, selected_milestones: List[str]) -> pd.Series:
    state = df["Current state"].astype(str).str.upper()
    reason = df["Current state reason"].astype(str).str.upper()

    is_completed = state.eq("COMPLETED")
    is_timedout_reason = reason.eq("TRACKING_TIMED_OUT")

    all_reported = pd.Series(True, index=df.index)
    for m in selected_milestones:
        all_reported &= milestone_reported(df, m)

    bucket = pd.Series("In Progress / Other", index=df.index)
    bucket[is_completed & ~is_timedout_reason] = "Completed"
    bucket[is_completed & is_timedout_reason & ~all_reported] = "Timed Out"
    bucket[is_completed & is_timedout_reason & all_reported] = "Completed"

    # Preserve other states text
    other = ~is_completed
    bucket[other] = df.loc[other, "Current state"].replace("", "Unknown")
    return bucket


# -----------------------------
# PDF (tables only)
# -----------------------------
@dataclass
class PdfConfig:
    customer_name: str
    date_range_label: str
    latency_minutes: int
    selected_milestones: List[str]
    logo_bytes: Optional[bytes]


def draw_footer(cnv: canvas.Canvas, cfg: PdfConfig, page_num: int, page_total: int, margin_left: float, margin_right: float, margin_bottom: float):
    w, _ = landscape(A4)
    cnv.setFont("Helvetica", 8.5)
    cnv.setFillColor(hex_to_rl(C["text_muted"]))
    y = margin_bottom - 8
    cnv.drawString(margin_left, y, cfg.customer_name)
    cnv.drawCentredString(w / 2, y, "Confidential – Customer Use Only")
    cnv.drawRightString(w - margin_right, y, f"Page {page_num} of {page_total}")


def draw_header(cnv: canvas.Canvas, cfg: PdfConfig, title: str, margin_left: float, margin_right: float, margin_top: float):
    w, h = landscape(A4)
    y = h - margin_top + 6

    if cfg.logo_bytes:
        try:
            img = ImageReader(io.BytesIO(cfg.logo_bytes))
            cnv.drawImage(img, margin_left, y - 18, width=60, height=18, mask="auto", preserveAspectRatio=True)
        except Exception:
            pass

    cnv.setFont("Helvetica-Bold", 14)
    cnv.setFillColor(hex_to_rl(C["text_primary"]))
    cnv.drawRightString(w - margin_right, y - 6, title)

    cnv.setStrokeColor(hex_to_rl(C["divider"]))
    cnv.setLineWidth(1)
    cnv.line(margin_left, y - 24, w - margin_right, y - 24)


def draw_cover(cnv: canvas.Canvas, cfg: PdfConfig):
    w, h = landscape(A4)
    margin_left = 32 * mm
    margin_top = 28 * mm

    cnv.setFillColor(hex_to_rl(C["bg"]))
    cnv.rect(0, 0, w, h, fill=1, stroke=0)

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


def draw_section_intro(cnv: canvas.Canvas, cfg: PdfConfig, section_title: str, section_desc: str):
    w, h = landscape(A4)
    margin_left = 32 * mm
    margin_top = 28 * mm

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


def draw_kpi_grid(cnv: canvas.Canvas, x: float, y: float, w: float, h: float, items: List[Dict]):
    gap = 8
    n = len(items)
    card_w = (w - gap * (n - 1)) / n
    for i, it in enumerate(items):
        cx = x + i * (card_w + gap)
        tone = it.get("tone", "neutral")
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
        cnv.drawString(cx + 10, y + h - 18, it["title"])

        cnv.setFont("Helvetica-Bold", 18)
        cnv.setFillColor(hex_to_rl(C["text_primary"]))
        cnv.drawString(cx + 10, y + h - 42, it["value"])

        cnv.setFont("Helvetica", 9.5)
        cnv.setFillColor(hex_to_rl(C["text_muted"]))
        cnv.drawString(cx + 10, y + 10, it.get("subtitle", ""))


def draw_table_block(
    cnv: canvas.Canvas,
    df: pd.DataFrame,
    title: str,
    margin_left: float,
    margin_right: float,
    margin_top: float,
    margin_bottom: float,
    max_rows: int = 14,
):
    """Draws a title + simple table (no wrapping). Caller paginates df."""
    w, h = landscape(A4)
    block_w = w - margin_left - margin_right

    cnv.setFont("Helvetica-Bold", 14)
    cnv.setFillColor(hex_to_rl(C["text_primary"]))
    cnv.drawString(margin_left, h - margin_top - 56, title)

    # table geometry
    row_h = 18
    table_x = margin_left
    table_y = margin_bottom + 30  # bottom of table
    table_rows = max_rows
    table_w = block_w

    cols = list(df.columns)
    if len(cols) == 0:
        return

    col_w = table_w / len(cols)

    # Header row
    cnv.setFillColor(hex_to_rl(C["table_header_bg"]))
    cnv.setStrokeColor(hex_to_rl(C["divider"]))
    cnv.setLineWidth(0.5)
    cnv.rect(table_x, table_y + row_h * table_rows, table_w, row_h, fill=1, stroke=1)

    cnv.setFont("Helvetica-Bold", 10.5)
    cnv.setFillColor(hex_to_rl(C["text_primary"]))
    for j, col in enumerate(cols):
        cnv.drawString(table_x + j * col_w + 6, table_y + row_h * table_rows + 6, str(col)[:40])

    # Body rows
    cnv.setFont("Helvetica", 10)
    for i in range(min(table_rows, len(df))):
        y = table_y + row_h * (table_rows - 1 - i)
        fill = C["bg"] if i % 2 == 0 else C["row_alt"]
        cnv.setFillColor(hex_to_rl(fill))
        cnv.rect(table_x, y, table_w, row_h, fill=1, stroke=1)

        cnv.setFillColor(hex_to_rl(C["text_secondary"]))
        for j, col in enumerate(cols):
            val = df.iloc[i, j]
            s = "" if pd.isna(val) else str(val)
            cnv.drawString(table_x + j * col_w + 6, y + 6, s[:46])


def build_pdf(
    cfg: PdfConfig,
    kpis: Dict[str, str],
    contract: Dict[str, float],
    state_summary: pd.DataFrame,
    milestone_reporting: pd.DataFrame,
    ontime_summary: pd.DataFrame,
    carrier_table: pd.DataFrame,
    lane_table: pd.DataFrame,
    rca_tables: Dict[str, pd.DataFrame],
) -> bytes:
    w, h = landscape(A4)

    # Margins (content pages)
    margin_left = 24 * mm
    margin_right = 20 * mm
    margin_top = 18 * mm
    margin_bottom = 18 * mm

    pages = []

    # Cover
    pages.append(lambda cnv, pn, pt: draw_cover(cnv, cfg))

    # Section intro
    pages.append(lambda cnv, pn, pt: draw_section_intro(
        cnv, cfg,
        "Executive Summary",
        "KPI overview, milestone reporting and on-time performance.\nTables only (no charts)."
    ))

    # Exec summary content
    def exec_page(cnv, pn, pt):
        cnv.setFillColor(hex_to_rl(C["bg"]))
        cnv.rect(0, 0, w, h, fill=1, stroke=0)
        draw_header(cnv, cfg, "Executive Summary", margin_left, margin_right, margin_top)

        # KPI grid
        grid_x = margin_left
        grid_y = h - margin_top - 120
        grid_w = w - margin_left - margin_right
        draw_kpi_grid(
            cnv, grid_x, grid_y, grid_w, 64,
            [
                {"title": "# Shipments", "value": kpis["shipments"], "subtitle": "Filtered", "tone": "accent"},
                {"title": "# Carriers", "value": kpis["carriers"], "subtitle": "Unique", "tone": "neutral"},
                {"title": "Reporting %", "value": kpis["reporting_pct"], "subtitle": "All selected milestones",
                 "tone": "good" if float(kpis["reporting_pct"].rstrip("%")) >= 80 else "warn"},
                {"title": "On-time %", "value": kpis["ontime_pct"], "subtitle": f"Latency: {cfg.latency_minutes}m",
                 "tone": "good" if float(kpis["ontime_pct"].rstrip("%")) >= 80 else "warn"},
            ]
        )

        # Contract line
        cnv.setFont("Helvetica-Bold", 14)
        cnv.setFillColor(hex_to_rl(C["text_primary"]))
        cnv.drawString(margin_left, grid_y - 28, "Contract Consumption")

        contracted = float(contract.get("contracted", 0.0))
        billed = float(contract.get("billed", 0.0))
        pct = (billed / contracted * 100) if contracted else 0.0
        tone = C["positive"] if billed <= contracted else C["critical"]

        cnv.setFont("Helvetica", 11)
        cnv.setFillColor(hex_to_rl(C["text_secondary"]))
        cnv.drawString(margin_left, grid_y - 48, f"Contracted volume: {contracted:,.0f}")
        cnv.drawString(margin_left, grid_y - 64, f"Billed volume: {billed:,.0f}")
        cnv.setFillColor(hex_to_rl(tone))
        cnv.drawString(margin_left, grid_y - 80, f"Consumption: {pct:.1f}%")

        # Tables (stack 3 small tables on one page: state, milestone reporting, on-time)
        # We'll show top rows to keep it readable.
        # State summary
        draw_table_block(
            cnv,
            state_summary.head(10),
            "Shipment State Summary",
            margin_left, margin_right, margin_top + 120, margin_bottom,
            max_rows=6,
        )
        # Milestone reporting
        draw_table_block(
            cnv,
            milestone_reporting,
            "Milestone Reporting",
            margin_left, margin_right, margin_top + 260, margin_bottom,
            max_rows=6,
        )
        # On-time
        draw_table_block(
            cnv,
            ontime_summary,
            "On-time Performance (Planned End vs Actual)",
            margin_left, margin_right, margin_top + 400, margin_bottom,
            max_rows=6,
        )

        draw_footer(cnv, cfg, pn, pt, margin_left, margin_right, margin_bottom)

    pages.append(exec_page)

    # Carrier section intro
    pages.append(lambda cnv, pn, pt: draw_section_intro(
        cnv, cfg,
        "Carrier Performance",
        "Carrier-wise shipment volume, milestone reporting, and on-time performance.\nPaginated tables (≤ 14 rows per page)."
    ))

    # Carrier table pages
    rows_per_page = 14
    carrier_pages = max(1, math.ceil(len(carrier_table) / rows_per_page))
    for p in range(carrier_pages):
        start = p * rows_per_page
        chunk = carrier_table.iloc[start:start + rows_per_page].copy()

        def _page_factory(df_chunk=chunk):
            def _p(cnv, pn, pt):
                cnv.setFillColor(hex_to_rl(C["bg"]))
                cnv.rect(0, 0, w, h, fill=1, stroke=0)
                draw_header(cnv, cfg, "Carrier Performance", margin_left, margin_right, margin_top)
                draw_table_block(cnv, df_chunk, "Carrier Summary", margin_left, margin_right, margin_top, margin_bottom, max_rows=rows_per_page)
                draw_footer(cnv, cfg, pn, pt, margin_left, margin_right, margin_bottom)
            return _p

        pages.append(_page_factory())

    # Lane section intro
    pages.append(lambda cnv, pn, pt: draw_section_intro(
        cnv, cfg,
        "Lane Insights",
        "Lane-wise (Origin → Destination) reporting and on-time.\nPaginated tables."
    ))

    lane_pages = max(1, math.ceil(len(lane_table) / rows_per_page))
    for p in range(lane_pages):
        start = p * rows_per_page
        chunk = lane_table.iloc[start:start + rows_per_page].copy()

        def _page_factory(df_chunk=chunk):
            def _p(cnv, pn, pt):
                cnv.setFillColor(hex_to_rl(C["bg"]))
                cnv.rect(0, 0, w, h, fill=1, stroke=0)
                draw_header(cnv, cfg, "Lane Insights", margin_left, margin_right, margin_top)
                draw_table_block(cnv, df_chunk, "Lane Summary", margin_left, margin_right, margin_top, margin_bottom, max_rows=rows_per_page)
                draw_footer(cnv, cfg, pn, pt, margin_left, margin_right, margin_bottom)
            return _p

        pages.append(_page_factory())

    # RCA intro
    pages.append(lambda cnv, pn, pt: draw_section_intro(
        cnv, cfg,
        "Root Cause Analysis",
        "Carrier-wise affected shipments for Creation, Tracking and Milestone RCA.\nTables only."
    ))

    # One page per RCA type (top 14 rows)
    for title, df_rca in rca_tables.items():
        def _rca_page_factory(t=title, d=df_rca.head(rows_per_page)):
            def _p(cnv, pn, pt):
                cnv.setFillColor(hex_to_rl(C["bg"]))
                cnv.rect(0, 0, w, h, fill=1, stroke=0)
                draw_header(cnv, cfg, "Root Cause Analysis", margin_left, margin_right, margin_top)
                draw_table_block(cnv, d, t, margin_left, margin_right, margin_top, margin_bottom, max_rows=rows_per_page)
                draw_footer(cnv, cfg, pn, pt, margin_left, margin_right, margin_bottom)
            return _p
        pages.append(_rca_page_factory())

    # Render PDF
    buf = io.BytesIO()
    cnv = canvas.Canvas(buf, pagesize=landscape(A4))
    total = len(pages)

    for idx, fn in enumerate(pages, start=1):
        fn(cnv, idx, total)
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Monthly Business Review Report Builder (Tables + PDF, No Charts)")

with st.sidebar:
    st.header("Upload files")
    shipments_file = st.file_uploader("Shipment details (CSV/Excel)", type=["csv", "xlsx", "xls"])
    rca_creation_file = st.file_uploader("RCA – Creation (Excel/CSV)", type=["xlsx", "xls", "csv"])
    rca_tracking_file = st.file_uploader("RCA – Tracking (Excel/CSV)", type=["xlsx", "xls", "csv"])
    rca_milestone_file = st.file_uploader("RCA – Milestone (Excel/CSV)", type=["xlsx", "xls", "csv"])
    logo_file = st.file_uploader("Project44 logo (PNG)", type=["png"])

    st.divider()
    st.header("Report inputs")
    customer_name = st.text_input("Customer name", value="Customer Name")
    contracted_volume = st.number_input("Contracted volume", min_value=0.0, value=0.0, step=1.0)
    billed_volume = st.number_input("Billed volume", min_value=0.0, value=0.0, step=1.0)

    latency_minutes = st.selectbox("Latency allowed (minutes)", options=[5, 10, 15, 30, 60], index=3)

    st.divider()
    st.header("Filters")
    date_basis = st.radio("Date filter based on", ["Destination latest planned arrival end time", "Created time"], index=0)
    selected_milestones = st.multiselect("Milestones / events", list(MILESTONES.keys()),
                                         default=["Arrival at Destination", "Departure at Origin", "Arrival at Origin"])

    run = st.button("Run report", type="primary")


if not shipments_file:
    st.info("Upload the shipment details file to start.")
    st.stop()

# Load data (cached)
try:
    shipments_df = load_shipments_cached(shipments_file.getvalue(), shipments_file.name)
except Exception as e:
    st.error(str(e))
    st.stop()

rca_creation_df = load_rca_cached(rca_creation_file.getvalue(), rca_creation_file.name) if rca_creation_file else pd.DataFrame()
rca_tracking_df = load_rca_cached(rca_tracking_file.getvalue(), rca_tracking_file.name) if rca_tracking_file else pd.DataFrame()
rca_milestone_df = load_rca_cached(rca_milestone_file.getvalue(), rca_milestone_file.name) if rca_milestone_file else pd.DataFrame()

# Choose date column
date_col = "Destination latest planned arrival end time" if date_basis.startswith("Destination") else "Created time"
if shipments_df[date_col].isna().all():
    st.warning(f"Date column '{date_col}' has no parseable dates.")
    st.stop()

min_d = shipments_df[date_col].min()
max_d = shipments_df[date_col].max()

c1, c2 = st.columns(2)
with c1:
    date_from = st.date_input("From", value=min_d.date())
with c2:
    date_to = st.date_input("To", value=max_d.date())

# Carrier filters (computed on full set)
carriers_all = sorted([c for c in shipments_df["Current carrier"].dropna().unique() if str(c).strip() and str(c).lower() != "nan"])
c3, c4 = st.columns(2)
with c3:
    carrier_mode = st.radio("Carriers filter", ["Include selected", "Exclude selected"], horizontal=True, index=0)
with c4:
    carrier_pick = st.multiselect("Carriers", options=carriers_all, default=[])

if not run:
    st.caption("Set filters and click **Run report**.")
    st.stop()

# Apply filters
df = shipments_df.copy()
mask_range = (df[date_col] >= pd.Timestamp(date_from)) & (df[date_col] < (pd.Timestamp(date_to) + pd.Timedelta(days=1)))
df = df[mask_range].copy()

if carrier_pick:
    if carrier_mode == "Include selected":
        df = df[df["Current carrier"].isin(carrier_pick)].copy()
    else:
        df = df[~df["Current carrier"].isin(carrier_pick)].copy()

if df.empty:
    st.warning("No shipments match the selected filters.")
    st.stop()

# Derived fields
df["Lane"] = build_lane_cols(df)
df["Completion bucket"] = completion_bucket(df, selected_milestones)
df["Exception list"] = df["Exceptions"].apply(parse_exceptions)
df["Has exceptions"] = df["Exception list"].apply(lambda x: len(x) > 0)

for m in selected_milestones:
    df[f"Reported: {m}"] = milestone_reported(df, m)
    df[f"On-time: {m}"] = on_time(df, m, latency_minutes)

df["All selected milestones reported"] = True
for m in selected_milestones:
    df["All selected milestones reported"] &= df[f"Reported: {m}"]

df["All selected milestones on-time"] = True
for m in selected_milestones:
    df["All selected milestones on-time"] &= df[f"On-time: {m}"]

df["Destination ETA accurate"] = destination_eta_accurate(df, latency_minutes)

# KPIs
num_shipments = df["Shipment ID"].nunique()
num_carriers = df["Current carrier"].nunique()
reporting_all_pct = float(df["All selected milestones reported"].mean() * 100)
ontime_all_pct = float(df["All selected milestones on-time"].mean() * 100)
eta_acc_pct = float(df["Destination ETA accurate"].mean() * 100)

st.subheader("Overview (tables)")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("# Shipments", f"{num_shipments:,}")
k2.metric("# Carriers", f"{num_carriers:,}")
k3.metric("Reporting % (all)", f"{reporting_all_pct:.1f}%")
k4.metric("On-time % (all)", f"{ontime_all_pct:.1f}%")
k5.metric("Destination ETA accuracy %", f"{eta_acc_pct:.1f}%")

# State summary
state_summary = df["Completion bucket"].value_counts(dropna=False).reset_index()
state_summary.columns = ["State / Bucket", "Shipments"]

# Milestone reporting summary
mr_rows = []
ot_rows = []
for m in selected_milestones:
    rep = int(df[f"Reported: {m}"].sum())
    rep_pct = rep / len(df) * 100
    ot = int(df[f"On-time: {m}"].sum())
    ot_pct = ot / len(df) * 100
    mr_rows.append([m, rep, f"{rep_pct:.1f}%"])
    ot_rows.append([m, ot, f"{ot_pct:.1f}%"])

milestone_reporting_df = pd.DataFrame(mr_rows, columns=["Milestone", "Shipments reported", "Reporting %"])
ontime_df = pd.DataFrame(ot_rows, columns=["Milestone", "On-time shipments", "On-time %"])

# Exceptions breakdown
ex_none = int((~df["Has exceptions"]).sum())
ex_yes = int(df["Has exceptions"].sum())

st.markdown("### Shipment state summary")
st.dataframe(state_summary, use_container_width=True)

st.markdown("### Milestone reporting")
st.dataframe(milestone_reporting_df, use_container_width=True)

st.markdown("### On-time performance (event-wise)")
st.dataframe(ontime_df, use_container_width=True)

st.markdown("### Exceptions summary")
st.write(f"- Shipments without exceptions: **{ex_none:,}**")
st.write(f"- Shipments with exceptions: **{ex_yes:,}**")

# Carrier table
g = df.groupby("Current carrier", dropna=False)
carrier_rows = []
for carrier, dfx in g:
    carrier = str(carrier).strip() if str(carrier).strip() else "Unknown"
    carrier_rows.append([
        carrier,
        dfx["Shipment ID"].nunique(),
        f"{dfx['All selected milestones reported'].mean()*100:.1f}%",
        f"{dfx['All selected milestones on-time'].mean()*100:.1f}%",
        f"{dfx['Has exceptions'].mean()*100:.1f}%",
    ])
carrier_table = pd.DataFrame(
    carrier_rows, columns=["Carrier", "# Shipments", "Reporting %", "On-time %", "Exceptions %"]
).sort_values("# Shipments", ascending=False)

st.markdown("### Carrier view")
st.dataframe(carrier_table, use_container_width=True)

# Lane table
lg = df.groupby("Lane", dropna=False)
lane_rows = []
for lane, dfx in lg:
    lane_rows.append([
        str(lane),
        dfx["Shipment ID"].nunique(),
        f"{dfx['All selected milestones reported'].mean()*100:.1f}%",
        f"{dfx['All selected milestones on-time'].mean()*100:.1f}%",
    ])
lane_table = pd.DataFrame(
    lane_rows, columns=["Lane", "# Shipments", "Reporting %", "On-time %"]
).sort_values("# Shipments", ascending=False)

st.markdown("### Lane insights")
st.dataframe(lane_table, use_container_width=True)

# RCA summaries
def rca_carrier_summary(df_rca: pd.DataFrame, rca_type: str) -> pd.DataFrame:
    if df_rca.empty:
        return pd.DataFrame(columns=["Carrier", "Affected shipments"])

    df2 = df_rca.copy()
    df2["Carrier"] = df2.get("Carrier", "").astype(str).replace("nan", "").fillna("Unknown")
    df2["match_shipment_id"] = df2["match_shipment_id"].astype(str).replace("nan", "").fillna("")

    # For tracking/milestone: must match shipment details
    if rca_type in ["tracking", "milestone"]:
        df2 = df2[df2["match_shipment_id"].isin(df["Shipment ID"].astype(str))].copy()

    out = df2.groupby("Carrier", dropna=False)["match_shipment_id"].nunique().reset_index()
    out.columns = ["Carrier", "Affected shipments"]
    return out.sort_values("Affected shipments", ascending=False)

st.markdown("### RCA analysis (carrier-wise)")
cA, cB, cC = st.columns(3)
with cA:
    st.write("**Creation RCA**")
    st.dataframe(rca_carrier_summary(rca_creation_df, "creation"), use_container_width=True, height=260)
with cB:
    st.write("**Tracking RCA**")
    st.dataframe(rca_carrier_summary(rca_tracking_df, "tracking"), use_container_width=True, height=260)
with cC:
    st.write("**Milestone RCA**")
    st.dataframe(rca_carrier_summary(rca_milestone_df, "milestone"), use_container_width=True, height=260)

# PDF export
st.divider()
st.subheader("Export PDF (tables only)")

logo_bytes = logo_file.getvalue() if logo_file else None
date_range_label = f"{date_from.isoformat()} to {date_to.isoformat()}"

kpis = {
    "shipments": f"{num_shipments:,}",
    "carriers": f"{num_carriers:,}",
    "reporting_pct": f"{reporting_all_pct:.1f}%",
    "ontime_pct": f"{ontime_all_pct:.1f}%",
}

rca_tables = {
    "Creation issues – affected shipments by carrier": rca_carrier_summary(rca_creation_df, "creation"),
    "Tracking issues – affected shipments by carrier": rca_carrier_summary(rca_tracking_df, "tracking"),
    "Milestone issues – affected shipments by carrier": rca_carrier_summary(rca_milestone_df, "milestone"),
}

pdf_cfg = PdfConfig(
    customer_name=customer_name,
    date_range_label=date_range_label,
    latency_minutes=int(latency_minutes),
    selected_milestones=selected_milestones,
    logo_bytes=logo_bytes,
)

if st.button("Generate PDF"):
    pdf_bytes = build_pdf(
        cfg=pdf_cfg,
        kpis=kpis,
        contract={"contracted": float(contracted_volume), "billed": float(billed_volume)},
        state_summary=state_summary,
        milestone_reporting=milestone_reporting_df,
        ontime_summary=ontime_df,
        carrier_table=carrier_table,
        lane_table=lane_table,
        rca_tables=rca_tables,
    )
    filename = f"Monthly Business Review Report - {customer_name} - {date_range_label}.pdf"
    st.download_button("Download PDF", data=pdf_bytes, file_name=filename, mime="application/pdf")
