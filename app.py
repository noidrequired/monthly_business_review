import io
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# -----------------------------
# Theme / Design Tokens
# -----------------------------
COLORS = {
    "text_primary": colors.HexColor("#1F2933"),
    "text_secondary": colors.HexColor("#4B5563"),
    "text_muted": colors.HexColor("#9CA3AF"),
    "bg": colors.HexColor("#FFFFFF"),
    "divider": colors.HexColor("#E5E7EB"),
    "accent": colors.HexColor("#1D4ED8"),
    "pos": colors.HexColor("#16A34A"),
    "amber": colors.HexColor("#D97706"),
    "critical": colors.HexColor("#DC2626"),
    "neutral": colors.HexColor("#64748B"),
    "table_header_bg": colors.HexColor("#F3F4F6"),
    "row_alt": colors.HexColor("#F9FAFB"),
    "kpi_pos_bg": colors.HexColor("#DCFCE7"),
    "kpi_risk_bg": colors.HexColor("#FEF3C7"),
    "kpi_crit_bg": colors.HexColor("#FEE2E2"),
}

PAGE_W, PAGE_H = landscape(A4)

MARGINS_CONTENT = dict(left=24 * mm, right=20 * mm, top=18 * mm, bottom=18 * mm)
MARGINS_COVER = dict(left=32 * mm, right=32 * mm, top=28 * mm, bottom=28 * mm)


# -----------------------------
# Expected Columns
# -----------------------------
EXPECTED_COLS = [
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


MILESTONES = {
    "Arrival at Origin": {
        "actual": "Origin actual arrival time",
        "planned_end": "Origin latest planned arrival end time",
        "estimated": "Origin estimated arrival time",
    },
    "Departure from Origin": {
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


# -----------------------------
# Utilities
# -----------------------------
def safe_to_datetime(series: pd.Series) -> pd.Series:
    """
    Parse datetimes robustly, leaving blanks/invalid as NaT.
    Works for sample formats like '07/01/26 11:46' etc.
    """
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Trim column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Check missing columns (we won't hard-fail; we’ll warn and continue)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        st.warning(
            "Some expected columns are missing. The app will run with what it has.\n\n"
            f"Missing: {', '.join(missing)}"
        )

    # Parse key datetime columns if present
    dt_cols = [c for c in EXPECTED_COLS if "time" in c.lower()]
    for c in dt_cols:
        if c in df.columns:
            df[c] = safe_to_datetime(df[c])

    # Basic cleanup for text columns
    for c in [
        "Shipment ID",
        "Shipment type",
        "Current state",
        "Current state reason",
        "Exceptions",
        "Current carrier",
        "Origin",
        "Origin city",
        "Origin state",
        "Origin country",
        "Destination",
        "Destination city",
        "Destination state",
        "Destination country",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("").str.strip()

    # Lane field
    if "Origin" in df.columns and "Destination" in df.columns:
        df["Lane"] = (df["Origin"].fillna("") + " → " + df["Destination"].fillna("")).str.strip()
    else:
        df["Lane"] = ""

    # Shipment unique id as string
    if "Shipment ID" in df.columns:
        df["Shipment ID"] = df["Shipment ID"].astype("string")

    return df


def read_uploaded_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        # Excel
        df = pd.read_excel(uploaded, engine="openpyxl")
    return df


def apply_include_exclude_filter(
    df: pd.DataFrame,
    col: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    mode: str,
) -> pd.DataFrame:
    """
    mode: "Include" keeps rows inside [start, end]
          "Exclude" removes rows inside [start, end]
    """
    if col not in df.columns:
        return df
    if start is None or end is None:
        return df

    s = df[col]
    mask = (s >= start) & (s <= end)
    return df[mask] if mode == "Include" else df[~mask]


def apply_include_exclude_list(
    df: pd.DataFrame,
    col: str,
    selected: List[str],
    mode: str,
) -> pd.DataFrame:
    if col not in df.columns:
        return df
    if not selected:
        return df
    mask = df[col].isin(selected)
    return df[mask] if mode == "Include" else df[~mask]


def reported_mask(df: pd.DataFrame, milestone_name: str) -> pd.Series:
    actual_col = MILESTONES[milestone_name]["actual"]
    if actual_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[actual_col].notna()


def all_selected_reported_mask(df: pd.DataFrame, selected_milestones: List[str]) -> pd.Series:
    if not selected_milestones:
        return pd.Series([False] * len(df), index=df.index)
    masks = [reported_mask(df, m) for m in selected_milestones]
    out = masks[0].copy()
    for m in masks[1:]:
        out &= m
    return out


def compute_completion_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule:
    - If Current state == COMPLETED:
        - If Current state reason == TRACKING_TIMED_OUT:
            - If ALL 3 core milestone actual timestamps exist -> Completed
            - Else -> Timed Out
        - Else -> Completed
    - Else -> Not Completed
    """
    out = df.copy()
    state = out["Current state"] if "Current state" in out.columns else ""
    reason = out["Current state reason"] if "Current state reason" in out.columns else ""

    core = ["Arrival at Origin", "Departure from Origin", "Arrival at Destination"]
    core_reported = all_selected_reported_mask(out, core)

    out["Completion Bucket"] = "Not Completed"
    completed_mask = state.eq("COMPLETED")

    timed_out_mask = completed_mask & reason.eq("TRACKING_TIMED_OUT")
    out.loc[completed_mask & ~timed_out_mask, "Completion Bucket"] = "Completed"
    out.loc[timed_out_mask & core_reported, "Completion Bucket"] = "Completed"
    out.loc[timed_out_mask & ~core_reported, "Completion Bucket"] = "Timed Out"

    return out


def on_time_mask(df: pd.DataFrame, milestone_name: str, latency_minutes: int) -> pd.Series:
    """
    On-time uses planned END time (always) vs actual time.
    On-time if actual <= planned_end + latency.
    """
    actual_col = MILESTONES[milestone_name]["actual"]
    planned_col = MILESTONES[milestone_name]["planned_end"]

    if actual_col not in df.columns or planned_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    actual = df[actual_col]
    planned_end = df[planned_col]
    # must have both actual and planned_end to assess
    valid = actual.notna() & planned_end.notna()
    delta_ok = actual <= (planned_end + pd.to_timedelta(latency_minutes, unit="m"))
    return valid & delta_ok


def estimated_accuracy_mask(df: pd.DataFrame, milestone_name: str, latency_minutes: int) -> pd.Series:
    """
    "Estimated was accurate with actual" = abs(actual - estimated) <= latency
    """
    actual_col = MILESTONES[milestone_name]["actual"]
    est_col = MILESTONES[milestone_name]["estimated"]
    if actual_col not in df.columns or est_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    actual = df[actual_col]
    est = df[est_col]
    valid = actual.notna() & est.notna()
    diff = (actual - est).abs()
    return valid & (diff <= pd.to_timedelta(latency_minutes, unit="m"))


def kpi_bg_color(consumption_ratio: float) -> colors.Color:
    if consumption_ratio <= 1.0:
        return COLORS["kpi_pos_bg"]
    if consumption_ratio <= 1.05:
        return COLORS["kpi_risk_bg"]
    return COLORS["kpi_crit_bg"]


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.1f}%"


def fmt_int(x) -> str:
    if pd.isna(x):
        return "—"
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


# -----------------------------
# Simple chart helpers (matplotlib)
# -----------------------------
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def bar_chart(series: pd.Series, title: str, xlabel: str = "", ylabel: str = "") -> bytes:
    fig = plt.figure(figsize=(7.6, 3.3))
    ax = fig.add_subplot(111)
    s = series.copy()
    s = s.sort_values(ascending=False)
    ax.bar(s.index.astype(str), s.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig_to_png_bytes(fig)


# -----------------------------
# PDF generation
# -----------------------------
@dataclass
class ReportMeta:
    customer_name: str
    date_range_label: str
    report_title: str


def build_styles():
    # We’ll use Helvetica by default (PDF-safe). If you later upload Inter TTF files,
    # you can register them here.
    base = getSampleStyleSheet()

    base.add(
        ParagraphStyle(
            name="H1",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=COLORS["text_primary"],
            spaceAfter=10,
        )
    )
    base.add(
        ParagraphStyle(
            name="H2",
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=COLORS["text_primary"],
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            name="Body",
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=COLORS["text_primary"],
        )
    )
    base.add(
        ParagraphStyle(
            name="Muted",
            fontName="Helvetica",
            fontSize=9.5,
            leading=12,
            textColor=COLORS["text_secondary"],
        )
    )
    base.add(
        ParagraphStyle(
            name="CoverTitle",
            fontName="Helvetica-Bold",
            fontSize=36,
            leading=42,
            textColor=COLORS["text_primary"],
            spaceAfter=14,
        )
    )
    base.add(
        ParagraphStyle(
            name="CoverSub",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=26,
            textColor=COLORS["text_secondary"],
            spaceAfter=10,
        )
    )
    base.add(
        ParagraphStyle(
            name="CoverSmall",
            fontName="Helvetica",
            fontSize=14,
            leading=18,
            textColor=COLORS["text_secondary"],
        )
    )
    base.add(
        ParagraphStyle(
            name="Foot",
            fontName="Helvetica",
            fontSize=8.5,
            leading=10,
            textColor=COLORS["text_muted"],
        )
    )
    return base


def footer_and_header(canvas, doc, meta: ReportMeta, logo_bytes: Optional[bytes]):
    canvas.saveState()

    # Header (logo)
    if logo_bytes:
        try:
            # place at top-left within content margins
            x = doc.leftMargin
            y = PAGE_H - doc.topMargin + 3 * mm
            img = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")
            # keep it small and neat
            target_h = 10 * mm
            ratio = img.width / img.height
            target_w = target_h * ratio
            canvas.drawImage(
                ImageReader(io.BytesIO(logo_bytes)),
                x,
                y - target_h,
                width=target_w,
                height=target_h,
                mask="auto",
            )
        except Exception:
            pass

    # Footer
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(COLORS["text_muted"])

    left = meta.customer_name
    center = "Confidential – Customer Use Only"
    right = f"Page {doc.page} of {doc.pageCount}" if hasattr(doc, "pageCount") else f"Page {doc.page}"

    y = doc.bottomMargin - 8 * mm
    canvas.drawString(doc.leftMargin, y, left)
    canvas.drawCentredString(PAGE_W / 2, y, center)
    canvas.drawRightString(PAGE_W - doc.rightMargin, y, right)

    canvas.restoreState()


# ReportLab needs this helper for drawImage with bytes.
from reportlab.lib.utils import ImageReader


def make_kpi_table(kpis: List[Tuple[str, str]]) -> Table:
    data = []
    for label, value in kpis:
        data.append([label, value])

    t = Table(data, colWidths=[80 * mm, 60 * mm])
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("TEXTCOLOR", (0, 0), (-1, -1), COLORS["text_primary"]),
                ("BACKGROUND", (0, 0), (-1, -1), COLORS["bg"]),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, COLORS["divider"]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def styled_table(df: pd.DataFrame, max_rows: int = 14) -> Table:
    show = df.head(max_rows).copy()
    data = [list(show.columns)] + show.astype(str).values.tolist()

    t = Table(data, repeatRows=1)
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10.5),
        ("BACKGROUND", (0, 0), (-1, 0), COLORS["table_header_bg"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["text_primary"]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, COLORS["divider"]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, COLORS["divider"]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]

    # alternating row backgrounds
    for r in range(1, len(data)):
        bg = COLORS["row_alt"] if r % 2 == 0 else COLORS["bg"]
        style_cmds.append(("BACKGROUND", (0, r), (-1, r), bg))

    t.setStyle(TableStyle(style_cmds))
    return t


def generate_pdf(
    meta: ReportMeta,
    kpis: Dict,
    carrier_table: pd.DataFrame,
    otp_table: pd.DataFrame,
    lane_milestone_table: pd.DataFrame,
    lane_otp_table: pd.DataFrame,
    charts: Dict[str, bytes],
    logo_bytes: Optional[bytes],
    screenshot_bytes: Optional[bytes],
) -> bytes:
    buf = io.BytesIO()

    styles = build_styles()

    # We'll generate in 2 passes to get total page count:
    # Use a custom doc template where we set pageCount after build.
    class CountingDoc(SimpleDocTemplate):
        def afterFlowable(self, flowable):
            pass

    # First build story to count pages is heavier; a simpler way:
    # Build once, then re-build with the known pagecount isn't trivial.
    # We'll do a pragmatic approach: footer shows "Page X" (no total).
    # If you *must* have total pages, we can implement a two-pass canvas later.

    doc = CountingDoc(
        buf,
        pagesize=landscape(A4),
        leftMargin=MARGINS_CONTENT["left"],
        rightMargin=MARGINS_CONTENT["right"],
        topMargin=MARGINS_CONTENT["top"],
        bottomMargin=MARGINS_CONTENT["bottom"],
        title=meta.report_title,
        author="Streamlit App",
    )

    story = []

    # --- Cover page (different margins visually via spacing) ---
    story.append(Spacer(1, 18 * mm))
    story.append(Paragraph("Monthly Business Review Report", styles["CoverTitle"]))
    story.append(Paragraph(meta.customer_name, styles["CoverSub"]))
    story.append(Paragraph(meta.date_range_label, styles["CoverSmall"]))
    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph("Prepared for external sharing", styles["Muted"]))
    story.append(PageBreak())

    # --- Section intro ---
    story.append(Paragraph("Executive Summary", styles["H1"]))
    story.append(Paragraph("High-level KPIs and reporting performance overview.", styles["Body"]))
    story.append(Spacer(1, 10 * mm))

    # KPI block
    kpi_pairs = [
        ("Number of Shipments", fmt_int(kpis["num_shipments"])),
        ("Number of Carriers", fmt_int(kpis["num_carriers"])),
        ("Shipments reporting ALL selected milestones", fmt_int(kpis["num_all_selected"])),
        ("% reporting ALL selected milestones", fmt_pct(kpis["pct_all_selected"])),
        ("Marked Completed", fmt_int(kpis["num_marked_completed"])),
        ("Timed Out (per rule)", fmt_int(kpis["num_timed_out"])),
    ]
    story.append(make_kpi_table(kpi_pairs))
    story.append(Spacer(1, 6 * mm))

    # Contract consumption
    story.append(Paragraph("Contract Consumption", styles["H2"]))
    story.append(Paragraph(kpis["contract_text"], styles["Body"]))
    story.append(Spacer(1, 6 * mm))
    story.append(PageBreak())

    # --- Milestone Reporting page ---
    story.append(Paragraph("Milestone Reporting", styles["H1"]))
    story.append(Paragraph("Reporting counts and percentages for selected milestones.", styles["Body"]))
    story.append(Spacer(1, 4 * mm))

    if charts.get("milestone_reporting"):
        img = RLImage(io.BytesIO(charts["milestone_reporting"]), width=240 * mm, height=90 * mm)
        story.append(img)
        story.append(Spacer(1, 6 * mm))

    story.append(styled_table(kpis["milestone_table"], max_rows=14))
    story.append(PageBreak())

    # --- Carrier-wise page ---
    story.append(Paragraph("Carrier-wise Summary", styles["H1"]))
    story.append(Paragraph("Shipment volumes and reporting performance by carrier.", styles["Body"]))
    story.append(Spacer(1, 6 * mm))
    story.append(styled_table(carrier_table, max_rows=14))
    story.append(PageBreak())

    # --- OTP page ---
    story.append(Paragraph("On-time Performance (OTP)", styles["H1"]))
    story.append(Paragraph("On-time is calculated using latest planned END time + allowed latency.", styles["Body"]))
    story.append(Spacer(1, 4 * mm))

    if charts.get("otp_by_event"):
        img = RLImage(io.BytesIO(charts["otp_by_event"]), width=240 * mm, height=90 * mm)
        story.append(img)
        story.append(Spacer(1, 6 * mm))

    story.append(styled_table(otp_table, max_rows=14))
    story.append(PageBreak())

    # --- Lane insights page ---
    story.append(Paragraph("Lane Insights", styles["H1"]))
    story.append(Paragraph("Milestone reporting % and OTP lane-wise (Origin → Destination).", styles["Body"]))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Top lanes by shipment volume – Milestone reporting", styles["H2"]))
    story.append(styled_table(lane_milestone_table, max_rows=12))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("Top lanes by shipment volume – On-time performance", styles["H2"]))
    story.append(styled_table(lane_otp_table, max_rows=12))

    # --- Optional screenshot page ---
    if screenshot_bytes:
        story.append(PageBreak())
        story.append(Paragraph("Embedded Screenshot", styles["H1"]))
        story.append(Paragraph("As provided by the user for this report.", styles["Body"]))
        story.append(Spacer(1, 6 * mm))
        img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        # Fit to page content area
        content_w = PAGE_W - (MARGINS_CONTENT["left"] + MARGINS_CONTENT["right"])
        content_h = PAGE_H - (MARGINS_CONTENT["top"] + MARGINS_CONTENT["bottom"] + 20 * mm)
        # Keep aspect
        ratio = img.width / img.height
        target_w = content_w
        target_h = target_w / ratio
        if target_h > content_h:
            target_h = content_h
            target_w = target_h * ratio
        story.append(RLImage(io.BytesIO(screenshot_bytes), width=target_w, height=target_h))

    def on_page(canvas, doc_):
        # Header+footer on each page
        footer_and_header(canvas, doc_, meta, logo_bytes)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    buf.seek(0)
    return buf.read()


# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Monthly Business Review Report", layout="wide")

st.title("Monthly Business Review Report Builder")
st.caption("Upload your shipment file, filter, review the dashboard, and export a beautifully formatted PDF.")


with st.sidebar:
    st.header("1) Upload")
    uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
    st.divider()

    st.header("2) Report Details")
    customer_name = st.text_input("Customer name", value="Customer")
    contracted_volume = st.number_input("Contracted volume", min_value=0.0, value=1000.0, step=10.0)
    billed_volume = st.number_input("Billed volume", min_value=0.0, value=900.0, step=10.0)

    st.divider()
    st.header("3) Assets")
    st.caption("Logo: defaults to project44 if not provided.")
    logo_up = st.file_uploader("Upload logo (PNG)", type=["png"], key="logo")
    screenshot_up = st.file_uploader("Upload screenshot to embed (PNG/JPG)", type=["png", "jpg", "jpeg"], key="shot")

    st.divider()
    st.header("4) Filters")

    date_basis = st.selectbox(
        "Date filter basis",
        options=["Destination latest planned arrival end time", "Created time"],
        index=0,
    )
    date_mode = st.radio("Date filter mode", options=["Include", "Exclude"], horizontal=True)

    carrier_mode = st.radio("Carrier filter mode", options=["Include", "Exclude"], horizontal=True)

    st.divider()
    st.header("5) Milestones & OTP")
    selected_milestones = st.multiselect(
        "Milestones/events to evaluate reporting",
        options=list(MILESTONES.keys()),
        default=["Arrival at Origin", "Departure from Origin", "Arrival at Destination"],
    )
    latency_minutes = st.selectbox("Allowed latency", options=[0, 5, 10, 15, 30, 60], index=1)
    st.caption("On-time uses latest planned END time + allowed latency.")


if not uploaded:
    st.info("Upload your Excel/CSV to begin.")
    st.stop()

raw_df = read_uploaded_file(uploaded)
df = normalize_df(raw_df)
df = compute_completion_status(df)

# Determine date column
date_col = "Destination latest planned arrival end time" if date_basis.startswith("Destination") else "Created time"
if date_col in df.columns and df[date_col].notna().any():
    min_dt = df[date_col].min()
    max_dt = df[date_col].max()
else:
    min_dt = None
    max_dt = None

# Date picker
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    start_date = st.date_input(
        "Start date",
        value=min_dt.date() if min_dt is not None else datetime.today().date(),
    )
with colB:
    end_date = st.date_input(
        "End date",
        value=max_dt.date() if max_dt is not None else datetime.today().date(),
    )

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)

# Carrier selection based on data
carriers = sorted([c for c in df["Current carrier"].dropna().unique().tolist() if str(c).strip() != ""]) if "Current carrier" in df.columns else []
selected_carriers = st.multiselect("Carriers", options=carriers, default=carriers[: min(len(carriers), 10)])

# Apply filters
fdf = df.copy()
fdf = apply_include_exclude_filter(fdf, date_col, start_ts, end_ts, date_mode)
fdf = apply_include_exclude_list(fdf, "Current carrier", selected_carriers, carrier_mode)

# Milestone reporting
all_selected_mask = all_selected_reported_mask(fdf, selected_milestones)
num_shipments = fdf["Shipment ID"].nunique() if "Shipment ID" in fdf.columns else len(fdf)
num_carriers = fdf["Current carrier"].nunique() if "Current carrier" in fdf.columns else 0
num_all_selected = int(all_selected_mask.sum())
pct_all_selected = (num_all_selected / num_shipments * 100) if num_shipments else float("nan")

# State summary with rule
state_counts = fdf["Current state"].value_counts(dropna=False) if "Current state" in fdf.columns else pd.Series(dtype=int)
completion_counts = fdf["Completion Bucket"].value_counts(dropna=False) if "Completion Bucket" in fdf.columns else pd.Series(dtype=int)

num_marked_completed = int((fdf["Current state"] == "COMPLETED").sum()) if "Current state" in fdf.columns else 0
num_timed_out = int((fdf["Completion Bucket"] == "Timed Out").sum()) if "Completion Bucket" in fdf.columns else 0

# Exceptions summary
exceptions_series = fdf["Exceptions"] if "Exceptions" in fdf.columns else pd.Series([""] * len(fdf))
no_exc = int((exceptions_series.fillna("").str.strip() == "").sum())
with_exc = len(fdf) - no_exc
exc_counts = (
    exceptions_series.fillna("")
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
    .dropna()
    .value_counts()
)

# Milestone eventwise counts
rows = []
for m in selected_milestones:
    rep = reported_mask(fdf, m)
    rows.append(
        {
            "Milestone": m,
            "Reported (#)": int(rep.sum()),
            "Reported (%)": round((rep.sum() / num_shipments * 100), 1) if num_shipments else None,
        }
    )
milestone_table = pd.DataFrame(rows)

# OTP tables eventwise
otp_rows = []
for m in selected_milestones:
    rep = reported_mask(fdf, m)
    ot = on_time_mask(fdf, m, latency_minutes)
    denom = int((rep & fdf[MILESTONES[m]["planned_end"]].notna()).sum()) if MILESTONES[m]["planned_end"] in fdf.columns else 0
    otp_rows.append(
        {
            "Event": m,
            "Assessable (#)": denom,
            "On-time (#)": int(ot.sum()),
            "On-time (%)": round((ot.sum() / denom * 100), 1) if denom else None,
        }
    )
otp_table = pd.DataFrame(otp_rows)

# Destination estimated accuracy
dest_est_acc = estimated_accuracy_mask(fdf, "Arrival at Destination", latency_minutes)
dest_est_assessable = int(
    (fdf[MILESTONES["Arrival at Destination"]["actual"]].notna() & fdf[MILESTONES["Arrival at Destination"]["estimated"]].notna()).sum()
) if (
    MILESTONES["Arrival at Destination"]["actual"] in fdf.columns
    and MILESTONES["Arrival at Destination"]["estimated"] in fdf.columns
) else 0

dest_est_accurate = int(dest_est_acc.sum())
dest_est_pct = round((dest_est_accurate / dest_est_assessable * 100), 1) if dest_est_assessable else None

# Carrier-wise view
carrier_group = []
if "Current carrier" in fdf.columns and "Shipment ID" in fdf.columns:
    for carrier, g in fdf.groupby("Current carrier"):
        ship = g["Shipment ID"].nunique()
        all_sel = int(all_selected_reported_mask(g, selected_milestones).sum())
        carrier_group.append(
            {
                "Carrier": carrier if str(carrier).strip() else "(Blank)",
                "Shipments": ship,
                "All selected milestones (#)": all_sel,
                "All selected milestones (%)": round(all_sel / ship * 100, 1) if ship else None,
            }
        )
carrier_table = pd.DataFrame(carrier_group).sort_values("Shipments", ascending=False)

# Lane insights
lane_m_table = pd.DataFrame()
lane_o_table = pd.DataFrame()
if "Lane" in fdf.columns and "Shipment ID" in fdf.columns and fdf["Lane"].astype(str).str.strip().ne("").any():
    lane_group = []
    for lane, g in fdf.groupby("Lane"):
        ship = g["Shipment ID"].nunique()
        all_sel = int(all_selected_reported_mask(g, selected_milestones).sum())
        lane_group.append(
            {
                "Lane": lane,
                "Shipments": ship,
                "All selected milestones (%)": round(all_sel / ship * 100, 1) if ship else None,
            }
        )
    lane_m_table = pd.DataFrame(lane_group).sort_values("Shipments", ascending=False)

    lane_o_group = []
    for lane, g in fdf.groupby("Lane"):
        ship = g["Shipment ID"].nunique()
        # OTP for destination arrival as a headline; also average OTP across selected events
        event_otps = []
        for m in selected_milestones:
            rep = reported_mask(g, m)
            ot = on_time_mask(g, m, latency_minutes)
            denom = int((rep & g[MILESTONES[m]["planned_end"]].notna()).sum()) if MILESTONES[m]["planned_end"] in g.columns else 0
            if denom:
                event_otps.append(float(ot.sum()) / denom * 100)
        avg_otp = sum(event_otps) / len(event_otps) if event_otps else None
        lane_o_group.append(
            {
                "Lane": lane,
                "Shipments": ship,
                "Avg OTP across selected events (%)": round(avg_otp, 1) if avg_otp is not None else None,
            }
        )
    lane_o_table = pd.DataFrame(lane_o_group).sort_values("Shipments", ascending=False)

# Contract consumption text
consumption_ratio = (billed_volume / contracted_volume) if contracted_volume else float("inf")
consumption_text = (
    f"Contracted: {contracted_volume:,.0f} | Billed: {billed_volume:,.0f} | "
    f"Consumption: {consumption_ratio*100:,.1f}%"
)

# -----------------------------
# Dashboard Layout
# -----------------------------
st.subheader("Summary")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Shipments", f"{num_shipments:,}")
k2.metric("Carriers", f"{num_carriers:,}")
k3.metric("All selected milestones (#)", f"{num_all_selected:,}")
k4.metric("All selected milestones (%)", fmt_pct(pct_all_selected))

c1, c2, c3 = st.columns(3)
c1.metric("Marked Completed", f"{num_marked_completed:,}")
c2.metric("Timed Out (per rule)", f"{num_timed_out:,}")
c3.metric("Destination ETA accurate (assessable)", f"{dest_est_accurate:,} / {dest_est_assessable:,}")

# Contract consumption card-like block
bg = kpi_bg_color(consumption_ratio)
st.markdown(
    f"""
<div style="padding:14px;border:1px solid #E5E7EB;border-radius:14px;background:{bg.hexval()};
color:#1F2933;">
  <div style="font-size:16px;font-weight:700;margin-bottom:6px;">Contract Consumption</div>
  <div style="font-size:13px;">{consumption_text}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

left, right = st.columns([1.1, 1.0])

with left:
    st.subheader("Shipment state summary")
    if not state_counts.empty:
        st.dataframe(state_counts.reset_index().rename(columns={"index": "Current state", "Current state": "Count"}), use_container_width=True)
    else:
        st.write("No state data available.")

    st.subheader("Completion bucket (with timed-out override rule)")
    if not completion_counts.empty:
        st.dataframe(completion_counts.reset_index().rename(columns={"index": "Completion Bucket", "Completion Bucket": "Count"}), use_container_width=True)
    else:
        st.write("No completion data available.")

with right:
    st.subheader("Exceptions summary")
    st.write(f"Shipments without exceptions: **{no_exc:,}**")
    st.write(f"Shipments with exceptions: **{with_exc:,}**")
    if not exc_counts.empty:
        st.dataframe(exc_counts.reset_index().rename(columns={"index": "Exception", "Exceptions": "Count"}), use_container_width=True)
    else:
        st.write("No exceptions present (or column missing).")

st.divider()

st.subheader("Milestone reporting (selected)")
st.dataframe(milestone_table, use_container_width=True)

st.subheader("Carrier-wise table")
st.dataframe(carrier_table, use_container_width=True)

st.subheader("On-time performance (event-wise)")
st.dataframe(otp_table, use_container_width=True)

if not lane_m_table.empty and not lane_o_table.empty:
    st.subheader("Lane insights (top lanes by shipment volume)")
    a, b = st.columns(2)
    with a:
        st.caption("Milestone reporting")
        st.dataframe(lane_m_table.head(20), use_container_width=True)
    with b:
        st.caption("On-time performance")
        st.dataframe(lane_o_table.head(20), use_container_width=True)
else:
    st.info("Lane insights will appear when Origin and Destination fields are populated.")

# -----------------------------
# Build charts for PDF
# -----------------------------
charts = {}

# milestone reporting chart
if not milestone_table.empty:
    s = milestone_table.set_index("Milestone")["Reported (%)"].fillna(0)
    charts["milestone_reporting"] = bar_chart(s, "Milestone reporting (%)", ylabel="Percent")

# otp chart
if not otp_table.empty:
    s2 = otp_table.set_index("Event")["On-time (%)"].fillna(0)
    charts["otp_by_event"] = bar_chart(s2, "On-time performance (%)", ylabel="Percent")

# -----------------------------
# PDF Export
# -----------------------------
# Date range label
date_range_label = f"{start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}"
report_title = f"Monthly Business Review Report - {customer_name} - {date_range_label}"

# Logo bytes
default_logo_path = "/mnt/data/p44_logo.png"  # provided in this environment; in Streamlit Cloud, you’ll upload it to repo
logo_bytes = None
if logo_up:
    logo_bytes = logo_up.read()
else:
    try:
        with open(default_logo_path, "rb") as f:
            logo_bytes = f.read()
    except Exception:
        logo_bytes = None

screenshot_bytes = screenshot_up.read() if screenshot_up else None

kpis = {
    "num_shipments": num_shipments,
    "num_carriers": num_carriers,
    "num_all_selected": num_all_selected,
    "pct_all_selected": pct_all_selected,
    "num_marked_completed": num_marked_completed,
    "num_timed_out": num_timed_out,
    "contract_text": consumption_text,
    "milestone_table": milestone_table,
}

meta = ReportMeta(customer_name=customer_name, date_range_label=date_range_label, report_title=report_title)

if st.button("Generate PDF Report"):
    pdf_bytes = generate_pdf(
        meta=meta,
        kpis=kpis,
        carrier_table=carrier_table,
        otp_table=otp_table.assign(
            **{
                "Destination ETA accurate (%)": [dest_est_pct] + [""] * (len(otp_table) - 1) if len(otp_table) else []
            }
        ),
        lane_milestone_table=lane_m_table.head(12) if not lane_m_table.empty else pd.DataFrame({"Lane": [], "Shipments": [], "All selected milestones (%)": []}),
        lane_otp_table=lane_o_table.head(12) if not lane_o_table.empty else pd.DataFrame({"Lane": [], "Shipments": [], "Avg OTP across selected events (%)": []}),
        charts=charts,
        logo_bytes=logo_bytes,
        screenshot_bytes=screenshot_bytes,
    )

    st.success("PDF generated!")
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=f"{report_title}.pdf",
        mime="application/pdf",
    )
