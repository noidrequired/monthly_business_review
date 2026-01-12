import io
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
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
    KeepTogether,
)
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth


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
    "kpi_neutral_bg": colors.HexColor("#F8FAFC"),
}

PAGE_W, PAGE_H = landscape(A4)
MARGINS_CONTENT = dict(left=24 * mm, right=20 * mm, top=18 * mm, bottom=18 * mm)


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
# Data utilities
# -----------------------------
def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        st.warning(
            "Some expected columns are missing. The app will run with what it has.\n\n"
            f"Missing: {', '.join(missing)}"
        )

    dt_cols = [c for c in EXPECTED_COLS if "time" in c.lower()]
    for c in dt_cols:
        if c in df.columns:
            df[c] = safe_to_datetime(df[c])

    text_cols = [
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
    ]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("").str.strip()

    if "Origin" in df.columns and "Destination" in df.columns:
        df["Lane"] = (df["Origin"].fillna("") + " → " + df["Destination"].fillna("")).str.strip()
    else:
        df["Lane"] = ""

    if "Shipment ID" in df.columns:
        df["Shipment ID"] = df["Shipment ID"].astype("string")

    return df


def read_uploaded_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded, engine="openpyxl")


def apply_include_exclude_filter(
    df: pd.DataFrame,
    col: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    mode: str,
) -> pd.DataFrame:
    if col not in df.columns or start is None or end is None:
        return df
    s = df[col]
    mask = (s >= start) & (s <= end)
    return df[mask] if mode == "Include" else df[~mask]


def apply_include_exclude_list(df: pd.DataFrame, col: str, selected: List[str], mode: str) -> pd.DataFrame:
    if col not in df.columns or not selected:
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
    out = df.copy()
    if "Current state" not in out.columns:
        out["Completion Bucket"] = "Not Completed"
        return out

    state = out["Current state"]
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
    actual_col = MILESTONES[milestone_name]["actual"]
    planned_col = MILESTONES[milestone_name]["planned_end"]
    if actual_col not in df.columns or planned_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    actual = df[actual_col]
    planned_end = df[planned_col]
    valid = actual.notna() & planned_end.notna()
    return valid & (actual <= (planned_end + pd.to_timedelta(latency_minutes, unit="m")))


def estimated_accuracy_mask(df: pd.DataFrame, milestone_name: str, latency_minutes: int) -> pd.Series:
    actual_col = MILESTONES[milestone_name]["actual"]
    est_col = MILESTONES[milestone_name]["estimated"]
    if actual_col not in df.columns or est_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    actual = df[actual_col]
    est = df[est_col]
    valid = actual.notna() & est.notna()
    diff = (actual - est).abs()
    return valid & (diff <= pd.to_timedelta(latency_minutes, unit="m"))


def fmt_pct(x: float) -> str:
    if pd.isna(x) or x is None:
        return "—"
    return f"{x:.1f}%"


def fmt_int(x) -> str:
    if pd.isna(x) or x is None:
        return "—"
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def kpi_bg_hex(consumption_ratio: float) -> str:
    if consumption_ratio <= 1.0:
        return "#DCFCE7"
    if consumption_ratio <= 1.05:
        return "#FEF3C7"
    return "#FEE2E2"


# -----------------------------
# PDF: Dataclasses + styles + header/footer
# -----------------------------
@dataclass
class EmbeddedImage:
    header: str
    subtitle: str
    bytes_: bytes


@dataclass
class ReportMeta:
    customer_name: str
    date_range_label: str
    report_title: str


def build_styles():
    base_toggle = getSampleStyleSheet()

    base_toggle.add(
        ParagraphStyle(
            name="H1",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=COLORS["text_primary"],
            spaceAfter=10,
        )
    )
    base_toggle.add(
        ParagraphStyle(
            name="Body",
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=COLORS["text_primary"],
        )
    )
    base_toggle.add(
        ParagraphStyle(
            name="Muted",
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=COLORS["text_secondary"],
        )
    )
    base_toggle.add(
        ParagraphStyle(
            name="CoverTitle",
            fontName="Helvetica-Bold",
            fontSize=36,
            leading=42,
            textColor=COLORS["text_primary"],
            spaceAfter=14,
        )
    )
    base_toggle.add(
        ParagraphStyle(
            name="CoverSub",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=26,
            textColor=COLORS["text_secondary"],
            spaceAfter=10,
        )
    )
    base_toggle.add(
        ParagraphStyle(
            name="CoverSmall",
            fontName="Helvetica",
            fontSize=14,
            leading=18,
            textColor=COLORS["text_secondary"],
        )
    )
    base_toggle.add(
        ParagraphStyle(
            name="TinyNote",
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            textColor=COLORS["text_secondary"],
        )
    )
    return base_toggle


def draw_header_footer(canvas, doc, meta: ReportMeta, logo_bytes: Optional[bytes]):
    canvas.saveState()

    # Header logo RIGHT, larger
    if logo_bytes:
        try:
            img = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")
            target_h = 18 * mm  # bigger and cleaner
            ratio = img.width / img.height
            target_w = target_h * ratio

            x = PAGE_W - doc.rightMargin - target_w
            y = PAGE_H - doc.topMargin + 7 * mm
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
    y = doc.bottomMargin - 8 * mm
    canvas.drawString(doc.leftMargin, y, meta.customer_name)
    canvas.drawCentredString(PAGE_W / 2, y, "Confidential – Customer Use Only")
    canvas.drawRightString(PAGE_W - doc.rightMargin, y, f"Page {doc.page}")

    canvas.restoreState()


# -----------------------------
# PDF: Auto-fit tables (NOW scales UP for small tables)
# -----------------------------
def _calc_col_widths(df: pd.DataFrame, font_name: str, font_size: float, max_total_width: float) -> List[float]:
    cols = list(df.columns)
    sample = df.head(18).astype(str)

    raw_widths = []
    for c in cols:
        header_w = stringWidth(str(c), font_name, font_size) + 12
        cell_w = 0
        if not sample.empty:
            cell_w = max(stringWidth(v, font_name, font_size) for v in sample[c].tolist())
        raw_widths.append(max(header_w, cell_w + 12))

    total = sum(raw_widths)
    if total <= max_total_width:
        return raw_widths

    scale = max_total_width / total if total else 1.0
    return [w * scale for w in raw_widths]


def auto_fit_table_dynamic(
    df: pd.DataFrame,
    avail_w: float,
    avail_h: float,
    max_rows: int = 80,
    min_font: float = 7.5,
    max_font: float = 14.0,
) -> Tuple[Table, bool]:
    """
    Returns: (table, truncated_flag)
    - Tries LARGE fonts first so small tables look 'page-worthy'
    - Shrinks as needed to avoid overflow
    - If still too tall, truncates rows (one-page rule)
    """
    truncated = False
    show = df.copy()
    if len(show) > max_rows:
        show = show.head(max_rows).copy()
        truncated = True

    data = [list(show.columns)] + show.astype(str).values.tolist()

    font_name_header = "Helvetica-Bold"
    font_name_body = "Helvetica"

    # Try from big -> small (so small tables become nicely large)
    font_candidates = [14.0, 13.0, 12.0, 11.0, 10.5, 10.0, 9.5, 9.0, 8.5, 8.0, 7.5]
    font_candidates = [fs for fs in font_candidates if min_font <= fs <= max_font]

    for fs in font_candidates:
        # padding rises with bigger font
        if fs >= 13:
            pad = 9
            vpad = 7
        elif fs >= 12:
            pad = 8
            vpad = 6
        elif fs >= 11:
            pad = 7
            vpad = 5
        elif fs >= 10:
            pad = 6
            vpad = 4
        else:
            pad = 5
            vpad = 3

        col_widths = _calc_col_widths(show, font_name_body, fs, avail_w)
        t = Table(data, colWidths=col_widths, repeatRows=1)

        style_cmds = [
            ("FONTNAME", (0, 0), (-1, 0), font_name_header),
            ("FONTSIZE", (0, 0), (-1, 0), fs),
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["table_header_bg"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["text_primary"]),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, COLORS["divider"]),
            ("FONTNAME", (0, 1), (-1, -1), font_name_body),
            ("FONTSIZE", (0, 1), (-1, -1), fs - 0.5),
            ("GRID", (0, 0), (-1, -1), 0.7, COLORS["divider"]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), pad),
            ("RIGHTPADDING", (0, 0), (-1, -1), pad),
            ("TOPPADDING", (0, 0), (-1, -1), vpad),
            ("BOTTOMPADDING", (0, 0), (-1, -1), vpad),
        ]

        for r in range(1, len(data)):
            bg = COLORS["row_alt"] if r % 2 == 0 else COLORS["bg"]
            style_cmds.append(("BACKGROUND", (0, r), (-1, r), bg))

        t.setStyle(TableStyle(style_cmds))

        w, h = t.wrap(avail_w, avail_h)
        if w <= avail_w + 1 and h <= avail_h + 1:
            return t, truncated

    # If it still doesn't fit, truncate more aggressively and use smallest font
    truncated = True
    show2 = df.head(18).copy()
    data2 = [list(show2.columns)] + show2.astype(str).values.tolist()
    fs = min_font
    col_widths = _calc_col_widths(show2, "Helvetica", fs, avail_w)
    t2 = Table(data2, colWidths=col_widths, repeatRows=1)
    t2.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), fs),
                ("BACKGROUND", (0, 0), (-1, 0), COLORS["table_header_bg"]),
                ("GRID", (0, 0), (-1, -1), 0.6, COLORS["divider"]),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), fs),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t2, truncated


def centered_flowable(flowable, avail_h: float, flowable_h: float) -> List:
    """
    Adds spacers above/below to center the flowable vertically within avail_h.
    """
    free = max(0, avail_h - flowable_h)
    top = free * 0.18  # slightly top-biased so it feels more 'designed'
    bottom = free - top
    return [Spacer(1, top), flowable, Spacer(1, bottom)]


# -----------------------------
# PDF: KPI Tiles (clean grid, label top-left, value centered)
# -----------------------------
def kpi_tiles_grid(kpis: List[Dict], content_w: float, cols: int = 4) -> Table:
    """
    kpis item format:
      {
        "label": str,
        "value": str,
        "tone": "neutral|pos|amber|critical",
        "subvalue": Optional[str]  # small line under centered value (optional)
      }
    """
    styles = getSampleStyleSheet()
    label_style = styles["BodyText"]
    value_style = styles["BodyText"]
    sub_style = styles["BodyText"]

    gutter = 6
    tile_w = (content_w - ((cols - 1) * gutter)) / cols

    def tile_bg(tone: str):
        if tone == "pos":
            return COLORS["kpi_pos_bg"]
        if tone == "amber":
            return COLORS["kpi_risk_bg"]
        if tone == "critical":
            return COLORS["kpi_crit_bg"]
        return COLORS["kpi_neutral_bg"]

    tiles: List = []
    for k in kpis:
        label = Paragraph(f"<font size='10' color='#4B5563'>{k['label']}</font>", label_style)
        value = Paragraph(f"<para alignment='center'><font size='24'><b>{k['value']}</b></font></para>", value_style)

        sub = None
        if k.get("subvalue"):
            sub = Paragraph(f"<para alignment='center'><font size='10' color='#4B5563'>{k['subvalue']}</font></para>", sub_style)

        # Build a fixed-height tile using 3 rows: label / spacer / centered value(+sub)
        inner_rows = []
        inner_rows.append([label])
        inner_rows.append([Spacer(1, 10)])  # breathing room

        if sub:
            inner_rows.append([value])
            inner_rows.append([Spacer(1, 4)])
            inner_rows.append([sub])
        else:
            inner_rows.append([value])

        inner = Table(inner_rows, colWidths=[tile_w])
        inner.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), tile_bg(k.get("tone", "neutral"))),
                    ("BOX", (0, 0), (-1, -1), 0.8, COLORS["divider"]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        tiles.append(inner)

    # Arrange into grid
    rows = []
    row = []
    for i, t in enumerate(tiles, start=1):
        row.append(t)
        if len(row) == cols:
            rows.append(row)
            row = []
    if row:
        # pad remaining
        while len(row) < cols:
            row.append(Spacer(1, 1))
        rows.append(row)

    outer = Table(rows, colWidths=[tile_w] * cols, hAlign="LEFT")
    outer.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("COLSPACING", (0, 0), (-1, -1), gutter),
                ("ROWSPACING", (0, 0), (-1, -1), gutter),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return outer


# -----------------------------
# PDF: Image fit (fill nicely but never overflow)
# -----------------------------
def fit_image_to_content(image_bytes: bytes, content_w: float, content_h: float) -> RLImage:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ratio = img.width / img.height

    target_w = content_w
    target_h = target_w / ratio
    if target_h > content_h:
        target_h = content_h
        target_w = target_h * ratio

    return RLImage(io.BytesIO(image_bytes), width=target_w, height=target_h)


# -----------------------------
# PDF generation
# -----------------------------
def generate_pdf(
    meta: ReportMeta,
    kpi_tiles: List[Dict],
    manual_summary_text: str,
    tables_in_order: List[Dict],
    embedded_images: List[EmbeddedImage],
    logo_bytes: Optional[bytes],
) -> bytes:
    buf = io.BytesIO()
    styles = build_styles()

    doc = SimpleDocTemplate(
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

    content_w = PAGE_W - (MARGINS_CONTENT["left"] + MARGINS_CONTENT["right"])
    content_h = PAGE_H - (MARGINS_CONTENT["top"] + MARGINS_CONTENT["bottom"])

    # Cover
    story.append(Spacer(1, 18 * mm))
    story.append(Paragraph("Monthly Business Review Report", styles["CoverTitle"]))
    story.append(Paragraph(meta.customer_name, styles["CoverSub"]))
    story.append(Paragraph(meta.date_range_label, styles["CoverSmall"]))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("Prepared for external sharing", styles["Muted"]))
    story.append(PageBreak())

    # Executive Summary
    story.append(Paragraph("Executive Summary", styles["H1"]))
    story.append(Paragraph("High-level KPIs and reporting performance overview.", styles["Body"]))
    story.append(Spacer(1, 10 * mm))

    tiles_tbl = kpi_tiles_grid(kpi_tiles, content_w, cols=4)
    story.append(tiles_tbl)
    story.append(Spacer(1, 12 * mm))

    if manual_summary_text.strip():
        story.append(Paragraph("Narrative Summary", styles["H1"]))
        story.append(Paragraph("Notes added manually for external sharing.", styles["Body"]))
        story.append(Spacer(1, 6 * mm))
        for para in manual_summary_text.strip().split("\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), styles["Body"]))
                story.append(Spacer(1, 2 * mm))

    story.append(PageBreak())

    # Tables: one per page, dynamic size + centered
    header_block_h = 26 * mm  # title + subtitle space
    note_h = 6 * mm

    table_avail_w = content_w
    table_avail_h = content_h - header_block_h - note_h

    for item in tables_in_order:
        title = item.get("title", "Table")
        subtitle = item.get("subtitle", "")
        df = item.get("df", pd.DataFrame())
        max_rows = item.get("max_rows", 80)

        story.append(Paragraph(title, styles["H1"]))
        if subtitle:
            story.append(Paragraph(subtitle, styles["Body"]))
        story.append(Spacer(1, 6 * mm))

        if df is None or df.empty:
            story.append(Paragraph("No data available for this view based on current filters.", styles["Body"]))
            story.append(PageBreak())
            continue

        tbl, truncated = auto_fit_table_dynamic(df, avail_w=table_avail_w, avail_h=table_avail_h, max_rows=max_rows)

        # Center the table vertically so page doesn't look empty
        w, h = tbl.wrap(table_avail_w, table_avail_h)
        centered_parts = centered_flowable(tbl, table_avail_h, h)
        story.extend(centered_parts)

        if truncated:
            story.append(Paragraph(f"Note: showing first {max_rows} rows to fit on one page.", styles["TinyNote"]))

        story.append(PageBreak())

    # Embedded images: each on its own page with header + subtitle, centered
    if embedded_images:
        img_header_h = 30 * mm
        img_avail_w = content_w
        img_avail_h = content_h - img_header_h

        for ei in embedded_images:
            story.append(Paragraph(ei.header or "Embedded Image", styles["H1"]))
            if ei.subtitle:
                story.append(Paragraph(ei.subtitle, styles["Body"]))
            story.append(Spacer(1, 6 * mm))

            img_flow = fit_image_to_content(ei.bytes_, img_avail_w, img_avail_h)
            # Center image vertically in remaining space
            w, h = img_flow.wrap(img_avail_w, img_avail_h)
            story.extend(centered_flowable(img_flow, img_avail_h, h))

            story.append(PageBreak())

    def on_page(canvas, doc_):
        draw_header_footer(canvas, doc_, meta, logo_bytes)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buf.seek(0)
    return buf.read()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Monthly Business Review Report", layout="wide")
st.title("Monthly Business Review Report Builder")
st.caption("Upload your shipment file, filter, review tables, and export a formatted PDF (designed for external sharing).")

with st.sidebar:
    st.header("1) Upload")
    uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
    st.divider()

    st.header("2) Report Details")
    customer_name = st.text_input("Customer name", value="Customer")
    contracted_volume = st.number_input("Contracted volume", min_value=0.0, value=1000.0, step=10.0)
    billed_volume = st.number_input("Billed volume", min_value=0.0, value=900.0, step=10.0)

    st.divider()
    st.header("3) Manual PDF Summary")
    manual_summary = st.text_area(
        "Type an executive summary (included in PDF)",
        placeholder="Example:\n- Reporting improved vs last month\n- Key gaps remain on Arrival at Destination\n- OTP below target on Lane A → B",
        height=180,
    )

    st.divider()
    st.header("4) Assets")
    st.caption("Logo: upload here, or add a repo logo at assets/p44_logo.png.")
    logo_up = st.file_uploader("Upload logo (PNG)", type=["png"], key="logo")

    st.caption("Embedded images (multiple). Each becomes a separate PDF page.")
    embedded_files = st.file_uploader(
        "Upload images to embed (PNG/JPG) — select multiple",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="embeds",
    )

    st.divider()
    st.header("5) Filters")
    date_basis = st.selectbox(
        "Date filter basis",
        options=["Destination latest planned arrival end time", "Created time"],
        index=0,
    )
    date_mode = st.radio("Date filter mode", options=["Include", "Exclude"], horizontal=True)
    carrier_mode = st.radio("Carrier filter mode", options=["Include", "Exclude"], horizontal=True)

    st.divider()
    st.header("6) Milestones & OTP")
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

date_col = "Destination latest planned arrival end time" if date_basis.startswith("Destination") else "Created time"

if date_col in df.columns and df[date_col].notna().any():
    min_dt = df[date_col].min()
    max_dt = df[date_col].max()
else:
    min_dt, max_dt = None, None

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

carriers = sorted(
    [c for c in df["Current carrier"].dropna().unique().tolist() if str(c).strip() != ""]
) if "Current carrier" in df.columns else []
selected_carriers = st.multiselect("Carriers", options=carriers, default=carriers[: min(len(carriers), 10)])

fdf = df.copy()
fdf = apply_include_exclude_filter(fdf, date_col, start_ts, end_ts, date_mode)
fdf = apply_include_exclude_list(fdf, "Current carrier", selected_carriers, carrier_mode)

# Core metrics
all_selected_mask = all_selected_reported_mask(fdf, selected_milestones)
num_shipments = fdf["Shipment ID"].nunique() if "Shipment ID" in fdf.columns else len(fdf)
num_carriers = fdf["Current carrier"].nunique() if "Current carrier" in fdf.columns else 0
num_all_selected = int(all_selected_mask.sum())
pct_all_selected = (num_all_selected / num_shipments * 100) if num_shipments else float("nan")

num_marked_completed = int((fdf["Current state"] == "COMPLETED").sum()) if "Current state" in fdf.columns else 0
num_timed_out = int((fdf["Completion Bucket"] == "Timed Out").sum()) if "Completion Bucket" in fdf.columns else 0

# Exceptions
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

# Milestone reporting table
milestone_rows = []
for m in selected_milestones:
    rep = reported_mask(fdf, m)
    milestone_rows.append(
        {
            "Milestone": m,
            "Reported (#)": int(rep.sum()),
            "Reported (%)": round((rep.sum() / num_shipments * 100), 1) if num_shipments else None,
        }
    )
milestone_table = pd.DataFrame(milestone_rows)

# OTP table
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

# Carrier table
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

# Lane tables
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

# Contract consumption (as tile)
consumption_ratio = (billed_volume / contracted_volume) if contracted_volume else float("inf")
consumption_pct = (consumption_ratio * 100) if contracted_volume else None

# -----------------------------
# Dashboard (tables + KPIs) — unchanged (Streamlit side)
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
c3.metric("Destination ETA accurate", f"{dest_est_accurate:,} / {dest_est_assessable:,}")

st.divider()
st.subheader("Milestone reporting (selected)")
st.dataframe(milestone_table, use_container_width=True)

st.subheader("Carrier-wise table")
st.dataframe(carrier_table, use_container_width=True)

st.subheader("On-time performance (event-wise)")
st.dataframe(otp_table, use_container_width=True)

if not lane_m_table.empty:
    st.subheader("Lane insights (top lanes)")
    a, b = st.columns(2)
    with a:
        st.caption("Milestone reporting")
        st.dataframe(lane_m_table.head(20), use_container_width=True)
    with b:
        st.caption("On-time performance")
        st.dataframe(lane_o_table.head(20), use_container_width=True)

# -----------------------------
# Embedded images metadata UI
# -----------------------------
embedded_images: List[EmbeddedImage] = []
if embedded_files:
    st.subheader("Embedded images setup (for PDF)")
    st.caption("For each image, provide a page header and a subtitle. Each becomes its own PDF page.")
    for i, f in enumerate(embedded_files, start=1):
        col1, col2 = st.columns([1, 1])
        with col1:
            header = st.text_input(f"Image {i} page header", value=f.name, key=f"img_head_{i}")
        with col2:
            subtitle = st.text_input(f"Image {i} subtitle", value="", key=f"img_sub_{i}")
        st.image(f, caption=f"Preview: {f.name}", use_container_width=True)
        embedded_images.append(EmbeddedImage(header=header, subtitle=subtitle, bytes_=f.getvalue()))

# -----------------------------
# PDF Export
# -----------------------------
date_range_label = f"{start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}"
report_title = f"Monthly Business Review Report - {customer_name} - {date_range_label}"
meta = ReportMeta(customer_name=customer_name, date_range_label=date_range_label, report_title=report_title)

# Logo bytes: upload takes precedence; else fallback to assets/p44_logo.png
logo_bytes = None
if logo_up:
    logo_bytes = logo_up.getvalue()
else:
    try:
        with open("assets/p44_logo.png", "rb") as f:
            logo_bytes = f.read()
    except Exception:
        logo_bytes = None

# Contract tile tone based on consumption
if contracted_volume and billed_volume:
    if consumption_ratio <= 1.0:
        contract_tone = "pos"
    elif consumption_ratio <= 1.05:
        contract_tone = "amber"
    else:
        contract_tone = "critical"
else:
    contract_tone = "neutral"

contract_sub = f"Contracted: {contracted_volume:,.0f} | Billed: {billed_volume:,.0f}"
contract_value = fmt_pct(consumption_pct) if consumption_pct is not None else "—"

kpi_tiles = [
    {"label": "Number of Shipments", "value": fmt_int(num_shipments), "tone": "neutral"},
    {"label": "Number of Carriers", "value": fmt_int(num_carriers), "tone": "neutral"},
    {"label": "All selected milestones (#)", "value": fmt_int(num_all_selected), "tone": "neutral"},
    {"label": "All selected milestones (%)", "value": fmt_pct(pct_all_selected), "tone": "neutral"},
    {"label": "Marked Completed", "value": fmt_int(num_marked_completed), "tone": "pos"},
    {"label": "Timed Out (per rule)", "value": fmt_int(num_timed_out), "tone": "amber" if num_timed_out else "pos"},
    {"label": "No Exceptions", "value": fmt_int(no_exc), "tone": "pos"},
    {"label": "With Exceptions", "value": fmt_int(with_exc), "tone": "amber" if with_exc else "pos"},
    # Contract consumption as part of tiles grid:
    {"label": "Contract Consumption", "value": contract_value, "subvalue": contract_sub, "tone": contract_tone},
]

# Put contract tile at the end: it will naturally create a 3rd row (still aligned)
# If you want it to always appear as the FIRST tile of row 3, this is perfect.

tables_in_order = [
    {
        "title": "Milestone Reporting",
        "subtitle": "Reporting counts and percentages for selected milestones.",
        "df": milestone_table,
        "max_rows": 80,
    },
    {
        "title": "Carrier-wise Summary",
        "subtitle": "Shipments and reporting performance by carrier.",
        "df": carrier_table,
        "max_rows": 80,
    },
    {
        "title": "On-time Performance (OTP)",
        "subtitle": f"On-time uses latest planned END time + {latency_minutes} minutes latency.",
        "df": otp_table,
        "max_rows": 80,
    },
]

if not lane_m_table.empty:
    tables_in_order.append(
        {
            "title": "Lane Insights — Milestone Reporting",
            "subtitle": "Top lanes by shipment volume.",
            "df": lane_m_table.head(300),
            "max_rows": 80,
        }
    )

if not lane_o_table.empty:
    tables_in_order.append(
        {
            "title": "Lane Insights — On-time Performance",
            "subtitle": "Top lanes by shipment volume.",
            "df": lane_o_table.head(300),
            "max_rows": 80,
        }
    )

if not exc_counts.empty:
    exc_table = exc_counts.reset_index()
    exc_table.columns = ["Exception", "Count"]
    tables_in_order.append(
        {
            "title": "Exceptions Breakdown",
            "subtitle": "Exception-wise distribution (non-empty exceptions).",
            "df": exc_table,
            "max_rows": 80,
        }
    )

dest_acc_df = pd.DataFrame(
    [
        {
            "Metric": "Destination ETA accuracy (abs(actual - estimated) <= latency)",
            "Accurate (#)": dest_est_accurate,
            "Assessable (#)": dest_est_assessable,
            "Accurate (%)": dest_est_pct,
        }
    ]
)
tables_in_order.append(
    {
        "title": "Destination ETA Accuracy",
        "subtitle": f"Latency threshold = {latency_minutes} minutes.",
        "df": dest_acc_df,
        "max_rows": 20,
    }
)

if st.button("Generate PDF Report"):
    pdf_bytes = generate_pdf(
        meta=meta,
        kpi_tiles=kpi_tiles,
        manual_summary_text=manual_summary,
        tables_in_order=tables_in_order,
        embedded_images=embedded_images,
        logo_bytes=logo_bytes,
    )

    st.success("PDF generated!")
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=f"{report_title}.pdf",
        mime="application/pdf",
    )
