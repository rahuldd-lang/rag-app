"""
generate_sample_pdf.py
──────────────────────
Generates "TechCorp Annual Report 2023.pdf" — a synthetic but realistic
business report with rich numerical data for demonstrating RAG + visualization.

Run:
    python data/generate_sample_pdf.py

Produces: data/TechCorp_Annual_Report_2023.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

OUTPUT_PATH = Path(__file__).parent / "TechCorp_Annual_Report_2023.pdf"

ACCENT  = colors.HexColor("#2563EB")   # Blue
DARK    = colors.HexColor("#1E293B")
LIGHT   = colors.HexColor("#F8FAFC")


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=24, spaceAfter=12, textColor=ACCENT, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=16, spaceBefore=18, spaceAfter=6, textColor=DARK,
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey, alignment=TA_CENTER,
    ))
    return styles


def table_style(header_color=ACCENT):
    return TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 10),
        ("ALIGN",       (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN",       (0, 0), (0, -1), "LEFT"),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def generate():
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
    )
    s = build_styles()
    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 3*cm),
        Paragraph("TechCorp Inc.", s["H1"]),
        Paragraph("Annual Report 2023", s["H2"]),
        Spacer(1, 0.5*cm),
        Paragraph(
            "Delivering Innovation Across Cloud, AI, and Enterprise Software",
            s["Caption"],
        ),
        Spacer(1, 1*cm),
        Paragraph(
            "Founded in 2010, TechCorp Inc. is a publicly listed technology company "
            "headquartered in San Francisco, CA. We serve over 12,000 enterprise customers "
            "across 45 countries. In fiscal year 2023, TechCorp achieved record revenue "
            "driven by strong growth in our Cloud Services and AI Platform divisions.",
            s["Body"],
        ),
        PageBreak(),
    ]

    # ── Executive Summary ────────────────────────────────────────────────────
    story += [
        Paragraph("Executive Summary", s["H2"]),
        Paragraph(
            "Fiscal year 2023 was a landmark year for TechCorp. Total revenue reached "
            "$2.84 billion, representing a 23% increase over 2022. Net income grew to "
            "$412 million (14.5% net margin), compared to $298 million in 2022. "
            "Operating expenses totalled $2.18 billion, with R&D investment of $520 million "
            "— the highest in company history.",
            s["Body"],
        ),
        Paragraph(
            "Headcount expanded from 14,200 to 17,800 employees globally. Customer "
            "retention rate remained strong at 94%. The company returned $180 million "
            "to shareholders via share buybacks and initiated a quarterly dividend of "
            "$0.15 per share starting Q4 2023.",
            s["Body"],
        ),
    ]

    # ── Revenue Table ────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.4*cm),
        Paragraph("Annual Revenue by Division (USD Millions)", s["H2"]),
        Paragraph(
            "Cloud Services remains the largest and fastest-growing division. "
            "Enterprise Software contributes steady recurring revenue while the "
            "newly-launched AI Platform is scaling rapidly.",
            s["Body"],
        ),
    ]

    rev_data = [
        ["Division",            "2021",  "2022",  "2023",  "YoY Growth"],
        ["Cloud Services",      "820",   "1,050", "1,380", "+31%"],
        ["Enterprise Software", "540",   "620",   "710",   "+15%"],
        ["AI Platform",         "60",    "120",   "380",   "+217%"],
        ["Professional Services","180",  "210",   "250",   "+19%"],
        ["Maintenance & Support","105",  "110",   "120",   "+9%"],
        ["Total",               "1,705", "2,110", "2,840", "+35%"],
    ]
    t = Table(rev_data, colWidths=[4.5*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.5*cm])
    t.setStyle(table_style())
    story += [t, Spacer(1, 0.2*cm),
              Paragraph("Table 1: Revenue by Division, 2021–2023", s["Caption"])]

    # ── Operating Expenses ───────────────────────────────────────────────────
    story += [
        Spacer(1, 0.6*cm),
        Paragraph("Operating Expense Breakdown 2023 (USD Millions)", s["H2"]),
        Paragraph(
            "R&D investment increased to $520 million (18.3% of revenue) as we "
            "accelerated hiring of AI researchers and engineers. Sales & Marketing "
            "spend grew modestly to $410 million while Cost of Revenue remained "
            "well-managed at $680 million.",
            s["Body"],
        ),
    ]

    opex_data = [
        ["Category",             "Amount (USD M)", "% of Revenue"],
        ["Cost of Revenue",      "680",            "23.9%"],
        ["R&D",                  "520",             "18.3%"],
        ["Sales & Marketing",    "410",             "14.4%"],
        ["General & Admin",      "250",              "8.8%"],
        ["Depreciation & Amort.","120",              "4.2%"],
        ["Total OpEx",           "1,980",            "69.7%"],
    ]
    t2 = Table(opex_data, colWidths=[5*cm, 4*cm, 3.5*cm])
    t2.setStyle(table_style(colors.HexColor("#7C3AED")))
    story += [t2, Spacer(1, 0.2*cm),
              Paragraph("Table 2: Operating Expense Breakdown, FY2023", s["Caption"])]

    story.append(PageBreak())

    # ── Geographic Revenue ───────────────────────────────────────────────────
    story += [
        Paragraph("Revenue by Geography 2023", s["H2"]),
        Paragraph(
            "North America continues to be our primary market, contributing 48% of total "
            "revenue. Europe, Middle East & Africa (EMEA) grew to 28%. Asia-Pacific "
            "is the fastest-expanding region at 19%, driven by enterprise deals in "
            "Japan, Singapore, and Australia. Latin America represents the remaining 5%.",
            s["Body"],
        ),
    ]

    geo_data = [
        ["Region",         "Revenue (USD M)", "% of Total", "YoY Growth"],
        ["North America",  "1,363",           "48%",        "+18%"],
        ["EMEA",           "795",             "28%",        "+27%"],
        ["Asia-Pacific",   "540",             "19%",        "+41%"],
        ["Latin America",  "142",              "5%",        "+22%"],
    ]
    t3 = Table(geo_data, colWidths=[4.5*cm, 3.5*cm, 2.5*cm, 2.5*cm])
    t3.setStyle(table_style(colors.HexColor("#059669")))
    story += [t3, Spacer(1, 0.2*cm),
              Paragraph("Table 3: Revenue by Geography, FY2023", s["Caption"])]

    # ── Quarterly Revenue ────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.6*cm),
        Paragraph("Quarterly Financial Performance 2023", s["H2"]),
        Paragraph(
            "Revenue showed consistent quarter-over-quarter acceleration through 2023. "
            "Q1 revenue was $612 million, Q2 $682 million, Q3 $745 million, and "
            "Q4 $801 million. Gross margin expanded from 68% in Q1 to 72% in Q4 "
            "as cloud infrastructure efficiency improved.",
            s["Body"],
        ),
    ]

    qtr_data = [
        ["Quarter", "Revenue (USD M)", "Gross Margin", "Operating Income"],
        ["Q1 2023", "612",             "68%",          "$98M"],
        ["Q2 2023", "682",             "69%",          "$112M"],
        ["Q3 2023", "745",             "71%",          "$130M"],
        ["Q4 2023", "801",             "72%",          "$152M"],
        ["FY 2023", "2,840",           "70% avg",      "$492M"],
    ]
    t4 = Table(qtr_data, colWidths=[3*cm, 4*cm, 3.5*cm, 4*cm])
    t4.setStyle(table_style(colors.HexColor("#DC2626")))
    story += [t4, Spacer(1, 0.2*cm),
              Paragraph("Table 4: Quarterly Performance Summary, 2023", s["Caption"])]

    story.append(PageBreak())

    # ── Customer Metrics ─────────────────────────────────────────────────────
    story += [
        Paragraph("Customer & Product Metrics", s["H2"]),
        Paragraph(
            "Total enterprise customers grew from 9,800 in 2022 to 12,200 in 2023. "
            "Net Revenue Retention (NRR) reached 118%, indicating strong upsell momentum. "
            "Average Contract Value (ACV) increased 14% to $233,000.",
            s["Body"],
        ),
        Paragraph(
            "Product adoption highlights in 2023: TechCorp AI Copilot reached 4,200 "
            "active enterprise deployments, up from 800 at end of 2022. "
            "Our flagship CloudOS platform processed 2.4 trillion API calls, "
            "maintaining 99.97% uptime. The developer platform crossed 1 million "
            "registered developers in September 2023.",
            s["Body"],
        ),
    ]

    cust_data = [
        ["Metric",                   "2021",  "2022",    "2023"],
        ["Enterprise Customers",     "7,200", "9,800",   "12,200"],
        ["Net Revenue Retention",    "108%",  "112%",    "118%"],
        ["Avg Contract Value (USD)", "$172K", "$205K",   "$233K"],
        ["Customer Retention Rate",  "91%",   "93%",     "94%"],
        ["Total Employees",          "11,400","14,200",  "17,800"],
    ]
    t5 = Table(cust_data, colWidths=[5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    t5.setStyle(table_style(colors.HexColor("#0891B2")))
    story += [t5, Spacer(1, 0.2*cm),
              Paragraph("Table 5: Key Customer & Operational Metrics, 2021–2023", s["Caption"])]

    # ── Outlook 2024 ─────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.6*cm),
        Paragraph("Outlook for Fiscal Year 2024", s["H2"]),
        Paragraph(
            "TechCorp management provides the following guidance for FY2024: "
            "Total revenue is expected to be in the range of $3.35–$3.50 billion, "
            "representing approximately 18–23% year-over-year growth. "
            "Non-GAAP operating margin is projected at 19–21%. "
            "The company plans to hire approximately 2,500 additional employees, "
            "primarily in AI research, cloud infrastructure, and customer success.",
            s["Body"],
        ),
        Paragraph(
            "Capital expenditure for 2024 is planned at $380 million, with 60% "
            "directed to data center expansion across three new regions: "
            "Mumbai, São Paulo, and Frankfurt. The remaining 40% will fund "
            "on-premise infrastructure refresh for the Enterprise Software division.",
            s["Body"],
        ),
    ]

    doc.build(story)
    print(f"✓ Sample PDF generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
