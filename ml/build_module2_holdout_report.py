"""Build the shareable Module 2 holdout evaluation report."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether


ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "output/module2_holdout_accuracy_2025_2026.json").read_text())
SAFE = json.loads((ROOT / "output/module2_holdout_accuracy_2025_2026_safety_revision.json").read_text())
RESEARCH = json.loads((ROOT / "output/module2_price_model_research.json").read_text())
DEST = ROOT / "output/pdf/module2_later_year_hold_sell_validation_2025_2026.pdf"


ink = colors.HexColor("#172238")
blue = colors.HexColor("#244985")
light = colors.HexColor("#eaf0f8")
muted = colors.HexColor("#58657a")
red = colors.HexColor("#a12e3b")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
                          fontSize=18, leading=22, textColor=ink, spaceAfter=10, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="Deck", parent=styles["Normal"], fontSize=9.5, leading=14,
                          textColor=muted, spaceAfter=12))
styles.add(ParagraphStyle(name="HeadCustom", parent=styles["Heading2"], fontName="Helvetica-Bold",
                          fontSize=11, leading=14, textColor=blue, spaceBefore=11, spaceAfter=6))
styles.add(ParagraphStyle(name="BodyCustom", parent=styles["Normal"], fontSize=8.8,
                          leading=12.6, textColor=ink, spaceAfter=6))
styles.add(ParagraphStyle(name="SmallCustom", parent=styles["Normal"], fontSize=7.8,
                          leading=10.7, textColor=muted, spaceAfter=5))
styles.add(ParagraphStyle(name="Callout", parent=styles["Normal"], fontName="Helvetica-Bold",
                          fontSize=10, leading=14, textColor=red, spaceAfter=4))


def p(text: str, name: str = "BodyCustom") -> Paragraph:
    return Paragraph(text, styles[name])


def table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    data = [[p(h, "SmallCustom") for h in headers]] + [[p(str(v), "BodyCustom") for v in row] for row in rows]
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), light),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), .5, blue),
        ("LINEBELOW", (0, 1), (-1, -2), .25, colors.HexColor("#d5dce6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return result


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d5dce6"))
    canvas.line(42, 41, 570, 41)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(muted)
    canvas.drawString(42, 29, "IPO Intelligence | Module 2 independent price-only evaluation | 16 October 2026")
    canvas.drawRightString(570, 29, str(doc.page))
    canvas.restoreState()


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(DEST), pagesize=(612, 792), rightMargin=42,
                            leftMargin=42, topMargin=39, bottomMargin=53)
    story = [p("Module 2: later-year hold/sell accuracy", "TitleCustom"),
             p("A dated replay of IPOs listed in 2025-2026. This evaluates the post-listing monitor, not the pre-listing Module 1 classifier.", "Deck"),
             p("Main finding", "HeadCustom"),
             p(f"The original rule correctly called {BASE['day30']['model_accuracy_pct']:.1f}% of 30-session directions on {BASE['day30']['n']} eligible IPOs. The always-sell baseline was {BASE['day30']['always_sell_accuracy_pct']:.1f}%. This is <b>not evidence of a useful hold/sell predictor</b>.", "Callout"),
             p("We changed one correctness rule after examining that result: absent filings no longer count as proof that anchors did not sell. The revised rule scored 50.8%, still below the 59.4% always-sell baseline. This revision is exploratory, not a fresh blind holdout.", "BodyCustom"),
             p("Decision and outcome definition", "HeadCustom"),
             p("One decision per IPO at its 31st observed daily close (index 30). The monitor could read prices only through that date. 'Sell' and 'take profit' predict a non-positive return; other verdicts predict a positive return. We measured the next session and the following 30 sessions separately. A trading illustration executes at the next close and checks value 30 sessions later. All returns here are close-to-close, not opening-price predictions.", "BodyCustom"),
             p("Primary test: direction after 30 trading sessions", "HeadCustom"),
             table(["Rule", "Correct", "Gap vs always sell", "Mean net action return"], [
                 ["Original deployed heuristic", "45.3%", "-14.1 pp", "-0.96%"],
                 ["Evidence-gated safety revision", "50.8%", "-8.6 pp", "-0.37%"],
                 ["8% trailing-stop baseline", "50.0%", "-9.4 pp", "-0.59%"],
                 ["Always sell to cash", "59.4%", "reference", "-0.30%"],
                 ["Always hold", "40.6%", "-18.8 pp", "-1.63%"],
             ], [156, 66, 122, 154]),
             Spacer(1, 5),
             p("Action-return illustration includes a stated 0.30% cost for selling and no cost to continue holding. It ignores tax, bid-ask spread, slippage, dividends and gaps. It is not an investable backtest.", "SmallCustom"),
             p("First-session direction and error counts", "HeadCustom"),
             table(["Test", "Original", "Safety revision", "Simple reference"], [
                 ["Next-session accuracy", "52.3%", "53.1%", "50.8% always hold"],
                 ["30-session confusion: TN / FP / FN / TP", "28 / 48 / 22 / 30", "36 / 40 / 23 / 29", "76 / 0 / 52 / 0 always sell"],
                 ["95% bootstrap gap vs always sell", "-27.3 to -0.8 pp", "-21.1 to +3.9 pp", "paired IPO resampling"],
             ], [174, 112, 112, 100]),
             p("Cohort and source integrity", "HeadCustom"),
             p("The historical file had no prices for 2025-26 listings. A separate Yahoo Finance historical-close pull matched 175 NSE-symbol IPOs. Of those, 128 had enough future sessions. We excluded 43 without a mature 30-session outcome and four whose first same-day price conflicted with the listing record by more than 2% (possible split adjustment). Three NSE symbols returned no historical observations; six IPOs lacked NSE symbols. All exclusions are in the saved audit output.", "BodyCustom"),
             p("Median first-day close discrepancy among the matched records was effectively zero, but the Yahoo feed is secondary. Exchange OHLC verification, corporate-action reconciliation and documented publication times remain required for a paper-grade claim.", "SmallCustom"),
             p("Year-by-year stability", "HeadCustom"),
             table(["Listing year", "N", "Positive after 30 sessions", "Original rule", "Always sell"], [
                 ["2025", "100", "32.0%", "43.0%", "68.0%"],
                 ["2026 matured", "28", "71.4%", "53.6%", "28.6%"],
             ], [90, 40, 180, 106, 82]),
             p("The direction base rate reversed between years. This explains why an apparently safe unconditional action can be good in one year and poor in the next. It does not establish that the monitor detects regime changes in time.", "BodyCustom"),
             p("Where the original rule missed", "HeadCustom"),
             p("In the 2025 subset, 68 of 100 IPOs fell over the next 30 sessions. The monitor nevertheless issued 61 stay-like calls; 43 of those were followed by a non-positive return. In the smaller 2026 subset, 20 of 28 rose, while the monitor issued 11 sell-like calls; eight of those missed a positive return. The rule was not adapting reliably to the sharp change in base rates.", "BodyCustom"),
             p("The current monitor also treated a missing disclosure list as if no anchor investor had sold. That is a data-provenance error, not a predictive signal. The safety revision removes that assumption. Its higher retrospective score does not validate it: the same outcomes were already visible during revision.", "BodyCustom"),
             p("Replacement model experiment", "HeadCustom"),
             p("A regularized logistic model used only price and benchmark features visible by session 31. It trained on 2021-23, selected regularization by 2024 Brier score, then refit on 2021-24. On the same later-year set it scored 57.8% direction accuracy, AUC 0.541 and Brier 0.245; always sell still scored 59.4%. The 2024 validation AUC was about 0.506 for the selected setting. This candidate is <b>not promoted</b>. The 2025-26 outcomes had already been inspected for the original rule, so this comparison cannot be called fully blind.", "BodyCustom"),
             p("What can go into the paper", "HeadCustom"),
             p("Report these as an honest <b>price-only post-listing diagnostic</b>, with the outcome horizon, cohort, baselines and limitations. Do not label Module 1 AUC as Module 2 accuracy. Do not describe the monitor as predictive or claim its buy/sell advice beats simple rules. A publishable full-Module-2 result needs exchange-checked historical prices, as-published filings/news, point-in-time signals, a frozen policy, and an untouched prospective cohort.", "BodyCustom"),
             p("Suggested paper wording", "HeadCustom"),
             p("'In a retrospective price-only evaluation of 128 eligible 2025-2026 IPOs, the original post-listing heuristic achieved 45.3% accuracy for the sign of the subsequent 30-session return, compared with 59.4% for an always-sell reference. A post-hoc evidence-gating correction improved the measured accuracy to 50.8%, but did not beat the reference. We therefore treat Module 2 as an exploratory monitoring interface rather than a validated return predictor.'", "BodyCustom"),
             p("Audit files and source", "HeadCustom"),
             p("The per-IPO dated decision ledger, source-file SHA-256 hashes, exclusions and exact metric calculations are saved with this report in the project's output and data/audit directories. Source price pull: Yahoo Finance historical daily closes (retrieved 16 October 2026); primary verification target: NSE security-wise historical price-volume data, https://www.nseindia.com/historical/price-and-volume-data-per-security. No news or filings were replayed. Results are retrospective and subject to price-source correction.", "SmallCustom")]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(DEST)


if __name__ == "__main__":
    main()
