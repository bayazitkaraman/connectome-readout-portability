"""Create a standalone two-page collaborator brief from the Project 1 results."""
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "output/pdf/project1-collaborator-brief.pdf"
W, H = letter
LEFT, WIDTH = 46, 520
INK = colors.HexColor("#182c36")
BODY = colors.HexColor("#29424c")
MUTED = colors.HexColor("#48626e")
TEAL = colors.HexColor("#09665f")
PALE = colors.HexColor("#eaf4f1")
LINE = colors.HexColor("#c5d5d7")
COPPER = colors.HexColor("#9b4d27")


def text(c, x, top, s, size=10, bold=False, color=BODY):
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.setFillColor(color)
    c.drawString(x, H-top-size, s)


def para(c, top, s, size=11, leading=16, color=BODY, x=LEFT, width=WIDTH):
    style = ParagraphStyle("body", fontName="Helvetica", fontSize=size, leading=leading, textColor=color)
    block = Paragraph(s, style)
    _, height = block.wrap(width, H)
    block.drawOn(c, x, H-top-height)
    return top+height


def header(c, section):
    text(c, LEFT, 32, "PROJECT 1   /   CONNECTOME-BASED DECODING", 9, True, TEAL)
    c.setFont("Helvetica", 9)
    c.setFillColor(MUTED)
    c.drawRightString(W-LEFT, H-41, section)
    c.setStrokeColor(LINE)
    c.line(LEFT, H-53, W-LEFT, H-53)


def footer(c, page):
    c.setStrokeColor(LINE)
    c.line(LEFT, 38, W-LEFT, 38)
    text(c, LEFT, H-30, "Research brief for discussion   /   19 November 2026", 8.8, color=MUTED)
    text(c, W-LEFT-24, H-30, f"{page} / 2", 8.8, color=MUTED)


def label(c, y, value):
    text(c, LEFT, y, value, 15, True, INK)
    return y + 24


def link(c, x, y, title, url):
    text(c, x, y, title, 9.2, True, TEAL)
    width = c.stringWidth(title, "Helvetica-Bold", 9.2)
    c.linkURL(url, (x, H-y-12, x+width, H-y+1), relative=0, thickness=0)
    return x + width + 20


def table(c, top, xs, data):
    c.setFillColor(PALE)
    c.rect(LEFT, H-top-25, WIDTH, 25, fill=1, stroke=0)
    for offset, heading in zip(xs, data[0]):
        text(c, LEFT+offset, top+8, heading, 9, True, TEAL)
    y = top+25
    for row in data[1:]:
        for offset, item in zip(xs, row):
            text(c, LEFT+offset, y+8, item, 11, True, INK)
        y += 28
        c.setStrokeColor(LINE)
        c.line(LEFT, H-y, LEFT+WIDTH, H-y)
    return y


def overview(c):
    header(c, "Zero-label transfer")
    text(c, LEFT, 72, "Can wiring similarity help", 27, True, INK)
    text(c, LEFT, 103, "choose a decoder?", 27, True, INK)
    y = para(c, 150, "Across two repeat-scan datasets, choosing another person's decoder by the similarity of "
             "their wiring improved simulated signal recall before the target supplied any labels. The "
             "evidence for an additional benefit from choosing the same person was limited.",
             12.2, 17, TEAL)
    y = label(c, y+18, "What the decoder predicts")
    y = para(c, y, "Each diffusion MRI scan supplies a structural wiring matrix for a conn2res recurrent network. "
             "A random signal drives eight regions. The decoder estimates the signal at ten past time steps "
             "from activity in the other regions. All inputs, labels and test sequences are simulated.")
    y = label(c, y+15, "Held-out R² on the repeat scan, zero target labels")
    y = table(c, y, [8,338,430], [
        ("First scan used to train the decoder", "NKI (25)", "Beijing (57)"),
        ("Mean across other people's decoders", "0.024", "0.103"),
        ("Other person with the closest wiring match", "0.114", "0.167"),
        ("Same person", "0.043", "0.153")])
    y = para(c, y+9, "Selection used only the correlation between the two wiring matrices. Compared with the "
             "average other person, selecting the closest match improved R² by 0.090 in NKI (95% interval "
             "0.068 to 0.113) and 0.063 in Beijing (0.048 to 0.079). Intervals condition on the observed donors.",
             10.3, 14)
    y = label(c, y+15, "Where does recall accuracy go?")
    y = para(c, y, "On its own source scan, a decoder tuned for that scan scored 0.965 in NKI and 0.831 in "
             "Beijing. Tuning for transfer lowered these source scores to 0.191 and 0.237. Freezing those "
             "weights and replaying identical fresh signals on the person's repeat scan lowered R² further, "
             "to 0.043 and 0.153. The scan change alone cost 0.148 and 0.085 R², respectively.")
    y = para(c, y+8, "The original cross-visit ridge penalties were 10 in NKI and 1 in Beijing, within a grid "
             "extending to 10,000. A denser, wider grid gave personal cross-visit R² of 0.041 and 0.155. "
             "The separate pooled experiment found zero calibration examples saved at its R² = 0.50 threshold.",
             10.3, 14)
    y = label(c, y+14, "Data and simulation sources")
    x = LEFT
    for title, url in (("NKI release and split methods", "https://zenodo.org/records/4041549"),
                       ("Beijing acquisition", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4623457/"),
                       ("Beijing matrices", "https://docs.neurodata.io/m2g/paper/data.html"),
                       ("conn2res paper", "https://pmc.ncbi.nlm.nih.gov/articles/PMC10803782/")):
        x = link(c, x, y, title, url)
    assert y + 12 < H-40, y
    footer(c, 1)
    c.showPage()


def visit_plot(c, rows, visit, x, top, color):
    text(c, x, top, f"Target visit {visit}  /  25 people", 13, True, INK)
    sel = [row for row in rows if row["target_visit"] == str(visit) and
           row["target_half"] == "even" and row["src_half"] == "odd"]
    own = {row["target_id"]:float(row["alt_own_r2"]) for row in sel if row["src_visit"] == str(visit)}
    cross = {row["target_id"]:float(row["alt_own_r2"]) for row in sel if row["src_visit"] != str(visit)}
    assert len(own) == 25 and own.keys() == cross.keys()
    l, r, b, t = x+47, x+193, H-top-169, H-top-32
    def ordinate(score):
        assert -.46 <= score <= .36
        return b + (score+.46)/.82*(t-b)
    for tick in (-.4,-.2,0,.2):
        py = ordinate(tick)
        c.setLineWidth(.6)
        c.setStrokeColor(LINE)
        c.line(l-10,py,r+10,py)
        c.setFont("Helvetica",8)
        c.setFillColor(MUTED)
        c.drawRightString(l-15,py-3,f"{tick:.1f}")
    c.setStrokeColor(colors.HexColor("#b6c9ca"))
    c.setFillColor(colors.HexColor("#829c9c"))
    c.setLineWidth(.65)
    for person in own:
        a, d = ordinate(own[person]), ordinate(cross[person])
        c.line(l,a,r,d)
        c.circle(l,a,1.65,fill=1,stroke=0)
        c.circle(r,d,1.65,fill=1,stroke=0)
    a, d = ordinate(sum(own.values())/25), ordinate(sum(cross.values())/25)
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(2.1)
    c.line(l,a,r,d)
    c.circle(l,a,3.3,fill=1,stroke=0)
    c.circle(r,d,3.3,fill=1,stroke=0)
    c.setFont("Helvetica-Bold",9)
    c.drawCentredString(l,b-14,"Same visit")
    c.drawCentredString(r,b-14,"Other visit")


def followup(c):
    header(c, "Within-acquisition check")
    text(c, LEFT, 72, "How much does a change", 25, True, INK)
    text(c, LEFT, 102, "of visit matter?", 25, True, INK)
    y = para(c, 146, "Only NKI provided two reconstructions per visit, with 64 diffusion directions in each half. "
             "We trained on odd directions and tested on the same even-direction target. The held-out signals "
             "and ridge penalty of 10 were identical for the same-visit and other-visit comparisons.")
    with (RESULTS/"nki-split-half/subject_scores.csv").open(encoding="utf-8",newline="") as handle:
        data = list(csv.DictReader(handle))
    top = y+18
    visit_plot(c, data, 1, LEFT, top, TEAL)
    visit_plot(c, data, 2, LEFT+268, top, COPPER)
    y = para(c, top+198, "Vertical axis gives zero-label held-out R² for the person's own decoder. Each thin line "
             "follows one person across the two source choices. The colored line connects cohort means. Both panels "
             "use the same people and scale.", 9, 12, MUTED)
    y = label(c, y+14, "The same-person comparison at a fixed penalty")
    y = para(c, y, "Changing the source from the same visit to the other visit reduced mean R² from 0.153 to "
             "0.059 at visit 1 (25/25 people) and from 0.176 to 0.048 at visit 2 (24/25). The halves identified "
             "all 25 people by wiring at both visits. With the same penalty, the source scored about 0.190 "
             "on itself at each visit. The halves shared nine baseline volumes and preprocessing. "
             "Acquisition, reconstruction and anatomy can all vary between visits.")
    y = label(c, y+14, "How far does wiring similarity explain transfer?")
    y = para(c, y, "Across visits, we compared the personal decoder with strangers whose wiring similarities "
             "bracketed its own for the same target. Mean residual personal gain was -0.004 R² in NKI (95% "
             "interval -0.020 to 0.012) and -0.003 in Beijing (-0.032 to 0.021). This comparison covered "
             "16/25 and 33/57 people. Within a single NKI acquisition, no stranger came within 0.05 of the "
             "personal wiring correlation for the same target and scan combination.")
    y = para(c, y+10, "Excluding the flagged NKI participant from testing, donor selection and tuning changed "
             "the original personal gain from 0.019 to 0.022 R² and the similarity-gain rank correlation from "
             "0.762 to 0.748. All 25 people contribute to the main analysis.", 10.3, 14)
    assert y < H-44, y
    footer(c, 2)
    c.showPage()


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(OUT), pagesize=letter)
    canvas.setTitle("Project 1 | Can wiring similarity help choose a decoder?")
    overview(canvas)
    followup(canvas)
    canvas.save()
