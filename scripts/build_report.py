"""Build the feasibility report and recall figure from the saved test scores.

Run with Python, pandas, ReportLab and the Poppler pdftoppm executable.
The figure uses ReportLab's vector charts and is exported at 300 dpi.
"""
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INK = "#172E36"
MUTED = "#435C65"
TEAL = "#00675F"
ORANGE = "#A74F20"
RULE = "#C8D5D4"
PAGE_WIDTH, PAGE_HEIGHT = letter
LEFT, RIGHT = 48, 564


def make_figure(scores, summary):
    primary = scores[(scores.gain == .90) & scores.decoder.isin(["individual", "shared"])]
    means = primary.pivot_table(index="delay", columns="decoder", values="r2")
    published = summary[(summary.gain == .90) & (summary.decoder == "shared")].copy()
    published = published[published.delays != "1-10"].set_index("delays")
    published.index = published.index.astype(int)
    assert means.index.tolist() == list(range(1, 11))
    assert primary.groupby(["delay", "decoder"]).size().eq(70).all()
    np.testing.assert_allclose(means.shared, published.loc[means.index, "mean_r2"])
    np.testing.assert_allclose(means.individual, published.loc[means.index, "mean_own_r2"])

    chart = Drawing(510, 273)
    chart.add(Rect(0,0,510,273,fillColor=HexColor("#FFFFFF"),strokeColor=None))
    chart.add(String(1,255,"Recall of earlier inputs",fontName="Helvetica-Bold",
                     fontSize=15,fillColor=HexColor(INK)))
    chart.add(String(1,238,"Prespecified analysis  |  Mean held-out R² across 70 connectomes  |  Spectral radius 0.90",
                     fontName="Helvetica",fontSize=9.5,fillColor=HexColor(MUTED)))
    chart.add(String(1,210,"R²",fontName="Helvetica-Bold",fontSize=10,fillColor=HexColor(INK)))
    x0, y0, plot_width, plot_height = 39, 43, 457, 164
    x_min, x_max, y_min, y_max = .80, 10.20, -.075, 1.06
    for level in [0,.25,.5,.75,1]:
        y = y0 + (level-y_min)*plot_height/(y_max-y_min)
        chart.add(Line(x0,y,x0+plot_width,y,strokeColor=HexColor("#697D80" if level==0 else "#D5DEDD"),
                       strokeWidth=.9 if level==0 else .55))

    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = x0, y0, plot_width, plot_height
    plot.data = [list(zip(means.index,means[column])) for column in ["individual","shared"]]
    for axis in [plot.xValueAxis, plot.yValueAxis]:
        axis.visibleGrid = False
        axis.visibleTicks = False
        axis.visibleAxis = False
        axis.labels.fontName = "Helvetica"
        axis.labels.fontSize = 9.4
        axis.labels.fillColor = HexColor(MUTED)
    plot.xValueAxis.valueMin,plot.xValueAxis.valueMax = x_min,x_max
    plot.xValueAxis.valueSteps = list(range(1,11))
    plot.xValueAxis.labels.dy = -3
    plot.yValueAxis.valueMin,plot.yValueAxis.valueMax = y_min,y_max
    plot.yValueAxis.valueSteps = [0,.25,.5,.75,1]
    plot.yValueAxis.labelTextFormat = lambda v: f"{v:.2f}"
    for index,(color,shape) in enumerate([(TEAL,"FilledCircle"),(ORANGE,"FilledSquare")]):
        style=plot.lines[index]
        style.strokeColor=HexColor(color)
        style.strokeWidth=2
        style.symbol=makeMarker(shape)
        style.symbol.size=5.2
        style.symbol.strokeColor=HexColor(color)
        style.symbol.fillColor=HexColor(color)
    chart.add(plot)
    for x,label,color,is_square in [
        (200,"Individual calibration",TEAL,False),(361,"Frozen shared readout",ORANGE,True)
    ]:
        col=HexColor(color)
        chart.add(Line(x,216,x+20,216,strokeColor=col,strokeWidth=2))
        marker = Rect(x+7.5,213.5,5,5) if is_square else Circle(x+10,216,2.5)
        marker.strokeColor=marker.fillColor=col
        chart.add(marker)
        chart.add(String(x+25,212.7,label,fontName="Helvetica",fontSize=9.5,fillColor=HexColor(INK)))
    chart.add(String(270,9,"Recall delay (simulation steps)",textAnchor="middle",
                     fontName="Helvetica",fontSize=10.5,fillColor=HexColor(INK)))
    return chart


def paragraph(doc,text,top,size=10.9,leading=15.4,face="Helvetica",color=INK,
              left=LEFT,width=RIGHT-LEFT):
    style=ParagraphStyle("body",fontName=face,fontSize=size,leading=leading,
                         textColor=HexColor(color),spaceBefore=0,spaceAfter=0)
    item=Paragraph(text,style)
    _,height=item.wrap(width,900)
    item.drawOn(doc,left,PAGE_HEIGHT-top-height)
    return top+height


def rule(doc,top):
    doc.setLineWidth(.6)
    doc.setStrokeColor(HexColor(RULE))
    doc.line(LEFT,PAGE_HEIGHT-top,RIGHT,PAGE_HEIGHT-top)


def page_frame(doc,page,label):
    paragraph(doc,"CONNECTOME COMPUTING  /  FEASIBILITY RESULTS",29,8.2,10,
              "Helvetica-Bold",TEAL)
    doc.setFont("Helvetica",8.2)
    doc.setFillColor(HexColor(MUTED))
    doc.drawRightString(RIGHT,PAGE_HEIGHT-37,"19 NOVEMBER 2026")
    rule(doc,47)
    rule(doc,750)
    paragraph(doc,label,759,8,10,color=MUTED)
    doc.setFont("Helvetica-Bold",8.2)
    doc.drawRightString(RIGHT,PAGE_HEIGHT-767,f"{page} / 2")


def heading(doc,text,top):
    return paragraph(doc,text,top,17,21,"Times-Bold")+5


def link(label,url):
    return f'<a href="{url}" color="{TEAL}"><u>{label}</u></a>'


def report(figure, scores, summary, sensitivity):
    def result(gain,decoder="shared",delays="1-10"):
        match=summary[(summary.gain==gain)&(summary.decoder==decoder)&(summary.delays==delays)]
        assert len(match)==1
        return match.iloc[0]

    main=result(.90)
    short,ninth=result(.90,delays="1"),result(.90,delays="9")
    rich,template=result(.90,"shared_all_rows"),result(.90,"group_average")
    expanded=sensitivity[sensitivity.delays=="1-10"].set_index(["map_key","decoder"])
    expanded_shared=expanded.loc[10,"shared"]
    expanded_template=expanded.loc[10,"group_average"]
    alternate=expanded.xs("shared",level="decoder").drop(10).sort_index()
    expanded_lag9=sensitivity[(sensitivity.delays=="9")&(sensitivity.decoder=="shared")&
                              (sensitivity.map_key==10)].iloc[0]
    medians=scores[(scores.gain==.90)&scores.decoder.isin(["individual","shared"])]
    pair=medians.groupby(["participant","decoder"]).normalized_mse.mean().unstack()
    person_loss=pair.shared-pair.individual
    rich_n=int(scores[(scores.gain==.90)&(scores.decoder=="shared_all_rows")].fit_rows.iloc[0])
    runinfo=json.loads((RESULTS/"run_info.json").read_text())

    doc=Canvas(str(RESULTS/"feasibility-report.pdf"),pagesize=letter)
    doc.setTitle("Transferring a memory readout across Lausanne connectomes")
    page_frame(doc,1,"Lausanne structural networks and simulated memory")
    y=paragraph(doc,"Transferring a memory readout<br/>across Lausanne connectomes",62,26,30,"Times-Bold")
    y=paragraph(doc,f"On unseen wiring maps, one shared readout recovered the previous input well. "
                    f"At nine steps, its average R² was {ninth.mean_r2:.3f}, compared with "
                    f"{ninth.mean_own_r2:.3f} after calibrating the readout to each network.",
                y+9,12.7,17.4,"Times-Roman",TEAL)
    y+=13
    rule(doc,y)
    y+=8
    renderPDF.draw(figure,doc,LEFT,PAGE_HEIGHT-y-figure.height)
    y+=figure.height+1
    y=paragraph(doc,"Each point averages one score per held-out participant. R² = 0 is the "
                    "error of predicting the test-target mean. Both readouts use equal fitting budgets.",
                y,9.7,13.6,color=MUTED)
    y=heading(doc,"How much did transfer cost?",y+15)
    y=paragraph(doc,f"Averaged across ten delays, normalized mean squared error was "
                    f"{main.mean_normalized_mse:.3f} for the frozen readout and {main.mean_own_normalized_mse:.3f} "
                    f"for individual calibration. The paired excess error was <b>{main.nmse_loss:.3f}</b> "
                    f"(descriptive 95% interval {main.nmse_loss_ci_low:.3f} to {main.nmse_loss_ci_high:.3f}). "
                    f"Mean R² was {main.mean_r2:.3f} versus {main.mean_own_r2:.3f}, a loss of "
                    f"{main.r2_loss:.3f} R² points ({main.r2_loss_ci_low:.3f} to {main.r2_loss_ci_high:.3f}).",y)
    y=paragraph(doc,f"All {len(person_loss)} participants had lower average error after individual calibration. "
                    f"The median participant's ten-delay excess error was {person_loss.median():.3f}. "
                    f"At one step, the frozen readout retained R² = {short.mean_r2:.3f}, while the "
                    f"individual readouts approached 1.",y+8)
    y=paragraph(doc,f"With a wider range of ridge penalties in a post hoc check, mean R² reached "
                    f"{expanded_shared.mean_r2:.3f} for the shared decoder and "
                    f"{expanded_shared.mean_own_r2:.3f} for individual calibration. The gap persisted. "
                    f"Changing the input regions also preserved its direction (page 2).",y+8)
    assert y<738
    doc.showPage()

    page_frame(doc,2,"Model, comparisons and interpretation")
    y=paragraph(doc,"What was held out",62,23,28,"Times-Bold")
    y=paragraph(doc,"The released scale-1 data supply 70 weighted networks with 68 cortical regions "
                    "each. We removed diagonal weights, scaled each matrix to the specified spectral "
                    "radius, and simulated a tanh echo-state network. Four positive and four negative "
                    "input weights drove eight fixed regions with scalar uniform noise. The readout used "
                    "the other 60 states.",y+9)
    y=paragraph(doc,"Five participant folds kept 14 test networks out of each shared fit. The shared "
                    "readout used 42 labeled time points from each of the 56 training participants "
                    "(2,352 total). The individual reference used 2,352 independent training points on "
                    "its own network. We selected ridge penalties on validation sequences and tested both "
                    "decoders on the same four new 512-step sequences per person after 200 warm-up "
                    "steps per sequence.",y+7)
    y=heading(doc,"Sensitivity to gain and training data",y+12)
    columns=[(LEFT,"Readout"),(337,"Radius"),(429,"Mean R²"),(RIGHT,"Excess error")]
    for x,label in columns:
        doc.setFont("Helvetica-Bold",9.2);doc.setFillColor(HexColor(MUTED))
        if x==LEFT:
            doc.drawString(x,PAGE_HEIGHT-y-11,label)
        else:
            doc.drawRightString(x,PAGE_HEIGHT-y-11,label)
    y+=16
    for label,record in [("Frozen shared",result(.70)),("Frozen shared (primary)",main),
                         ("Frozen shared",result(1.)),("Frozen shared, all training rows",rich),
                         ("Trained on average wiring",template)]:
        rule(doc,y)
        doc.setFont("Helvetica",10)
        doc.setFillColor(HexColor(INK))
        doc.drawString(LEFT,PAGE_HEIGHT-y-14,label)
        for (x,_),val in zip(columns[1:],[f"{record.gain:.2f}",f"{record.mean_r2:.3f}",f"{record.nmse_loss:.3f}"]):
            doc.drawRightString(x,PAGE_HEIGHT-y-14,val)
        y+=21
    y=paragraph(doc,f"Excess error is the paired MSE difference divided by the input variance (1/3). "
                    f"Using {rich_n:,} pooled rows lifted mean R² to {rich.mean_r2:.3f}. "
                    f"The decoder trained on average wiring predicted worse than the test-target mean "
                    f"(negative R²) under the original tuning rule.",y+4,10.3,14.1)
    y=heading(doc,"Post hoc regularization and input-map checks",y+9)
    y=paragraph(doc,f"In all five folds the shared and average-wiring fits selected the strongest "
                    f"allowed penalty, and individual fits selected the weakest. Widening the grid "
                    f"raised the average-wiring R² from {template.mean_r2:.3f} to {expanded_template.mean_r2:.3f}. "
                    f"With the original input map, the shared readout still lost {expanded_shared.r2_loss:.3f} "
                    f"R² points ({expanded_shared.r2_loss_ci_low:.3f} to {expanded_shared.r2_loss_ci_high:.3f}). "
                    f"At delay 9, shared and individual R² were {expanded_lag9.mean_r2:.3f} and "
                    f"{expanded_lag9.mean_own_r2:.3f}. The two alternative eight-region input maps "
                    f"gave losses of {alternate.r2_loss.iloc[0]:.3f} and {alternate.r2_loss.iloc[1]:.3f} R² points. "
                    f"Both used the same 70 participants and test sequences. Individual fits used "
                    f"direct singular value decomposition to stabilize weakly penalized estimates.",y,10.3,14.1)
    y=heading(doc,"Scope of inference",y+10)
    y=paragraph(doc,"The intervals resample 70 paired participant means 10,000 times, conditional on "
                    "the fitted folds and streams. The readouts had full numerical access to noiseless "
                    "simulated states. The experiment leaves anatomical specificity, repeat-scan "
                    "reliability and associations with human memory open.",y,10.3,14.1)
    y=paragraph(doc,"Data and software. "+link("Griffa, Alemán-Gómez and Hagmann, Lausanne release",
                    "https://zenodo.org/records/2872624")+" · "+link("conn2res toolbox",
                    "https://doi.org/10.1038/s41467-024-44900-4")
                    +f". See PROTOCOL.md, DIAGNOSTICS.md and run_feasibility.py. Tables in results/ "
                    f"include summary.csv and sensitivity_summary.csv. Split and map seed "
                    f"{runinfo['seed']}.",y+9,9.0,12.6,color=MUTED)
    assert y<741,y
    doc.showPage()
    doc.save()


def main():
    scores=pd.read_csv(RESULTS/"participant_delay_scores.csv")
    summary=pd.read_csv(RESULTS/"summary.csv",dtype={"delays":str})
    sensitivity=pd.read_csv(RESULTS/"sensitivity_summary.csv",dtype={"delays":str})
    figure=make_figure(scores,summary)
    report(figure,scores,summary,sensitivity)
    # The report embeds vector graphics; the separate export is a 300-dpi PNG.
    with tempfile.TemporaryDirectory() as directory:
        vector=Path(directory)/"figure.pdf"
        renderPDF.drawToFile(figure,str(vector))
        subprocess.run(["pdftoppm","-r","300","-png","-singlefile",str(vector),
                        str(RESULTS/"recall-by-delay")],check=True)


if __name__ == "__main__":
    main()
