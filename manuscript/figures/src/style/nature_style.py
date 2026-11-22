"""House matplotlib style: Nature figure standards + the Nature-branded palette.

    import sys; sys.path.insert(0, '<repo>/house/figures')
    import nature_style as ns
    ns.apply()
    fig, axes = ns.figure(width='2', height_mm=90, nrows=1, ncols=3)   # 183 mm wide
    ns.panel_letter(axes[0], 'a')
    ns.save(fig, 'fig_name', outdir)        # PDF (TrueType fonts embedded) + 300 dpi PNG + editable SVG

Rules (see FIGURE_RULES.md): Helvetica/Arial, all text 7 pt (panel letters 8 pt bold lowercase), black text,
lines 0.5 to 1.5 pt, no title inside the figure, no gradients, axes labelled with units, colour-blind-safe palette.
Widths: '1' = 89 mm, '1.5' = 120 mm, '2' = 183 mm. Maximum height 247 mm.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

MM = 1 / 25.4
WIDTHS = {"1": 89 * MM, "1.5": 120 * MM, "2": 183 * MM}
_HERE = os.path.dirname(os.path.abspath(__file__))
NAT = json.load(open(os.path.join(_HERE, "nature_palette.json")))["palette"]   # family -> 6 levels (0 lightest)
TXT = "#000000"
# accent colours, level 4 of each family (index 3), plus dark variants (index 4)
BLUE, ORANGE, TEAL, RED, PURPLE, YELLOW, GREEN, OLIVE = (NAT[k][3] for k in
    ("blue", "orange", "teal", "red", "purple", "yellow", "green", "olive"))
BLUE_D, ORANGE_D, TEAL_D, RED_D, PURPLE_D, YELLOW_D, GREEN_D = (NAT[k][4] for k in
    ("blue", "orange", "teal", "red", "purple", "yellow", "green"))
GREY_L, GREY, GREY_D = NAT["grey"][0], NAT["grey"][3], NAT["grey"][5]
CYCLE = [BLUE, ORANGE, TEAL, PURPLE, YELLOW, RED]      # avoid putting RED next to GREEN in one panel


def _sans():
    """Helvetica where matplotlib can reach its bold face; Arial otherwise.

    macOS ships Helvetica only as a .ttc collection. matplotlib before 3.11 reads the first face of a
    collection for every weight, so bold panel letters silently come out regular. Arial is a set of
    separate .ttf files, so on those versions it is the font that keeps bold bold (Nature accepts either)."""
    major, minor = (int(x) for x in matplotlib.__version__.split(".")[:2])
    return ["Helvetica", "Arial", "DejaVu Sans"] if (major, minor) >= (3, 11) else ["Arial", "Helvetica", "DejaVu Sans"]


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": _sans(),
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "figure.titlesize": 7, "text.color": TXT, "axes.labelcolor": TXT,
        "xtick.color": TXT, "ytick.color": TXT, "axes.edgecolor": TXT,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0, "lines.markersize": 3,
        "axes.labelpad": 2.0, "axes.prop_cycle": cycler(color=CYCLE), "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600,
        "mathtext.default": "regular", "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "image.cmap": "viridis",
    })


def figure(width="2", height_mm=80, **kw):
    """New figure of Nature width ('1', '1.5' or '2' column) and height in mm (<= 247)."""
    assert height_mm <= 247, "Nature maximum figure height is 247 mm"
    apply()
    fig, ax = plt.subplots(figsize=(WIDTHS[width], height_mm * MM), constrained_layout=kw.pop("constrained_layout", True), **kw)
    return fig, ax


def panel_letter(ax, letter, x=-0.18, y=1.05):
    """Bold lowercase 8 pt panel letter at the top left of an axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", color=TXT)


def svg_to_pdf(svg_path, pdf_path=None, png_dpi=300):
    """SVG master -> PDF (TrueType only) + PNG with rsvg-convert.

    Cairo writes the space glyph as a Type 3 font, which journal preflight flags. The PDF is therefore drawn
    from a copy of the SVG in which each space inside <text> becomes a tspan offset (0.278 em, the Helvetica
    space); the SVG master keeps its spaces so it stays editable."""
    import re, subprocess, tempfile
    from pathlib import Path
    svg_path = Path(svg_path); pdf_path = Path(pdf_path or svg_path.with_suffix(".pdf"))
    s = svg_path.read_text()
    size = float(re.search(r'font-size="([\d.]+)"', s).group(1)) if 'font-size="' in s else 7.0

    def words(m):
        parts = m.group(2).split(" ")
        return m.group(1) + parts[0] + "".join(f'<tspan dx="{0.278 * size:.2f}">{w}</tspan>' for w in parts[1:]) + "</text>"
    s = re.sub(r"(<text[^>]*>)([^<]*)</text>", words, s)
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as t:
        t.write(s)
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), t.name], check=True)
    subprocess.run(["rsvg-convert", "-f", "png", "-d", str(png_dpi), "-p", str(png_dpi), "-o",
                    str(pdf_path.with_suffix(".png")), t.name], check=True)
    os.unlink(t.name)


def save(fig, stem, outdir):
    """PDF (fonts embedded, vector), 300 dpi PNG for Word, and an editable SVG with live text."""
    from pathlib import Path
    outdir = Path(outdir)
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png", dpi=300)
    fig.savefig(outdir / f"{stem}.svg")
    plt.close(fig)
