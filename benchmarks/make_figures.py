"""Render the README figures from the benchmark result/curve JSON files.

Kept separate from the benchmark scripts so the (expensive) fits run once and
figure design can be iterated freely. Every figure is emitted twice — a light
and a dark variant stepped from the same palette — so the README can serve the
right one via ``<picture media="(prefers-color-scheme: dark)">``.

Design rules applied (see the project's data-viz guidance):
  * Forms: DOT PLOT for the head-to-head (see below); MULTI-LINE for the ATP
    skill curves (the series *are* the subject); SMALL MULTIPLES for NBA
    franchises (5 converging lines would be spaghetti).
  * Why dots and not bars for the comparison: every system scores between 0.60
    and 0.69, so a zero-baseline bar chart spends all its width on the 0.0-0.6
    stretch nobody is comparing and renders a 4% quality gap as six visually
    identical bars. A dot encodes value by POSITION, which makes cropping the
    axis honest — there is no bar length to misread proportionally — and the
    differences legible. Panels keep independent scales on purpose: comparing
    football's 3-way log-loss against the 2-way ones would be meaningless.
  * Marks: >= 7px dots with a 1.2px surface ring, 2px lines with round caps,
    hairline solid recessive leaders and gridlines.
  * Text never wears a series colour; identity comes from the coloured mark
    beside the label. Values are direct-labelled next to each dot; the README
    carries the same numbers as tables (the table-view twin).
  * Colour separation in the comparison is carried by LIGHTNESS within one hue,
    which is what keeps the three marks distinct under every CVD type. Note the
    dot palette is deliberately not the shared THEMES pair: dark mode's
    ``accent_soft`` was tuned to recede behind 24px bars and collapses into the
    accent at dot size, so the dots step *lighter* than the accent instead.
  * The line-chart slots were validated with the palette validator in both modes
    (adjacent CVD dE 9.1 light / 8.4 dark, normal-vision 19.6 / 19.3). Three
    light slots sit under 3:1 vs the surface, so the relief rule applies: direct
    labels + the README tables.

Run:  uv run --with pandas --with matplotlib python benchmarks/make_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DPI = 160

# --------------------------------------------------------------------------- #
# Theme parameters (palette values; dark is *stepped*, never an auto-flip)
# --------------------------------------------------------------------------- #
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "accent": "#2a78d6",  # categorical slot 1 (blue)
        "accent_soft": "#86b6ef",  # blue step 250 — same hue, second shade
        "context": "#b4b2ab",  # recessive fill for reference/baseline bars
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "accent": "#3987e5",
        # A deep step of the SAME hue (blue 600). Adjacent steps (e.g. 500)
        # were indistinguishable from the accent on the dark surface, which
        # killed the emphasis; this sits ~200 ramp steps away, mirroring the
        # light theme's accent/soft gap. Like light mode's soft step it is
        # deliberately low-contrast (recessive), with the relief rule covered by
        # the direct labels and the README tables.
        "accent_soft": "#184f95",
        "context": "#5c5b56",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
    },
}

SANS = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"]


def _apply_rc(t: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": SANS,
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "text.color": t["ink"],
            "axes.labelcolor": t["ink_secondary"],
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "axes.edgecolor": t["axis"],
            "axes.linewidth": 1.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.titlesize": 10,
            "figure.dpi": DPI,
        }
    )


def _clean_axes(ax, t: dict, *, grid_axis="y") -> None:
    """Hairline solid recessive grid; only the baseline spine kept."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(
        axis=grid_axis,
        color=t["grid"],
        linewidth=1.0,
        linestyle="-",
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


class Stack:
    """Top-down, point-based vertical layout in figure coordinates.

    Deterministic placement (``fig.add_axes``) rather than tight/constrained
    layout: the bar-thickness and corner-radius specs are in *pixels*, so the
    axes rectangle must be final before those are converted to data units.
    """

    def __init__(self, height_in: float):
        self.total = height_in * 72.0
        self.cursor = self.total

    def skip(self, pt: float) -> None:
        self.cursor -= pt

    def band(self, pt: float) -> tuple[float, float]:
        """Reserve `pt` of height; return (y0, height) as figure fractions."""
        self.cursor -= pt
        return self.cursor / self.total, pt / self.total

    def at(self, offset_pt: float = 0.0) -> float:
        """Current cursor as a figure fraction (optionally offset upward)."""
        return (self.cursor + offset_pt) / self.total


# --------------------------------------------------------------------------- #
# Figure 1 — model comparison (emphasis bars, one panel per benchmark)
# --------------------------------------------------------------------------- #
# role: "flagship" (accent) | "variant" (same hue, softer) | "context" (gray)
# The head-to-head panels read `versus_*.json`: every model in a panel was
# fitted on the same training prefix, tuned on the same validation season and
# scored on the same test season, so the bars inside a panel are directly
# comparable. Rows are sorted by log-loss at render time -- the ranking comes
# out of the data, it is not authored here.
CONSTANT_MODELS = {"coin_flip"}

PANELS = [
    {
        "key": "nba",
        "title": "NBA 2018-19  ·  1,312 games, trained on 41,279  ·  2-way log-loss",
        "note": "WHR vs the reference implementations, plus FiveThirtyEight's "
        "own published pre-game probabilities on the identical games",
        "rows": [
            ("whr", "WHR", "flagship"),
            ("kickscore", "KickScore", "variant"),
            ("ttt", "TrueSkill Through Time", "variant"),
            ("fte_raptor", "538 RAPTOR", "context"),
            ("fte_elo", "538 Elo", "context"),
            ("base_rate", "home-rate baseline", "context"),
        ],
    },
    {
        "key": "tennis",
        "title": "ATP tennis 2014  ·  2,816 matches, trained on 44,405  ·  2-way log-loss",
        "note": "the setup TrueSkill Through Time was published on",
        "rows": [
            ("whr", "WHR", "flagship"),
            ("kickscore", "KickScore", "variant"),
            ("ttt", "TrueSkill Through Time", "variant"),
            ("coin_flip", "coin flip", "context"),
        ],
    },
    {
        "key": "football",
        "title": "Football big-5 2022-23  ·  1,826 matches, 25% draws  ·  3-way log-loss",
        "note": "three-outcome prediction: WHR's Davidson draw model vs each "
        "library's own draw handling",
        "rows": [
            ("whr", "WHR (Davidson)", "flagship"),
            ("kickscore", "KickScore (ternary)", "variant"),
            ("ttt", "TrueSkill Through Time", "variant"),
            ("base_rate", "H/D/A base rate", "context"),
        ],
    },
]


def _load(name: str) -> dict:
    with open(os.path.join(RESULTS, f"versus_{name}.json")) as f:
        return json.load(f)


def figure_comparison(theme: str) -> None:
    t = THEMES[theme]
    _apply_rc(t)
    data = {p["key"]: _load(p["key"]) for p in PANELS}

    ROW_PT, HEAD_PT, GAP_PT = 25.0, 30.0, 16.0
    height_in = (
        46.0
        + sum(HEAD_PT + len(p["rows"]) * ROW_PT for p in PANELS)
        + GAP_PT * len(PANELS)
        + 20.0
    ) / 72.0
    fig = plt.figure(figsize=(8.6, height_in))
    stack = Stack(height_in)
    left, width = 0.205, 0.79

    stack.skip(20.0)
    fig.text(
        0.014,
        stack.at(),
        "Head-to-head on real competition data — predictive log-loss (lower is better)",
        ha="left",
        va="top",
        fontsize=12.5,
        color=t["ink"],
        fontweight="bold",
    )
    stack.skip(26.0)

    # Dot-plot-specific palette, deliberately not the shared THEMES values.
    # ``accent_soft`` on dark is #184f95, a deep navy picked to recede behind
    # 24px-tall bars; shrunk to an 8px dot on a near-black surface it collapses
    # into the accent and the legend swatches become indistinguishable. Dots
    # need the second shade to separate by going *lighter* than the accent, not
    # darker. Separation here is carried by lightness, which is what keeps the
    # three marks distinct under every CVD type as well.
    colour = {
        "light": {
            "flagship": t["accent"],
            "variant": "#86b6ef",
            "context": t["context"],
        },
        "dark": {
            "flagship": t["accent"],
            "variant": "#a8cbf5",
            "context": "#8a8880",
        },
    }[theme]

    for panel in PANELS:
        models = data[panel["key"]]["models"]
        rows = [r for r in panel["rows"] if r[0] in models]
        rows.sort(key=lambda r: models[r[0]]["log_loss"])  # best first, from data
        vals = [models[k]["log_loss"] for k, _, _ in rows]
        # A constant-probability model has no accuracy worth printing: at exactly
        # p = 0.5 every prediction is a tie, so the number that comes out (47.4%)
        # is just which way argmax breaks ties, and reads as a bug.
        accs = [
            None if k in CONSTANT_MODELS else models[k]["accuracy"] for k, _, _ in rows
        ]

        # panel header: title then note, both above the axes
        fig.text(
            left - 0.19,
            stack.at(-2.0),
            panel["title"],
            ha="left",
            va="top",
            fontsize=9.5,
            color=t["ink"],
            fontweight="bold",
        )
        fig.text(
            left - 0.19,
            stack.at(-15.0),
            panel["note"],
            ha="left",
            va="top",
            fontsize=8,
            color=t["muted"],
        )
        stack.skip(HEAD_PT)

        y0, h = stack.band(len(rows) * ROW_PT)
        ax = fig.add_axes((left, y0, width, h))
        # Dots, not bars, and an axis cropped to the values. Every system here
        # scores between 0.60 and 0.69, so a zero-baseline bar chart spends all
        # its width on the 0.0-0.6 stretch nobody is comparing and renders a
        # 12% quality gap as six visually identical bars. A dot encodes value by
        # POSITION, so cropping the axis is honest -- there is no bar length to
        # read proportionally -- and the differences become legible. The panels
        # deliberately keep independent scales: comparing football's 3-way
        # log-loss against the 2-way ones would be meaningless.
        span = max(vals) - min(vals)
        pad = max(span * 0.16, max(vals) * 0.006)
        x_lo, x_hi = min(vals) - pad, max(vals) + pad
        ax.set_xlim(x_lo, x_hi + span * 1.15 + pad)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks(range(len(rows)))
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        fig.canvas.draw()  # axes rect is final -> px specs convert correctly

        best_v = min(vals)
        acc_x = x_hi + span * 0.34  # clear of the widest value label
        for i, (v, (_k, _lab, role)) in enumerate(zip(vals, rows, strict=True)):
            # a hairline from the panel's best score to this dot: the reader sees
            # the deficit as a length without the axis pretending to start at 0
            ax.plot(
                [best_v, v],
                [i, i],
                color=t["grid"],
                linewidth=1.0,
                solid_capstyle="butt",
                zorder=1,
            )
            ax.plot(
                [v],
                [i],
                marker="o",
                markersize=8.5 if role == "flagship" else 7.0,
                color=colour[role],
                markeredgecolor=t["surface"],
                markeredgewidth=1.2,
                zorder=3,
            )
            ax.text(
                v + pad * 0.7,
                i,
                f"{v:.3f}",
                va="center",
                ha="left",
                fontsize=8.6,
                color=t["ink"],
                fontweight="bold" if role == "flagship" else "normal",
            )
            if accs[i] is not None:
                # fixed column, not offset from each dot: trailing the marker
                # left the secondary numbers in a ragged staircase
                ax.text(
                    acc_x,
                    i,
                    f"{accs[i] * 100:.1f}% acc",
                    va="center",
                    ha="left",
                    fontsize=7.6,
                    color=t["muted"],
                )
        # mark the panel's best score, so "how far behind" has a visible origin
        ax.axvline(best_v, color=t["axis"], linewidth=1.0, zorder=2)

        ax.set_yticklabels(
            [lab for _, lab, _ in rows], fontsize=9, color=t["ink_secondary"]
        )
        for tick, (_k, _lab, role) in zip(ax.get_yticklabels(), rows, strict=True):
            if role == "flagship":
                tick.set_color(t["ink"])
                tick.set_fontweight("bold")
        stack.skip(GAP_PT)

    # swatches mirror the marks: same colours, same round marker
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=colour[role])
        for role in ("flagship", "variant", "context")
    ]
    leg = fig.legend(
        handles,
        [
            "WHR (this library)",
            "reference implementation",
            "published probs / baseline",
        ],
        loc="lower left",
        bbox_to_anchor=(0.012, 0.004),
        ncol=3,
        frameon=False,
        fontsize=8.2,
        handletextpad=0.5,
        columnspacing=1.6,
    )
    for txt in leg.get_texts():
        txt.set_color(t["ink_secondary"])

    out = os.path.join(RESULTS, f"bench_comparison_{theme}.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out)}")


# --------------------------------------------------------------------------- #
# Figure 2 — ATP skill over time (multi-line, direct end labels + legend)
# --------------------------------------------------------------------------- #
# The three subjects of the story carry the hues; everyone else is context.
# Three slots are the all-pairs-validated ceiling for this palette (see the
# comment in figure_tennis).
EMPHASISED = ["Roger Federer", "Rafael Nadal", "Novak Djokovic"]


def _declutter(ys: list[float], min_gap: float) -> list[float]:
    """Push labels apart just enough to stop collisions, preserving order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for pos, i in enumerate(order):
        if pos == 0:
            continue
        prev = out[order[pos - 1]]
        if out[i] - prev < min_gap:
            out[i] = prev + min_gap
    return out


def figure_tennis(theme: str) -> None:
    t = THEMES[theme]
    _apply_rc(t)
    with open(os.path.join(RESULTS, "tennis_curves.json")) as f:
        curves = json.load(f)["curves"]
    if not curves:
        print("(no tennis curves yet)")
        return

    height_in = 4.6
    fig = plt.figure(figsize=(8.6, height_in))
    stack = Stack(height_in)
    stack.skip(18.0)
    fig.text(
        0.014,
        stack.at(),
        "WHR skill over time — ATP singles 2000-2015",
        ha="left",
        va="top",
        fontsize=12,
        color=t["ink"],
        fontweight="bold",
    )
    stack.skip(15.0)
    fig.text(
        0.014,
        stack.at(),
        "Fitted on 48,335 matches, 1,948 players. Display rating = WHR elo + 1500 "
        "(only differences are meaningful).",
        ha="left",
        va="top",
        fontsize=8.2,
        color=t["muted"],
    )
    stack.skip(14.0)

    names = list(curves)
    x_max = max(pt[0] for c in curves.values() for pt in c)
    y0, h = stack.band(stack.cursor - 46.0)
    ax = fig.add_axes((0.072, y0, 0.925, h))
    # right-hand padding must fit the longest (bold) end label without clipping
    ax.set_xlim(1999.6, x_max + 3.3)
    # integer year ticks only: the x range is padded to make room for the end
    # labels, so letting matplotlib auto-tick produces meaningless "2002.5"s.
    x_min_year = int(min(pt[0] for c in curves.values() for pt in c))
    ax.set_xticks([y for y in range(x_min_year, int(x_max) + 1) if y % 5 == 0])
    _clean_axes(ax, t, grid_axis="y")

    # EMPHASIS, not six equal categories. A spaghetti multi-line chart puts
    # *every* pair of series visually adjacent, so it must clear the validator's
    # all-pairs gates -- and past three slots this palette cannot (magenta vs
    # aqua collapses to CVD dE 1.6 for deuteranopes; yellow vs orange sits at
    # 10.6 for normal vision, under the hard floor of 15). Three slots DO pass
    # all-pairs in both modes, so the three subjects of the story carry the hues
    # and the rest become recessive gray context, each still directly labelled.
    ends = []
    for name in names:
        rank = EMPHASISED.index(name) if name in EMPHASISED else None
        colour = t["series"][rank] if rank is not None else t["context"]
        width = 2.0 if rank is not None else 1.3
        xs = [p[0] for p in curves[name]]
        ys = [p[1] for p in curves[name]]
        ax.plot(
            xs,
            ys,
            color=colour,
            linewidth=width,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3 if rank is not None else 2,
        )
        # >= 8px end marker with a 2px surface ring
        ax.plot(
            xs[-1],
            ys[-1],
            marker="o",
            markersize=5.0,
            color=colour,
            markeredgecolor=t["surface"],
            markeredgewidth=2.0,
            zorder=4,
            linestyle="",
        )
        ends.append((xs[-1], ys[-1], name, rank))

    # direct end labels with leader lines where curves converge
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    label_y = _declutter([e[1] for e in ends], min_gap=span * 0.052)
    for (x_end, y_end, name, rank), y_lab in zip(ends, label_y, strict=True):
        ax.plot(
            [x_end, x_end + 0.42],
            [y_end, y_lab],
            color=t["axis"],
            linewidth=1.0,
            zorder=2,
        )
        ax.text(
            x_end + 0.55,
            y_lab,
            name,
            va="center",
            ha="left",
            fontsize=8.4,
            color=t["ink"] if rank is not None else t["muted"],
            fontweight="bold" if rank is not None else "normal",
        )

    ax.set_xlabel("year", fontsize=8.5, labelpad=5)
    ax.set_ylabel("rating", fontsize=8.5)

    present = [n for n in EMPHASISED if n in names]
    handles = [
        plt.Line2D([], [], color=t["series"][EMPHASISED.index(n)], linewidth=2.0)
        for n in present
    ]
    labels = list(present)
    if any(n not in EMPHASISED for n in names):
        handles.append(plt.Line2D([], [], color=t["context"], linewidth=1.3))
        labels.append("others (context)")
    leg = ax.legend(
        handles,
        labels,
        loc="lower left",
        ncol=4,
        frameon=False,
        fontsize=7.8,
        handlelength=1.5,
        columnspacing=1.4,
    )
    for txt in leg.get_texts():
        txt.set_color(t["ink_secondary"])

    out = os.path.join(RESULTS, f"tennis_history_{theme}.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out)}")


# --------------------------------------------------------------------------- #
# Figure 3 — NBA franchise eras (small multiples, single hue)
# --------------------------------------------------------------------------- #
NBA_LABEL = {
    "BOS": "Boston Celtics",
    "LAL": "Los Angeles Lakers",
    "CHI": "Chicago Bulls",
    "GSW": "Golden State Warriors",
    "SAS": "San Antonio Spurs",
}


def figure_nba(theme: str) -> None:
    t = THEMES[theme]
    _apply_rc(t)
    with open(os.path.join(RESULTS, "nba_curves.json")) as f:
        curves = json.load(f)["curves"]
    if not curves:
        print("(no nba curves yet)")
        return

    names = [k for k in ["BOS", "LAL", "CHI", "GSW", "SAS"] if k in curves]
    lo = min(p[1] for c in curves.values() for p in c)
    hi = max(p[1] for c in curves.values() for p in c)
    pad = (hi - lo) * 0.10
    x_lo = min(p[0] for c in curves.values() for p in c)
    x_hi = max(p[0] for c in curves.values() for p in c)

    PANEL_PT, HEAD_PT, GAP_PT, AXIS_PT = 60.0, 16.0, 8.0, 38.0
    height_in = (
        44.0 + len(names) * (HEAD_PT + PANEL_PT) + GAP_PT * (len(names) - 1) + AXIS_PT
    ) / 72.0
    fig = plt.figure(figsize=(8.6, height_in))
    stack = Stack(height_in)
    left, width = 0.068, 0.925

    stack.skip(18.0)
    fig.text(
        0.014,
        stack.at(),
        "WHR recovers NBA franchise eras — full history, 1947-2020",
        ha="left",
        va="top",
        fontsize=12,
        color=t["ink"],
        fontweight="bold",
    )
    stack.skip(15.0)
    fig.text(
        0.014,
        stack.at(),
        "One panel per franchise, shared scale. Display rating = WHR elo + 1500.",
        ha="left",
        va="top",
        fontsize=8.2,
        color=t["muted"],
    )
    stack.skip(11.0)

    last_ax = None
    for name in names:
        fig.text(
            left,
            stack.at(-3.0),
            NBA_LABEL.get(name, name),
            ha="left",
            va="top",
            fontsize=9,
            color=t["ink"],
            fontweight="bold",
        )
        stack.skip(HEAD_PT)
        y0, h = stack.band(PANEL_PT)
        ax = fig.add_axes((left, y0, width, h))
        last_ax = ax

        xs = [p[0] for p in curves[name]]
        ys = [p[1] for p in curves[name]]
        ax.set_xlim(x_lo - 0.5, x_hi + 0.5)
        ax.set_ylim(lo - pad, hi + pad)
        _clean_axes(ax, t, grid_axis="y")
        ax.spines["bottom"].set_visible(False)
        ax.set_yticks([1400, 1600, 1800])
        decades = [d for d in range(1950, 2021, 10) if x_lo - 0.5 <= d <= x_hi + 0.5]
        ax.set_xticks(decades)
        # only the bottom panel carries the shared x-axis band
        ax.tick_params(axis="x", labelbottom=(name == names[-1]), length=0)
        ax.plot(
            xs,
            ys,
            color=t["accent"],
            linewidth=1.6,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
        # area wash at ~10% opacity (never a saturated block)
        ax.fill_between(xs, lo - pad, ys, color=t["accent"], alpha=0.10, zorder=2)

        # selective direct label: the extreme only
        i_peak = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot(
            xs[i_peak],
            ys[i_peak],
            marker="o",
            markersize=5.0,
            color=t["accent"],
            markeredgecolor=t["surface"],
            markeredgewidth=2.0,
            linestyle="",
            zorder=4,
        )
        # flip the label inside when the peak sits near the right edge
        near_edge = xs[i_peak] > x_hi - 14.0
        ax.annotate(
            f"peak {int(round(xs[i_peak]))}",
            (xs[i_peak], ys[i_peak]),
            textcoords="offset points",
            xytext=(-8, 0) if near_edge else (8, 0),
            fontsize=7.6,
            color=t["muted"],
            va="center",
            ha="right" if near_edge else "left",
        )
        stack.skip(GAP_PT)

    if last_ax is not None:
        last_ax.set_xlabel("season", fontsize=8.5, labelpad=6)

    out = os.path.join(RESULTS, f"nba_history_{theme}.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out)}")


def main() -> None:
    for theme in ("light", "dark"):
        figure_comparison(theme)
        if os.path.exists(os.path.join(RESULTS, "tennis_curves.json")):
            figure_tennis(theme)
        if os.path.exists(os.path.join(RESULTS, "nba_curves.json")):
            figure_nba(theme)


if __name__ == "__main__":
    main()
