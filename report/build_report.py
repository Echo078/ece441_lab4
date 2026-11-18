"""Builds lab4_report.html and lab4_report.pdf from report_template.html.

Run from anywhere:  python build_report.py
Needs report/results.json and report/img/ (run ../lab4_analysis.py first) and
Microsoft Edge or Google Chrome for the PDF.

The template uses {{key}} placeholders: dotted paths into results.json
(e.g. {{runs.run1.main.n_circles}}) or the derived values built in derived_values().
"""

import html
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
TEMPLATE = os.path.join(HERE, "report_template.html")
RESULTS = os.path.join(HERE, "results.json")
OUT_HTML = os.path.join(HERE, "lab4_report.html")
OUT_PDF = os.path.join(HERE, "lab4_report.pdf")
PLACEHOLDER = re.compile(r"\{\{([A-Za-z0-9_.]+)\}\}")
P_THRESHOLD = 0.05
WINDOW_LABELS = ("main", "extended")

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def find_browser() -> str:
    for path in BROWSERS:
        if os.path.exists(path):
            return path
    found = shutil.which("msedge") or shutil.which("chrome")
    if found is None:
        sys.exit("Edge or Chrome not found; open lab4_report.html and print it to PDF instead.")
    return found


def fmt_p(p: float, n_perm: int) -> str:
    return f"&lt;&nbsp;{1 / n_perm:.3f}" if p == 0 else f"{p:.3f}"


def datasets(res: Dict) -> List[Tuple[str, str, Dict]]:
    """(key, column title, block holder) for every run and the pooled data."""
    out = [(tag, f"Run {i + 1}", run) for i, (tag, run) in enumerate(res["runs"].items())]
    if res.get("pooled"):
        out.append(("pooled", "Pooled", res["pooled"]))
    return out


def stats_table(res: Dict, label: str) -> str:
    """Channel rows; for each run and the pooled data: cluster window, sum of t, p (corrected p)."""
    sets = datasets(res)
    n_perm = res["n_permutations"]
    top = "<tr><th rowspan=\"2\">Channel</th>" + "".join(f"<th colspan=\"3\" class=\"grp\">{title}</th>" for _, title, _ in sets) + "</tr>"
    sub = "<tr>" + "<th class=\"n\">cluster (ms)</th><th class=\"n\">&Sigma;t</th><th class=\"n\">p (corr.)</th>" * len(sets) + "</tr>"
    rows = []
    channels = [r["channel"] for r in sets[0][2][label]["channels"]]
    for i, channel in enumerate(channels):
        cells = [f"<td>{channel}</td>"]
        for _, _, holder in sets:
            r = holder[label]["channels"][i]
            window = "&ndash;" if r["cluster_start_ms"] is None else f"{r['cluster_start_ms']} to {r['cluster_end_ms']}"
            cls = ' class="n sig"' if r["significant"] else ' class="n"'
            cells.append(f"<td class=\"n\">{window}</td><td class=\"n\">{r['cluster_stat']:.0f}</td>")
            cells.append(f"<td{cls}>{fmt_p(r['p_value'], n_perm)} <span class=\"corr\">({fmt_p(r['p_value_corrected'], n_perm)})</span></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    counts = "<tr class=\"foot\"><td>epochs</td>" + "".join(
        f"<td colspan=\"3\" class=\"n\">{h[label]['n_circles']} circles / {h[label]['n_squares']} squares</td>" for _, _, h in sets
    ) + "</tr>"
    return "<table class=\"stats\">" + top + sub + "".join(rows) + counts + "</table>"


def quality_table(res: Dict) -> str:
    runs = list(res["runs"].items())
    head = "<tr><th></th>" + "".join(f"<th class=\"n\">Run {i + 1} ({html.escape(r['recording'])})</th>" for i, (_, r) in enumerate(runs)) + "</tr>"

    def row(title: str, values: List[str]) -> str:
        return f"<tr><td>{title}</td>" + "".join(f"<td class=\"n\">{v}</td>" for v in values) + "</tr>"

    lines = [
        row("Duration (s)", [f"{r['duration_s']}" for _, r in runs]),
        row("D13 column", [f"<code>{r['photoresistor_column']}</code>" for _, r in runs]),
        row("Photoresistor onsets", [str(r["n_onsets_total"]) for _, r in runs]),
        row("Sync onsets detected (of 5 flashes)", [str(r["sync_onsets_detected"]) for _, r in runs]),
        row("Median TTL pulse (ms)", [str(r["median_flash_ms"]) for _, r in runs]),
        row("Trials matched / 90", [str(r["n_trials_matched"]) for _, r in runs]),
        row("Trials missed", [", ".join(str(k + 1) for k in r["missed_trials"]) or "none" for _, r in runs]),
        row("Cropped window (s)", [f"{r['crop_start_s']}&ndash;{r['crop_stop_s']}" for _, r in runs]),
        row("find_events in crop", [str(r["n_events_in_crop"]) for _, r in runs]),
        row("Lost samples / counter resets", [f"{r['lost_samples']} / {r['counter_resets']}" for _, r in runs]),
    ]
    for ch in runs[0][1]["channel_std_uv"]:
        lines.append(row(f"SD {ch} (&micro;V)", [f"{r['channel_std_uv'][ch]:.1f}" for _, r in runs]))
    return "<table class=\"quality\">" + head + "".join(lines) + "</table>"


def significant_figure(key: str, title: str, label: str, block: Dict, fig_no: int) -> str:
    sig = block["significant_channels"]
    name = f"{key}_{label}_erp_significant.png"
    if not sig or not os.path.exists(os.path.join(HERE, "img", name)):
        return f"<p class=\"cap\"><b>Figure {fig_no}</b> is not produced for {title}: no channel reached p &lt; {P_THRESHOLD}.</p>"
    return (
        f'<figure class="narrow"><img src="img/{name}" alt=""><figcaption><b>Figure {fig_no}.</b> {title}: the channels with a '
        f"permutation p-value below {P_THRESHOLD} ({', '.join(sig)}), mean &plusmn; SEM z-scored response to circles (red) and "
        "squares (blue); the shaded band is the channel&rsquo;s largest cluster.</figcaption></figure>"
    )


def mne_max_difference(res: Dict) -> float:
    diffs = [
        abs(r["p_value"] - r["p_value_mne"])
        for _, _, holder in datasets(res)
        for label in WINDOW_LABELS
        for r in holder[label]["channels"]
        if r["p_value_mne"] is not None
    ]
    return max(diffs) if diffs else 0.0


def read_code(name: str) -> str:
    with open(os.path.join(CODE_DIR, name), encoding="utf-8") as handle:
        return html.escape(handle.read())


def derived_values(res: Dict) -> Dict[str, str]:
    values = {
        "BANNER": (
            '<div class="demo">DEMO &mdash; built from a <b>synthetic</b> recording (lab4_analysis.py --demo). '
            "Every number and figure will change with the real data.</div>"
            if res["is_demo"]
            else ""
        ),
        "TABLE_QUALITY": quality_table(res),
        "MNE_MAX_DIFF": f"{mne_max_difference(res):.3f}",
        "CODE_STATS": read_code("lab4_stats.py"),
        "CODE_ANALYSIS": read_code("lab4_analysis.py"),
    }
    for label in WINDOW_LABELS:
        values[f"TABLE_{label.upper()}"] = stats_table(res, label)
    for tag, run in res["runs"].items():
        values[f"MISSED_{tag}"] = ", ".join(str(k + 1) for k in run["missed_trials"]) or "none"
    for key, title, holder in datasets(res):
        for label in WINDOW_LABELS:
            values[f"SIG_{key}_{label}"] = ", ".join(holder[label]["significant_channels"]) or "none"
    return values


def figure_values(res: Dict, numbering: Dict[str, Tuple[str, str, str, int]]) -> Dict[str, str]:
    """{{FIGSIG_<id>}} placeholders: id -> (dataset key, title, window label, figure number)."""
    holders = {key: holder for key, _, holder in datasets(res)}
    return {
        f"FIGSIG_{fid}": significant_figure(key, title, label, holders[key][label], no)
        for fid, (key, title, label, no) in numbering.items()
    }


SIGNIFICANT_FIGURES = {
    "run2_main": ("run2", "Run 2", "main", 7),
    "pooled_main": ("pooled", "Both runs pooled", "main", 9),
    "run2_extended": ("run2", "Run 2, &minus;0.2 to 0.8 s", "extended", 13),
    "pooled_extended": ("pooled", "Both runs pooled, &minus;0.2 to 0.8 s", "extended", 14),
}


def format_value(key: str, value: object, n_perm: int) -> str:
    """p-values with 3 decimals ('<0.001' for 0), other floats compactly."""
    if isinstance(value, float):
        return fmt_p(value, n_perm) if "p_value" in key.rsplit(".", 1)[-1] else f"{value:g}"
    return str(value)


def lookup(res: Dict, derived: Dict[str, str], key: str) -> str:
    """{{a.b.0.c}}: dict keys and list indices; derived values first."""
    if key in derived:
        return derived[key]
    value: object = res
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise KeyError(key)
    return format_value(key, value, res["n_permutations"])


def _missing(res: Dict, derived: Dict[str, str], key: str) -> bool:
    try:
        lookup(res, derived, key)
        return False
    except KeyError:
        return True


def render(template: str, res: Dict) -> str:
    derived = {**derived_values(res), **figure_values(res, SIGNIFICANT_FIGURES)}
    missing = sorted({k for k in PLACEHOLDER.findall(template) if _missing(res, derived, k)})
    if missing:
        sys.exit("Unknown placeholders in the template: " + ", ".join(missing))
    return PLACEHOLDER.sub(lambda m: lookup(res, derived, m.group(1)), template)


def main() -> None:
    if not os.path.exists(RESULTS):
        sys.exit("report/results.json not found: run lab4_analysis.py first.")
    with open(RESULTS, encoding="utf-8") as handle:
        res = json.load(handle)
    with open(TEMPLATE, encoding="utf-8") as handle:
        page = render(handle.read(), res)
    with open(OUT_HTML, "w", encoding="utf-8") as handle:
        handle.write(page)
    subprocess.run(
        [find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--print-to-pdf=" + OUT_PDF, "file:///" + OUT_HTML.replace("\\", "/")],
        check=True,
        timeout=180,
    )
    print("Wrote", OUT_HTML)
    print("Wrote", OUT_PDF)


if __name__ == "__main__":
    main()
