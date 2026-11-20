from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, ctx

# ============================================================
# DATA CONNECTIONS (unchanged)
# ============================================================
PROJECT = Path(__file__).resolve().parents[2]
FORECAST_DIR = PROJECT / "models" / "forecast_evaluation"
INVENTORY_DIR = PROJECT / "models" / "inventory_optimization"
DARTS_DIR = PROJECT / "models" / "darts_reconciliation"

COLORS = {
    "navy": "#14243A",
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#059669",
    "orange": "#D97706",
    "red": "#DC2626",
    "background": "#F3F6FA",
    "white": "#FFFFFF",
    "text": "#243247",
    "muted": "#64748B",
    "border": "#E2E8F0"
}

# Friendly check: tell the user exactly which data file is missing
# (the files themselves are read exactly as before, below).
_REQUIRED_FILES = [
    FORECAST_DIR / "prophet_nationwide_actual_vs_forecast.csv",
    FORECAST_DIR / "lightgbm_enhanced_30490_predictions.csv",
    FORECAST_DIR / "lightgbm_enhanced_30490_feature_importance.csv",
    FORECAST_DIR / "fixed_origin_predictions.csv",
    FORECAST_DIR / "combined_prediction_intervals.csv",
    INVENTORY_DIR / "corrected_portfolio_daily_plan.csv",
    INVENTORY_DIR / "corrected_portfolio_item_summary.csv",
    DARTS_DIR / "m5_hobbies_ca1_28day_reconciled_forecast.parquet",
]
_missing_files = [str(p) for p in _REQUIRED_FILES if not p.exists()]
if _missing_files:
    raise FileNotFoundError(
        "The dashboard could not find these data files:\n  - "
        + "\n  - ".join(_missing_files)
        + f"\nExpected project folder: {PROJECT}"
    )

prophet = pd.read_csv(FORECAST_DIR / "prophet_nationwide_actual_vs_forecast.csv")
lgbm = pd.read_csv(FORECAST_DIR / "lightgbm_enhanced_30490_predictions.csv")
importance = pd.read_csv(FORECAST_DIR / "lightgbm_enhanced_30490_feature_importance.csv")
fixed = pd.read_csv(FORECAST_DIR / "fixed_origin_predictions.csv")
intervals = pd.read_csv(FORECAST_DIR / "combined_prediction_intervals.csv")
daily = pd.read_csv(INVENTORY_DIR / "corrected_portfolio_daily_plan.csv")
items = pd.read_csv(INVENTORY_DIR / "corrected_portfolio_item_summary.csv")
hierarchy = pd.read_parquet(DARTS_DIR / "m5_hobbies_ca1_28day_reconciled_forecast.parquet")

for frame, column in [
    (prophet, "ds"), (lgbm, "date"), (fixed, "date"),
    (intervals, "date"), (daily, "date")
]:
    frame[column] = pd.to_datetime(frame[column])

daily_summary = daily.groupby("date", as_index=False).agg({
    "forecast_demand": "sum",
    "recommended_order": "sum",
    "ending_inventory": "sum",
    "unmet_demand": "sum"
})

items["meaningful_shortage"] = items["unmet_demand"].abs() > 1e-8
shortages = items[items["meaningful_shortage"]].sort_values(
    "unmet_demand", ascending=False
).head(10)

parents = [
    "CAT_HOBBIES_CA_1",
    "DEPT_HOBBIES_1_CA_1",
    "DEPT_HOBBIES_2_CA_1"
]
bottom_columns = [c for c in hierarchy.columns if c not in parents]
hierarchy_dates = pd.date_range("2016-03-28", periods=len(hierarchy), freq="D")

# ============================================================
# DESIGN SYSTEM (styles only)
# ============================================================
FONT_STACK = "Inter, 'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif"
COLORWAY = [COLORS["blue"], COLORS["cyan"], COLORS["green"],
            COLORS["orange"], COLORS["red"], "#7C3AED"]

# Chart colours for each theme
CHART_THEME = {
    "light": {
        "text": "#243247", "title": "#14243A", "axis_title": "#64748B",
        "grid": "#EEF2F7", "axis_line": "#CBD5E1", "pie_line": "#FFFFFF",
        "hover_bg": "#14243A", "hover_border": "#14243A",
        "navy_line": "#14243A", "interval_fill": "rgba(37,99,235,0.12)"
    },
    "dark": {
        "text": "#CBD5E1", "title": "#F1F5F9", "axis_title": "#94A3B8",
        "grid": "#1E2D45", "axis_line": "#2F4160", "pie_line": "#111B2E",
        "hover_bg": "#0B1220", "hover_border": "#3B4C69",
        "navy_line": "#E2E8F0", "interval_fill": "rgba(96,165,250,0.22)"
    }
}

APP_CSS = """
.theme-light {
  --bg:#F1F5F9; --surface:#FFFFFF; --soft:#F8FAFC;
  --text:#243247; --heading:#14243A; --muted:#64748B; --border:#E2E8F0;
  --th-bg:#14243A; --row-hover:#EEF4FF;
  --nav-bg:rgba(255,255,255,.94); --nav-hover:#EEF3FA; --nav-active:#E8F0FE; --accent-text:#2563EB;
  --icon:#94A3B8; --reco-bg:#E8F0FE;
  --ok-bg:#ECFDF5; --ok-border:#A7F3D0; --ok-text:#065F46;
  --warn-bg:#FFF7ED; --warn-border:#FED7AA; --warn-text:#7C2D12;
  --badge-ok-bg:#D1FAE5; --badge-ok-text:#065F46; --badge-warn-bg:#FEF3C7; --badge-warn-text:#92400E;
  --neg:#DC2626;
  --shadow-sm:0 1px 2px rgba(20,36,58,.06);
  --shadow:0 2px 6px rgba(20,36,58,.06), 0 10px 28px rgba(20,36,58,.08);
}
.theme-dark {
  --bg:#0B1220; --surface:#111B2E; --soft:#0F1829;
  --text:#CBD5E1; --heading:#F1F5F9; --muted:#94A3B8; --border:#22314A;
  --th-bg:#1B2B45; --row-hover:#17263F;
  --nav-bg:rgba(11,18,32,.92); --nav-hover:#16243B; --nav-active:rgba(59,130,246,.20); --accent-text:#93C5FD;
  --icon:#64748B; --reco-bg:rgba(59,130,246,.20);
  --ok-bg:rgba(16,185,129,.12); --ok-border:rgba(16,185,129,.38); --ok-text:#6EE7B7;
  --warn-bg:rgba(245,158,11,.10); --warn-border:rgba(245,158,11,.38); --warn-text:#FCD34D;
  --badge-ok-bg:rgba(16,185,129,.20); --badge-ok-text:#6EE7B7; --badge-warn-bg:rgba(245,158,11,.20); --badge-warn-text:#FCD34D;
  --neg:#F87171;
  --shadow-sm:0 1px 2px rgba(0,0,0,.35);
  --shadow:0 2px 6px rgba(0,0,0,.35), 0 10px 28px rgba(0,0,0,.35);
}
:root { --blue:#2563EB; --cyan:#0891B2; --green:#059669; --orange:#D97706; --red:#DC2626; --radius:14px; }
* { box-sizing:border-box; }
html { background:#F1F5F9; }
body {
  margin:0; background:#F1F5F9;
  font-family:Inter,'Segoe UI',system-ui,-apple-system,Roboto,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.app-shell { min-height:100vh; background:var(--bg); color:var(--text); transition:background .2s ease, color .2s ease; }

/* ---------- Header ---------- */
.app-header { background:linear-gradient(115deg,#0D1A2D 0%,#14304F 55%,#1D4ED8 150%); color:#fff; }
.app-header-inner {
  max-width:1480px; margin:0 auto; padding:28px 32px 26px;
  display:flex; align-items:center; justify-content:space-between; gap:20px; flex-wrap:wrap;
}
.brand { display:flex; align-items:center; gap:14px; min-width:0; }
.brand-mark {
  width:46px; height:46px; flex:0 0 46px; border-radius:13px;
  display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#3B82F6,#06B6D4);
  font-weight:800; font-size:16px; letter-spacing:.5px; color:#fff;
  box-shadow:0 8px 20px rgba(8,145,178,.35);
}
.brand-eyebrow { font-size:11px; font-weight:700; letter-spacing:1.4px; color:#7DD3FC; margin-bottom:6px; }
.brand-title { font-size:28px; font-weight:800; letter-spacing:-.8px; color:#fff; line-height:1.15; }
.header-stats { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
.stat-chip {
  font-size:11.5px; font-weight:600; color:#E2EEF9; padding:5px 11px; border-radius:999px;
  background:rgba(255,255,255,.08); border:1px solid rgba(220,232,245,.22); white-space:nowrap;
}
.brand-sub { font-size:13.5px; color:#BCCBE0; margin-top:6px; line-height:1.5; }
.header-actions { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.header-pill {
  font-size:10px; font-weight:700; letter-spacing:.8px; color:#DCE8F5;
  border:1px solid rgba(220,232,245,.35); background:rgba(255,255,255,.06);
  border-radius:999px; padding:7px 13px; white-space:nowrap;
}
.theme-switch {
  display:inline-flex; padding:3px; gap:2px; border-radius:999px;
  background:rgba(255,255,255,.10); border:1px solid rgba(220,232,245,.25);
}
.theme-btn {
  border:none; background:transparent; color:#DCE8F5; font-family:inherit;
  font-size:12px; font-weight:600; padding:7px 13px; border-radius:999px; cursor:pointer;
  transition:background .15s ease, color .15s ease; white-space:nowrap;
}
.theme-btn:hover { background:rgba(255,255,255,.12); }
.theme-btn.active { background:#fff; color:#14243A; }
.theme-btn:focus-visible { outline:2px solid #93C5FD; outline-offset:2px; }

/* ---------- Navigation ---------- */
.nav-wrap {
  position:sticky; top:0; z-index:1000;
  background:var(--nav-bg); backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border); box-shadow:var(--shadow-sm);
}
.nav {
  max-width:1480px; margin:0 auto; padding:0 24px;
  display:flex; gap:2px; align-items:stretch; overflow-x:auto; scrollbar-width:none;
}
.nav::-webkit-scrollbar { display:none; }
.nav-btn {
  position:relative; display:inline-flex; align-items:center; gap:8px;
  padding:14px 11px 13px; border:none; border-bottom:3px solid transparent; background:transparent;
  color:var(--muted); font-family:inherit; font-size:13px; font-weight:600;
  cursor:pointer; white-space:nowrap; flex:0 0 auto;
  transition:background .15s ease, color .15s ease, border-color .15s ease;
}
.nav-btn:hover { background:var(--nav-hover); color:var(--heading); }
.nav-btn.active { color:var(--accent-text); border-bottom-color:var(--blue); background:var(--nav-active); }
.nav-btn:focus-visible { outline:2px solid var(--blue); outline-offset:-2px; }
.nav-num {
  width:22px; height:22px; border-radius:7px; flex:0 0 22px;
  display:inline-flex; align-items:center; justify-content:center;
  font-size:11px; font-weight:800; color:var(--muted);
  background:var(--nav-hover); border:1px solid var(--border);
}
.nav-btn.active .nav-num { background:var(--blue); border-color:var(--blue); color:#fff; }

/* ---------- Page ---------- */
.content { max-width:1480px; margin:0 auto; padding:30px 32px 60px; }
.page { animation:pageIn .3s ease; }
@keyframes pageIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }

.page-head { margin-bottom:22px; padding-bottom:18px; border-bottom:1px solid var(--border); }
.page-title { margin:0; font-size:28px; font-weight:800; letter-spacing:-.6px; color:var(--heading); }
.page-sub { margin:8px 0 0; color:var(--muted); font-size:14px; line-height:1.6; max-width:920px; }

.section-title {
  display:flex; align-items:center; gap:10px; margin:36px 0 14px;
  font-size:17px; font-weight:700; letter-spacing:-.2px; color:var(--heading);
}
.section-title::before {
  content:""; width:4px; height:18px; border-radius:2px; flex:0 0 4px;
  background:linear-gradient(var(--blue),var(--cyan));
}
.note { color:var(--muted); font-size:12.5px; line-height:1.7; margin:10px 0; }

/* ---------- Grids ---------- */
.grid { display:grid; gap:16px; margin:16px 0; }
.cols-1 { grid-template-columns:minmax(0,1fr); }
.cols-2 { grid-template-columns:repeat(2,minmax(0,1fr)); }
.cols-3 { grid-template-columns:repeat(3,minmax(0,1fr)); }
.cols-4 { grid-template-columns:repeat(4,minmax(0,1fr)); }
.cols-5 { grid-template-columns:repeat(5,minmax(0,1fr)); }
.cols-2-1 { grid-template-columns:minmax(0,2fr) minmax(0,1fr); }
.align-start { align-items:start; }
.grid > .insight:last-child:nth-child(odd),
.grid > .reco:last-child:nth-child(odd) { grid-column:1 / -1; }
@media (max-width:1200px) {
  .cols-3, .cols-4, .cols-5 { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media (max-width:900px) {
  .cols-2, .cols-2-1, .cols-3, .cols-4, .cols-5 { grid-template-columns:minmax(0,1fr); }
  .content { padding:22px 16px 48px; }
  .app-header-inner { padding:18px 16px; }
  .nav { padding:8px 12px; }
  .page-title { font-size:24px; }
  .brand-title { font-size:21px; }
  .brand-mark { display:none; }
}
@media (max-width:520px) {
  .kpi-value { font-size:24px; }
  .panel { padding:18px 16px; }
}

/* ---------- KPI cards ---------- */
.kpi {
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:18px 20px 17px; box-shadow:var(--shadow-sm);
  transition:transform .15s ease, box-shadow .15s ease, background .2s ease;
}
.kpi:hover { transform:translateY(-2px); box-shadow:var(--shadow); }
.kpi-label {
  display:flex; align-items:center; gap:8px;
  font-size:11px; font-weight:700; letter-spacing:.6px; color:var(--muted);
}
.kpi-dot { width:8px; height:8px; border-radius:50%; flex:0 0 8px; }
.kpi-value { font-size:27px; font-weight:800; letter-spacing:-.6px; color:var(--heading); margin:11px 0 6px; overflow-wrap:anywhere; }
.kpi-sub { font-size:12px; color:var(--muted); line-height:1.5; }

/* ---------- Panels, charts ---------- */
.panel {
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:22px 26px; box-shadow:var(--shadow-sm); line-height:1.75; color:var(--text);
}
.panel p { margin:0 0 12px; }
.panel p:last-child { margin-bottom:0; }
.panel-title { margin:18px 0 8px; font-size:15px; font-weight:700; color:var(--heading); }
.panel-title:first-child { margin-top:0; }
.chart-card {
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:8px 8px 2px; box-shadow:var(--shadow-sm); min-width:0; overflow:hidden;
}

/* ---------- Tables ---------- */
.table-wrap {
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  overflow:auto; box-shadow:var(--shadow-sm); margin:12px 0 16px;
}
.tbl { width:100%; min-width:640px; border-collapse:collapse; table-layout:fixed; font-size:13px; color:var(--text); }
.tbl-compact { min-width:0; }
.tbl th {
  padding:13px 14px; background:var(--th-bg); color:#fff; text-align:left;
  font-weight:600; font-size:12px; letter-spacing:.3px; overflow-wrap:anywhere;
}
.tbl td {
  padding:12px 14px; border-bottom:1px solid var(--border);
  vertical-align:top; line-height:1.55; overflow-wrap:anywhere;
}
.tbl tbody tr:nth-child(even) { background:var(--soft); }
.tbl tbody tr:hover { background:var(--row-hover); }
.tbl tbody tr:last-child td { border-bottom:none; }
.tbl .num { text-align:right; font-variant-numeric:tabular-nums; }
.tbl .ctr { text-align:center; }
.tbl td.strong { font-weight:700; color:var(--heading); }
.tbl td.neg { font-weight:700; color:var(--neg); }
.panel .table-wrap { box-shadow:none; margin:6px 0 0; }

.badge { display:inline-block; padding:3px 11px; border-radius:999px; font-size:11px; font-weight:700; }
.badge-optimal { background:var(--badge-ok-bg); color:var(--badge-ok-text); }
.badge-feasible { background:var(--badge-warn-bg); color:var(--badge-warn-text); }

/* ---------- Callouts ---------- */
.callout {
  border:1px solid; border-radius:12px; padding:14px 18px; margin:14px 0;
  font-size:13.5px; line-height:1.65;
}
.callout ul { margin:0; padding-left:20px; }
.callout li { margin-bottom:10px; line-height:1.65; }
.callout li:last-child { margin-bottom:0; }
.callout-success { background:var(--ok-bg); border-color:var(--ok-border); color:var(--ok-text); font-weight:600; }
.callout-warn { background:var(--warn-bg); border-color:var(--warn-border); color:var(--warn-text); }

/* ---------- Insights ---------- */
.insight {
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:20px 22px; box-shadow:var(--shadow-sm); color:var(--text);
}
.insight h4 { margin:0 0 10px; font-size:15px; color:var(--heading); }
.insight .finding { margin:0 0 8px; font-weight:600; line-height:1.55; }
.insight .detail { margin:0; color:var(--muted); font-size:13px; line-height:1.65; }

.reco {
  display:flex; gap:14px; align-items:flex-start;
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:18px 20px; box-shadow:var(--shadow-sm);
}
.reco-num {
  flex:0 0 32px; height:32px; border-radius:10px; background:var(--reco-bg); color:var(--accent-text);
  display:flex; align-items:center; justify-content:center; font-weight:800; font-size:14px;
}
.reco h4 { margin:2px 0 6px; font-size:14.5px; color:var(--heading); }
.reco p { margin:0; color:var(--muted); font-size:13px; line-height:1.65; }

.bullets { margin:10px 0 0; padding-left:20px; line-height:1.7; }
.bullets li { margin-bottom:8px; }
.panel-accent { border-left:4px solid var(--blue); }
.strong-text { font-weight:700; color:var(--heading); }
"""

app = Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Demand Forecasting & Inventory Intelligence"
app.index_string = (
    "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
    "<style>" + APP_CSS + "</style></head>"
    "<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"
)

# ---------- UI building blocks ----------
def card(title, value, subtitle, accent):
    return html.Div([
        html.Div([
            html.Span(className="kpi-dot", style={"background": accent}),
            title
        ], className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(subtitle, className="kpi-sub")
    ], className="kpi", style={"borderTop": f"3px solid {accent}"})

def card_grid(cards, columns=3):
    return html.Div(cards, className=f"grid cols-{columns}")

def page_header(title, subtitle):
    return html.Div([
        html.H2(title, className="page-title"),
        html.P(subtitle, className="page-sub")
    ], className="page-head")

def section(title):
    return html.H3(title, className="section-title")

def note(text):
    return html.P(text, className="note")

def callout(content, kind="success"):
    return html.Div(content, className=f"callout callout-{kind}")

def panel(children, extra=""):
    return html.Div(children, className=f"panel {extra}".strip())

def chart_grid(*charts, layout="cols-2", extra=""):
    return html.Div(list(charts), className=f"grid {layout} {extra}".strip())

def graph(source_figure, theme="light"):
    """Return a styled chart card. Works on a copy, so one figure can be shown in both themes."""
    t = CHART_THEME[theme]
    figure = go.Figure(source_figure.to_dict())

    is_pie = len(figure.data) > 0 and isinstance(figure.data[0], go.Pie)

    def recolor(trace):
        line = getattr(trace, "line", None)
        if line is not None and getattr(line, "color", None) == COLORS["navy"]:
            trace.update(line_color=t["navy_line"])
        if getattr(trace, "fillcolor", None) == "rgba(37,99,235,0.12)":
            trace.update(fillcolor=t["interval_fill"])

    figure.for_each_trace(recolor)

    figure.update_layout(
        autosize=True,
        font=dict(family=FONT_STACK, size=12, color=t["text"]),
        colorway=COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            x=0, xanchor="left",
            font=dict(size=15, color=t["title"], family=FONT_STACK)
        ),
        # Axis labels and titles make room for themselves (automargin below),
        # so wide labels and the secondary axis are never clipped on small screens.
        margin=dict(l=40, r=40, t=75, b=65),
        hoverlabel=dict(
            bgcolor=t["hover_bg"],
            font=dict(size=12, family=FONT_STACK, color="white"),
            bordercolor=t["hover_border"]
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=t["text"])
        ),
        bargap=0.28, bargroupgap=0.06
    )
    if is_pie:
        figure.update_layout(showlegend=False)

    figure.update_xaxes(
        gridcolor=t["grid"], zeroline=False, linecolor=t["axis_line"],
        tickcolor=t["axis_line"], automargin=True,
        title_font=dict(color=t["axis_title"])
    )
    figure.update_yaxes(
        gridcolor=t["grid"], zeroline=False, automargin=True,
        ticks="outside", ticklen=8, tickcolor="rgba(0,0,0,0)",
        title_font=dict(color=t["axis_title"])
    )
    if is_pie:
        figure.update_traces(marker_line_color=t["pie_line"], marker_line_width=2)

    return html.Div(
        dcc.Graph(
            figure=figure,
            config={"displaylogo": False, "responsive": True, "displayModeBar": "hover"},
            style={"width": "100%"}
        ),
        className="chart-card"
    )

# ---------- Tables ----------
ALIGN_CLASS = {"left": "", "right": "num", "center": "ctr"}

def table_header(text, width=None, align="left"):
    return html.Th(
        text,
        className=ALIGN_CLASS[align],
        style={"width": width} if width else None
    )

def table_cell(content, align="left", extra=None):
    classes = " ".join(c for c in [ALIGN_CLASS[align], extra] if c)
    return html.Td(content, className=classes)

def make_table(columns, rows, compact=False):
    """columns: [(label, width, align)]; each cell is content or (content, extra_class)."""
    header = html.Thead(html.Tr([
        table_header(label, width, align) for label, width, align in columns
    ]))
    body = html.Tbody([
        html.Tr([
            table_cell(
                cell[0] if isinstance(cell, tuple) else cell,
                columns[j][2],
                cell[1] if isinstance(cell, tuple) else None
            )
            for j, cell in enumerate(row)
        ])
        for row in rows
    ])
    return html.Div(
        html.Table([header, body], className="tbl tbl-compact" if compact else "tbl"),
        className="table-wrap"
    )

# ============================================================
# FIGURES (data and labels unchanged)
# ============================================================
# Executive forecast chart
fig_prophet = go.Figure()
fig_prophet.add_trace(go.Scatter(
    x=prophet.ds, y=prophet.actual, name="Actual Demand",
    line=dict(color=COLORS["navy"], width=2.5)
))
fig_prophet.add_trace(go.Scatter(
    x=prophet.ds, y=prophet.yhat_upper, line=dict(width=0),
    showlegend=False, hoverinfo="skip"
))
fig_prophet.add_trace(go.Scatter(
    x=prophet.ds, y=prophet.yhat_lower, fill="tonexty",
    fillcolor="rgba(37,99,235,0.12)", line=dict(width=0),
    name="Prediction Interval", hoverinfo="skip"
))
fig_prophet.add_trace(go.Scatter(
    x=prophet.ds, y=prophet.yhat, name="Prophet Forecast",
    line=dict(color=COLORS["blue"], width=2.5, dash="dash")
))
fig_prophet.update_layout(
    title="Nationwide Demand | Actual vs Forecast",
    xaxis_title="Date", yaxis_title="Units Sold",
    template="plotly_white", hovermode="x unified", height=390
)

fig_daily = go.Figure()
fig_daily.add_trace(go.Bar(
    x=daily_summary.date, y=daily_summary.forecast_demand,
    name="Forecast Demand", marker_color=COLORS["blue"]
))
fig_daily.add_trace(go.Bar(
    x=daily_summary.date, y=daily_summary.recommended_order,
    name="Recommended Orders", marker_color=COLORS["green"]
))
fig_daily.update_layout(
    title="Daily Demand vs Recommended Orders",
    barmode="group", template="plotly_white", height=390,
    xaxis_title="Date", yaxis_title="Units"
)

# Sales distribution
fig_sales = go.Figure(go.Pie(
    labels=["Zero Sales", "Positive Sales"],
    values=[39777094, 18550276], hole=0.58,
    marker=dict(colors=[COLORS["cyan"], COLORS["blue"]]),
    textinfo="label+percent"
))
fig_sales.update_layout(title="Historical Sales Distribution", height=320, template="plotly_white")

# Fixed-origin comparison
metrics = pd.DataFrame([
    {"Model": "Seasonal Naive", "MAE": 1.403214, "RMSE": 3.555931, "WAPE": 113.325645, "RMSSE": 1.088713},
    {"Model": "Enhanced LightGBM", "MAE": 1.148244, "RMSE": 2.621038, "WAPE": 92.733829, "RMSSE": 0.812631},
    {"Model": "TFT", "MAE": 1.057395, "RMSE": 2.728429, "WAPE": 85.396774, "RMSSE": 0.809256}
])
fig_compare = go.Figure()
fig_compare.add_trace(go.Bar(x=metrics.Model, y=metrics.MAE, name="MAE", marker_color=COLORS["blue"]))
fig_compare.add_trace(go.Bar(x=metrics.Model, y=metrics.RMSE, name="RMSE", marker_color=COLORS["cyan"]))
fig_compare.update_layout(
    title="Fixed-Origin Model Comparison | 100 Series",
    barmode="group", template="plotly_white", height=330,
    xaxis_title="Model", yaxis_title="Error"
)

fig_wape = go.Figure(go.Bar(
    x=metrics.Model, y=metrics.WAPE,
    marker_color=[COLORS["orange"], COLORS["blue"], COLORS["green"]],
    text=metrics.WAPE.map(lambda x: f"{x:.2f}%"),
    textposition="outside"
))
fig_wape.update_layout(
    title="Fixed-Origin WAPE Comparison", template="plotly_white",
    height=310,
    yaxis=dict(
        title="WAPE (%)",
        range=[0, 125],
        automargin=True
    ),
    margin=dict(t=75, b=65)
)

importance_col = "importance"
feature_col = "feature"
top_importance = importance.sort_values(importance_col).tail(18)
fig_importance = go.Figure(go.Bar(
    x=top_importance[importance_col], y=top_importance[feature_col],
    orientation="h", marker_color=COLORS["blue"]
))
fig_importance.update_layout(
    title="Enhanced LightGBM | Feature Importance",
    template="plotly_white", height=450,
    margin=dict(l=140, r=25, t=60, b=35)
)

# Interval coverage
coverage = pd.DataFrame([
    {"Model": "TFT 80%", "Nominal": 80, "Actual": 77.821429},
    {"Model": "TFT 96%", "Nominal": 96, "Actual": 94.571429}
])
fig_coverage = go.Figure()
fig_coverage.add_trace(go.Bar(
    x=coverage.Model, y=coverage.Nominal, name="Nominal Coverage",
    marker_color=COLORS["cyan"]
))
fig_coverage.add_trace(go.Bar(
    x=coverage.Model, y=coverage.Actual, name="Actual Coverage",
    marker_color=COLORS["blue"],
    text=coverage.Actual.map(lambda x: f"{x:.2f}%"),
    textposition="outside"
))
fig_coverage.update_layout(
    title="Prediction Interval | Nominal vs Actual Coverage",
    barmode="group", yaxis=dict(range=[0, 110]),
    yaxis_title="Coverage (%)", template="plotly_white", height=330
)

# Hierarchy charts
fig_hierarchy = go.Figure()
for col, label in [
    (parents[0], "HOBBIES Category"),
    (parents[1], "HOBBIES Department 1"),
    (parents[2], "HOBBIES Department 2")
]:
    fig_hierarchy.add_trace(go.Scatter(
        x=hierarchy_dates, y=hierarchy[col],
        mode="lines+markers", name=label
    ))
fig_hierarchy.update_layout(
    title="Darts MinT Reconciled Forecast | CA_1 HOBBIES",
    xaxis_title="Forecast Date", yaxis_title="Forecast Units",
    template="plotly_white", hovermode="x unified", height=340
)

fig_hierarchy_structure = go.Figure(go.Bar(
    x=["Category", "Departments", "Bottom Item-Series"],
    y=[1, 2, len(bottom_columns)],
    marker_color=[COLORS["blue"], COLORS["cyan"], COLORS["green"]],
    text=["1", "2", str(len(bottom_columns))],
    textposition="outside"
))
fig_hierarchy_structure.update_layout(
    title="M5 Hierarchy Composition | CA_1 HOBBIES",
    yaxis=dict(title="Components (log scale)", type="log", range=[-0.15, 3.05]),
    template="plotly_white", height=380, showlegend=False
)

# Inventory charts
fig_inventory = go.Figure()
fig_inventory.add_trace(go.Scatter(
    x=daily_summary.date, y=daily_summary.ending_inventory,
    mode="lines+markers", name="Ending Inventory",
    line=dict(color=COLORS["green"], width=2.5), yaxis="y"
))
fig_inventory.add_trace(go.Scatter(
    x=daily_summary.date, y=daily_summary.unmet_demand,
    mode="lines+markers", name="Unmet Demand",
    line=dict(color=COLORS["red"], width=2.5),
    marker=dict(size=7), yaxis="y2"
))
fig_inventory.update_layout(
    title="Daily Inventory Position and Unmet Demand",
    xaxis=dict(title="Date", showgrid=False),
    yaxis=dict(title="Ending Inventory (units)", rangemode="tozero", gridcolor=COLORS["border"]),
    yaxis2=dict(title="Unmet Demand (units)", overlaying="y", side="right", rangemode="tozero", showgrid=False),
    template="plotly_white", hovermode="x unified", height=350,
    legend=dict(orientation="h", y=1.12, x=0)
)

costs = pd.DataFrame({
    "Component": ["Purchase Cost", "Ordering Cost", "Holding Cost", "Stockout Cost"],
    "Cost": [574800, 56250, 59565.036547, 22914.500367]
})
fig_cost = go.Figure(go.Pie(
    labels=costs.Component, values=costs.Cost,
    hole=0.55, textinfo="label+percent"
))
fig_cost.update_layout(title="Inventory Optimization Cost Breakdown", height=300, template="plotly_white")

solver_counts = items.solver_status.value_counts().reindex(["OPTIMAL", "FEASIBLE"]).fillna(0)
fig_solver = go.Figure(go.Bar(
    x=solver_counts.index, y=solver_counts.values,
    marker_color=[COLORS["green"], COLORS["orange"]],
    text=solver_counts.values, textposition="outside"
))
fig_solver.update_layout(title="OR-Tools Solver Results", height=330, template="plotly_white")

# ============================================================
# CONTENT (text unchanged)
# ============================================================
insights = [
    (
        "Forecasting Performance",
        "TFT achieved the strongest MAE, WAPE and RMSSE in the verified fixed-origin comparison.",
        "Across the same 100 CA_1 HOBBIES series and 28-day horizon, TFT recorded 1.0574 MAE and 85.40% WAPE. Enhanced LightGBM recorded the lowest RMSE at 2.6210. Both beat the seasonal-naive baseline on all four metrics."
    ),
    (
        "Large-Scale Demand Forecasting",
        "Enhanced LightGBM evaluated all 30,490 item-store series.",
        "Its 69.85% WAPE comes from rolling one-day-ahead validation with 150 sampled training dates per series. This is a separate evaluation from the 100-series fixed-origin comparison."
    ),
    (
        "Intermittent Demand Behaviour",
        "Approximately 68.2% of historical sales observations were zero.",
        "Zero-sales days are valid observations in the M5 dataset. This level of intermittency makes demand planning more challenging and supports evaluating forecasts at the item-store level."
    ),
    (
        "Hierarchical Consistency",
        "The 568-component CA_1 HOBBIES reconciliation passed its numerical coherence audit.",
        "Three parent-child relationships were checked across all 28 forecast days, giving 84 daily checks. Maximum absolute coherence error was 0.0005493 against a tolerance of 0.001."
    ),
    (
        "Inventory Planning",
        "The 565-item portfolio generated 9,580 recommended order units.",
        "Against forecast demand of 13,309.65 units, the simulated plan recorded 152.76 units of unmet demand across 264 items with meaningful shortages."
    ),
    (
        "Optimization Solver",
        "All 565 item-level optimization problems returned valid solutions.",
        "472 were OPTIMAL, with optimality proven; 93 were FEASIBLE, with valid solutions found but optimality not proven within the available solving time."
    ),
    (
        "Prediction Uncertainty",
        "TFT's empirical interval coverage was 77.82% and 94.57%.",
        "These results correspond to nominal 80% and 96% intervals, respectively, on 100 series over one fixed-origin 28-day validation window. They are empirical measurements, not future coverage guarantees."
    )
]

limitations = [
    "M5 provides sales, calendar and price data, but not actual inventory levels, lead times, supplier costs or stockout penalties. Inventory optimization therefore uses explicitly simulated assumptions.",
    "Price-change indicators are proxies and do not establish that a promotion occurred.",
    "The complete M5 hierarchy contains 42,840 nodes. The Darts MinT experiment covers 568 components in the CA_1 HOBBIES subset, not the complete hierarchy.",
    "TFT and the fixed-origin model comparison use 100 item-store series, while the enhanced LightGBM rolling evaluation covers all 30,490 series.",
    "Prophet forecasts nationwide aggregate demand, so its 6.19% WAPE is not directly comparable with item-store model results.",
    "The reported fixed-origin RMSSE is an unweighted mean across series, not official M5 WRMSSE.",
    "The inventory results depend on the chosen simulated cost, initial-stock, lead-time and order-quantity assumptions."
]

recommendations = [
    ("Use forecasts at the appropriate planning level",
     "Use item-store forecasts for localized replenishment decisions, and aggregate forecasts for broader capacity and demand planning."),
    ("Account for intermittent demand",
     "Avoid treating zero-sales observations as missing data. Consider demand variability and item-level history when setting replenishment policies."),
    ("Monitor forecast errors and uncertainty",
     "Track MAE, RMSE and WAPE alongside prediction interval coverage. Reassess models as new sales observations become available."),
    ("Validate replenishment assumptions before deployment",
     "Replace simulated inventory, lead-time and cost inputs with real operational data before using the optimization outputs for purchasing decisions."),
    ("Preserve aggregation consistency",
     "Use reconciliation checks when forecasts are consumed across item, department and category planning levels.")
]

# ============================================================
# PAGES (built once per theme, so switching is instant)
# ============================================================
def build_pages(theme):
    def g(figure):
        return graph(figure, theme)

    overview = html.Div([
        page_header(
            "Executive Overview",
            "End-to-end demand forecasting, hierarchical reconciliation and inventory optimization."
        ),
        card_grid([
            card("ITEM-STORE SERIES", "30,490", "Products × stores", COLORS["blue"]),
            card("ITEM-STORE-DAY SALES RECORDS", "58.33M", "1,913 days × 30,490 item-store series", COLORS["cyan"]),
            card("FORECAST HORIZON", "28 Days", "Daily forecast window", COLORS["green"]),
            card("INVENTORY ITEMS", "565", "CA_1 HOBBIES subset", COLORS["orange"]),
            card("RECOMMENDED ORDERS", "9,580", "Total order quantity", COLORS["green"]),
            card("UNMET DEMAND", "152.76", "Portfolio shortage units", COLORS["red"])
        ]),

        section("Model Performance Snapshot"),
        note("Evaluation populations and protocols differ; these WAPE values are not a direct model ranking."),
        make_table(
            [("Model", "20%", "left"), ("Evaluation Protocol", "32%", "left"),
             ("Population", "20%", "left"), ("Horizon", "13%", "left"), ("WAPE", "15%", "right")],
            [
                ("Seasonal Naive", "Fixed-origin baseline", "100 series", "28 days", "113.33%"),
                ("Enhanced LightGBM", "Recursive fixed-origin", "100 series", "28 days", "92.73%"),
                ("TFT", "Direct fixed-origin", "100 series", "28 days", "85.40%"),
                ("Enhanced LightGBM", "Rolling one-day-ahead", "30,490 series", "1 day", "69.85%"),
                ("Prophet", "Nationwide aggregate holdout", "1 aggregate series", "28 days", "6.19%")
            ]
        ),
        note("The first three rows form the comparable fixed-origin experiment. The final two rows use different populations and evaluation protocols and are reported separately for context."),
        callout("Among the three models evaluated on the same 100-series fixed-origin experiment, TFT achieved the lowest WAPE of 85.40%."),
        note("Prophet forecasts nationwide aggregate sales, while LightGBM and TFT results below are evaluated at item-store level. Their WAPE values therefore measure different forecasting tasks and should not be ranked directly."),

        card_grid([
            card("PROPHET WAPE", "6.19%", "Nationwide aggregate · 28-day validation", COLORS["blue"]),
            card("LIGHTGBM WAPE", "69.85%", "30,490 series · Rolling one-day-ahead", COLORS["cyan"]),
            card("LIGHTGBM · FIXED-ORIGIN WAPE", "92.73%", "100 series · Recursive 28-day forecast", COLORS["orange"]),
            card("TFT WAPE", "85.40%", "100 series · Fixed-origin 28-day forecast", COLORS["green"])
        ], columns=4),

        chart_grid(g(fig_prophet), g(fig_daily)),
        note("Inventory optimization uses simulated inventory and cost assumptions.")
    ], className="page")

    data_page = html.Div([
        page_header(
            "Data Intelligence",
            "M5 Forecasting dataset profile and historical demand characteristics."
        ),
        card_grid([
            card("ITEM-STORE SERIES", "30,490", "3,049 products × 10 stores", COLORS["blue"]),
            card("ITEM-STORE-DAY SALES RECORDS", "58.33M", "1,913 daily observations per series", COLORS["cyan"]),
            card("CALENDAR COVERAGE", "1,969 days", "Calendar records", COLORS["green"]),
            card("SELL PRICE RECORDS", "6.84M", "Price history records", COLORS["orange"]),
            card("ZERO SALES", "68.2%", "39.78 million observations", COLORS["red"]),
            card("POSITIVE SALES", "31.8%", "18.55 million observations", COLORS["green"])
        ]),
        chart_grid(
            g(fig_sales),
            panel([
                html.H4("Dataset Structure", className="panel-title"),
                make_table(
                    [("Dimension", "55%", "left"), ("Coverage", "45%", "center")],
                    [
                        ("Products", "3,049"), ("Stores", "10"),
                        ("Departments", "7"), ("Categories", "3"),
                        ("States", "CA, TX, WI"), ("Sales missing values", "0"),
                        ("Negative sales values", "0"), ("Missing sell prices", "21.09%")
                    ],
                    compact=True
                )
            ]),
            extra="align-start"
        ),
        note("Missing sell prices reflect unavailable price-history records; they are not treated as zero prices.")
    ], className="page")

    forecast_page = html.Div([
        page_header(
            "Forecasting Intelligence",
            "Model evaluation, feature importance and prediction uncertainty."
        ),
        section("Full-Dataset Enhanced LightGBM"),
        card_grid([
            card("VALIDATION SERIES", "30,490", "All item-store series", COLORS["blue"]),
            card("VALIDATION OBSERVATIONS", "853,720", "28 days × 30,490 series", COLORS["cyan"]),
            card("MAE", "0.9684", "Mean absolute error", COLORS["green"]),
            card("RMSE", "1.9353", "Root mean squared error", COLORS["orange"]),
            card("WAPE", "69.85%", "One-day-ahead rolling evaluation", COLORS["red"])
        ], columns=5),

        section("Fixed-Origin Model Comparison | 100 Series"),
        note("Verified fixed-origin evaluation | 100 CA_1 HOBBIES series | Origin: March 27, 2016 | Forecast: March 28 – April 24, 2016. TFT leads on MAE, WAPE and RMSSE; Enhanced LightGBM leads on RMSE."),
        chart_grid(g(fig_compare), g(fig_wape)),

        section("LightGBM Feature Importance"),
        g(fig_importance),

        section("Prediction Interval Calibration"),
        g(fig_coverage),
        note("Verified TFT prediction intervals: 77.82% empirical coverage for the nominal 80% interval and 94.57% for the nominal 96% interval. Evaluation covers 100 item-store series over one 28-day fixed-origin window; empirical coverage is not a guarantee of future coverage.")
    ], className="page")

    hierarchy_page = html.Div([
        page_header(
            "Hierarchical Reconciliation",
            "Darts MinT reconciliation | California Store CA_1 | HOBBIES category"
        ),
        card_grid([
            card("TOTAL COMPONENTS", "568", "Subset hierarchy", COLORS["blue"]),
            card("CATEGORY NODES", "1", "HOBBIES category", COLORS["cyan"]),
            card("DEPARTMENT NODES", "2", "HOBBIES departments", COLORS["green"]),
            card("BOTTOM SERIES", "565", "Item-store series", COLORS["orange"]),
            card("FORECAST HORIZON", "28 Days", "Daily reconciled forecasts", COLORS["blue"]),
            card("MAX COHERENCE ERROR", "0.0005493", "84 checks · Tolerance 0.001", COLORS["green"])
        ]),
        note("Numerical coherence validation passed: 3 parent-child relationships checked over all 28 forecast days (84 daily checks). Mean absolute coherence error: 0.0001787."),
        chart_grid(g(fig_hierarchy), g(fig_hierarchy_structure), layout="cols-2-1"),
        panel([
            html.H4("Understanding the Dataset and Hierarchy", className="panel-title"),
            html.P("The 58.33 million figure represents daily sales observations: 30,490 item-store series across 1,913 historical days. The 42,840 figure represents nodes in the complete M5 aggregation hierarchy across its different levels. These are different measures: observations count time-series records, while hierarchy nodes count series at bottom and aggregate levels."),
            html.P("This experiment reconciles a manageable CA_1 HOBBIES subset containing 565 bottom-level item series and 3 aggregate nodes, for 568 components in total."),
            html.H4("Reconciliation Method", className="panel-title"),
            html.P("Darts MinT (Minimum Trace) reconciliation was applied to the CA_1 HOBBIES hierarchy using historical covariance information."),
            html.H4("Hierarchy Structure", className="panel-title"),
            html.P("The hierarchy contains one category node, two department nodes and 565 bottom-level item series."),
            html.H4("Scope and Interpretation", className="panel-title"),
            html.P("This is a subset demonstration, not a reconciliation of the complete 42,840-node M5 hierarchy. The 28-day forecast uses seasonal-naive base forecasts followed by MinT reconciliation.")
        ])
    ], className="page")

    if len(shortages) > 0:
        shortage_block = make_table(
            [("Item ID", "38%", "left"), ("Forecast Demand", "15%", "right"),
             ("Recommended Orders", "17%", "right"), ("Unmet Demand", "15%", "right"),
             ("Solver Status", "15%", "center")],
            [
                (
                    (row.item_id, "strong"),
                    f"{row.forecast_demand:.2f}",
                    f"{row.total_ordered:.0f}",
                    (f"{row.unmet_demand:.2f}", "neg"),
                    html.Span(
                        row.solver_status,
                        className="badge " + (
                            "badge-optimal" if row.solver_status == "OPTIMAL"
                            else "badge-feasible" if row.solver_status == "FEASIBLE" else ""
                        )
                    )
                )
                for row in shortages.itertuples(index=False)
            ]
        )
    else:
        shortage_block = callout("No items have a meaningful shortage in the current data.")

    inventory_page = html.Div([
        page_header(
            "Inventory Intelligence",
            "OR-Tools inventory planning and optimization results."
        ),
        card_grid([
            card("OPTIMIZED ITEMS", "565", "CA_1 HOBBIES subset", COLORS["blue"]),
            card("FORECAST DEMAND", "13,309.65", "28-day forecast units", COLORS["cyan"]),
            card("RECOMMENDED ORDERS", "9,580", "Total order quantity", COLORS["green"]),
            card("UNMET DEMAND", "152.76", "Portfolio shortage units", COLORS["red"]),
            card("OPTIMAL SOLUTIONS", "472", "Optimal solver status", COLORS["green"]),
            card("FEASIBLE SOLUTIONS", "93", "Optimality not proven", COLORS["orange"])
        ]),
        chart_grid(g(fig_inventory), g(fig_cost)),

        section("Optimization Assumptions"),
        note("The following values are simulated business assumptions because the M5 dataset does not provide actual inventory or operating costs."),
        make_table(
            [("Assumption", "38%", "left"), ("Value", "20%", "right"), ("Unit / Meaning", "42%", "left")],
            [
                (label, (value, "strong"), unit)
                for label, value, unit in [
                    ("Lead time", "2", "Days"),
                    ("Holding cost", "2.00", "Per unit per day"),
                    ("Stockout penalty", "150.00", "Per unit"),
                    ("Ordering cost", "50.00", "Per order"),
                    ("Unit purchase cost", "60.00", "Per unit"),
                    ("Maximum order quantity", "100", "Units"),
                    ("Initial stock", "7", "Days of forecast demand")
                ]
            ]
        ),

        section("Top 10 Items by Unmet Demand"),
        shortage_block,
        note("Recommended order quantity represents the additional units the optimization model suggests ordering for an item on a particular day. It is based on forecast demand, simulated starting inventory, replenishment constraints and cost assumptions; it is not an actual purchase order."),

        panel([
            html.H4("Understanding Solver Outcomes", className="panel-title"),
            html.P("472 OPTIMAL solutions: the solver found and proved an optimal solution for these item-level problems."),
            html.P("93 FEASIBLE solutions: the solver found valid solutions, but did not prove global optimality within the available solving time."),
            html.P("Both statuses represent valid solutions. FEASIBLE should not be interpreted as a failed optimization.", className="note")
        ]),
        note("Optimization uses simulated inventory, lead-time and cost assumptions. The M5 dataset does not contain actual inventory positions or procurement costs.")
    ], className="page")

    insights_page = html.Div([
        page_header(
            "Business Insights",
            "What the forecasting, reconciliation and inventory experiments reveal for demand planning."
        ),

        section("Key Findings"),
        html.Div([
            html.Div([
                html.H4(title),
                html.P(finding, className="finding"),
                html.P(detail, className="detail")
            ], className="insight", style={"borderLeft": f"4px solid {accent}"})
            for (title, finding, detail), accent in zip(insights, [
                COLORS["blue"], COLORS["cyan"], COLORS["orange"],
                COLORS["green"], COLORS["blue"], COLORS["red"], COLORS["cyan"]
            ])
        ], className="grid cols-2"),

        section("Business Recommendations"),
        html.Div([
            html.Div([
                html.Div(str(i), className="reco-num"),
                html.Div([html.H4(title), html.P(text)])
            ], className="reco")
            for i, (title, text) in enumerate(recommendations, start=1)
        ], className="grid cols-2"),

        section("Project Limitations and Assumptions"),
        callout(
            html.Ul([html.Li(x) for x in limitations]),
            kind="warn"
        ),

        section("Project Conclusion"),
        panel([
            html.P("This project demonstrates an end-to-end demand forecasting and inventory planning workflow using the M5 dataset. The full dataset was profiled, and enhanced LightGBM evaluation covered all 30,490 item-store series."),
            html.P("A controlled fixed-origin experiment compared Seasonal Naive, Enhanced LightGBM and TFT on the same 100-series subset. TFT achieved the strongest MAE, WAPE and RMSSE, while LightGBM achieved the lowest RMSE. Hierarchical reconciliation passed the defined coherence tolerance, and the inventory optimization prototype produced valid solutions for all 565 selected items."),
            html.P(
                "The results demonstrate the value of combining forecasting, hierarchical consistency and constrained inventory planning. Further operational validation with actual inventory, procurement and service-level data would be required before production deployment.",
                className="strong-text"
            )
        ], extra="panel-accent")
    ], className="page")

    config_page = html.Div([
        page_header(
            "Experimental Configuration",
            "Dataset scope, evaluation windows, model configurations and computational boundaries used in the experiments."
        ),

        section("Dataset and Evaluation Scope"),
        make_table(
            [("Experiment", "17%", "left"), ("Population / Data", "27%", "left"),
             ("Training / Input", "26%", "left"), ("Validation / Horizon", "30%", "left")],
            [
                ("M5 dataset", "30,490 series · 58,327,370 observations", "1,913 historical days", "Calendar contains 1,969 days"),
                ("Enhanced LightGBM", "All 30,490 series", "4,573,500 sampled training rows · 150 dates per series", "853,720 rows · 28-day rolling one-day-ahead validation"),
                ("Fixed-origin comparison", "100 CA_1 HOBBIES series · 2,800 predictions", "Origin: March 27, 2016", "March 28 – April 24, 2016 · 28 days"),
                ("Prophet", "Nationwide aggregate · 1,913 observations", "First 1,885 days", "Final 28 days"),
                ("Darts MinT", "565 bottom series · 568 components", "CA_1 HOBBIES subset", "28-day reconciled forecast"),
                ("Inventory optimization", "565 CA_1 HOBBIES item-series", "OR-Tools optimization", "28-day planning horizon")
            ]
        ),

        section("Model Configuration"),
        make_table(
            [("Model", "19%", "left"), ("Configuration", "51%", "left"), ("Evaluation", "30%", "left")],
            [
                ("Enhanced LightGBM", "Poisson objective · 1,000 estimators · Learning rate 0.05 · 31 leaves · Column and row sampling 0.8 · Random seed 42 · 2 jobs", "18 features · 150 sampled training dates per series"),
                ("TFT", "Encoder 56 days · Prediction 28 days · Hidden size 16 · 4 attention heads · Dropout 0.1 · Hidden continuous size 8 · 7 quantiles", "Direct 28-day forecast · Median quantile used for point metrics"),
                ("Seasonal Naive", "Seasonal lag-7 baseline", "Same 100-series fixed-origin evaluation"),
                ("Prophet", "Yearly and weekly seasonality · No daily seasonality", "Nationwide aggregate · Final 28-day holdout"),
                ("Darts MinT", "Historical covariance-based MinT reconciliation", "565 bottom series and 3 aggregate nodes")
            ]
        ),

        section("Evaluation Metrics and Interpretation"),
        panel(html.Ul([
            html.Li("MAE and RMSE measure point-forecast errors."),
            html.Li("WAPE is reported separately for each evaluation population and protocol; values from different populations are not directly comparable."),
            html.Li("RMSSE in the fixed-origin comparison is an unweighted mean across series, not official M5 WRMSSE."),
            html.Li("TFT interval coverage is empirical coverage on one 28-day validation window, not a guarantee of future coverage.")
        ], className="bullets")),

        section("Computational Scope and Reproducibility"),
        panel([
            html.P("The full M5 dataset was profiled and the enhanced LightGBM evaluation covered all 30,490 item-store series. Computationally intensive TFT, fixed-origin comparison, Darts reconciliation and inventory optimization experiments used explicitly identified subsets. The Darts experiment reconciled 568 components rather than the complete 42,840-node M5 hierarchy."),
            html.P("The LightGBM training sample used 150 sampled dates per series; its validation covered all 853,720 observations. The TFT experiment used 100 series. These scopes are stated separately to distinguish full-dataset analysis from subset-based experiments.", className="note")
        ])
    ], className="page")

    return {
        "overview": overview,
        "data": data_page,
        "forecast": forecast_page,
        "hierarchy": hierarchy_page,
        "inventory": inventory_page,
        "insights": insights_page,
        "config": config_page
    }

PAGES_BY_THEME = {"light": build_pages("light"), "dark": build_pages("dark")}

NAV = [
    ("overview", "Executive Overview"),
    ("data", "Data Intelligence"),
    ("forecast", "Forecasting Intelligence"),
    ("hierarchy", "Hierarchical Reconciliation"),
    ("inventory", "Inventory Intelligence"),
    ("insights", "Business Insights"),
    ("config", "Experimental Configuration")
]

NAV_ICONS = {
    "overview": "▦",
    "data": "◫",
    "forecast": "⌁",
    "hierarchy": "⌘",
    "inventory": "▤",
    "insights": "✧",
    "config": "⚙"
}

def make_navigation(active):
    return [
        html.Button(
            [
                html.Span(str(i), className="nav-num"),
                html.Span(label)
            ],
            id=f"nav-{key}",
            n_clicks=0,
            className="nav-btn active" if key == active else "nav-btn"
        )
        for i, (key, label) in enumerate(NAV, start=1)
    ]

# ============================================================
# LAYOUT AND NAVIGATION
# ============================================================
DEFAULT_STATE = {"page": "overview", "theme": "light"}

app.layout = html.Div([
    dcc.Store(id="view-state", storage_type="local", data=DEFAULT_STATE),

    html.Div(
        html.Div([
            html.Div([
                html.Div("DF", className="brand-mark"),
                html.Div([
                    html.Div("M5 FORECASTING PROJECT  ·  DS-02", className="brand-eyebrow"),
                    html.Div("Demand Forecasting & Inventory Intelligence", className="brand-title"),
                    html.Div(
                        "End-to-end demand forecasting, hierarchical reconciliation & inventory optimization",
                        className="brand-sub"
                    ),
                    html.Div([
                        html.Span("30,490 series", className="stat-chip"),
                        html.Span("Prophet · LightGBM · TFT", className="stat-chip"),
                        html.Span("MinT reconciliation", className="stat-chip"),
                        html.Span("OR-Tools optimization", className="stat-chip")
                    ], className="header-stats")
                ], style={"minWidth": "0"})
            ], className="brand"),
            html.Div([
                html.Div([
                    html.Button("☀ Light", id="theme-light", n_clicks=0, className="theme-btn active"),
                    html.Button("☾ Dark", id="theme-dark", n_clicks=0, className="theme-btn")
                ], className="theme-switch"),
            ], className="header-actions")
        ], className="app-header-inner"),
        className="app-header"
    ),

    html.Div(
        html.Div(make_navigation("overview"), id="top-navigation", className="nav"),
        className="nav-wrap"
    ),

    html.Div(PAGES_BY_THEME["light"]["overview"], id="main-content", className="content")
], id="app-shell", className="app-shell theme-light")

@app.callback(
    Output("app-shell", "className"),
    Output("main-content", "children"),
    Output("top-navigation", "children"),
    Output("theme-light", "className"),
    Output("theme-dark", "className"),
    Output("view-state", "data"),
    *[Input(f"nav-{key}", "n_clicks") for key, _ in NAV],
    Input("theme-light", "n_clicks"),
    Input("theme-dark", "n_clicks"),
    State("view-state", "data")
)
def update_view(*args):
    saved = args[-1]
    page, theme = DEFAULT_STATE["page"], DEFAULT_STATE["theme"]
    # Restore the saved page and theme (ignore anything unexpected)
    if isinstance(saved, dict):
        if saved.get("page") in PAGES_BY_THEME["light"]:
            page = saved["page"]
        if saved.get("theme") in PAGES_BY_THEME:
            theme = saved["theme"]

    trigger = ctx.triggered_id
    if isinstance(trigger, str):
        if trigger.startswith("nav-") and trigger.replace("nav-", "") in PAGES_BY_THEME["light"]:
            page = trigger.replace("nav-", "")
        elif trigger == "theme-light":
            theme = "light"
        elif trigger == "theme-dark":
            theme = "dark"

    return (
        f"app-shell theme-{theme}",
        PAGES_BY_THEME[theme][page],
        make_navigation(page),
        "theme-btn active" if theme == "light" else "theme-btn",
        "theme-btn active" if theme == "dark" else "theme-btn",
        {"page": page, "theme": theme}
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8051, debug=False)