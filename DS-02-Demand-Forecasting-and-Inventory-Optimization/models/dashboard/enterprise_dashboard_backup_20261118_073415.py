
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, ctx

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

def card(title, value, subtitle, accent):
    return html.Div([
        html.Div(title, style={"fontSize": "11px", "color": COLORS["muted"], "fontWeight": "bold"}),
        html.Div(value, style={"fontSize": "25px", "fontWeight": "bold", "color": COLORS["navy"], "margin": "9px 0"}),
        html.Div(subtitle, style={"fontSize": "11px", "color": COLORS["muted"]})
    ], style={
        "background": "white", "padding": "19px", "borderRadius": "10px",
        "border": f'1px solid {COLORS["border"]}',
        "borderTop": f"3px solid {accent}"
    })

def card_grid(cards, columns=3):
    return html.Div(cards, style={
        "display": "grid",
        "gridTemplateColumns": f"repeat({columns}, minmax(0, 1fr))",
        "gap": "16px",
        "margin": "18px 0"
    })

def graph(figure):
    return dcc.Graph(figure=figure, config={"displaylogo": False, "responsive": True})

def table_style():
    return {"width": "100%", "borderCollapse": "collapse", "background": "white", "fontSize": "12px"}

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
fig_sales.update_layout(title="Historical Sales Distribution", height=370, template="plotly_white")

# Fixed-origin comparison
metrics = pd.DataFrame([
    {"Model": "Naive", "MAE": 1.2259, "RMSE": 2.9879, "WAPE": 99.0089},
    {"Model": "LightGBM", "MAE": 1.1409, "RMSE": 2.6103, "WAPE": 92.1377},
    {"Model": "TFT", "MAE": 1.0627, "RMSE": 2.7775, "WAPE": 85.8247}
])
fig_compare = go.Figure()
fig_compare.add_trace(go.Bar(x=metrics.Model, y=metrics.MAE, name="MAE", marker_color=COLORS["blue"]))
fig_compare.add_trace(go.Bar(x=metrics.Model, y=metrics.RMSE, name="RMSE", marker_color=COLORS["cyan"]))
fig_compare.update_layout(
    title="Fixed-Origin Model Comparison | 100 Series",
    barmode="group", template="plotly_white", height=370,
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
    height=350, yaxis_title="WAPE (%)"
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
    template="plotly_white", height=560,
    margin=dict(l=140, r=25, t=60, b=35)
)

# Interval coverage
coverage = pd.DataFrame([
    {"Model": "TFT 80%", "Nominal": 80, "Actual": 83.571},
    {"Model": "TFT 96%", "Nominal": 96, "Actual": 96.786},
    {"Model": "LightGBM 80%", "Nominal": 80, "Actual": 90.286},
    {"Model": "LightGBM 96%", "Nominal": 96, "Actual": 98.000}
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
    yaxis_title="Coverage (%)", template="plotly_white", height=380
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
    template="plotly_white", hovermode="x unified", height=400
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
    template="plotly_white", hovermode="x unified", height=420,
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
fig_cost.update_layout(title="Inventory Optimization Cost Breakdown", height=390, template="plotly_white")

solver_counts = items.solver_status.value_counts().reindex(["OPTIMAL", "FEASIBLE"]).fillna(0)
fig_solver = go.Figure(go.Bar(
    x=solver_counts.index, y=solver_counts.values,
    marker_color=[COLORS["green"], COLORS["orange"]],
    text=solver_counts.values, textposition="outside"
))
fig_solver.update_layout(title="OR-Tools Solver Results", height=330, template="plotly_white")

# Business insights
insights = [
    ("Forecasting Performance",
     "TFT achieved the lowest MAE (1.0627) and WAPE (85.82%) in the fixed-origin comparison.",
     "LightGBM achieved the lowest RMSE (2.6103). Both outperformed the Naive baseline across the reported metrics."),
    ("Large-Scale Demand Forecasting",
     "Enhanced LightGBM achieved 69.85% WAPE across 30,490 item-store series.",
     "This is a one-day-ahead rolling validation using 150 sampled training dates per series, not a fixed-origin 28-day evaluation."),
    ("Hierarchical Consistency",
     "Darts MinT reconciliation produced 568 components across the CA_1 HOBBIES subset.",
     "The hierarchy includes one category, two departments and 565 bottom-level item series."),
    ("Inventory Planning",
     "The optimizer recommended 9,580 units of orders against forecast demand of 13,309.65 units.",
     "The portfolio has 152.76 units of unmet demand across 264 items with meaningful shortages."),
    ("Optimization Solver",
     "472 item-level solutions were OPTIMAL and 93 were FEASIBLE.",
     "FEASIBLE means a valid solution was found, but optimality was not proven."),
    ("Prediction Uncertainty",
     "TFT's 80% interval achieved 83.57% empirical coverage; its 96% interval achieved 96.79%.",
     "Coverage was evaluated on 100 item-store series over one 28-day validation window.")
]

limitations = [
    "M5 does not provide actual inventory levels, lead times, supplier costs or stockout penalties. Inventory optimization uses simulated assumptions.",
    "Price-change indicators are proxies and do not confirm promotional activity.",
    "The full 39,791-node M5 hierarchy was not reconciled; the Darts demonstration covers the CA_1 HOBBIES subset.",
    "TFT and fixed-origin model comparisons use a 100-series subset.",
    "The enhanced LightGBM evaluation is one-day-ahead rolling validation, while fixed-origin evaluation measures a 28-day horizon.",
    "Reported RMSSE is an unweighted mean across series, not official M5 WRMSSE."
]

overview = html.Div([
    html.H2("Executive Overview"),
    html.P("End-to-end demand forecasting, hierarchical reconciliation and inventory optimization.", style={"color": COLORS["muted"]}),
    card_grid([
        card("ITEM-STORE SERIES", "30,490", "Products × stores", COLORS["blue"]),
        card("HISTORICAL OBSERVATIONS", "58.33M", "Daily sales observations", COLORS["cyan"]),
        card("FORECAST HORIZON", "28 Days", "Daily forecast window", COLORS["green"]),
        card("INVENTORY ITEMS", "565", "CA_1 HOBBIES subset", COLORS["orange"]),
        card("RECOMMENDED ORDERS", "9,580", "Total order quantity", COLORS["green"]),
        card("UNMET DEMAND", "152.76", "Portfolio shortage units", COLORS["red"])
    ]),
    html.H3("Model Performance Snapshot"),
    html.P("Evaluation populations and protocols differ; these WAPE values are not a direct model ranking.", style={"color": COLORS["muted"]}),
    card_grid([
        card("PROPHET WAPE", "6.19%", "Nationwide aggregate · 28-day validation", COLORS["blue"]),
        card("LIGHTGBM WAPE", "69.85%", "30,490 series · Rolling one-day-ahead", COLORS["cyan"]),
        card("TFT WAPE", "85.82%", "100 series · Fixed-origin 28-day forecast", COLORS["green"])
    ]),
    html.Div([graph(fig_prophet), graph(fig_daily)], style={"display": "grid", "gridTemplateColumns": "repeat(2,minmax(0,1fr))", "gap": "20px"}),
    html.P("Inventory optimization uses simulated inventory and cost assumptions.", style={"color": COLORS["muted"], "fontSize": "12px"})
])

data_page = html.Div([
    html.H2("Data Intelligence"),
    html.P("M5 Forecasting dataset profile and historical demand characteristics.", style={"color": COLORS["muted"]}),
    card_grid([
        card("ITEM-STORE SERIES", "30,490", "3,049 products × 10 stores", COLORS["blue"]),
        card("HISTORICAL OBSERVATIONS", "58.33M", "1,913 daily observations per series", COLORS["cyan"]),
        card("CALENDAR COVERAGE", "1,969 days", "Calendar records", COLORS["green"]),
        card("SELL PRICE RECORDS", "6.84M", "Price history records", COLORS["orange"]),
        card("ZERO SALES", "68.2%", "39.78 million observations", COLORS["red"]),
        card("POSITIVE SALES", "31.8%", "18.55 million observations", COLORS["green"])
    ]),
    html.Div([
        graph(fig_sales),
        html.Div([
            html.H3("Dataset Structure"),
            html.Table([
                html.Tr([html.Th("Dimension"), html.Th("Coverage")]),
                *[html.Tr([html.Td(a), html.Td(b)]) for a, b in [
                    ("Products", "3,049"), ("Stores", "10"),
                    ("Departments", "7"), ("Categories", "3"),
                    ("States", "CA, TX, WI"), ("Sales missing values", "0"),
                    ("Negative sales values", "0"), ("Missing sell prices", "21.09%")
                ]]
            ], style=table_style())
        ], style={"background": "white", "padding": "20px", "borderRadius": "10px", "border": f'1px solid {COLORS["border"]}'})
    ], style={"display": "grid", "gridTemplateColumns": "repeat(2,minmax(0,1fr))", "gap": "20px"}),
    html.P("Missing sell prices reflect unavailable price-history records; they are not treated as zero prices.", style={"color": COLORS["muted"], "fontSize": "12px"})
])

forecast_page = html.Div([
    html.H2("Forecasting Intelligence"),
    html.P("Model evaluation, feature importance and prediction uncertainty.", style={"color": COLORS["muted"]}),
    html.H3("Full-Dataset Enhanced LightGBM"),
    card_grid([
        card("VALIDATION SERIES", "30,490", "All item-store series", COLORS["blue"]),
        card("VALIDATION OBSERVATIONS", "853,720", "28 days × 30,490 series", COLORS["cyan"]),
        card("MAE", "0.9684", "Mean absolute error", COLORS["green"]),
        card("RMSE", "1.9353", "Root mean squared error", COLORS["orange"]),
        card("WAPE", "69.85%", "One-day-ahead rolling evaluation", COLORS["red"])
    ]),
    html.H3("Fixed-Origin Model Comparison | 100 Series"),
    html.P("TFT performs best on MAE and WAPE; LightGBM performs best on RMSE.", style={"color": COLORS["muted"]}),
    graph(fig_compare), graph(fig_wape),
    html.H3("LightGBM Feature Importance"), graph(fig_importance),
    html.H3("Prediction Interval Calibration"), graph(fig_coverage),
    html.P("Prediction intervals were evaluated on 100 item-store series over one 28-day fixed-origin validation window. Empirical coverage is not a guarantee of future coverage.", style={"color": COLORS["muted"], "fontSize": "12px"})
])

hierarchy_page = html.Div([
    html.H2("Hierarchical Reconciliation"),
    html.P("Darts MinT reconciliation | California Store CA_1 | HOBBIES category", style={"color": COLORS["muted"]}),
    card_grid([
        card("TOTAL COMPONENTS", "568", "Subset hierarchy", COLORS["blue"]),
        card("CATEGORY NODES", "1", "HOBBIES category", COLORS["cyan"]),
        card("DEPARTMENT NODES", "2", "HOBBIES departments", COLORS["green"]),
        card("BOTTOM SERIES", "565", "Item-store series", COLORS["orange"]),
        card("FORECAST HORIZON", "28 Days", "Daily reconciled forecasts", COLORS["blue"])
    ]),
    graph(fig_hierarchy), graph(fig_hierarchy_structure),
    html.Div([
        html.H3("Reconciliation Method"),
        html.P("Darts MinT (Minimum Trace) reconciliation was applied to the CA_1 HOBBIES hierarchy using historical covariance information."),
        html.H3("Hierarchy Structure"),
        html.P("The hierarchy contains one category node, two department nodes and 565 bottom-level item series."),
        html.H3("Scope and Interpretation"),
        html.P("This is a subset demonstration, not a reconciliation of the complete 39,791-node M5 hierarchy. The 28-day forecast uses seasonal-naive base forecasts followed by MinT reconciliation.")
    ], style={"background": "white", "padding": "22px", "borderRadius": "10px", "lineHeight": "1.7"})
])

shortage_table = html.Table([
    html.Thead(html.Tr([
        html.Th("Item ID"), html.Th("Forecast Demand"),
        html.Th("Recommended Orders"), html.Th("Unmet Demand"),
        html.Th("Solver Status")
    ], style={"background": COLORS["navy"], "color": "white"})),
    html.Tbody([
        html.Tr([
            html.Td(row.item_id),
            html.Td(f"{row.forecast_demand:.2f}", style={"textAlign": "right"}),
            html.Td(f"{row.total_ordered:.0f}", style={"textAlign": "right"}),
            html.Td(f"{row.unmet_demand:.2f}", style={"textAlign": "right", "color": COLORS["red"], "fontWeight": "bold"}),
            html.Td(row.solver_status)
        ], style={"background": "#F8FAFC" if i % 2 == 0 else "white", "borderBottom": f'1px solid {COLORS["border"]}'})
        for i, row in enumerate(shortages.itertuples(index=False))
    ])
], style=table_style())

inventory_page = html.Div([
    html.H2("Inventory Intelligence"),
    html.P("OR-Tools inventory planning and optimization results.", style={"color": COLORS["muted"]}),
    card_grid([
        card("OPTIMIZED ITEMS", "565", "CA_1 HOBBIES subset", COLORS["blue"]),
        card("FORECAST DEMAND", "13,309.65", "28-day forecast units", COLORS["cyan"]),
        card("RECOMMENDED ORDERS", "9,580", "Total order quantity", COLORS["green"]),
        card("UNMET DEMAND", "152.76", "Portfolio shortage units", COLORS["red"]),
        card("OPTIMAL SOLUTIONS", "472", "Optimal solver status", COLORS["green"]),
        card("FEASIBLE SOLUTIONS", "93", "Optimality not proven", COLORS["orange"])
    ]),
    html.Div([graph(fig_inventory), graph(fig_cost)], style={"display": "grid", "gridTemplateColumns": "repeat(2,minmax(0,1fr))", "gap": "20px"}),
    graph(fig_solver),
    html.H3("Top 10 Items by Unmet Demand"),
    shortage_table,
    html.P("Optimization uses simulated inventory, lead-time and cost assumptions. FEASIBLE solutions are valid but not proven optimal.", style={"color": COLORS["muted"], "fontSize": "12px"})
])

insight_cards = html.Div([
    html.Div([
        html.H3(title, style={"color": COLORS["navy"]}),
        html.P(finding, style={"fontWeight": "bold", "lineHeight": "1.5"}),
        html.P(detail, style={"color": COLORS["muted"], "fontSize": "13px", "lineHeight": "1.6"})
    ], style={"background": "white", "padding": "20px", "borderRadius": "10px", "border": f'1px solid {COLORS["border"]}', "borderLeft": f"4px solid {accent}"})
    for (title, finding, detail), accent in zip(insights, [
        COLORS["blue"], COLORS["cyan"], COLORS["green"],
        COLORS["orange"], COLORS["red"], COLORS["blue"]
    ])
], style={"display": "grid", "gridTemplateColumns": "repeat(2,minmax(0,1fr))", "gap": "16px"})

insights_page = html.Div([
    html.H2("Business Insights"),
    html.P("Key findings, operational implications and project limitations.", style={"color": COLORS["muted"]}),
    insight_cards,
    html.Div([
        html.H3("Project Limitations"),
        html.Ul([html.Li(x, style={"marginBottom": "12px", "lineHeight": "1.6"}) for x in limitations])
    ], style={"background": "#FFF7ED", "border": "1px solid #FED7AA", "borderRadius": "10px", "padding": "22px", "marginTop": "24px"})
])

PAGES = {
    "overview": overview,
    "data": data_page,
    "forecast": forecast_page,
    "hierarchy": hierarchy_page,
    "inventory": inventory_page,
    "insights": insights_page
}

NAV = [
    ("overview", "Executive Overview"),
    ("data", "Data Intelligence"),
    ("forecast", "Forecasting Intelligence"),
    ("hierarchy", "Hierarchical Reconciliation"),
    ("inventory", "Inventory Intelligence"),
    ("insights", "Business Insights")
]

app = Dash(__name__)
app.title = "Demand Forecasting & Inventory Intelligence"

def nav_button(key, label, active):
    return html.Button(label, id=f"nav-{key}", n_clicks=0, style={
        "display": "block", "width": "100%", "textAlign": "left",
        "padding": "13px 12px", "marginBottom": "8px",
        "border": "none", "borderRadius": "7px",
        "background": COLORS["blue"] if key == active else "transparent",
        "color": "white", "cursor": "pointer", "fontSize": "13px"
    })


def nav_style(active, key):
    selected = active == key
    return {
        "padding": "12px 18px",
        "border": "none",
        "borderBottom": f"3px solid {COLORS['blue']}" if selected else "3px solid transparent",
        "borderRadius": "7px 7px 0 0",
        "background": "#EFF6FF" if selected else "transparent",
        "color": COLORS["blue"] if selected else COLORS["muted"],
        "fontWeight": "bold" if selected else "500",
        "fontSize": "13px",
        "cursor": "pointer",
        "whiteSpace": "nowrap",
        "flex": "1 1 140px"
    }

def make_navigation(active):
    return [
        html.Button(
            label,
            id=f"nav-{key}",
            n_clicks=0,
            style=nav_style(active, key)
        )
        for key, label in NAV
    ]

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Div(
                    "DEMAND INTELLIGENCE",
                    style={
                        "fontSize": "21px",
                        "fontWeight": "bold",
                        "letterSpacing": "0.4px",
                        "color": "white"
                    }
                ),
                html.Div(
                    "Forecasting & Inventory Optimization",
                    style={
                        "fontSize": "12px",
                        "color": "#CBD5E1",
                        "marginTop": "5px"
                    }
                )
            ]),
            html.Div(
                "M5 FORECASTING PROJECT",
                style={
                    "fontSize": "11px",
                    "fontWeight": "bold",
                    "letterSpacing": "1px",
                    "color": "#BFDBFE",
                    "border": "1px solid #45617F",
                    "borderRadius": "20px",
                    "padding": "8px 13px"
                }
            )
        ], style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "gap": "16px",
            "maxWidth": "1600px",
            "margin": "0 auto",
            "padding": "19px 30px"
        })
    ], style={"background": COLORS["navy"]}),

    html.Div([
        html.Div(
            make_navigation("overview"),
            id="top-navigation",
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "5px",
                "alignItems": "center",
                "maxWidth": "1600px",
                "margin": "0 auto",
                "padding": "8px 24px 0"
            }
        )
    ], style={
        "background": "white",
        "borderBottom": f'1px solid {COLORS["border"]}',
        "position": "sticky",
        "top": "0",
        "zIndex": "1000"
    }),

    html.Div(
        PAGES["overview"],
        id="main-content",
        style={
            "maxWidth": "1600px",
            "width": "100%",
            "boxSizing": "border-box",
            "margin": "0 auto",
            "padding": "30px"
        }
    )
], style={
    "background": COLORS["background"],
    "minHeight": "100vh",
    "fontFamily": "Arial, sans-serif",
    "color": COLORS["text"],
    "margin": "0"
})

@app.callback(
    Output("main-content", "children"),
    Output("top-navigation", "children"),
    *[Input(f"nav-{key}", "n_clicks") for key, _ in NAV]
)
def navigate(*clicks):
    active = ctx.triggered_id.replace("nav-", "") if ctx.triggered_id else "overview"
    return PAGES[active], make_navigation(active)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8051, debug=False)
