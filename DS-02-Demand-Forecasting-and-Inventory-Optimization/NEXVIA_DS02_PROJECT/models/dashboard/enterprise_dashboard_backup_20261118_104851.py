
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
    left_margin = 150 if figure is fig_importance else 65
    right_margin = 85 if figure is fig_inventory else 35

    figure.update_layout(
        autosize=True,
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=COLORS["text"]
        ),
        margin=dict(
            l=left_margin,
            r=right_margin,
            t=75,
            b=65
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial, sans-serif",
            bordercolor=COLORS["border"]
        )
    )

    return dcc.Graph(
        figure=figure,
        config={
            "displaylogo": False,
            "responsive": True
        },
        style={"width": "100%"}
    )

def table_style():
    return {
        "width": "100%",
        "borderCollapse": "collapse",
        "background": "white",
        "fontSize": "12px",
        "tableLayout": "fixed",
        "marginBottom": "16px"
    }

def table_header(text, width=None, align="left"):
    style = {
        "padding": "12px 10px",
        "textAlign": align,
        "color": "white",
        "background": COLORS["navy"],
        "fontWeight": "bold",
        "whiteSpace": "normal",
        "overflowWrap": "anywhere",
        "verticalAlign": "middle"
    }
    if width:
        style["width"] = width
    return html.Th(text, style=style)

def table_cell(text, align="left"):
    return html.Td(text, style={
        "padding": "10px",
        "textAlign": align,
        "verticalAlign": "top",
        "whiteSpace": "normal",
        "overflowWrap": "anywhere",
        "lineHeight": "1.5",
        "borderBottom": f'1px solid {COLORS["border"]}'
    })

def table_row(cells, index):
    return html.Tr(
        cells,
        style={"background": "#F8FAFC" if index % 2 == 0 else "white"}
    )

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

# Business insights
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
    "The complete M5 hierarchy contains 39,791 nodes. The Darts MinT experiment covers 568 components in the CA_1 HOBBIES subset, not the complete hierarchy.",
    "TFT and the fixed-origin model comparison use 100 item-store series, while the enhanced LightGBM rolling evaluation covers all 30,490 series.",
    "Prophet forecasts nationwide aggregate demand, so its 6.19% WAPE is not directly comparable with item-store model results.",
    "The reported fixed-origin RMSSE is an unweighted mean across series, not official M5 WRMSSE.",
    "The inventory results depend on the chosen simulated cost, initial-stock, lead-time and order-quantity assumptions."
]

insight_cards = html.Div([
    html.Div([
        html.H3(title, style={"color": COLORS["navy"]}),
        html.P(finding, style={"fontWeight": "bold", "lineHeight": "1.5"}),
        html.P(detail, style={"color": COLORS["muted"], "fontSize": "13px", "lineHeight": "1.6"})
    ], style={
        "background": "white",
        "padding": "20px",
        "borderRadius": "10px",
        "border": f'1px solid {COLORS["border"]}',
        "borderLeft": f"4px solid {accent}"
    })
    for (title, finding, detail), accent in zip(insights, [
        COLORS["blue"], COLORS["cyan"], COLORS["orange"],
        COLORS["green"], COLORS["blue"], COLORS["red"], COLORS["cyan"]
    ])
], style={
    "display": "grid",
    "gridTemplateColumns": "repeat(2,minmax(0,1fr))",
    "gap": "16px"
})

insights_page = html.Div([
    html.H2("Business Insights"),
    html.P(
        "What the forecasting, reconciliation and inventory experiments reveal for demand planning.",
        style={"color": COLORS["muted"], "lineHeight": "1.6"}
    ),

    html.H3("Key Findings"),
    insight_cards,

    html.H3("Business Recommendations", style={"marginTop": "30px"}),
    html.Div([
        html.Div([
            html.H4("1. Use forecasts at the appropriate planning level"),
            html.P("Use item-store forecasts for localized replenishment decisions, and aggregate forecasts for broader capacity and demand planning.")
        ]),
        html.Div([
            html.H4("2. Account for intermittent demand"),
            html.P("Avoid treating zero-sales observations as missing data. Consider demand variability and item-level history when setting replenishment policies.")
        ]),
        html.Div([
            html.H4("3. Monitor forecast errors and uncertainty"),
            html.P("Track MAE, RMSE and WAPE alongside prediction interval coverage. Reassess models as new sales observations become available.")
        ]),
        html.Div([
            html.H4("4. Validate replenishment assumptions before deployment"),
            html.P("Replace simulated inventory, lead-time and cost inputs with real operational data before using the optimization outputs for purchasing decisions.")
        ]),
        html.Div([
            html.H4("5. Preserve aggregation consistency"),
            html.P("Use reconciliation checks when forecasts are consumed across item, department and category planning levels.")
        ])
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(2,minmax(0,1fr))",
        "gap": "14px",
        "background": "white",
        "padding": "20px",
        "border": f'1px solid {COLORS["border"]}',
        "borderRadius": "10px",
        "lineHeight": "1.6"
    }),

    html.H3("Project Limitations and Assumptions", style={"marginTop": "30px"}),
    html.Div([
        html.Ul([
            html.Li(x, style={"marginBottom": "11px", "lineHeight": "1.6"})
            for x in limitations
        ])
    ], style={
        "background": "#FFF7ED",
        "border": "1px solid #FED7AA",
        "borderRadius": "10px",
        "padding": "20px 24px"
    }),

    html.H3("Project Conclusion", style={"marginTop": "30px"}),
    html.Div([
        html.P(
            "This project demonstrates an end-to-end demand forecasting and inventory planning workflow using the M5 dataset. The full dataset was profiled, and enhanced LightGBM evaluation covered all 30,490 item-store series."
        ),
        html.P(
            "A controlled fixed-origin experiment compared Seasonal Naive, Enhanced LightGBM and TFT on the same 100-series subset. TFT achieved the strongest MAE, WAPE and RMSSE, while LightGBM achieved the lowest RMSE. Hierarchical reconciliation passed the defined coherence tolerance, and the inventory optimization prototype produced valid solutions for all 565 selected items."
        ),
        html.P(
            "The results demonstrate the value of combining forecasting, hierarchical consistency and constrained inventory planning. Further operational validation with actual inventory, procurement and service-level data would be required before production deployment.",
            style={"fontWeight": "bold"}
        )
    ], style={
        "background": "white",
        "padding": "22px",
        "border": f'1px solid {COLORS["border"]}',
        "borderLeft": f"4px solid {COLORS['blue']}",
        "borderRadius": "10px",
        "lineHeight": "1.8"
    })
])


overview = html.Div([
    html.H2("Executive Overview"),
    html.P("End-to-end demand forecasting, hierarchical reconciliation and inventory optimization.", style={"color": COLORS["muted"]}),
    card_grid([
        card("ITEM-STORE SERIES", "30,490", "Products × stores", COLORS["blue"]),
        card("ITEM-STORE-DAY SALES RECORDS", "58.33M", "1,913 days × 30,490 item-store series", COLORS["cyan"]),
        card("FORECAST HORIZON", "28 Days", "Daily forecast window", COLORS["green"]),
        card("INVENTORY ITEMS", "565", "CA_1 HOBBIES subset", COLORS["orange"]),
        card("RECOMMENDED ORDERS", "9,580", "Total order quantity", COLORS["green"]),
        card("UNMET DEMAND", "152.76", "Portfolio shortage units", COLORS["red"])
    ]),
    html.H3("Model Performance Snapshot"),
    html.P("Evaluation populations and protocols differ; these WAPE values are not a direct model ranking.", style={"color": COLORS["muted"]}),
    html.Table([
        html.Thead(html.Tr([
            table_header("Model", "20%"),
            table_header("Evaluation Protocol", "32%"),
            table_header("Population", "20%"),
            table_header("Horizon", "13%"),
            table_header("WAPE", "15%", "right")
        ])),
        html.Tbody([
            table_row([table_cell(value) for value in row[:-1]] + [table_cell(row[-1], "right")], i)
            for i, row in enumerate([
                ("Seasonal Naive", "Fixed-origin baseline", "100 series", "28 days", "113.33%"),
                ("Enhanced LightGBM", "Recursive fixed-origin", "100 series", "28 days", "92.73%"),
                ("TFT", "Direct fixed-origin", "100 series", "28 days", "85.40%"),
                ("Enhanced LightGBM", "Rolling one-day-ahead", "30,490 series", "1 day", "69.85%"),
                ("Prophet", "Nationwide aggregate holdout", "1 aggregate series", "28 days", "6.19%")
            ])
        ])
    ], style={**table_style(), "tableLayout": "fixed", "marginBottom": "18px"}),
    html.P("The first three rows form the comparable fixed-origin experiment. The final two rows use different populations and evaluation protocols and are reported separately for context.", style={"color": COLORS["muted"], "fontSize": "12px", "lineHeight": "1.6"}),
    html.P("Among the three models evaluated on the same 100-series fixed-origin experiment, TFT achieved the lowest WAPE of 85.40%.", style={"color": COLORS["green"], "fontWeight": "bold", "fontSize": "13px", "lineHeight": "1.6"}),
    html.P("Prophet forecasts nationwide aggregate sales, while LightGBM and TFT results below are evaluated at item-store level. Their WAPE values therefore measure different forecasting tasks and should not be ranked directly.", style={"color": COLORS["muted"], "fontSize": "12px", "lineHeight": "1.6"}),
    card_grid([
        card("PROPHET WAPE", "6.19%", "Nationwide aggregate · 28-day validation", COLORS["blue"]),
        card("LIGHTGBM WAPE", "69.85%", "30,490 series · Rolling one-day-ahead", COLORS["cyan"]),
        card("LIGHTGBM · FIXED-ORIGIN WAPE", "92.73%", "100 series · Recursive 28-day forecast", COLORS["orange"]),
        card("TFT WAPE", "85.40%", "100 series · Fixed-origin 28-day forecast", COLORS["green"])
    ]),
    html.Div([graph(fig_prophet), graph(fig_daily)], style={"display": "grid", "gridTemplateColumns": "repeat(2,minmax(0,1fr))", "gap": "20px"}),
    html.P("Inventory optimization uses simulated inventory and cost assumptions.", style={"color": COLORS["muted"], "fontSize": "12px"})
])

data_page = html.Div([
    html.H2("Data Intelligence"),
    html.P("M5 Forecasting dataset profile and historical demand characteristics.", style={"color": COLORS["muted"]}),
    card_grid([
        card("ITEM-STORE SERIES", "30,490", "3,049 products × 10 stores", COLORS["blue"]),
        card("ITEM-STORE-DAY SALES RECORDS", "58.33M", "1,913 daily observations per series", COLORS["cyan"]),
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
                html.Thead(html.Tr([
                    table_header("Dimension", "55%"),
                    table_header("Coverage", "45%", "center")
                ])),
                html.Tbody([
                    table_row([
                        table_cell(a),
                        table_cell(b, "center")
                    ], i)
                    for i, (a, b) in enumerate([
                        ("Products", "3,049"), ("Stores", "10"),
                        ("Departments", "7"), ("Categories", "3"),
                        ("States", "CA, TX, WI"), ("Sales missing values", "0"),
                        ("Negative sales values", "0"), ("Missing sell prices", "21.09%")
                    ])
                ])
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
    html.P("Verified fixed-origin evaluation | 100 CA_1 HOBBIES series | Origin: March 27, 2016 | Forecast: March 28 – April 24, 2016. TFT leads on MAE, WAPE and RMSSE; Enhanced LightGBM leads on RMSE.", style={"color": COLORS["muted"]}),
    graph(fig_compare), graph(fig_wape),
    html.H3("LightGBM Feature Importance"), graph(fig_importance),
    html.H3("Prediction Interval Calibration"), graph(fig_coverage),
    html.P("Verified TFT prediction intervals: 77.82% empirical coverage for the nominal 80% interval and 94.57% for the nominal 96% interval. Evaluation covers 100 item-store series over one 28-day fixed-origin window; empirical coverage is not a guarantee of future coverage.", style={"color": COLORS["muted"], "fontSize": "12px"})
])

hierarchy_page = html.Div([
    html.H2("Hierarchical Reconciliation"),
    html.P("Darts MinT reconciliation | California Store CA_1 | HOBBIES category", style={"color": COLORS["muted"]}),
    card_grid([
        card("TOTAL COMPONENTS", "568", "Subset hierarchy", COLORS["blue"]),
        card("CATEGORY NODES", "1", "HOBBIES category", COLORS["cyan"]),
        card("DEPARTMENT NODES", "2", "HOBBIES departments", COLORS["green"]),
        card("BOTTOM SERIES", "565", "Item-store series", COLORS["orange"]),
        card("FORECAST HORIZON", "28 Days", "Daily reconciled forecasts", COLORS["blue"]),
        card("MAX COHERENCE ERROR", "0.0005493", "84 checks · Tolerance 0.001", COLORS["green"])
    ]),
    html.P(
        "Numerical coherence validation passed: 3 parent-child relationships checked over all 28 forecast days (84 daily checks). Mean absolute coherence error: 0.0001787.",
        style={"color": COLORS["muted"], "fontSize": "12px"}
    ),
    graph(fig_hierarchy), graph(fig_hierarchy_structure),
    html.Div([
        html.H3("Understanding the Dataset and Hierarchy"),
        html.P("The 58.33 million figure represents daily sales observations: 30,490 item-store series across 1,913 historical days. The 39,791 figure represents nodes in the complete M5 aggregation hierarchy across its different levels. These are different measures: observations count time-series records, while hierarchy nodes count series at bottom and aggregate levels."),
        html.P("This experiment reconciles a manageable CA_1 HOBBIES subset containing 565 bottom-level item series and 3 aggregate nodes, for 568 components in total."),
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
        html.Th("Item ID", style={"width": "38%", "textAlign": "left", "padding": "12px 10px"}),
        html.Th("Forecast Demand", style={"width": "15%", "textAlign": "right", "padding": "10px", "padding": "12px 10px"}),
        html.Th("Recommended Orders", style={"width": "17%", "textAlign": "right", "padding": "10px", "padding": "12px 10px"}),
        html.Th("Unmet Demand", style={"width": "15%", "textAlign": "right", "padding": "10px", "padding": "12px 10px"}),
        html.Th("Solver Status", style={"width": "15%", "textAlign": "center", "padding": "10px", "padding": "12px 10px"})
    ], style={
        "background": COLORS["navy"],
        "color": "white"
    })),
    html.Tbody([
        html.Tr([
            html.Td(row.item_id, style={"textAlign": "left", "fontWeight": "500", "padding": "10px"}),
            html.Td(f"{row.forecast_demand:.2f}", style={"textAlign": "right", "padding": "10px"}),
            html.Td(f"{row.total_ordered:.0f}", style={"textAlign": "right", "padding": "10px"}),
            html.Td(f"{row.unmet_demand:.2f}", style={
                "textAlign": "right", "padding": "10px",
                "color": COLORS["red"],
                "fontWeight": "bold"
            }),
            html.Td(row.solver_status, style={"textAlign": "center", "padding": "10px"})
        ], style={
            "background": "#F8FAFC" if i % 2 == 0 else "white",
            "borderBottom": f'1px solid {COLORS["border"]}'
        })
        for i, row in enumerate(shortages.itertuples(index=False))
    ])
], style={
    **table_style(),
    "tableLayout": "fixed",
    "fontSize": "12px",
    "border": f'1px solid {COLORS["border"]}',
    "marginBottom": "18px"
})
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
    html.H3("Optimization Assumptions"),
    html.P(
        "The following values are simulated business assumptions because the M5 dataset does not provide actual inventory or operating costs.",
        style={"color": COLORS["muted"], "fontSize": "12px"}
    ),
    html.Table([
        html.Thead(html.Tr([
            html.Th("Assumption", style={"width": "38%", "padding": "12px 16px", "textAlign": "left"}),
            html.Th("Value", style={"width": "20%", "padding": "12px 16px", "textAlign": "right"}),
            html.Th("Unit / Meaning", style={"width": "42%", "padding": "12px 16px", "textAlign": "left"})
        ], style={"background": COLORS["navy"], "color": "white"})),
        html.Tbody([
            html.Tr([
                html.Td(label, style={"padding": "11px 16px", "textAlign": "left"}),
                html.Td(value, style={"padding": "11px 16px", "textAlign": "right", "fontWeight": "bold"}),
                html.Td(unit, style={"padding": "11px 16px", "textAlign": "left"})
            ], style={
                "background": "#F8FAFC" if i % 2 == 0 else "white",
                "borderBottom": f'1px solid {COLORS["border"]}'
            })
            for i, (label, value, unit) in enumerate([
                ("Lead time", "2", "Days"),
                ("Holding cost", "2.00", "Per unit per day"),
                ("Stockout penalty", "150.00", "Per unit"),
                ("Ordering cost", "50.00", "Per order"),
                ("Unit purchase cost", "60.00", "Per unit"),
                ("Maximum order quantity", "100", "Units"),
                ("Initial stock", "7", "Days of forecast demand")
            ])
        ])
    ], style={
        **table_style(),
        "tableLayout": "fixed",
        "marginBottom": "24px",
        "border": f'1px solid {COLORS["border"]}'
    }),
    html.H3("Top 10 Items by Unmet Demand"),
    shortage_table,
    html.P("Recommended order quantity represents the additional units the optimization model suggests ordering for an item on a particular day. It is based on forecast demand, simulated starting inventory, replenishment constraints and cost assumptions; it is not an actual purchase order.", style={"color": COLORS["muted"], "fontSize": "12px", "lineHeight": "1.7"}),
    html.Div([
        html.H3("Understanding Solver Outcomes"),
        html.P("472 OPTIMAL solutions: the solver found and proved an optimal solution for these item-level problems."),
        html.P("93 FEASIBLE solutions: the solver found valid solutions, but did not prove global optimality within the available solving time."),
        html.P("Both statuses represent valid solutions. FEASIBLE should not be interpreted as a failed optimization.", style={"color": COLORS["muted"], "fontSize": "12px"})
    ], style={"background": "white", "padding": "20px", "borderRadius": "10px", "border": f'1px solid {COLORS["border"]}', "marginTop": "16px", "lineHeight": "1.7"}),
    html.P("Optimization uses simulated inventory, lead-time and cost assumptions. The M5 dataset does not contain actual inventory positions or procurement costs.", style={"color": COLORS["muted"], "fontSize": "12px"})
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


config_page = html.Div([
    html.H2("Experimental Configuration"),
    html.P(
        "Dataset scope, evaluation windows, model configurations and computational boundaries used in the experiments.",
        style={"color": COLORS["muted"]}
    ),
    html.H3("Dataset and Evaluation Scope"),
    html.Table([
        html.Thead(html.Tr([
            table_header("Experiment", "17%"),
            table_header("Population / Data", "27%"),
            table_header("Training / Input", "26%"),
            table_header("Validation / Horizon", "30%")
        ])),
        html.Tbody([
            table_row([table_cell(value) for value in row], i)
            for i, row in enumerate([
                ("M5 dataset", "30,490 series · 58,327,370 observations", "1,913 historical days", "Calendar contains 1,969 days"),
                ("Enhanced LightGBM", "All 30,490 series", "4,573,500 sampled training rows · 150 dates per series", "853,720 rows · 28-day rolling one-day-ahead validation"),
                ("Fixed-origin comparison", "100 CA_1 HOBBIES series · 2,800 predictions", "Origin: March 27, 2016", "March 28 – April 24, 2016 · 28 days"),
                ("Prophet", "Nationwide aggregate · 1,913 observations", "First 1,885 days", "Final 28 days"),
                ("Darts MinT", "565 bottom series · 568 components", "CA_1 HOBBIES subset", "28-day reconciled forecast"),
                ("Inventory optimization", "565 CA_1 HOBBIES item-series", "OR-Tools optimization", "28-day planning horizon")
            ])
        ])
    ], style=table_style()),
    html.H3("Model Configuration"),
    html.Table([
        html.Thead(html.Tr([
            table_header("Model", "19%"),
            table_header("Configuration", "51%"),
            table_header("Evaluation", "30%")
        ])),
        html.Tbody([
            table_row([table_cell(value) for value in row], i)
            for i, row in enumerate([
                ("Enhanced LightGBM", "Poisson objective · 1,000 estimators · Learning rate 0.05 · 31 leaves · Column and row sampling 0.8 · Random seed 42 · 2 jobs", "18 features · 150 sampled training dates per series"),
                ("TFT", "Encoder 56 days · Prediction 28 days · Hidden size 16 · 4 attention heads · Dropout 0.1 · Hidden continuous size 8 · 7 quantiles", "Direct 28-day forecast · Median quantile used for point metrics"),
                ("Seasonal Naive", "Seasonal lag-7 baseline", "Same 100-series fixed-origin evaluation"),
                ("Prophet", "Yearly and weekly seasonality · No daily seasonality", "Nationwide aggregate · Final 28-day holdout"),
                ("Darts MinT", "Historical covariance-based MinT reconciliation", "565 bottom series and 3 aggregate nodes")
            ])
        ])
    ], style=table_style()),
    html.H3("Evaluation Metrics and Interpretation"),
    html.Ul([
        html.Li("MAE and RMSE measure point-forecast errors."),
        html.Li("WAPE is reported separately for each evaluation population and protocol; values from different populations are not directly comparable."),
        html.Li("RMSSE in the fixed-origin comparison is an unweighted mean across series, not official M5 WRMSSE."),
        html.Li("TFT interval coverage is empirical coverage on one 28-day validation window, not a guarantee of future coverage.")
    ]),
    html.H3("Computational Scope and Reproducibility"),
    html.P(
        "The full M5 dataset was profiled and the enhanced LightGBM evaluation covered all 30,490 item-store series. Computationally intensive TFT, fixed-origin comparison, Darts reconciliation and inventory optimization experiments used explicitly identified subsets. The Darts experiment reconciled 568 components rather than the complete 39,791-node M5 hierarchy."
    ),
    html.P(
        "The LightGBM training sample used 150 sampled dates per series; its validation covered all 853,720 observations. The TFT experiment used 100 series. These scopes are stated separately to distinguish full-dataset analysis from subset-based experiments.",
        style={"color": COLORS["muted"], "fontSize": "12px"}
    )
])

PAGES = {
    "overview": overview,
    "data": data_page,
    "forecast": forecast_page,
    "hierarchy": hierarchy_page,
    "inventory": inventory_page,
    "insights": insights_page,
    "config": config_page
}

NAV = [
    ("overview", "Executive Overview"),
    ("data", "Data Intelligence"),
    ("forecast", "Forecasting Intelligence"),
    ("hierarchy", "Hierarchical Reconciliation"),
    ("inventory", "Inventory Intelligence"),
    ("insights", "Business Insights"),
    ("config", "Experimental Configuration")
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


NAV_ICONS = {
    "overview": "▦",
    "data": "◫",
    "forecast": "⌁",
    "hierarchy": "⌘",
    "inventory": "▤",
    "insights": "✧",
    "config": "⚙"
}

def nav_style(active, key):
    selected = active == key
    return {
        "display": "inline-flex",
        "alignItems": "center",
        "justifyContent": "center",
        "gap": "8px",
        "padding": "15px 12px 13px",
        "margin": "0 3px",
        "border": "none",
        "borderBottom": f"2px solid {COLORS['blue']}" if selected else "2px solid transparent",
        "borderRadius": "0",
        "background": "transparent",
        "color": COLORS["blue"] if selected else "#64748B",
        "fontWeight": "700" if selected else "500",
        "fontSize": "13px",
        "cursor": "pointer",
        "whiteSpace": "nowrap",
        "flex": "0 0 auto"
    }

def make_navigation(active):
    return [
        html.Button(
            [
                html.Span(
                    NAV_ICONS[key],
                    style={
                        "fontSize": "17px",
                        "lineHeight": "1",
                        "color": COLORS["blue"] if key == active else "#94A3B8"
                    }
                ),
                html.Span(label)
            ],
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
                "flexWrap": "nowrap",
                "gap": "5px",
                "alignItems": "center",
                "justifyContent": "flex-start",
                "maxWidth": "1600px",
                "margin": "0 auto",
                "padding": "0 20px",
                "background": "white",
                "overflowX": "auto",
                "overflowY": "hidden",
                "whiteSpace": "nowrap",
                "scrollbarWidth": "thin"
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
