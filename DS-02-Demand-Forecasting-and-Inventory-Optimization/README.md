![Demand Forecasting and Inventory Optimization banner](docs/screenshots/banner.png)

# Nationwide Demand Forecasting and Inventory Optimization

**Nexvia Technologies | DS-02 Project**

An end-to-end demand forecasting and inventory optimization system built on the M5 Forecasting dataset. It combines statistical forecasting, machine learning, hierarchical reconciliation, uncertainty estimation and operations research, and presents the results in an interactive Plotly Dash dashboard.

![Executive Overview](docs/screenshots/01-executive-overview.png)

## Highlights

| Area | Result |
|---|---|
| Scale | 30,490 item-store series · 58.33M daily sales records · 1,913 days |
| Best model (same 100-series, 28-day fixed-origin test) | **TFT**: MAE 1.0574, WAPE 85.40%, RMSSE 0.8093 |
| Full-dataset LightGBM (rolling one-day-ahead, all 30,490 series) | MAE 0.9684, RMSE 1.9353, WAPE 69.85% |
| Prediction intervals (TFT, nominal 80% / 96%) | 77.82% / 94.57% empirical coverage |
| Reconciliation (Darts MinT, CA_1 HOBBIES, 568 components) | Passed coherence audit: max error 0.0005493 vs tolerance 0.001 |
| Inventory optimization (OR-Tools, 565 items) | 9,580 units recommended · 472 OPTIMAL, 93 FEASIBLE solutions |

> WAPE values come from different populations and evaluation protocols (for example, 100 series vs. 30,490 series, 28-day vs. one-day-ahead, item-store vs. nationwide aggregate). Only the three models in the fixed-origin experiment are directly comparable.

## Project Overview

The project forecasts retail demand from historical Walmart sales data, keeps forecasts consistent across the product hierarchy, quantifies uncertainty, and turns the forecasts into replenishment recommendations.

## Objectives

- Analyze historical retail sales data and identify demand patterns.
- Perform time-series preprocessing and feature engineering.
- Develop and evaluate multiple demand forecasting approaches.
- Maintain consistency across hierarchical forecasts.
- Estimate demand uncertainty using prediction intervals.
- Optimize inventory replenishment using Google OR-Tools.
- Visualize forecasting performance and inventory recommendations through an interactive dashboard.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Data Storage | Parquet, CSV, JSON |
| Forecasting | Seasonal Naive, Prophet, LightGBM, Temporal Fusion Transformer (TFT) |
| Reconciliation | Darts (MinT) |
| Optimization | Google OR-Tools |
| Visualization | Plotly, Dash |
| Development | Google Colab, GitHub |

## Dataset

**M5 Forecasting Dataset** (Walmart): historical sales, calendar information and selling prices.

- 30,490 item-store time series
- 3,049 products · 10 stores · 3 states (CA, TX, WI)
- 1,913 days of historical data
- 68.2% of sales observations are zero (intermittent demand)

Source: [M5 Forecasting – Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

## System Architecture

![System Architecture](docs/architecture%202.png)

### Implementation Workflow

1. **Data Sources:** Historical sales, calendar events and selling prices.
2. **Preprocessing:** Data quality checks, time-series reshaping, feature engineering and Parquet generation.
3. **Forecasting:** Seasonal Naive, Prophet, LightGBM and Temporal Fusion Transformer.
4. **Evaluation:** Forecast accuracy assessment, model comparison and prediction interval analysis.
5. **Reconciliation:** Hierarchical forecast reconciliation using Darts.
6. **Inventory Optimization:** Demand-driven replenishment recommendations using Google OR-Tools.
7. **Dashboard:** Interactive forecasting, evaluation, reconciliation and inventory intelligence using Plotly and Dash.

## Interactive Dashboard

A seven-page dashboard with a Light / Dark theme toggle and a responsive layout. The selected page and theme are remembered in the browser.

| Page | What it shows |
|---|---|
| 1. Executive Overview | Headline KPIs, model performance snapshot, Prophet forecast and daily demand vs. recommended orders |
| 2. Data Intelligence | Dataset profile and zero-sales / intermittency analysis |
| 3. Forecasting Intelligence | Full-dataset LightGBM results, fixed-origin model comparison, feature importance and interval calibration |
| 4. Hierarchical Reconciliation | Darts MinT reconciled forecasts and coherence validation |
| 5. Inventory Intelligence | OR-Tools results, inventory position, cost breakdown, assumptions and top shortages |
| 6. Business Insights | Key findings, recommendations, limitations and conclusion |
| 7. Experimental Configuration | Dataset scope, evaluation windows and model configurations |

### Screenshots

**Forecasting Intelligence**
![Forecasting Intelligence](docs/screenshots/02-forecasting-intelligence.png)

**Feature Importance and Prediction Interval Calibration**
![Feature importance and interval calibration](docs/screenshots/03-feature-importance-calibration.png)

**Hierarchical Reconciliation**
![Hierarchical Reconciliation](docs/screenshots/04-hierarchical-reconciliation.png)

**Inventory Intelligence**
![Inventory Intelligence](docs/screenshots/05-inventory-intelligence.png)

**Business Insights**
![Business Insights](docs/screenshots/06-business-insights.png)

## Key Components

### 1. Demand Forecasting
- Statistical and machine learning forecasting approaches.
- Lag-based and rolling-window features.
- Nationwide aggregate forecasting.
- Recursive multi-step forecasting.

### 2. Forecast Evaluation
- Fixed-origin model comparison.
- Rolling-origin LightGBM evaluation.
- Forecast accuracy metrics.
- Prediction interval estimation and validation.

### 3. Hierarchical Reconciliation
- Hierarchical forecast consistency.
- Darts reconciliation workflow.
- Bottom-level and aggregate forecast alignment.
- Coherence validation.

### 4. Inventory Optimization
- Demand-based inventory planning.
- Simulated starting inventory.
- Lead-time and ordering cost assumptions.
- Holding cost and stockout penalty considerations.
- Replenishment and order recommendations.

## Evaluation Metrics

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **WAPE:** Weighted Absolute Percentage Error.
- **RMSSE:** Root Mean Squared Scaled Error (unweighted mean across series, not the official M5 WRMSSE).

Evaluation protocols and forecast horizons are documented separately so that comparisons stay meaningful.

## Running the Project Locally

**1. Clone the repository**

```bash
git clone https://github.com/dinesh15-hub/Nexvia.git
cd Nexvia/DS-02-Demand-Forecasting-and-Inventory-Optimization
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the dashboard**

```bash
python models/dashboard/enterprise_dashboard.py
```

The dashboard will be available at **http://localhost:8051**.

**Note:** The dashboard uses pre-generated model outputs stored in the `models/` directory. Raw M5 datasets are excluded from GitHub due to their size. Download and place the required datasets in the local `data/` directory if you want to run the complete data processing and forecasting pipeline.

## Repository Structure

```text
DS-02-Demand-Forecasting-and-Inventory-Optimization/
│
├── docs/
│   ├── screenshots/
│   ├── api_documentation.md
│   ├── architecture 2.png
│   ├── architecture.md
│   ├── dashboard_design.md
│   ├── data_dictionary.md
│   ├── inventory_optimization.md
│   └── testing.md
│
├── models/
│   ├── darts_reconciliation/
│   ├── dashboard/
│   ├── forecast_evaluation/
│   └── inventory_optimization/
│
├── notebooks/
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Experimental Scope and Assumptions

- Full dataset used for feature engineering and full-scale LightGBM evaluation.
- TFT, the fixed-origin comparison and hierarchical reconciliation were evaluated on selected subsets to fit available computational resources.
- Hierarchical reconciliation covers 568 components of the CA_1 HOBBIES subset, not the complete 42,840-node M5 hierarchy.
- Inventory optimization covers 565 item-store series.
- Evaluation populations and protocols differ; WAPE comparisons should use matched evaluation settings.
- Price-reduction features are price-change proxies, not confirmed promotional events.
- Inventory stock, lead times and costs are simulated assumptions rather than observed M5 variables, so the optimization results are a prototype and not purchasing advice.

## Project Deliverables

- Source code and implementation notebooks.
- Data preprocessing and feature engineering pipeline.
- Forecasting models and evaluation results.
- Hierarchical reconciliation outputs.
- Prediction interval analysis.
- Inventory optimization outputs.
- Interactive dashboard.
- System architecture and technical documentation.
- Testing and evaluation documentation.
- Project presentation and individual reflection.

## Project Status

**Completed – DS-02 Final Submission | Nexvia Technologies**

## Individual Reflection

**What I built.** I built a forecasting and inventory optimization system on the M5 Walmart dataset. It covers Seasonal Naive, Prophet, LightGBM and TFT forecasts, prediction intervals, hierarchical reconciliation with Darts MinT, and replenishment planning with Google OR-Tools. The results are shown in a seven-page Plotly Dash dashboard.

**What I learned.**
- Evaluation design matters as much as the model. My WAPE values come from different populations and protocols (100 series vs. 30,490 series, 28-day vs. one-day-ahead, item-store vs. nationwide). Only the three fixed-origin models are directly comparable, and I stated that on the dashboard rather than ranking everything together.
- About 68% of sales records are zero. Intermittent demand makes percentage errors large, so I looked at several metrics (MAE, RMSE, WAPE, RMSSE) instead of one.
- Prediction intervals need checking. TFT's 80% and 96% intervals covered 77.82% and 94.57% of actual values, close to nominal but not exact, and only on one validation window.
- Reconciliation made the forecasts consistent across item, department and category levels, and I checked it numerically against a tolerance.
- M5 has no real inventory, lead-time or cost data. The optimization results therefore use simulated assumptions, which I documented clearly so they are not mistaken for real purchasing advice.

**Challenges.** TFT, reconciliation and optimization were too heavy to run on the full dataset with the compute I had, so I ran them on defined subsets and stated the scope explicitly. Keeping the numbers and wording consistent across the README, dashboard and evaluation files took care.

**What I would improve.** Evaluate TFT on more series and several time windows, use the official M5 WRMSSE metric, reconcile the full hierarchy, and replace the simulated inventory inputs with real operational data.

---

## Acknowledgement

Developed as part of the **DS-02 Internship Project at Nexvia Technologies**.
Dataset: M5 Forecasting (Walmart).

## Author

**Dinesh Chandra Reddy** · [GitHub Profile](https://github.com/dinesh15-hub)
