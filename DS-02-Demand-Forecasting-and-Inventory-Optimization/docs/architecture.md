# System Architecture

## 1. Overview

The Nationwide Demand Forecasting and Inventory Optimization system follows a modular data processing and machine learning architecture.

It integrates historical retail sales data, forecasting algorithms, hierarchical reconciliation, prediction uncertainty estimation, and inventory optimization into a unified workflow.

## 2. Architecture Components

### 2.1 Data Ingestion
- M5 historical sales dataset
- Calendar dataset
- Product selling price dataset

### 2.2 Data Processing
- Data preprocessing and validation
- Calendar and price integration
- Feature engineering
- Lag-based features
- Rolling statistical features

### 2.3 Forecasting Engine

The system evaluates four forecasting approaches:

- Naive Forecasting: Baseline demand prediction.
- Prophet: Statistical time-series forecasting.
- LightGBM: Gradient boosting using engineered features.
- Temporal Fusion Transformer (TFT): Deep learning-based time-series forecasting.

### 2.4 Hierarchical Reconciliation

Darts is used to support hierarchical forecasting and reconciliation.

The objective is to maintain consistency between forecasts at different aggregation levels, such as product, department, category, store, and total sales.

### 2.5 Prediction Intervals

Prediction intervals represent uncertainty around demand forecasts.

Lower and upper forecast bounds will be used to support inventory risk assessment.

### 2.6 Inventory Optimization

Google OR-Tools is used to formulate inventory optimization problems.

The optimization module considers:
- Forecast demand
- Available inventory
- Safety stock
- Reorder quantities
- Inventory constraints

### 2.7 Dashboard

Plotly Dash provides an interactive interface for:

- Demand forecast visualization
- Actual versus predicted demand
- WAPE and RMSSE evaluation
- Prediction interval visualization
- Inventory recommendations

## 3. End-to-End Workflow

1. Load M5 sales, calendar, and price datasets.
2. Process historical sales observations.
3. Generate time-series features.
4. Train forecasting models.
5. Generate demand predictions.
6. Reconcile hierarchical forecasts.
7. Estimate prediction intervals.
8. Optimize inventory decisions.
9. Evaluate model performance.
10. Display results through the dashboard.

## 4. Technology Stack

| Layer | Technology |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Storage | Parquet |
| Development | Google Colab |
| Forecasting | Prophet, LightGBM, TFT |
| Reconciliation | Darts |
| Optimization | Google OR-Tools |
| Visualization | Plotly Dash |

## 5. Design Considerations

- Modular architecture for independent component development.
- Efficient processing of large-scale retail datasets.
- Separation of forecasting and inventory optimization.
- Reproducible model evaluation.
- Clear documentation of data and model outputs.

## 6. Implementation Status

This architecture represents the intended system design. Individual components will be marked complete after implementation and validation.
