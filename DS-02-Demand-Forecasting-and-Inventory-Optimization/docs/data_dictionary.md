# Data Dictionary

## 1. Dataset Overview

The project uses the M5 Forecasting dataset, consisting of historical Walmart sales, calendar events, and product pricing information.

## 2. Data Sources

| Dataset | Description | 
|---|---|
| sales_train_validation | Historical daily product sales |
| calendar | Dates, events, SNAP indicators |
| sell_prices | Product selling prices by store and week |

## 3. Key Fields

| Field | Description |
|---|---|
| id | Unique product identifier |
| item_id | Product identifier |
| dept_id | Department identifier |
| cat_id | Product category |
| store_id | Store identifier |
| state_id | State identifier |
| date | Sales date |
| sales | Daily units sold |
| wm_yr_wk | Walmart year-week identifier |
| sell_price | Product selling price |
| event_name_1 | Primary calendar event |
| event_type_1 | Primary event category |
| snap_CA | SNAP indicator for California |
| snap_TX | SNAP indicator for Texas |
| snap_WI | SNAP indicator for Wisconsin |

## 4. Engineered Features

| Feature | Description |
|---|---|
| Lag features | Historical sales values |
| Rolling mean | Average sales over a previous window |
| Rolling standard deviation | Historical sales variability |
| Calendar features | Date-related characteristics |
| Price features | Product pricing information |

## 5. Forecast Output Fields

| Field | Description |
|---|---|
| item_id | Forecasted product |
| store_id | Forecasted store |
| date | Forecast date |
| model | Forecasting model |
| forecast | Predicted demand |
| lower_bound | Lower prediction interval |
| upper_bound | Upper prediction interval |

## 6. Inventory Output Fields

| Field | Description |
|---|---|
| item_id | Product identifier |
| store_id | Store identifier |
| current_stock | Available inventory |
| forecast_demand | Expected demand |
| safety_stock | Additional stock buffer |
| reorder_quantity | Recommended replenishment |
| expected_shortage | Unmet demand estimate |

## 7. Notes

The final schema and field names will be aligned with the actual processed datasets and implementation outputs.
