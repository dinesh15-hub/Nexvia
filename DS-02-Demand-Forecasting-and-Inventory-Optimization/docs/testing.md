# Testing Strategy and Performance Evaluation

## 1. Overview

The project uses a structured testing approach to validate data processing, forecasting accuracy, hierarchical reconciliation, prediction intervals, and inventory optimization.

## 2. Data Pipeline Testing

| Test | Expected Result |
|---|---|
| Missing value validation | Missing values identified |
| Duplicate validation | Duplicate records identified |
| Date validation | Valid date ranges |
| Feature validation | Expected lag and rolling features generated |
| Dataset consistency | Correct product and store mappings |

## 3. Forecasting Model Evaluation

The following models will be evaluated:

- Naive Forecasting
- Prophet
- LightGBM
- Temporal Fusion Transformer (TFT)

### Evaluation Metrics

**Weighted Absolute Percentage Error (WAPE)**

WAPE measures total absolute forecasting error relative to total actual demand.

WAPE = Sum(|Actual - Forecast|) / Sum(Actual) × 100

**Root Mean Squared Scaled Error (RMSSE)**

RMSSE measures forecast error relative to a naive seasonal benchmark, with squared errors penalizing larger forecasting mistakes.

Lower values indicate better forecasting performance.

## 4. Hierarchical Reconciliation Testing

Validation includes:

- Consistency between aggregate and lower-level forecasts.
- Correct hierarchy mappings.
- No unexpected missing aggregation levels.
- Reconciled forecasts checked against aggregation constraints.

## 5. Prediction Interval Testing

Validation includes:

- Lower bounds do not exceed upper bounds.
- Forecasts fall within the generated interval where appropriate.
- Empirical coverage is measured against actual observations.
- Interval width is evaluated.

## 6. Inventory Optimization Testing

Validation includes:

- Non-negative reorder quantities.
- Non-negative ending inventory.
- Inventory balance consistency.
- Shortage calculations.
- Capacity and ordering constraints, where implemented.
- Feasibility of optimization solutions.

## 7. Dashboard Testing

Validation includes:

- Dashboard startup.
- Chart rendering.
- Filter functionality.
- Forecast data loading.
- Metric display.
- Inventory recommendation display.

## 8. Performance Evaluation

The final evaluation report will include:

| Component | Evaluation |
|---|---|
| Data pipeline | Processing time and dataset dimensions |
| Forecasting | WAPE and RMSSE |
| Reconciliation | Hierarchical consistency |
| Prediction intervals | Coverage and interval width |
| Optimization | Feasibility and objective value |
| Dashboard | Loading and interaction responsiveness |

## 9. Test Results

Actual test results, execution times, and model metrics will be added after implementation and validation.

No unverified performance values will be reported.

## 10. Conclusion

The testing strategy ensures that the forecasting and inventory optimization system is evaluated for correctness, reliability, and practical usefulness.   
