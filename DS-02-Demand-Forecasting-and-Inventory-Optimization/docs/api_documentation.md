# API Documentation

## 1. Overview

This document defines the proposed interfaces for connecting demand forecasting, inventory optimization, and dashboard components.

The initial implementation may consume local CSV or Parquet outputs directly. REST API endpoints are optional future integration points.

## 2. Proposed Endpoints

### GET /api/health

Purpose: Check application availability.

Expected response:

```json
{
  "status": "healthy"
}
```

### GET /api/forecasts

Purpose: Retrieve demand forecasts.

Query parameters:
- store_id
- item_id
- start_date
- end_date
- model

### GET /api/metrics

Purpose: Retrieve forecasting evaluation metrics.

Expected fields:
- model
- WAPE
- RMSSE

### GET /api/inventory/recommendations

Purpose: Retrieve inventory replenishment recommendations.

Expected fields:
- item_id
- store_id
- current_stock
- forecast_demand
- safety_stock
- reorder_quantity

## 3. Data Exchange

Supported data formats:

- JSON for API responses
- CSV for forecast exports
- Parquet for analytical datasets

## 4. Error Handling

Proposed HTTP status codes:

| Code | Meaning |
|---|---|
| 200 | Successful request |
| 400 | Invalid parameters |
| 404 | Requested data not found |
| 500 | Internal server error |

## 5. Security Considerations

- Validate incoming parameters.
- Avoid exposing credentials.
- Restrict access to sensitive resources.
- Keep configuration in environment variables.

## 6. Implementation Status

These endpoints are proposed interfaces, not claims of an existing deployed API. Their implementation will depend on the final integration architecture.
