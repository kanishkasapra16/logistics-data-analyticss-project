# Week 4 - Predictive Modeling and Optimization in Logistics

## Objective
Predict delivery time using Python and translate predictive insights into an illustrative
resource-allocation optimization.

## Dataset
A hypothetical/synthetic dataset of 500 shipments is included. Features include region,
transport mode, priority, shipment volume, distance, weather, congestion index and the target
`delivery_days`.

## Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Tuned Random Forest using 5-fold cross-validation and GridSearchCV

## Evaluation
- MAE
- RMSE
- R-squared
- 5-fold cross-validation

## Optimization
A linear programming example allocates 1,000 capacity units across four regions while applying
minimum/maximum capacity constraints and risk-adjusted costs.

## Run
```bash
pip install -r requirements.txt
python week4_predictive_optimization.py
```

## Important
The data and optimization costs are illustrative. They demonstrate methodology and should not
be interpreted as actual performance or operational recommendations for a real company.
