# Week 4 - Predictive Modeling and Optimization in Logistics

## Objective

This project focuses on predictive modeling and optimization in a logistics environment using Python.

## Problem Statement

The objective is to predict shipment delivery time and use the predictive results to support logistics resource allocation.

## Dataset

A hypothetical dataset of 500 simulated shipments is used. It contains:

- Region
- Transport mode
- Priority
- Shipment volume
- Distance
- Weather condition
- Congestion index
- Delivery time

The target variable is `delivery_days`.

## Predictive Models

The project compares:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Five-fold cross-validation and GridSearchCV are also used for model validation and hyperparameter tuning.

## Optimization

The project connects predictive analytics with an illustrative linear programming problem for regional capacity allocation.

The optimization considers:

- Regional capacity
- Risk-adjusted cost
- Minimum capacity constraints
- Maximum capacity constraints

## Project Structure

```text
Week4_Predictive_Modeling/
│
├── README.md
├── week4_predictive_optimization.py
├── requirements.txt
│
├── data/
│   └── hypothetical_logistics_prediction_data.csv
│
└── outputs/
    ├── 01_model_comparison_rmse.png
    ├── 02_actual_vs_predicted.png
    ├── 03_optimized_capacity.png
    └── flow.png
