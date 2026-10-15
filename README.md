---
title: MediRisk Actuarial Engine
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MediRisk: Healthcare Cost Actuarial Risk Prediction Engine

An interactive machine learning decision engine for predicting healthcare insurance charges and simulating actuarial risk counterfactuals.

## Overview
* **Model**: Random Forest Regressor (300 estimators, max depth 6)
* **Performance**: Test $R^2 = 0.8668$, Test $\text{MAE} = \$1,973.45$
* **Features**: Dynamic risk scoring, counterfactual simulations, peer benchmarking, and embedded actuarial research report.

## Endpoints
* `/` or `/app.html` - Interactive Actuarial Dashboard
* `/api/predict` - Real-time underwriting quote & component attribution
* `/api/simulate` - Multi-year trajectories & lifestyle intervention savings
* `/api/model_info` - Model hyperparameters and feature importance metrics
* `/report.pdf` - Complete research publication PDF
