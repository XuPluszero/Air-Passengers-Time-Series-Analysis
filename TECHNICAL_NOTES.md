# Air Passengers Analysis - Technical Documentation

## Overview

This project implements multiple time series forecasting approaches on the classic Air Passengers dataset to predict monthly international airline passenger volumes. The analysis demonstrates the importance of feature engineering, stationarity testing, and proper model evaluation for temporal data.

## Dataset Description

**Name**: Air Passengers  
**Period**: January 1949 - December 1960  
**Frequency**: Monthly  
**Observations**: 144  
**Unit**: Thousands of passengers  

**Characteristics**:
- Strong upward trend (~30k passengers/year increase)
- Clear seasonal pattern (12-month cycle)
- Multiplicative seasonality (variance increases with level)
- Non-stationary series

## Methodology

### Data Split Strategy

- **Training Set**: 120 months (1949-01 to 1958-12) - 83.3%
- **Test Set**: 24 months (1959-01 to 1960-12) - 16.7%
- **Rationale**: Preserves temporal order, sufficient test period for seasonal evaluation

### Models Implemented

#### 1. Linear Trend Model (Baseline)
- **Purpose**: Capture long-term growth
- **Features**: Time period (t)
- **Result**: Train R² = 0.836, Test R² = -0.003
- **Limitation**: Ignores seasonality

#### 2. AR(1) Model (Baseline)
- **Purpose**: Capture short-term autocorrelation
- **Features**: Previous observation (t-1)
- **Result**: Train R² = 0.910, Test R² = -3.619
- **Limitation**: Assumes stationarity, misses seasonal patterns

#### 3. Custom AR Model (Best)
- **Purpose**: Comprehensive temporal modeling
- **Features**: 
  - Time period (trend)
  - Lag-1 (short-term autocorrelation)
  - Lag-12 (seasonal autocorrelation)
  - Month indicators (seasonal effects)
- **Feature Selection**: Retained 8 features (p-value < 0.05)
- **Result**: Train R² = 0.983, Test R² = **0.934** ⭐
- **Success Factors**: Captures trend + seasonality + autocorrelation

#### 4. Random Forest Regressor
- **Purpose**: Non-linear ensemble approach
- **Hyperparameter Tuning**: 5-fold Time Series CV
- **Best Configuration**: 100 estimators
- **Result**: Train R² = 0.995, Test R² = 0.624
- **Observation**: Overfits compared to linear model

### Feature Engineering

**Temporal Features**:
```python
- TimePeriod: Sequential index (1 to 144)
- MonthNum: Month of year (1-12)
- Year: Calendar year (1949-1960)
```

**Lagged Features**:
```python
- Lag_1: Previous month's passengers (t-1)
- Lag_12: Same month last year (t-12)
```

**Seasonal Features**:
```python
- Month_2 through Month_12: Binary indicators
- (Month_1 excluded to avoid multicollinearity)
```

**Selected Features** (p < 0.05):
- Lag_1: 0.4906 (short-term momentum)
- Lag_12: 0.5301 (seasonal pattern) ⭐ Most important
- Month_3, Month_6, Month_7: Positive (spring/summer boost)
- Month_9, Month_10, Month_11: Negative (fall decline)

### Stationarity Analysis

**Augmented Dickey-Fuller (ADF) Test Results**:

| Transformation | Test Statistic | p-value | Stationary? |
|---------------|----------------|---------|-------------|
| Original (Pₜ) | 0.815 | 0.992 | ❌ No |
| First Diff (ΔPₜ) | -2.829 | 0.054 | ❌ No |
| Seasonal Diff (Δ₁₂Pₜ) | -3.383 | 0.012 | ✅ Yes |

**Implications**:
- Original series violates stationarity (trend + seasonality)
- First differencing removes trend but preserves seasonality
- Seasonal differencing (lag-12) achieves stationarity
- Enables application of classical time series methods (ARIMA, etc.)

## Key Findings

### 1. Feature Engineering Dominates Model Complexity
- Simple linear model with good features (R² = 0.934) >> Complex Random Forest (R² = 0.624)
- Lag-12 feature is crucial (coefficient = 0.53)
- Explicit seasonality modeling essential

### 2. Model Selection Considerations
- **Linear models**: Better for extrapolation, interpretable
- **Ensemble methods**: Risk overfitting with limited time series data
- **Feature quality** > Model sophistication

### 3. Stationarity Testing is Essential
- Non-stationary data violates assumptions of many models
- ADF test provides objective assessment
- Appropriate transformations enable broader model applicability

### 4. Time Series CV is Critical
- Standard k-fold CV violates temporal ordering
- TimeSeriesSplit preserves causality
- Prevents data leakage from future to past

## Evaluation Metrics

**R² (Coefficient of Determination)**:
```
R² = 1 - (SS_res / SS_tot)
```
- Measures proportion of variance explained
- Range: (-∞, 1], where 1 = perfect fit
- Negative values indicate predictions worse than mean

**Out-of-Sample R² (OSR²)**:
- R² computed on held-out test set
- True measure of generalization
- Primary metric for model selection

## Code Structure

```
analysis.py
├── load_and_prepare_data()      # Data loading and preprocessing
├── split_train_test()            # Temporal train-test split
├── exploratory_analysis()        # EDA and visualizations
├── fit_linear_trend()            # Baseline linear model
├── fit_ar1_model()              # Baseline AR(1) model
├── create_features()             # Feature engineering pipeline
├── fit_custom_ar_model()        # Best model with feature selection
├── fit_random_forest()          # Random Forest with CV
├── adf_test()                   # Stationarity testing
├── stationarity_analysis()      # Differencing analysis
└── main()                       # Orchestration pipeline
```

## Reproducibility

**Random Seeds**:
- Random Forest: seed = 42
- All results are deterministic and reproducible

**Dependencies**:
```
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
statsmodels >= 0.13.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

**Execution**:
```bash
python analysis.py
```

## Output Files

### Visualizations
1. `eda_visualization.png`: Time series decomposition
2. `custom_ar_model.png`: Best model predictions
3. `stationarity_analysis.png`: ADF test results

### Console Output
- Model equations
- Performance metrics
- Statistical test results
- Feature importance

## Extensions and Future Work

**Potential Improvements**:
1. SARIMA modeling (Seasonal ARIMA)
2. Prophet (Facebook's time series forecasting)
3. LSTM/GRU neural networks
4. Ensemble of multiple approaches
5. Confidence intervals for predictions
6. Anomaly detection

**Additional Analyses**:
1. Forecast horizon sensitivity
2. Walk-forward validation
3. Residual diagnostics
4. Multi-step ahead forecasting
5. Exogenous variables (economic indicators, holidays)

## References

**Dataset**:
- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.

**Methods**:
- Dickey, D. A., & Fuller, W. A. (1979). "Distribution of the estimators for autoregressive time series with a unit root." *Journal of the American Statistical Association*.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

## License

MIT License - See LICENSE file for details.

## Contact

For questions or contributions, please open an issue or submit a pull request.

---

**Last Updated**: November 2025
