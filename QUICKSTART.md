# Quick Start Guide

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/air-passengers-analysis.git
cd air-passengers-analysis
```

2. **Create a virtual environment** (recommended):
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the Analysis

**Basic usage**:
```bash
python analysis.py
```

This will:
- ✅ Load and analyze the Air Passengers dataset
- ✅ Train and evaluate all models
- ✅ Generate visualizations in the `outputs/` directory
- ✅ Print detailed results to console

### Expected Runtime
- **Total**: ~30 seconds
- **Random Forest CV**: ~15-20 seconds (most time-intensive)
- **Other models**: <10 seconds combined

## Output Files

After running, check the `outputs/` directory:

```
outputs/
├── eda_visualization.png          # Exploratory data analysis
├── custom_ar_model.png            # Best model predictions
└── stationarity_analysis.png      # ADF test results
```

## Customization

### Modify Train-Test Split

Edit `analysis.py`, line 23:
```python
TEST_SIZE_MONTHS = 24  # Change this value
```

### Change Random Forest Hyperparameters

Edit `analysis.py`, line 268:
```python
n_estimators_options = [10, 50, 100, 200]  # Add/remove values
```

### Add New Features

Modify the `create_features()` function (line 151):
```python
def create_features(df_input, lag_orders=[1, 12]):
    # Add your custom features here
    df_feat['custom_feature'] = ...
    return df_feat
```

## Interpreting Results

### Model Performance Metrics

**R² (Coefficient of Determination)**:
- Range: (-∞, 1]
- 1.0 = Perfect predictions
- 0.0 = Predictions as good as mean
- Negative = Predictions worse than mean

**Key Comparison**:
- **Train R²**: Model fit on training data (can overfit)
- **Test R²**: True generalization performance (what matters!)

### Expected Results

With the default configuration:
```
Model              Train R²    Test R²
─────────────────────────────────────
Linear Trend        0.8356    -0.0030
AR(1)               0.9099    -3.6192
Custom AR           0.9833     0.9343  ⭐ Best
Random Forest       0.9946     0.6239
```

## Troubleshooting

### Issue: Import errors
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Issue: FileNotFoundError for AirPassengers.csv
**Solution**: Ensure the CSV file is in the same directory as `analysis.py`

### Issue: Plots not displaying
**Solution**: Plots are saved to `outputs/` directory automatically. Check there first.

### Issue: Random Forest taking too long
**Solution**: Reduce CV folds or n_estimators options:
```python
# In fit_random_forest()
tscv = TimeSeriesSplit(n_splits=3)  # Reduce from 5 to 3
n_estimators_options = [50, 100]     # Test fewer options
```

## Understanding the Code

### Main Pipeline Flow

```
main()
  ├─ Load data (144 observations)
  ├─ Split train/test (120/24)
  ├─ Exploratory analysis
  │   └─ Visualize trend, seasonality, variance
  ├─ Fit baseline models
  │   ├─ Linear trend
  │   └─ AR(1)
  ├─ Feature engineering
  │   ├─ Create lags (1, 12)
  │   ├─ Create month dummies
  │   └─ Feature selection (p-value)
  ├─ Fit advanced models
  │   ├─ Custom AR (linear)
  │   └─ Random Forest (with CV)
  ├─ Stationarity analysis
  │   ├─ ADF test (original)
  │   ├─ ADF test (differenced)
  │   └─ Visualize transformations
  └─ Print summary
```

### Key Functions

**Data Processing**:
- `load_and_prepare_data()`: Loads CSV, adds features
- `split_train_test()`: Temporal split
- `create_features()`: Engineer lag and seasonal features

**Modeling**:
- `fit_linear_trend()`: Simple baseline
- `fit_ar1_model()`: Statsmodels AR(1)
- `fit_custom_ar_model()`: Best model with feature selection
- `fit_random_forest()`: Ensemble with Time Series CV

**Analysis**:
- `adf_test()`: Augmented Dickey-Fuller stationarity test
- `exploratory_analysis()`: EDA visualizations
- `stationarity_analysis()`: Differencing analysis

## Advanced Usage

### Use as a Module

```python
from analysis import load_and_prepare_data, fit_custom_ar_model

# Load data
df = load_and_prepare_data('AirPassengers.csv')

# Custom split
train = df.iloc[:-12]  # Use last 12 months as test
test = df.iloc[-12:]

# Fit model
model, features, train_r2, test_r2, *_ = fit_custom_ar_model(train, test, df)

print(f"Test R²: {test_r2:.4f}")
```

### Extend with New Models

```python
def fit_my_custom_model(train_df, test_df):
    """Your custom model here."""
    # ... implementation
    return model, train_r2, test_r2, y_pred_train, y_pred_test
```

Add to `main()`:
```python
my_model, my_train_r2, my_test_r2, _, _ = fit_my_custom_model(train_df, test_df)
print(f"My Model Test R²: {my_test_r2:.4f}")
```

## Contributing

We welcome contributions! Areas for improvement:
- Additional model implementations (SARIMA, Prophet, LSTM)
- More visualization options
- Interactive dashboards (Plotly, Streamlit)
- Multi-step ahead forecasting
- Confidence intervals

Please open an issue first to discuss proposed changes.

## Getting Help

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: See TECHNICAL_NOTES.md for detailed methodology

## Additional Resources

**Time Series Analysis**:
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [statsmodels Documentation](https://www.statsmodels.org/)
- [scikit-learn Time Series](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

**Python Data Science**:
- [pandas Documentation](https://pandas.pydata.org/)
- [matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

---

Happy forecasting! 📈
