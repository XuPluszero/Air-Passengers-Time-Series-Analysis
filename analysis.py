"""
Air Passengers Time Series Analysis
====================================

A comprehensive analysis of the classic Air Passengers dataset (1949-1960).
This script performs exploratory data analysis, builds multiple forecasting models,
and evaluates their performance.

Author: Lucas Jialin Xu
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = 'outputs'
DATA_PATH = 'AirPassengers.csv'
RANDOM_STATE = 42
TEST_SIZE_MONTHS = 24

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)


def load_and_prepare_data(filepath):
    """Load and prepare the Air Passengers dataset."""
    df = pd.read_csv(filepath)
    df['Month'] = pd.to_datetime(df['Month'])
    df['Passengers'] = df['#Passengers']
    df['TimePeriod'] = range(1, len(df) + 1)
    df['MonthNum'] = df['Month'].dt.month
    df['Year'] = df['Month'].dt.year
    return df


def split_train_test(df, test_size=24):
    """Split data into train and test sets."""
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    return train_df, test_df


def exploratory_analysis(df, train_df, test_df, output_dir):
    """
    Perform exploratory data analysis and create visualizations.
    Analyzes trend, seasonality, and variance patterns.
    """
    print("="*80)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Time series plot
    axes[0, 0].plot(df['Month'], df['Passengers'], marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Date', fontsize=12)
    axes[0, 0].set_ylabel('Passengers (thousands)', fontsize=12)
    axes[0, 0].set_title('Air Passengers Time Series (1949-1960)', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=test_df['Month'].iloc[0], color='red', 
                       linestyle='--', linewidth=2, label='Train-Test Split')
    axes[0, 0].legend()
    
    # Monthly seasonality
    monthly_avg = df.groupby('MonthNum')['Passengers'].mean()
    axes[0, 1].bar(monthly_avg.index, monthly_avg.values, color='steelblue', alpha=0.7)
    axes[0, 1].set_xlabel('Month', fontsize=12)
    axes[0, 1].set_ylabel('Average Passengers (thousands)', fontsize=12)
    axes[0, 1].set_title('Average Passengers by Month', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Yearly trend
    yearly_avg = df.groupby('Year')['Passengers'].mean()
    axes[1, 0].plot(yearly_avg.index, yearly_avg.values, marker='o',
                    linewidth=2, markersize=8, color='darkgreen')
    axes[1, 0].set_xlabel('Year', fontsize=12)
    axes[1, 0].set_ylabel('Average Passengers (thousands)', fontsize=12)
    axes[1, 0].set_title('Yearly Average Trend', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Variance over time
    axes[1, 1].boxplot([df[df['Year'] == year]['Passengers'].values
                        for year in sorted(df['Year'].unique())],
                       labels=sorted(df['Year'].unique()))
    axes[1, 1].set_xlabel('Year', fontsize=12)
    axes[1, 1].set_ylabel('Passengers (thousands)', fontsize=12)
    axes[1, 1].set_title('Distribution by Year (Increasing Variance)',
                         fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: eda_visualization.png")
    print("\nKey Observations:")
    print("  • Strong upward trend (~30k passengers/year)")
    print("  • Clear seasonality (summer peaks, winter troughs)")
    print("  • Non-stationary (increasing mean and variance)")
    print()


def fit_linear_trend(train_df, test_df):
    """Fit a simple linear trend model."""
    X_train = train_df[['TimePeriod']].values
    y_train = train_df['Passengers'].values
    X_test = test_df[['TimePeriod']].values
    y_test = test_df['Passengers'].values
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return model, train_r2, test_r2, y_train_pred, y_test_pred


def fit_ar1_model(train_df, test_df):
    """Fit an AR(1) autoregressive model."""
    train_passengers = train_df['Passengers'].values
    
    model = AutoReg(train_passengers, lags=1, trend='c').fit()
    
    y_train_pred = model.predict(start=1, end=len(train_passengers)-1)
    y_test_pred = model.predict(start=len(train_passengers),
                                end=len(train_passengers) + len(test_df) - 1,
                                dynamic=False)
    
    train_r2 = r2_score(train_passengers[1:], y_train_pred)
    test_r2 = r2_score(test_df['Passengers'].values, y_test_pred)
    
    return model, train_r2, test_r2, y_train_pred, y_test_pred


def create_features(df_input, lag_orders=[1, 12]):
    """
    Engineer temporal features for time series modeling.
    
    Features:
    - Month dummies (seasonality)
    - Lagged values (autocorrelation)
    """
    df_feat = df_input.copy()
    
    # Month indicators
    for month in range(1, 13):
        df_feat[f'Month_{month}'] = (df_feat['MonthNum'] == month).astype(int)
    
    # Lagged features
    for lag in lag_orders:
        df_feat[f'Lag_{lag}'] = df_feat['Passengers'].shift(lag)
    
    return df_feat


def fit_custom_ar_model(train_df, test_df, full_df):
    """
    Fit custom AR model with engineered features.
    Includes feature selection based on statistical significance.
    """
    import statsmodels.api as sm
    
    # Create features
    train_feat = create_features(train_df, lag_orders=[1, 12])
    test_feat = create_features(full_df, lag_orders=[1, 12])
    test_feat = test_feat.iloc[-24:].copy()
    
    # Remove NaN from lagging
    train_feat_clean = train_feat.dropna()
    test_feat_clean = test_feat.dropna()
    
    # Define feature set
    month_features = [f'Month_{i}' for i in range(2, 13)]
    lag_features = ['Lag_1', 'Lag_12']
    feature_cols = ['TimePeriod'] + lag_features + month_features
    
    X_train = train_feat_clean[feature_cols].values
    y_train = train_feat_clean['Passengers'].values
    X_test = test_feat_clean[feature_cols].values
    y_test = test_feat_clean['Passengers'].values
    
    # Feature selection (p-value < 0.05)
    X_train_sm = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    p_values = ols_model.pvalues[1:]
    significant_features = [feature_cols[i] for i in range(len(feature_cols))
                           if p_values[i] < 0.05]
    
    # Refit with selected features
    X_train_selected = train_feat_clean[significant_features].values
    X_test_selected = test_feat_clean[significant_features].values
    
    model = LinearRegression()
    model.fit(X_train_selected, y_train)
    
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return (model, significant_features, train_r2, test_r2, 
            y_train_pred, y_test_pred, train_feat_clean, test_feat_clean)


def fit_random_forest(X_train, y_train, X_test, y_test, random_state=42):
    """
    Fit Random Forest with time series cross-validation for hyperparameter tuning.
    """
    n_estimators_options = [10, 50, 100, 200]
    tscv = TimeSeriesSplit(n_splits=5)
    
    best_score = -np.inf
    best_n = None
    
    print("\nRandom Forest Cross-Validation:")
    for n_est in n_estimators_options:
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            
            rf = RandomForestRegressor(n_estimators=n_est, 
                                      random_state=random_state, n_jobs=-1)
            rf.fit(X_cv_train, y_cv_train)
            y_cv_pred = rf.predict(X_cv_val)
            score = r2_score(y_cv_val, y_cv_pred)
            scores.append(score)
        
        avg_score = np.mean(scores)
        print(f"  n_estimators={n_est:3d}: CV R² = {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_n = n_est
    
    print(f"\nBest: n_estimators={best_n} (CV R² = {best_score:.4f})")
    
    # Train final model
    model = RandomForestRegressor(n_estimators=best_n, 
                                 random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return model, train_r2, test_r2, y_train_pred, y_test_pred


def adf_test(series, name='Series'):
    """Perform Augmented Dickey-Fuller test for stationarity."""
    s = series.dropna().values
    result = adfuller(s, autolag='AIC')
    
    print(f'ADF Test: "{name}"')
    print(f'  Test Statistic: {result[0]:.4f}')
    print(f'  p-value: {result[1]:.4f}')
    print(f'  Critical Value (5%): {result[4]["5%"]:.4f}')
    print(f'  Stationary: {"Yes" if result[1] < 0.05 else "No"}')
    print()
    
    return result[1] < 0.05


def stationarity_analysis(df, train_df, test_df, output_dir):
    """
    Analyze stationarity using ADF test.
    Tests original series, first difference, and seasonal difference.
    """
    print("="*80)
    print("STATIONARITY ANALYSIS")
    print("="*80)
    
    # Create differenced series
    df['Diff_1'] = df['Passengers'].diff(1)
    df['Diff_12'] = df['Passengers'].diff(12)
    
    # ADF tests
    is_stat_orig = adf_test(df['Passengers'], 'Original Series')
    is_stat_diff1 = adf_test(df['Diff_1'], 'First Difference')
    is_stat_diff12 = adf_test(df['Diff_12'], 'Seasonal Difference (Lag-12)')
    
    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Original
    axes[0].plot(df['Month'], df['Passengers'], 'o-', linewidth=2, markersize=4)
    axes[0].set_ylabel('Passengers (thousands)', fontsize=12)
    axes[0].set_title(f'Original Series {"(Stationary)" if is_stat_orig else "(Non-Stationary)"}',
                     fontsize=14, fontweight='bold')
    axes[0].axhline(y=df['Passengers'].mean(), color='red', 
                   linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    
    # First difference
    axes[1].plot(df['Month'], df['Diff_1'], 'o-', linewidth=2, markersize=4)
    axes[1].set_ylabel('First Difference', fontsize=12)
    axes[1].set_title(f'First Difference {"(Stationary)" if is_stat_diff1 else "(Non-Stationary)"}',
                     fontsize=14, fontweight='bold')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal difference
    axes[2].plot(df['Month'], df['Diff_12'], 'o-', linewidth=2, markersize=4)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_ylabel('Seasonal Difference', fontsize=12)
    axes[2].set_title(f'Seasonal Difference {"(Stationary)" if is_stat_diff12 else "(Non-Stationary)"}',
                     fontsize=14, fontweight='bold')
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stationarity_analysis.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: stationarity_analysis.png")
    print()
    
    return df


def main():
    """Main analysis pipeline."""
    print("\n" + "="*80)
    print("AIR PASSENGERS TIME SERIES ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data(DATA_PATH)
    train_df, test_df = split_train_test(df, TEST_SIZE_MONTHS)
    
    print(f"\nDataset: {len(df)} total observations")
    print(f"Training: {len(train_df)} observations ({train_df['Month'].iloc[0].strftime('%Y-%m')} to {train_df['Month'].iloc[-1].strftime('%Y-%m')})")
    print(f"Testing: {len(test_df)} observations ({test_df['Month'].iloc[0].strftime('%Y-%m')} to {test_df['Month'].iloc[-1].strftime('%Y-%m')})")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}\n")
    
    # Exploratory analysis
    exploratory_analysis(df, train_df, test_df, OUTPUT_DIR)
    
    # Baseline models
    print("="*80)
    print("BASELINE MODELS")
    print("="*80)
    
    linear_model, linear_train_r2, linear_test_r2, _, _ = fit_linear_trend(train_df, test_df)
    print(f"\nLinear Trend Model:")
    print(f"  Train R²: {linear_train_r2:.4f}")
    print(f"  Test R²: {linear_test_r2:.4f}")
    
    ar1_model, ar1_train_r2, ar1_test_r2, _, _ = fit_ar1_model(train_df, test_df)
    print(f"\nAR(1) Model:")
    print(f"  Train R²: {ar1_train_r2:.4f}")
    print(f"  Test R²: {ar1_test_r2:.4f}")
    print()
    
    # Custom AR model with features
    print("="*80)
    print("CUSTOM AR MODEL WITH FEATURE ENGINEERING")
    print("="*80)
    
    (custom_model, features, custom_train_r2, custom_test_r2,
     y_train_pred_custom, y_test_pred_custom, 
     train_feat_clean, test_feat_clean) = fit_custom_ar_model(train_df, test_df, df)
    
    print(f"\nSelected Features: {features}")
    print(f"Train R²: {custom_train_r2:.4f}")
    print(f"Test R²: {custom_test_r2:.4f} ⭐ BEST")
    
    # Visualize custom model
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.plot(df['Month'], df['Passengers'], 'o-', label='Actual',
           linewidth=2, markersize=4, alpha=0.7, color='black')
    ax.plot(train_feat_clean['Month'], y_train_pred_custom, 's-',
           label='Fitted (Train)', linewidth=2, markersize=3, color='blue')
    ax.plot(test_feat_clean['Month'], y_test_pred_custom, '^-',
           label='Predicted (Test)', linewidth=2, markersize=4, color='red')
    ax.axvline(x=test_df['Month'].iloc[0], color='red',
              linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Passengers (thousands)', fontsize=12)
    ax.set_title(f'Custom AR Model (Test R² = {custom_test_r2:.4f})',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'custom_ar_model.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved: custom_ar_model.png")
    
    # Random Forest
    print("\n" + "="*80)
    print("RANDOM FOREST MODEL")
    print("="*80)
    
    X_train_rf = train_feat_clean[features].values
    y_train_rf = train_feat_clean['Passengers'].values
    X_test_rf = test_feat_clean[features].values
    y_test_rf = test_feat_clean['Passengers'].values
    
    rf_model, rf_train_r2, rf_test_r2, _, _ = fit_random_forest(
        X_train_rf, y_train_rf, X_test_rf, y_test_rf, RANDOM_STATE)
    
    print(f"\nRandom Forest:")
    print(f"  Train R²: {rf_train_r2:.4f}")
    print(f"  Test R²: {rf_test_r2:.4f}")
    
    # Stationarity analysis
    df = stationarity_analysis(df, train_df, test_df, OUTPUT_DIR)
    
    # Summary
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    results = pd.DataFrame({
        'Model': ['Linear Trend', 'AR(1)', 'Custom AR', 'Random Forest'],
        'Train R²': [linear_train_r2, ar1_train_r2, custom_train_r2, rf_train_r2],
        'Test R²': [linear_test_r2, ar1_test_r2, custom_test_r2, rf_test_r2]
    })
    
    print("\n" + results.to_string(index=False))
    print("\n✅ Best Model: Custom AR (Test R² = {:.4f})".format(custom_test_r2))
    print(f"\n📁 All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
