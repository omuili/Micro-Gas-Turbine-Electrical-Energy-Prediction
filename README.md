# Micro Gas Turbine Electrical Energy Prediction: A Numerical Modeling Approach

## Project Overview

This repository contains a comprehensive machine learning approach for predicting the electrical power output of a 3-kilowatt micro gas turbine based on input control voltage. The project addresses the challenging task of modeling the temporal dynamics of gas turbines, particularly the time delay between input changes and output response.

## Dataset

The dataset consists of 8 time series experiments (6 for training, 2 for testing) representing the gas turbine's behavior under diverse conditions. Each experiment contains:
- Time (seconds)
- Input control voltage (volts)
- Electrical output power (watts)

The time series vary in duration from 6,495 to 11,820 data points with a resolution of approximately 1 second, corresponding to approximately 1.8 to 3.3 hours.

## Features

- **Temporal feature engineering**: Lag features, moving averages, derivatives
- **Advanced modeling**: Multiple ML algorithms for time-series prediction
- **Comprehensive model comparison**: Testing and evaluation of 10 different algorithms
- **3D visualization**: Interactive plots showing relationships between time, voltage, and power
- **Hyperparameter optimization**: Fine-tuning for optimal performance

## Results

Our model comparison reveals that tree-based models perform best for this prediction task:

| Rank | Model | Train RMSE | Test RMSE | Train R² | Test R² | Train MAE | Test MAE |
|------|-------|------------|-----------|----------|---------|-----------|----------|
| 1 | XGBoost | 308.66 | 357.29 | 0.818 | 0.805 | 166.58 | 198.33 |
| 2 | Random Forest | 308.65 | 359.85 | 0.818 | 0.802 | 166.57 | 202.15 |
| 3 | MLP Regressor | 313.61 | 360.74 | 0.812 | 0.801 | 165.52 | 199.05 |
| 4 | Linear Regression | 318.66 | 361.02 | 0.806 | 0.801 | 175.96 | 206.92 |
| 5 | Ridge Regression | 318.86 | 361.20 | 0.806 | 0.801 | 174.87 | 204.97 |
| 6 | Gradient Boosting | 309.11 | 362.62 | 0.817 | 0.799 | 167.25 | 205.79 |
| 7 | LightGBM | 308.70 | 363.12 | 0.818 | 0.798 | 166.78 | 206.17 |
| 8 | Lasso Regression | 327.23 | 368.96 | 0.795 | 0.792 | 182.58 | 207.19 |
| 9 | ElasticNet | 333.81 | 373.96 | 0.787 | 0.786 | 207.47 | 231.14 |
| 10 | SVR | 331.51 | 377.76 | 0.790 | 0.782 | 148.18 | 183.93 |

**Best Performing Models**: XGBoost, Random Forest, MLP Regressor

## Repository Structure

```
micro-gas-turbine-prediction/
├── data/
│   ├── train.zip  # Training experiments
│   └── test.zip   # Testing experiments
├── notebooks/
│   └── main_analysis.ipynb  # Complete analysis workflow
├── src/
│   ├── data_processing.py   # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── models.py  # Model implementations
│   └── visualization.py  # Plotting and visualization tools
├── results/
│   ├── figures/  # Generated plots and visualizations
│   └── models/   # Saved model files
├── README.md
└── requirements.txt
```

## Key Findings

1. Time delay between input voltage changes and power response averages around 12-18 seconds
2. Feature engineering, particularly lag features, significantly improves prediction accuracy
3. Tree-based models (XGBoost, Random Forest) outperform other methods for this application
4. Different experiments show varying patterns, with some featuring rectangular changes and others showing continuous variations

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/micro-gas-turbine-prediction.git
cd micro-gas-turbine-prediction

# Install dependencies
pip install -r requirements.txt

# Run the main analysis notebook
jupyter notebook notebooks/main_analysis.ipynb
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM

## Future Work

- Real-time prediction system for operational environments
- Integration of physics-informed neural networks (PINNs)
- Exploration of anomaly detection for turbine monitoring
- Improved visualization tools for time-series analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code or methodology in your research, please cite:

```
@software{gas_turbine_prediction,
  author = {Your Name},
  title = {Micro Gas Turbine Electrical Energy Prediction},
  year = {2025},
  url = {https://github.com/yourusername/micro-gas-turbine-prediction}
}
```

## Acknowledgments

- Dataset provided by the UC Irvine Machine Learning Repository
- Original data collected by Pawel Bielski and Dustin Kottonau from Karlsruhe Institute of Technology
