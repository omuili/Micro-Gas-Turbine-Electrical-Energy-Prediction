# CSCA-5622-Supervised-Learning-Final-Project

# Micro Gas Turbine Electrical Energy Prediction

## Regular HCGT (Gas Turbine) Engagement NiceVCRNet (or similar branding)

This repository is for the 2025 STAT comprehensive Learning project. Real project is system-development featuring contributions to energy management applications and integrates MVG with applied machine learning methods.

The goal is to predict the electrical power output of a 3-kilowatt micro gas turbine based on input control voltage. We use multiple deep learning: CNN, RNNs, GRU, LSTM, XGBoost, SVR, MLP models, etc. By making use of a dataset of 71,225 temporal data points, primarily time, input_voltage, and el_power. (~1.8-3.3 hours)

Keywords: time series prediction, electrical energy forecasting, temporal dynamics, control systems, lag modeling, deep learning, machine learning, turbine modeling, and real-time prediction. We employ advanced preprocessing like lag conversion, sequence chunking, and feature engineering to transform raw temporal data into a complete model.

## Summary

| Feature Extraction Method | Modeling/Classification Method | Dataset | Accuracy | r² Score | Loss/RMSE (watts) |
|---|---|---|---|---|---|
| Raw Input Voltage | K-means | Total | 0.53240 | 0.23865 | 847.712 |
| Input Voltage + Lag(1) | XGBoost | Total | 0.78425 | 0.82376 | 377.894 |
| Input Voltage + Lag(1,5,10) | XGBoost | Total | 0.81459 | 0.85272 | 344.774 |
| Input Voltage + Moving Avg | Random Forest | Total | 0.82780 | 0.83713 | 361.825 |
| LSTM + RNN | Sequential Model | Total | 0.86582 | 0.87714 | 314.883 |
| LSTM + CNN + Moving Avg | Sequential Model | Total | 0.89815 | 0.89713 | 287.432 |
| LSTM + CNN + Lag | Augmentation (Tuned seq) | Total | 0.87844 | 0.88212 | 308.511 |
| LSTM + GRU | Augmentation (Tuned seq) | Total | 0.88030 | 0.88277 | 307.665 |
| LSTM + GRU + Moving | Augmentation (Tuned seq) | Total | 0.91422 | 0.91735 | 258.066 |
| LSTM + GRU + TIME | Augmentation (Tuned seq) | Total | 0.91760 | 0.91780 | 257.637 |
| LSTM + MLP | Augmented (3D Chunking) | Total | 0.90708 | 0.90895 | 271.022 |
| XGB | XGB | Total | 0.87875 | 0.87810 | 313.217 |
| SVR | SVR | Test | 0.78242 | 0.78210 | - |
| Linear | Linear | Test | 0.59883 | 0.59830 | - |
| GB | GB | Test | 0.85764 | 0.86340 | - |
| LGBM + GRU | SVR | Test | 0.87545 | 0.87340 | - |
| LGBM + CNN | GB | Novel | 0.86639 | 0.86612 | - |
| LGBM + RNN + Tuning | GB | Novel | 0.88690 | 0.89271 | - |
| LGBM + RNN + Tuning | SVR | Novel | 0.87234 | 0.87500 | - |

The results of using different feature extraction, preprocessing methods, and modeling methods have been organized above. Among them, the method using LSTM+GRU+TIME features with augmentation (Tuned sequential) method has the best accuracy and r² score, while the LSTM+GRU+Moving technique had the lowest RMSE. These algorithms work as black-box yet carry effective flow characteristics integration at granular turbine behavior, and can effectively capture the time-sensitive nature of the system's temporal-lag in input-output response through complex architecture. (STAT-NP)

Dynamically, modified net optimizes the absorption of the feature-space while reduces the latent variance. The recurrent approaches better handle the time-series aspect. Lag insertion has become critical to this architecture assembly, likely due to the micro gas turbine's inherent physical time delay of approximately 12-18 seconds for reaction to input control voltage changes. Shorter window extraction can sometimes confuse performance, and the inclusion of 3D-velocity scale also indicates that a wider temporal sensitivity partly necessary for micro turbine modeling. The increased utilization of movement-direction over raw-signal is reflected by the effect in moving average performance. The explicit time steps also appear beneficial for state transitions in the gas turbine's behavior.

Appropriate parameter tuning can allow the model to better adapt to the specific characteristics of the dataset. From the peaks in the table with optimized training parameters, tuning the input window parameters through cross-validation has proven effective.

## Architecture Diagram

```
                 Raw Temporal Data
                        ▼
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Feature Eng.    Preprocessing    Transformation
        │               │               │
        ▼               ▼               ▼
┌───────────────┐ ┌──────────┐ ┌───────────────┐
│Lag Features   │ │Scaling   │ │Time Windowing │
│Moving Averages│ │Chunking  │ │3D Sequencing  │
│Diff Features  │ │Filtering │ │State Detection│
└───────┬───────┘ └────┬─────┘ └───────┬───────┘
        │              │               │
        └──────────────┼───────────────┘
                       ▼
              ┌─────────────────┐
              │  Model Training │
              └────────┬────────┘
                       ▼
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
┌───────────┐   ┌─────────────┐   ┌───────────┐
│Traditional│   │Deep Learning│   │Ensembles  │
└─────┬─────┘   └──────┬──────┘   └─────┬─────┘
      ▼                ▼                ▼
┌───────────┐   ┌─────────────┐   ┌───────────┐
│Linear     │   │LSTM/GRU     │   │XGBoost    │
│Ridge      │   │CNN          │   │RandomForest│
│SVR        │   │MLP          │   │LightGBM   │
└─────┬─────┘   └──────┬──────┘   └─────┬─────┘
      │                │                │
      └────────────────┼────────────────┘
                       ▼
                ┌─────────────┐
                │Hyperparameter│
                │Optimization  │
                └──────┬───────┘
                       ▼
               ┌───────────────┐
               │ Final Model   │
               │ LSTM+GRU+TIME │
               └───────────────┘
```

## Directions for Improvement

* Explore more specialized data augmentation and selective preprocessing, particularly before training to enhance the model's generalization ability on new data

* Examine more effective methods of calculating discontinuously chunked embeddings to better visualize the time series of the data

* Conduct more detailed and comprehensive parameter tuning to find the optimal model configuration

* Attempt to combine multiple feature extraction and classification/learning methods, such as ensemble approaches for improved reliability and accuracy

* Add real-time prediction capabilities with feedback adaptation for continuous deployment in operational environments

* Investigate approaches to handle more extreme operational conditions and edge cases not seen in the current dataset

* Implement a physics-informed neural network (PINN) layer to integrate domain knowledge of gas turbine dynamics
