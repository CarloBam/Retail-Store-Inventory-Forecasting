# Retail Price Elasticity & Demand Forecasting System

**Target Audience**: Senior Data Science & Strategy Stakeholders  
**Tech Stack**: Python, Streamlit, Statsmodels, Prophet, TensorFlow, Plotly/Matplotlib

---

## 1. Business Context
Retailers face a dual challenge: avoiding stockouts while maximizing revenue through optimal pricing. This project delivers an **end-to-end analytical platform** that ingests raw sales data to solve two critical problems:
1.  **Demand Forecasting**: Predicting future inventory needs to optimize supply chain operations.
2.  **Pricing Intelligence**: Estimating price elasticity of demand to guide discount strategies and revenue optimization.

The solution is delivered as an interactive Streamlit web application, allowing non-technical stakeholders to explore data, run scenarios ("What-If analysis"), and export executive recommendations.

## 2. Dataset Description
The system is built to ingest standard retail transaction logs.

| Column | Type | Description |
| :--- | :--- | :--- |
| `Date` | Datetime | Transaction date (daily freq). |
| `Store ID` | Categorical | Unique identifier for retail location. |
| `Product ID` | Categorical | Unique identifier for SKU. |
| `Category` | Categorical | Product hierarchy (e.g., Electronics, Clothing). |
| `Units Sold` | Integer | **Target Variable**. Quantity demaned/sold. |
| `Price` | Float | Unit price at time of sale. |
| `Discount` | Float | Discount percentage applied (0-100). |
| `Competitor Pricing`| Float | Price of equivalent SKU at primary competitor. |
| `Holiday` / `Promo` | Binary | Flags for external events affecting demand. |

## 3. Analytical Approach

### A. Exploratory Data Analysis (EDA)
We employ a robust EDA pipeline to understand:
*   **Seasonality**: Weekly and annual cycles using decomposition.
*   **Segmentation**: High-value vs. low-value regions and categories.
*   **Distributions**: identifying price clustering and outlier sales events.

### B. Forecasting Service
We implement a unified interface `src.forecasting` supporting three distinct model architectures:
1.  **ARIMA (SARIMAX)**: Baseline statistical model capturing autocorrelation and seasonality.
2.  **Prophet**: Additive regression model effective for handling holidays and changepoints.
3.  **LSTM (Deep Learning)**: Recurrent Neural Network for capturing complex non-linear temporal dependencies (requires sufficient data depth).

### C. Pricing & Elasticity Engine
Central to the project is the **Log-Log Regression Model** used to estimate price elasticity ($\beta$).

$$ \log(\text{Units}) = \alpha + \beta \cdot \log(\text{Effective Price}) + \gamma \cdot \text{Controls} + \epsilon $$

*   **Controls**: We control for `Seasonality`, `Promotion Presence`, and `Competitor Gaps` to isolate the pure price effect.
*   **Robustness**: Models use **HC1 Robust Standard Errors** to account for heteroscedasticity frequent in sales data.

## 4. Key Results & Interpretation

*(Example output from the Recommendations Engine)*

### 🏷️ Price Sensitivity
| Category | Elasticity ($\beta$) | Interpretation | Recommendation |
| :--- | :--- | :--- | :--- |
| **Luxury Electronics** | **-1.85** | Highly Elastic | **Volume Driver**: Deep discounts will generate massive volume uplift, potentially offsetting margin loss. |
| **Basic Groceries** | **-0.45** | Inelastic | **Profit Driver**: Demand is sticky. Increasing price by 5% is expected to yield ~4.5% revenue growth. |

### ⚔️ Competitor Impact
*   **Observation**: A statistically significant positive coefficient ($0.8$) on `Price Gap` (MyPrice - CompPrice).
*   **Interpretation**: For every $1 our price exceeds the competitor, we lose ~0.8 units of sales.

## 5. Deployment & Execution
The application is container-ready but can be run locally.

### Prerequisites
*   Python 3.9+
*   Dependencies listed in `requirements.txt`

### Running Locally
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Launch the application:
    ```bash
    streamlit run app/Home.py
    ```
4.  Navigate to `http://localhost:8501`.

## 6. Repository Structure
```
├── app/
│   ├── Home.py                  # Entry point
│   └── pages/                   # Streamlit multipage modules
│       ├── 01_Data_Upload.py
│       ├── ...
│       ├── 08_Data_Science_View.py
├── src/
│   ├── data_io.py               # Ingestion & Session State
│   ├── cleaning.py              # Pipeline Logic
│   ├── features.py              # Time-series Feature Eng
│   ├── forecasting.py           # Modeling Wrappers (Arima/Prophet/LSTM)
│   ├── pricing.py               # Econometrics Module
│   ├── viz.py                   # Plotting utilities
│   └── ui_components.py         # Reusable UX elements
├── data/                        # Local data storage
├── requirements.txt
└── README.md
```

## 7. Limitations & Future Roadmap

 While robust, this observational model has inherent limitations to be addressed in Phase 2:

1.  **Cross-Price Elasticity**: Current models treat products in isolation. We do not yet model cannibalization (e.g., does discounting Coke hurt Pepsi sales?).
    *   *Next Step*: Implement Vector Autoregression (VAR) or demand systems (AIDS) for category-level interactions.
2.  **Causal Inference**: Coefficients are correlational. "Clearance sales" (low price when demand is naturally low) can bias elasticity towards zero.
    *   *Next Step*: Implement Instrumental Variables (IV) or meta-learners (DoubleML) if randomized price experiment data becomes available.
3.  **Store Clustering**: Pooling all stores assumes homogeneity.
    *   *Next Step*: K-Means clustering of stores based on sales velocity to run tiered pricing models.
4.  **Model Monitoring**: No automated drift detection.
    *   *Next Step*: Implement a dashboard for tracking prediction error (MAPE) over time to trigger retraining.
