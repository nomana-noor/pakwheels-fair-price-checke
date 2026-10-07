# 🚗 PakWheels Fair Price Checker — Used Car Price Prediction (Pakistan)

A data science project analyzing Pakistan's used car market (PakWheels listings) 
to uncover pricing patterns and build a model that estimates fair market value — 
deployed as an interactive "Fair Price Checker" web app.

**🔗 Live Demo:** https://pakwheels-fair-price-checke-brx8mbctb9ydeptwtqvawc.streamlit.app/

## Problem

Pakistan's used car market (via PakWheels) has thousands of listings with wildly 
varying prices, inconsistent data formatting, and no easy way for buyers/sellers 
to know if a price is fair. This project cleans real scraped market data, uncovers 
pricing patterns specific to the Pakistani market, and builds a predictive tool 
buyers can actually use.

## Dataset

- **Source:** PakWheels used car listings (scraped), ~5,770 rows
- **Features:** Brand, Model, City, Registered Year, Mileage, Engine, Fuel Type, 
  Transmission, Price
- Required significant cleaning: prices stored as mixed "lacs"/"crore" text, 
  engine values mixing cc (combustion) and kWh (electric), mileage with embedded 
  commas/units, and brand/model extraction from unstructured listing titles.

## Key Findings

1. **Market concentration:** Toyota, Honda, and Suzuki account for 92.9% of all 
   listings — extreme brand concentration driven by decades of import restrictions.

2. **City price gap:** Karachi listings are consistently ~10-12% cheaper than 
   Lahore/Islamabad — and this gap holds even when comparing the *same brand*, 
   not just different brand mixes across cities. Independently confirmed by SHAP 
   analysis on the trained model.

3. **The "Toyota holds value better" myth — not supported:** Despite popular 
   belief, this data shows Toyota, Honda, and Suzuki all retain value at similar 
   rates (30-34%) over a 15-year span. Toyota's advantage lies in higher absolute 
   prices and dominance in the near-new resale segment — not slower depreciation.

4. **Car age is the strongest price predictor** (correlation -0.78), followed by 
   mileage (-0.35) and transmission type (automatic cars command a premium).

## Modeling

| Model | MAE (PKR) | R² |
|-------|-----------|-----|
| Linear Regression | 284,167 | 0.860 |
| Random Forest (final) | 211,092 | 0.919 |
| XGBoost | 198,291 | 0.924 |

**Selected model:** Random Forest (deployed version, tuned for size/speed trade-off 
for deployment). 72% of test predictions fall within 10% of actual price.

Used SHAP for explainability — confirmed that Car_Age, Brand, City, and Transmission 
are the dominant price drivers, matching EDA findings independently.

## Tech Stack

Python, pandas, scikit-learn, SHAP, Streamlit (deployment), skops (model serialization)

## Limitations

- Does not account for accident history, specific trim/condition, or dealer markup
- City price comparison doesn't fully control for age/mileage mix within brands
- Dataset limited to 3 major cities (Karachi, Lahore, Islamabad)

## Project Structure
