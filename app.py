import streamlit as st
import pandas as pd
import numpy as np
import joblib
model = joblib.load('car_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')
st.set_page_config(page_title="PakWheels Fair Price Checker", page_icon="🚗")
st.title("🚗 Fair Price Checker — Used Cars Pakistan")
st.write("Enter your car's details to see if the asking price is fair, based on real PakWheels market data.")
col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", ["Toyota", "Honda", "Suzuki", "Nissan", "Mitsubishi", 
                                     "FAW", "Proton", "KIA", "Changan", "Daihatsu", "Other"])
    city = st.selectbox("City", ["Karachi", "Lahore", "Islamabad"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
with col2:
    reg_year = st.number_input("Registered Year", min_value=1960, max_value=2026, value=2018)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=80000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "CNG", "LPG"])
engine_value = st.number_input("Engine Size (cc)", min_value=600, max_value=5000, value=1300)
asking_price = st.number_input("Asking Price (PKR)", min_value=0, value=2500000)
if st.button("Check Fair Price"):
    car_age = 2026 - reg_year
    is_automatic = 1 if transmission == "Automatic" else 0
    input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    input_data['Car_Age'] = car_age
    input_data['Mileage_Km'] = mileage
    input_data['Engine_Value'] = engine_value
    if 'Is_Automatic' in input_data.columns:
        input_data['Is_Automatic'] = is_automatic
    brand_col = f'Brand_{brand}'
    if brand_col in input_data.columns:
        input_data[brand_col] = True
    city_col = f'City_{city}'
    if city_col in input_data.columns:
        input_data[city_col] = True
    fuel_col = f'Fuel Type_{fuel_type}'
    if fuel_col in input_data.columns:
        input_data[fuel_col] = True
    trans_col = f'Transmission_{transmission}'
    if trans_col in input_data.columns:
        input_data[trans_col] = True
    predicted_log = model.predict(input_data)[0]
    predicted_price = np.expm1(predicted_log)   
    st.subheader(f"Estimated Fair Price: PKR {predicted_price:,.0f}") 
    diff = asking_price - predicted_price
    diff_pct = (diff / predicted_price) * 100
    if diff_pct > 15:
        st.error(f"⚠️ This listing is priced {diff_pct:.1f}% ABOVE the estimated fair price. Consider negotiating.")
    elif diff_pct < -15:
        st.success(f"✅ This listing is priced {abs(diff_pct):.1f}% BELOW the estimated fair price — good deal!")
    else:
        st.info(f"This listing is priced close to fair market value (within {abs(diff_pct):.1f}%).")
st.caption("Model trained on PakWheels listings (Karachi, Lahore, Islamabad) | Random Forest, R²=0.92")