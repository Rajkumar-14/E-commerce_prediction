import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Ecommerce Profit Prediction",
    page_icon="📈",
    layout="wide"
)

# --- CUSTOM STYLES (CSS ONLY ADDITIONS) ---
st.markdown("""
<style>
    /* Background Image with Dark Overlay for High Contrast */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)), 
                    url("https://images.unsplash.com/photo-1557804506-669a67965ba0?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }

    /* Header & Typography Styling */
    h1 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    
    h2, h3, p, label, .stMarkdown {
        color: #E2E8F0 !important;
        font-weight: 500 !important;
    }

    /* Frosted Card Container for Input Form */
    div[data-testid="stHorizontalBlock"] {
        background: rgba(30, 41, 59, 0.75);
        backdrop-filter: blur(12px);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }

    /* Form Labels High Visibility */
    .stNumberInput label, .stSelectbox label, .stSlider label, .stDateInput label {
        color: #F8FAFC !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px;
    }

    /* ================= INPUT BOXES ================= */

/* Number Input */
.stNumberInput input {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}

/* Selected text in Selectbox */
.stSelectbox span {
    color: #FFFFFF !important;
}

/* Date Input */
.stDateInput input {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #475569 !important;
}

/* Placeholder */
input::placeholder {
    color: #CBD5E1 !important;
}

/* Hover */
.stNumberInput input:hover,
.stDateInput input:hover,
.stSelectbox div[data-baseweb="select"] > div:hover {
    border-color: #38BDF8 !important;
}

/* Force all input text to white */
input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Dropdown menu */
div[data-baseweb="popover"] {
    background-color: #1E293B !important;
}

div[data-baseweb="popover"] li {
    color: white !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #334155 !important;
    color: white !important;
}

    /* Custom Primary Button Styling */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39);
        transition: all 0.3s ease !important;
    }

    /* Primary Button Hover State */
    div.stButton > button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.6);
    }

    /* Result Alert Box Overrides */
    .stAlert {
        border-radius: 10px !important;
        backdrop-filter: blur(8px);
    }
</style>
""", unsafe_allow_html=True)

# --- ORIGINAL UNCHANGED LOGIC BELOW ---

model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Ecommerce Profit Prediction System")

st.markdown("Predict whether an ecommerce order will generate High Profit or Low Profit using Machine Learning.")

st.header("Enter Order Details")

col1, col2 = st.columns(2)

with col1:
    customer_age = st.number_input("Customer Age", min_value=18, max_value=80, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city = st.selectbox("City", ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Lucknow", "Mumbai", "Pune"])
    category = st.selectbox("Category", ["Beauty", "Electronics", "Fashion", "Furniture", "Grocery", "Sports"])
    qty = st.number_input("Quantity", min_value=1, value=2)

with col2:
    unit_price = st.number_input("Unit Price", min_value=1.0, value=500.0)
    discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0)
    shipping = st.number_input("Shipping Cost", min_value=0.0, value=100.0)
    delivery = st.number_input("Delivery Days", min_value=1, value=5)
    rating = st.slider("Customer Rating", 1.0, 5.0, 4.0, step=0.1)

order_date = st.date_input("Order Date", datetime.today())

month = order_date.month
year = order_date.year
day_of_week = order_date.weekday()
weekend = 1 if day_of_week >= 5 else 0

gender_num = 1 if gender == "Male" else 0

input_data = {
    "Customer_Age": customer_age,
    "Gender": gender_num,
    "Qty": qty,
    "Unit Price": unit_price,
    "Discount": discount,
    "Shipping": shipping,
    "Delivery": delivery,
    "Rating": rating,
    "Month": month,
    "Year": year,
    "Day_of_Week": day_of_week,
    "Weekend": weekend,
    "City_Chennai": 0,
    "City_Delhi": 0,
    "City_Hyderabad": 0,
    "City_Jaipur": 0,
    "City_Lucknow": 0,
    "City_Mumbai": 0,
    "City_Pune": 0,
    "Category_Electronics": 0,
    "Category_Fashion": 0,
    "Category_Furniture": 0,
    "Category_Grocery": 0,
    "Category_Sports": 0
}

if city != "Bengaluru":
    input_data[f"City_{city}"] = 1

if category != "Beauty":
    input_data[f"Category_{category}"] = 1

input_df = pd.DataFrame([input_data])

feature_order = [
    'Customer_Age',
    'Gender',
    'Qty',
    'Unit Price',
    'Discount',
    'Shipping',
    'Delivery',
    'Rating',
    'Month',
    'Year',
    'Day_of_Week',
    'Weekend',
    'City_Chennai',
    'City_Delhi',
    'City_Hyderabad',
    'City_Jaipur',
    'City_Lucknow',
    'City_Mumbai',
    'City_Pune',
    'Category_Electronics',
    'Category_Fashion',
    'Category_Furniture',
    'Category_Grocery',
    'Category_Sports'
]

input_df = input_df[feature_order]
input_scaled = scaler.transform(input_df)

if st.button("Predict Profit Category"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")

    # Probability Metrics
    high_profit_prob = round(probability[0][1] * 100, 2)
    low_profit_prob = round(probability[0][0] * 100, 2)

    if prediction == 1:
        st.success(f"High Profit (Confidence: {high_profit_prob}%)")
        
        st.markdown("### Strategic Recommendations to Maximize Profit:")
        st.info(
            f"**Healthy Margins Detected!**\n"
            f"* **Upsell & Cross-sell:** Consider bundling related products in the **{category}** category to raise order values.\n"
            f"* **Customer Loyalty:** Offer a repeat-purchase coupon since this order profile is highly profitable.\n"
            f"* **Replicate Strategy:** Use this order's discount structure ({discount}%) as a benchmark for similar products in {city}."
        )
    else:
        st.error(f"Low Profit (Confidence: {low_profit_prob}%)")
        
        # Diagnostic analysis of inputs
        st.markdown("### Actionable Suggestions to Improve Profitability:")
        
        suggestions = []
        
        # Rule-based suggestions derived from inputs
        if discount > 15.0:
            suggestions.append(f"**Reduce Discount:** Current discount is high ({discount}%). Cap discounts below 15% to protect gross margins.")
        
        if shipping > (unit_price * qty * 0.15):
            suggestions.append(f"**Optimize Shipping Cost:** Shipping (₹{shipping}) takes up over 15% of total order value (₹{unit_price * qty}). Re-evaluate logistics partners in {city}.")
            
        if delivery > 4:
            suggestions.append(f"**Shorten Delivery Time:** {delivery} days is high. Faster fulfillment improves customer satisfaction and lowers return rates.")
            
        if qty == 1:
            suggestions.append("**Incentivize Volume:** Offer 'Buy 2 Get 10% Off' minimum threshold to raise unit count.")
            
        if rating < 3.5:
            suggestions.append(f"**Quality Check:** Low product rating ({rating}/5.0). Improve product quality or supplier standards to prevent high return costs.")

        if not suggestions:
            suggestions.append("**Adjust Pricing:** Consider slightly raising the unit price or setting a minimum purchase amount for free shipping.")

        for item in suggestions:
            st.warning(item)

