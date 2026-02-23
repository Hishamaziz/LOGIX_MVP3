import streamlit as st
import pandas as pd
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="LOGIX – Dealer Upload Portal",
    layout="wide"
)

# --------------------------------------------------
# BRANDING HEADER
# --------------------------------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;">
    <h1 style="color: #e94560; margin: 0; font-size: 2.4rem;">LOGIX</h1>
    <p style="color: #a0a0b0; margin: 0.3rem 0 0 0; font-size: 1.1rem;">
        Dealership Analytics & Decision Support
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Upload your **Sales** and **Inventory** files to get started.

- Supports **CSV** or **Excel**  
- Validates required columns automatically  
- Download templates below if you need the correct format  
""")

# --------------------------------------------------
# REQUIRED COLUMN DEFINITIONS
# --------------------------------------------------
required_sales_cols = [
    "Stock_Number", "Stock_Type", "Sale_Date", "Inventory_Date",
    "Make", "Model", "Year", "Mileage",
    "Store_Name", "Price", "Front_Gross", "Back_Gross"
]

required_inventory_cols = [
    "Stock_Number", "Stock_Type", "Inventory_Date",
    "Make", "Model", "Year", "Mileage",
    "Store_Name", "Price", "Cost"
]

# --------------------------------------------------
# TEMPLATES
# --------------------------------------------------
sales_template = pd.DataFrame([{
    "Stock_Number": "12345A",
    "Stock_Type": "Used",
    "Sale_Date": "2024-01-10",
    "Inventory_Date": "2023-12-01",
    "Make": "Toyota",
    "Model": "Camry",
    "Year": 2021,
    "Mileage": 50100,
    "Store_Name": "Riverside Toyota",
    "Price": 23900,
    "Front_Gross": 400,
    "Back_Gross": 600
}])

inventory_template = pd.DataFrame([{
    "Stock_Number": "INV10001",
    "Stock_Type": "Used",
    "Inventory_Date": "2024-02-15",
    "Make": "Honda",
    "Model": "Civic",
    "Year": 2020,
    "Mileage": 62000,
    "Store_Name": "Riverside Honda",
    "Price": 18900,
    "Cost": 15000
}])


def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


# --------------------------------------------------
# SAMPLE DATA (from data/input/ folder)
# --------------------------------------------------
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "input")
sample_sales_path = os.path.join(SAMPLE_DIR, "sales_input_file.csv")
sample_inventory_path = os.path.join(SAMPLE_DIR, "inventory_input_file.csv")
sample_available = os.path.exists(sample_sales_path) and os.path.exists(sample_inventory_path)

if sample_available:
    st.markdown("---")
    st.markdown("#### 🧪 Quick Start — Load Sample Data")
    st.caption("Load pre-built sample data to explore the platform instantly.")

    if st.button("Load Sample Data", type="secondary", use_container_width=False):
        try:
            sales_df = pd.read_csv(sample_sales_path)
            inv_df = pd.read_csv(sample_inventory_path)

            s_missing = [c for c in required_sales_cols if c not in sales_df.columns]
            i_missing = [c for c in required_inventory_cols if c not in inv_df.columns]

            if s_missing or i_missing:
                if s_missing:
                    st.error(f"Sample sales file missing columns: {s_missing}")
                if i_missing:
                    st.error(f"Sample inventory file missing columns: {i_missing}")
            else:
                st.session_state["sales_df"] = sales_df
                st.session_state["inventory_df"] = inv_df
                st.session_state.pop("engine_results", None)
                st.success(f"✅ Sample data loaded — {len(sales_df):,} sales, {len(inv_df):,} inventory units")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading sample data: {e}")

    st.markdown("---")

# --------------------------------------------------
# UPLOAD LAYOUT — Side by Side
# --------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("📄 Sales File")
    st.download_button(
        label="📥 Download Sales Template",
        data=sales_template.to_csv(index=False),
        file_name="sales_input_template.csv",
        key="dl_sales"
    )
    sales_file = st.file_uploader("Upload Sales CSV/Excel", type=["csv", "xlsx"], key="up_sales")

    if sales_file:
        try:
            df = load_file(sales_file)
            missing = [c for c in required_sales_cols if c not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                st.success(f"✅ Sales file loaded — {len(df):,} rows")
                st.dataframe(df.head(5), use_container_width=True)
                st.session_state["sales_df"] = df
                st.session_state.pop("engine_results", None)
        except Exception as e:
            st.error(f"Error reading Sales file: {e}")

with right:
    st.subheader("📄 Inventory File")
    st.download_button(
        label="📥 Download Inventory Template",
        data=inventory_template.to_csv(index=False),
        file_name="inventory_input_template.csv",
        key="dl_inv"
    )
    inventory_file = st.file_uploader("Upload Inventory CSV/Excel", type=["csv", "xlsx"], key="up_inv")

    if inventory_file:
        try:
            df = load_file(inventory_file)
            missing = [c for c in required_inventory_cols if c not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                st.success(f"✅ Inventory file loaded — {len(df):,} rows")
                st.dataframe(df.head(5), use_container_width=True)
                st.session_state["inventory_df"] = df
                st.session_state.pop("engine_results", None)
        except Exception as e:
            st.error(f"Error reading Inventory file: {e}")

# --------------------------------------------------
# READY STATE
# --------------------------------------------------
st.markdown("---")
if "sales_df" in st.session_state and "inventory_df" in st.session_state:
    s_len = len(st.session_state["sales_df"])
    i_len = len(st.session_state["inventory_df"])
    st.success(f"🎉 Both files loaded ({s_len:,} sales, {i_len:,} inventory). "
               f"Navigate to **Data Summary** or **Intelligence Engine** from the sidebar.")
elif "sales_df" in st.session_state:
    st.info("Sales loaded. Upload Inventory to continue.")
elif "inventory_df" in st.session_state:
    st.info("Inventory loaded. Upload Sales to continue.")
else:
    st.info("Upload both files above to unlock analytics.")
