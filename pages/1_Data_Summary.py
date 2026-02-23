import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;">
    <h1 style="color: #e94560; margin: 0; font-size: 2rem;">📊 Data Summary</h1>
    <p style="color: #a0a0b0; margin: 0.3rem 0 0 0;">Sales performance &amp; inventory health at a glance</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if "sales_df" not in st.session_state:
    st.error("Upload Sales file on the Home page first.")
    st.stop()
if "inventory_df" not in st.session_state:
    st.error("Upload Inventory file on the Home page first.")
    st.stop()

sales = st.session_state["sales_df"].copy()
inventory = st.session_state["inventory_df"].copy()

# --------------------------------------------------
# PREP
# --------------------------------------------------
sales["Sale_Date"] = pd.to_datetime(sales["Sale_Date"], errors="coerce")
sales["Inventory_Date"] = pd.to_datetime(sales["Inventory_Date"], errors="coerce")
inventory["Inventory_Date"] = pd.to_datetime(inventory["Inventory_Date"], errors="coerce")

for c in ["Price", "Front_Gross", "Back_Gross", "Mileage", "Year"]:
    sales[c] = pd.to_numeric(sales[c], errors="coerce")
for c in ["Price", "Cost", "Mileage", "Year"]:
    inventory[c] = pd.to_numeric(inventory[c], errors="coerce")

sales["Days_On_Lot"] = (sales["Sale_Date"] - sales["Inventory_Date"]).dt.days
sales["Total_Gross"] = sales["Front_Gross"] + sales["Back_Gross"]

today = pd.to_datetime("today").normalize()
inventory["Days_In_Stock"] = (today - inventory["Inventory_Date"]).dt.days.clip(lower=0)
inventory["Margin"] = inventory["Price"] - inventory["Cost"]

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Filters")

store_opts = ["All"] + sorted(sales["Store_Name"].dropna().unique().tolist())
store_f = st.sidebar.selectbox("Store", store_opts, index=0)

stock_opts = ["All"] + sorted(sales["Stock_Type"].dropna().unique().tolist())
stock_f = st.sidebar.selectbox("Stock Type", stock_opts, index=0)

# Apply filters
sf = sales.copy()
inv_f = inventory.copy()

if store_f != "All":
    sf = sf[sf["Store_Name"] == store_f]
    inv_f = inv_f[inv_f["Store_Name"] == store_f]
if stock_f != "All":
    sf = sf[sf["Stock_Type"] == stock_f]
    inv_f = inv_f[inv_f["Stock_Type"] == stock_f]


# --------------------------------------------------
# HELPER: styled metric card
# --------------------------------------------------
def metric_card(label, value, subtitle="", color="#e94560"):
    return f"""
    <div style="background: #16213e; border-radius: 10px; padding: 1.2rem 1.5rem;
                border-left: 4px solid {color}; min-height: 100px;">
        <div style="color: #a0a0b0; font-size: 0.85rem; margin-bottom: 0.3rem;">{label}</div>
        <div style="color: #ffffff; font-size: 1.8rem; font-weight: 700;">{value}</div>
        <div style="color: #606080; font-size: 0.78rem; margin-top: 0.2rem;">{subtitle}</div>
    </div>
    """


# =====================================================
# SALES PERFORMANCE
# =====================================================
st.markdown("## Sales Performance")
st.caption(f"Based on {len(sf):,} deals  •  "
           f"{sf['Sale_Date'].min().strftime('%b %Y') if sf['Sale_Date'].notna().any() else '—'} to "
           f"{sf['Sale_Date'].max().strftime('%b %Y') if sf['Sale_Date'].notna().any() else '—'}")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(metric_card("Total Deals", f"{len(sf):,}"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Avg Sale Price", f"${sf['Price'].mean():,.0f}"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Avg Front Gross", f"${sf['Front_Gross'].mean():,.0f}",
                            subtitle=f"Median: ${sf['Front_Gross'].median():,.0f}"), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("Avg Back Gross", f"${sf['Back_Gross'].mean():,.0f}",
                            subtitle=f"Median: ${sf['Back_Gross'].median():,.0f}"), unsafe_allow_html=True)
with c5:
    st.markdown(metric_card("Avg Total Gross", f"${sf['Total_Gross'].mean():,.0f}",
                            subtitle=f"Front + Back per deal"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Sales by Stock Type ---
left, right = st.columns(2)

with left:
    st.markdown("#### Sales by Stock Type")
    type_summary = sf.groupby("Stock_Type").agg(
        Deals=("Price", "count"),
        Avg_Price=("Price", "mean"),
        Avg_Front=("Front_Gross", "mean"),
        Avg_Back=("Back_Gross", "mean"),
        Avg_DOL=("Days_On_Lot", "mean")
    ).round(0)
    type_summary["Avg_Price"] = type_summary["Avg_Price"].apply(lambda x: f"${x:,.0f}")
    type_summary["Avg_Front"] = type_summary["Avg_Front"].apply(lambda x: f"${x:,.0f}")
    type_summary["Avg_Back"] = type_summary["Avg_Back"].apply(lambda x: f"${x:,.0f}")
    type_summary["Avg_DOL"] = type_summary["Avg_DOL"].apply(lambda x: f"{x:.0f} days")
    type_summary["Deals"] = type_summary["Deals"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(type_summary, use_container_width=True)

with right:
    st.markdown("#### Sales by Store")
    store_summary = sf.groupby("Store_Name").agg(
        Deals=("Price", "count"),
        Avg_Price=("Price", "mean"),
        Avg_Total_Gross=("Total_Gross", "mean"),
        Avg_DOL=("Days_On_Lot", "mean")
    ).round(0)
    store_summary["Avg_Price"] = store_summary["Avg_Price"].apply(lambda x: f"${x:,.0f}")
    store_summary["Avg_Total_Gross"] = store_summary["Avg_Total_Gross"].apply(lambda x: f"${x:,.0f}")
    store_summary["Avg_DOL"] = store_summary["Avg_DOL"].apply(lambda x: f"{x:.0f} days")
    store_summary["Deals"] = store_summary["Deals"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(store_summary, use_container_width=True)

# --- Top Models ---
st.markdown("#### Top 10 Models by Volume")
top_models = sf.groupby("Model").agg(
    Deals=("Price", "count"),
    Avg_Price=("Price", "mean"),
    Avg_Gross=("Total_Gross", "mean"),
    Avg_DOL=("Days_On_Lot", "mean")
).sort_values("Deals", ascending=False).head(10)
top_models["Avg_Price"] = top_models["Avg_Price"].apply(lambda x: f"${x:,.0f}")
top_models["Avg_Gross"] = top_models["Avg_Gross"].apply(lambda x: f"${x:,.0f}")
top_models["Avg_DOL"] = top_models["Avg_DOL"].apply(lambda x: f"{x:.0f} days")
st.dataframe(top_models, use_container_width=True)


# =====================================================
# INVENTORY HEALTH
# =====================================================
st.markdown("---")
st.markdown("## Inventory Health")
st.caption(f"{len(inv_f):,} units currently in stock")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(metric_card("Units in Stock", f"{len(inv_f):,}", color="#4ecca3"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Avg Price", f"${inv_f['Price'].mean():,.0f}", color="#4ecca3"), unsafe_allow_html=True)
with c3:
    avg_margin = inv_f["Margin"].mean()
    st.markdown(metric_card("Avg Margin", f"${avg_margin:,.0f}",
                            subtitle=f"{avg_margin / inv_f['Price'].mean() * 100:.1f}% of price",
                            color="#4ecca3"), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("Avg Days in Stock", f"{inv_f['Days_In_Stock'].mean():.0f}",
                            subtitle=f"Median: {inv_f['Days_In_Stock'].median():.0f} days",
                            color="#4ecca3"), unsafe_allow_html=True)
with c5:
    aged_60 = (inv_f["Days_In_Stock"] >= 60).sum()
    pct_aged = aged_60 / len(inv_f) * 100 if len(inv_f) > 0 else 0
    color_aged = "#e94560" if pct_aged > 20 else "#f0a500" if pct_aged > 10 else "#4ecca3"
    st.markdown(metric_card("Aged 60+ Days", f"{aged_60} units",
                            subtitle=f"{pct_aged:.0f}% of inventory", color=color_aged), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Inventory by Stock Type ---
left, right = st.columns(2)

with left:
    st.markdown("#### Inventory by Stock Type")
    inv_type = inv_f.groupby("Stock_Type").agg(
        Units=("Price", "count"),
        Avg_Price=("Price", "mean"),
        Avg_Cost=("Cost", "mean"),
        Avg_Margin=("Margin", "mean"),
        Avg_Days=("Days_In_Stock", "mean")
    ).round(0)
    for mc in ["Avg_Price", "Avg_Cost", "Avg_Margin"]:
        inv_type[mc] = inv_type[mc].apply(lambda x: f"${x:,.0f}")
    inv_type["Avg_Days"] = inv_type["Avg_Days"].apply(lambda x: f"{x:.0f} days")
    st.dataframe(inv_type, use_container_width=True)

with right:
    st.markdown("#### Inventory by Store")
    inv_store = inv_f.groupby("Store_Name").agg(
        Units=("Price", "count"),
        Avg_Price=("Price", "mean"),
        Avg_Margin=("Margin", "mean"),
        Avg_Days=("Days_In_Stock", "mean")
    ).round(0)
    inv_store["Avg_Price"] = inv_store["Avg_Price"].apply(lambda x: f"${x:,.0f}")
    inv_store["Avg_Margin"] = inv_store["Avg_Margin"].apply(lambda x: f"${x:,.0f}")
    inv_store["Avg_Days"] = inv_store["Avg_Days"].apply(lambda x: f"{x:.0f} days")
    st.dataframe(inv_store, use_container_width=True)

# --- Aging Distribution ---
st.markdown("#### Inventory Aging Breakdown")
bins = [0, 15, 30, 45, 60, 90, 9999]
labels = ["0–15 days", "16–30 days", "31–45 days", "46–60 days", "61–90 days", "90+ days"]
inv_f["Age_Bucket"] = pd.cut(inv_f["Days_In_Stock"], bins=bins, labels=labels)

aging = inv_f.groupby("Age_Bucket", observed=True).agg(
    Units=("Price", "count"),
    Avg_Price=("Price", "mean"),
    Total_Value=("Price", "sum")
).fillna(0)
aging["Avg_Price"] = aging["Avg_Price"].apply(lambda x: f"${x:,.0f}" if x > 0 else "—")
aging["Total_Value"] = aging["Total_Value"].apply(lambda x: f"${x:,.0f}")
st.dataframe(aging, use_container_width=True)

# --- Top 10 Oldest ---
st.markdown("#### ⚠️ Top 10 Oldest Units")
oldest = inv_f.nlargest(10, "Days_In_Stock")[
    ["Stock_Number", "Make", "Model", "Year", "Store_Name", "Stock_Type",
     "Price", "Cost", "Margin", "Days_In_Stock"]
].copy()
oldest["Price"] = oldest["Price"].apply(lambda x: f"${x:,.0f}")
oldest["Cost"] = oldest["Cost"].apply(lambda x: f"${x:,.0f}")
oldest["Margin"] = oldest["Margin"].apply(lambda x: f"${x:,.0f}")
oldest.rename(columns={"Days_In_Stock": "Days"}, inplace=True)
st.dataframe(oldest, use_container_width=True, hide_index=True)
