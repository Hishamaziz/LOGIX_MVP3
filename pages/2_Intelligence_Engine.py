import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

st.set_page_config(layout="wide")

# ==========================================================
# CONFIGURATION
# ==========================================================
PRICE_THRESHOLD_PCT = 0.03       # 3% gap to trigger rec
MAX_CHANGE_PCT = 0.20            # cap suggested change at 20%
MOVE_MIN_DAYS_SAVED = 5          # min days faster to recommend move
MOVE_MIN_STORE_SALES = 5         # min sales at alt store
TRANSFER_COST = 300              # flat cost per vehicle move
LOW_DATA_THRESHOLD = 10          # min segment sales to trust benchmark
GBM_N_ESTIMATORS = 150
GBM_MAX_DEPTH = 6
GBM_LEARNING_RATE = 0.1
GBM_SUBSAMPLE = 0.8
GBM_MIN_SAMPLES_LEAF = 10
CURRENT_YEAR = 2026

# ==========================================================
# HEADER
# ==========================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;">
    <h1 style="color: #e94560; margin: 0; font-size: 2rem;">🤖 Intelligence Engine</h1>
    <p style="color: #a0a0b0; margin: 0.3rem 0 0 0;">
        ML-driven pricing benchmarks &amp; actionable inventory recommendations
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# HELPERS
# ==========================================================
def fmt_money(x):
    try:
        v = float(x)
        if v < 0:
            return f"-${abs(v):,.0f}"
        return f"${v:,.0f}"
    except Exception:
        return ""

def fmt_signed_money(x):
    try:
        v = float(x)
        if v > 0:
            return f"+${v:,.0f}"
        elif v < 0:
            return f"-${abs(v):,.0f}"
        return "$0"
    except Exception:
        return ""

# ==========================================================
# CHECK DATA
# ==========================================================
if "sales_df" not in st.session_state or "inventory_df" not in st.session_state:
    st.error("Upload both Sales and Inventory files on the Home page first.")
    st.stop()

# ==========================================================
# RUN BUTTON + CACHE
# ==========================================================
if "engine_results" not in st.session_state:
    st.info("Click **Run Analysis** to train the model on your sales data and generate recommendations.")

col_btn, col_status = st.columns([1, 3])
with col_btn:
    run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
with col_status:
    if "engine_results" in st.session_state:
        r = st.session_state["engine_results"]
        st.success(f"✅ Last run: {r['run_time']:.1f}s  •  {r['train_rows']:,} sales trained on  •  "
                   f"{r['inv_count']:,} units scored")

if not run_clicked and "engine_results" not in st.session_state:
    st.stop()

# ==========================================================
# TRAINING PIPELINE (only runs on button click)
# ==========================================================
if run_clicked:
    with st.spinner("Training model and scoring inventory..."):
        t_start = time.time()

        sales = st.session_state["sales_df"].copy()
        inventory = st.session_state["inventory_df"].copy()

        # ------------------------------------------------------
        # VALIDATE
        # ------------------------------------------------------
        required_sales = [
            "Stock_Number", "Stock_Type", "Sale_Date", "Inventory_Date",
            "Make", "Model", "Year", "Mileage", "Store_Name",
            "Price", "Front_Gross", "Back_Gross"
        ]
        required_inventory = [
            "Stock_Number", "Stock_Type", "Inventory_Date",
            "Make", "Model", "Year", "Mileage", "Store_Name",
            "Price", "Cost"
        ]

        for req, df, name in [(required_sales, sales, "Sales"),
                               (required_inventory, inventory, "Inventory")]:
            missing = [c for c in req if c not in df.columns]
            if missing:
                st.error(f"❌ {name} missing columns: {missing}")
                st.stop()

        # ------------------------------------------------------
        # AUTO-CLEAN (silent)
        # ------------------------------------------------------
        sales["Sale_Date"] = pd.to_datetime(sales["Sale_Date"], errors="coerce")
        sales["Inventory_Date"] = pd.to_datetime(sales["Inventory_Date"], errors="coerce")
        inventory["Inventory_Date"] = pd.to_datetime(inventory["Inventory_Date"], errors="coerce")

        for c in ["Year", "Mileage", "Price", "Front_Gross", "Back_Gross"]:
            sales[c] = pd.to_numeric(sales[c], errors="coerce")
        for c in ["Year", "Mileage", "Price", "Cost"]:
            inventory[c] = pd.to_numeric(inventory[c], errors="coerce")

        # Uppercase + strip text columns
        text_cols = ["Make", "Model", "Store_Name", "Stock_Type"]
        for c in text_cols:
            sales[c] = sales[c].astype(str).str.strip().str.upper()
            inventory[c] = inventory[c].astype(str).str.strip().str.upper()

        # Title case Stock_Type for display
        sales["Stock_Type"] = sales["Stock_Type"].str.title()
        inventory["Stock_Type"] = inventory["Stock_Type"].str.title()

        # ------------------------------------------------------
        # FEATURE ENGINEERING
        # ------------------------------------------------------
        sales["Vehicle_Age"] = CURRENT_YEAR - sales["Year"]
        inventory["Vehicle_Age"] = CURRENT_YEAR - inventory["Year"]

        sales["Days_On_Lot"] = (sales["Sale_Date"] - sales["Inventory_Date"]).dt.days.clip(lower=1)

        today = pd.to_datetime("today").normalize()
        inventory["Days_In_Stock"] = (today - inventory["Inventory_Date"]).dt.days.clip(lower=0)

        sales_train = sales.dropna(
            subset=["Year", "Mileage", "Price", "Days_On_Lot",
                    "Make", "Model", "Store_Name", "Stock_Type"]
        ).copy()

        if len(sales_train) < 200:
            st.error("Not enough valid sales rows to train (need 200+).")
            st.stop()

        # ------------------------------------------------------
        # SEGMENT STATS
        # ------------------------------------------------------
        seg_stats = sales_train.groupby(["Make", "Model", "Stock_Type"]).agg(
            Seg_Avg_Price=("Price", "mean"),
            Seg_Avg_Turn=("Days_On_Lot", "mean"),
            Seg_Sales_Count=("Price", "count")
        ).reset_index()

        store_seg_stats = sales_train.groupby(
            ["Make", "Model", "Stock_Type", "Store_Name"]
        ).agg(
            Store_Avg_Price=("Price", "mean"),
            Store_Avg_Turn=("Days_On_Lot", "mean"),
            Store_Sales_Count=("Price", "count")
        ).reset_index()

        # ------------------------------------------------------
        # ML: PRICE MODEL
        # ------------------------------------------------------
        cat_cols = ["Make", "Model", "Store_Name", "Stock_Type"]
        num_cols = ["Vehicle_Age", "Mileage"]

        all_cat = pd.concat([sales_train[cat_cols], inventory[cat_cols]],
                            ignore_index=True)
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        encoder.fit(all_cat)

        X_train_cat = pd.DataFrame(
            encoder.transform(sales_train[cat_cols]),
            columns=cat_cols, index=sales_train.index
        )
        X_train = pd.concat([
            X_train_cat,
            sales_train[num_cols].reset_index(drop=True)
        ], axis=1)
        X_train.index = sales_train.index
        y_price = sales_train["Price"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train, y_price, test_size=0.2, random_state=42
        )

        gbm_price = GradientBoostingRegressor(
            n_estimators=GBM_N_ESTIMATORS, max_depth=GBM_MAX_DEPTH,
            learning_rate=GBM_LEARNING_RATE, subsample=GBM_SUBSAMPLE,
            min_samples_leaf=GBM_MIN_SAMPLES_LEAF, random_state=42
        )
        gbm_price.fit(X_tr, y_tr)
        y_pred = gbm_price.predict(X_te)

        price_r2 = r2_score(y_te, y_pred)
        price_mae = mean_absolute_error(y_te, y_pred)

        # ------------------------------------------------------
        # PREDICT ON INVENTORY
        # ------------------------------------------------------
        inv_cat = pd.DataFrame(
            encoder.transform(inventory[cat_cols]),
            columns=cat_cols, index=inventory.index
        )
        inv_num = inventory[num_cols].copy()
        for c in num_cols:
            inv_num[c] = inv_num[c].fillna(sales_train[c].median())
        X_inv = pd.concat([
            inv_cat, inv_num.reset_index(drop=True)
        ], axis=1)
        X_inv.index = inventory.index

        inventory["Benchmark_Price"] = gbm_price.predict(X_inv).round(0)
        inventory["Price_vs_Benchmark"] = (
            inventory["Price"] - inventory["Benchmark_Price"]
        )
        inventory["Price_vs_Benchmark_Pct"] = (
            inventory["Price_vs_Benchmark"] / inventory["Benchmark_Price"]
        )

        # ------------------------------------------------------
        # PRICING RECOMMENDATIONS
        # ------------------------------------------------------
        def compute_recommendation(row):
            pct = row["Price_vs_Benchmark_Pct"]
            gap = row["Price_vs_Benchmark"]
            cur = row["Price"]

            if abs(pct) < PRICE_THRESHOLD_PCT:
                return "Well Priced", 0.0
            if pct > 0:
                suggested = min(gap, cur * MAX_CHANGE_PCT)
                return "Reduce Price", -round(suggested, 0)
            else:
                suggested = min(abs(gap), cur * MAX_CHANGE_PCT)
                return "Raise Price", round(suggested, 0)

        rec = inventory.apply(compute_recommendation, axis=1, result_type="expand")
        rec.columns = ["Recommendation", "Suggested_Change"]
        inventory["Recommendation"] = rec["Recommendation"]
        inventory["Suggested_Change"] = rec["Suggested_Change"]
        inventory["Suggested_Price"] = inventory["Price"] + inventory["Suggested_Change"]

        # ------------------------------------------------------
        # TURN ESTIMATE (segment-based)
        # ------------------------------------------------------
        inventory = inventory.merge(
            seg_stats[["Make", "Model", "Stock_Type",
                       "Seg_Avg_Turn", "Seg_Sales_Count"]],
            on=["Make", "Model", "Stock_Type"],
            how="left"
        )

        inventory["Est_Days_Saved"] = np.where(
            inventory["Recommendation"] == "Reduce Price",
            (inventory["Price_vs_Benchmark_Pct"].clip(lower=0)
             * inventory["Seg_Avg_Turn"]).round(1),
            0.0
        )

        # ------------------------------------------------------
        # LOW DATA FLAG
        # ------------------------------------------------------
        low_mask = inventory["Seg_Sales_Count"].fillna(0) < LOW_DATA_THRESHOLD
        inventory["Confidence"] = np.where(low_mask, "Low Data", "")
        inventory.loc[low_mask, "Recommendation"] = "Low Data"
        inventory.loc[low_mask, "Suggested_Price"] = inventory.loc[low_mask, "Price"]
        inventory.loc[low_mask, "Est_Days_Saved"] = 0.0

        # ------------------------------------------------------
        # MOVE-STOCK (stats-based, Used only, multi-store only)
        # ------------------------------------------------------
        num_stores = inventory["Store_Name"].nunique()
        multi_store = num_stores > 1

        inventory["Move_To_Store"] = ""
        inventory["Move_Days_Faster"] = np.nan
        inventory["Move_Price_Uplift"] = np.nan

        if multi_store:
            used_mask = (inventory["Stock_Type"].str.lower() == "used") & (~low_mask)
            for idx in inventory.index[used_mask]:
                row = inventory.loc[idx]
                current_store = row["Store_Name"]
                make, model = row["Make"], row["Model"]

                candidates = store_seg_stats[
                    (store_seg_stats["Make"] == make) &
                    (store_seg_stats["Model"] == model) &
                    (store_seg_stats["Stock_Type"] == "Used") &
                    (store_seg_stats["Store_Name"] != current_store) &
                    (store_seg_stats["Store_Sales_Count"] >= MOVE_MIN_STORE_SALES)
                ]
                if candidates.empty:
                    continue

                curr = store_seg_stats[
                    (store_seg_stats["Make"] == make) &
                    (store_seg_stats["Model"] == model) &
                    (store_seg_stats["Stock_Type"] == "Used") &
                    (store_seg_stats["Store_Name"] == current_store)
                ]

                if curr.empty:
                    curr_turn = (row["Seg_Avg_Turn"]
                                 if pd.notna(row["Seg_Avg_Turn"]) else 47)
                    curr_price = row["Benchmark_Price"]
                else:
                    curr_turn = curr.iloc[0]["Store_Avg_Turn"]
                    curr_price = curr.iloc[0]["Store_Avg_Price"]

                best = candidates.loc[candidates["Store_Avg_Turn"].idxmin()]
                days_faster = curr_turn - best["Store_Avg_Turn"]
                price_uplift = best["Store_Avg_Price"] - curr_price
                net_price = price_uplift - TRANSFER_COST

                if days_faster >= MOVE_MIN_DAYS_SAVED and net_price > -500:
                    inventory.at[idx, "Move_To_Store"] = best["Store_Name"]
                    inventory.at[idx, "Move_Days_Faster"] = round(days_faster, 1)
                    inventory.at[idx, "Move_Price_Uplift"] = round(net_price, 0)

        t_total = time.time() - t_start

        # ------------------------------------------------------
        # CACHE
        # ------------------------------------------------------
        st.session_state["engine_results"] = {
            "inventory": inventory,
            "price_r2": price_r2,
            "price_mae": price_mae,
            "train_rows": len(sales_train),
            "inv_count": len(inventory),
            "run_time": t_total,
            "multi_store": multi_store,
            "num_stores": num_stores,
        }

    st.rerun()

# ==========================================================
# DISPLAY (from cache — no retraining on page switch)
# ==========================================================
if "engine_results" not in st.session_state:
    st.stop()

res = st.session_state["engine_results"]
inventory = res["inventory"]
multi_store = res["multi_store"]

# ==========================================================
# SIDEBAR FILTERS
# ==========================================================
st.sidebar.header("Filters")

store_opts = ["All"] + sorted(inventory["Store_Name"].dropna().unique().tolist())
stock_opts = ["All"] + sorted(inventory["Stock_Type"].dropna().unique().tolist())
make_opts = ["All"] + sorted(inventory["Make"].dropna().unique().tolist())

store_f = st.sidebar.selectbox("Store", store_opts, index=0)
stock_f = st.sidebar.selectbox("Stock Type", stock_opts, index=0)
make_f = st.sidebar.selectbox("Make", make_opts, index=0)

# ==========================================================
# PRIORITY ACTIONS
# ==========================================================
n_reduce = (inventory["Recommendation"] == "Reduce Price").sum()
n_raise = (inventory["Recommendation"] == "Raise Price").sum()
n_move = (inventory["Move_To_Store"] != "").sum()
n_well = (inventory["Recommendation"] == "Well Priced").sum()
n_low = (inventory["Recommendation"] == "Low Data").sum()

total_over = inventory.loc[
    inventory["Recommendation"] == "Reduce Price", "Price_vs_Benchmark"
].sum()
total_raise = inventory.loc[
    inventory["Recommendation"] == "Raise Price", "Price_vs_Benchmark"
].abs().sum()

st.markdown("### Priority Actions")

if multi_store:
    a1, a2, a3, a4 = st.columns(4)
else:
    a1, a2, a4 = st.columns(3)

with a1:
    st.markdown(f"""
    <div style="background: #2d1520; border-radius: 10px; padding: 1.2rem;
                border-left: 4px solid #e94560;">
        <div style="color: #e94560; font-size: 2rem; font-weight: 700;">
            {n_reduce}</div>
        <div style="color: #c0c0c0; font-size: 0.9rem;">Units Overpriced</div>
        <div style="color: #808090; font-size: 0.8rem;">
            {fmt_money(total_over)} total above benchmark</div>
    </div>""", unsafe_allow_html=True)

with a2:
    st.markdown(f"""
    <div style="background: #152d20; border-radius: 10px; padding: 1.2rem;
                border-left: 4px solid #4ecca3;">
        <div style="color: #4ecca3; font-size: 2rem; font-weight: 700;">
            {n_raise}</div>
        <div style="color: #c0c0c0; font-size: 0.9rem;">Units Underpriced</div>
        <div style="color: #808090; font-size: 0.8rem;">
            {fmt_money(total_raise)} raise opportunity</div>
    </div>""", unsafe_allow_html=True)

if multi_store:
    with a3:
        st.markdown(f"""
        <div style="background: #1a2040; border-radius: 10px; padding: 1.2rem;
                    border-left: 4px solid #6c8cff;">
            <div style="color: #6c8cff; font-size: 2rem; font-weight: 700;">
                {n_move}</div>
            <div style="color: #c0c0c0; font-size: 0.9rem;">Move Candidates</div>
            <div style="color: #808090; font-size: 0.8rem;">
                Used units better suited elsewhere</div>
        </div>""", unsafe_allow_html=True)

with a4:
    low_note = f" · {n_low} Low Data" if n_low > 0 else ""
    st.markdown(f"""
    <div style="background: #1a1a2e; border-radius: 10px; padding: 1.2rem;
                border-left: 4px solid #a0a0b0;">
        <div style="color: #a0a0b0; font-size: 2rem; font-weight: 700;">
            {n_well}</div>
        <div style="color: #c0c0c0; font-size: 0.9rem;">
            Well Priced{low_note}</div>
        <div style="color: #808090; font-size: 0.8rem;">No action needed</div>
    </div>""", unsafe_allow_html=True)

# ==========================================================
# MODEL DETAILS (collapsible)
# ==========================================================
with st.expander("📐 Model Details", expanded=False):
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Price Model R²", f"{res['price_r2']:.2f}")
    mc2.metric("Price MAE", f"${res['price_mae']:,.0f}")
    mc3.metric("Training Rows", f"{res['train_rows']:,}")
    mc4.metric("Processing Time", f"{res['run_time']:.1f}s")

    st.caption(
        f"Benchmark Price is predicted using a Gradient Boosting model trained "
        f"on {res['train_rows']:,} historical sales. "
        f"Features: Make, Model, Store, Stock Type, Vehicle Age, Mileage. "
        f"Turn estimates use segment averages from sales history."
    )
    if multi_store:
        st.caption(
            "Move recommendations compare store-level performance "
            "for the same Make/Model."
        )
    if n_low > 0:
        st.caption(
            f"⚠️ {n_low} units marked **Low Data** — fewer than "
            f"{LOW_DATA_THRESHOLD} historical sales for that segment. "
            f"Benchmark may be unreliable."
        )

# ==========================================================
# SHARED FILTER LOGIC
# ==========================================================
def apply_base_filters(df):
    """Apply Store, Stock Type, Make filters."""
    m = pd.Series(True, index=df.index)
    if store_f != "All":
        m &= (df["Store_Name"] == store_f)
    if stock_f != "All":
        m &= (df["Stock_Type"] == stock_f)
    if make_f != "All":
        m &= (df["Make"] == make_f)
    return df[m].copy()

# ==========================================================
# TABS
# ==========================================================
st.markdown("---")

if multi_store:
    tab_pricing, tab_move = st.tabs(["💰 Pricing Recommendations", "🚚 Move Stock Recommendations"])
else:
    tab_pricing, = st.tabs(["💰 Pricing Recommendations"])

# ----------------------------------------------------------
# TAB 1: PRICING
# ----------------------------------------------------------
with tab_pricing:
    # Recommendation filter — only visible in this tab
    rec_values = sorted(inventory["Recommendation"].dropna().unique().tolist())
    rec_opts = ["All"] + rec_values
    rec_f = st.selectbox("Filter by Recommendation", rec_opts, index=0, key="rec_filter")

    # Build table
    out = inventory[[
        "Stock_Number", "Make", "Model", "Year", "Mileage",
        "Store_Name", "Stock_Type", "Days_In_Stock",
        "Price", "Cost", "Benchmark_Price", "Price_vs_Benchmark",
        "Recommendation", "Suggested_Price", "Est_Days_Saved"
    ]].copy()

    out_f = apply_base_filters(out)
    if rec_f != "All":
        out_f = out_f[out_f["Recommendation"] == rec_f].copy()

    # Sort
    sort_order = {"Reduce Price": 0, "Raise Price": 1, "Well Priced": 2, "Low Data": 3}
    out_f["_sort"] = out_f["Recommendation"].map(sort_order).fillna(4)
    out_f = out_f.sort_values(
        ["_sort", "Price_vs_Benchmark"], ascending=[True, False]
    ).drop(columns="_sort")

    # Keep raw copy for download before formatting
    download_pricing = out_f.copy()

    # Format for display
    display = out_f.copy()
    display["Days_In_Stock"] = display["Days_In_Stock"].apply(lambda x: f"{x:.0f}")
    display["Price"] = display["Price"].apply(fmt_money)
    display["Cost"] = display["Cost"].apply(fmt_money)
    display["Benchmark_Price"] = display["Benchmark_Price"].apply(fmt_money)
    display["Price_vs_Benchmark"] = display["Price_vs_Benchmark"].apply(fmt_signed_money)
    display["Suggested_Price"] = display["Suggested_Price"].apply(fmt_money)
    display["Est_Days_Saved"] = display["Est_Days_Saved"].apply(
        lambda x: f"~{x:.0f} days" if x > 0 else "—"
    )

    display.rename(columns={
        "Days_In_Stock": "Days",
        "Benchmark_Price": "Benchmark Price",
        "Price_vs_Benchmark": "vs Benchmark",
        "Suggested_Price": "Suggested Price",
        "Est_Days_Saved": "Est. Days Saved",
    }, inplace=True)

    # Row count + table
    st.caption(f"Showing {len(display):,} units")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.download_button(
        f"📥 Download Pricing Results ({len(download_pricing)} units)",
        data=download_pricing.to_csv(index=False),
        file_name="logix_pricing_recommendations.csv",
        mime="text/csv",
        key="dl_pricing",
    )

# ----------------------------------------------------------
# TAB 2: MOVE STOCK (multi-store only)
# ----------------------------------------------------------
if multi_store:
    with tab_move:
        move_data = inventory[inventory["Move_To_Store"] != ""].copy()

        if move_data.empty:
            st.info("No move recommendations generated. "
                    "This can happen if all Used units are already at the best-performing store.")
        else:
            st.caption(
                "Used vehicles that could sell faster at another store. "
                f"Transfer cost of ${TRANSFER_COST:,.0f} per vehicle already "
                f"deducted from net uplift."
            )

            move_f = apply_base_filters(move_data)

            if move_f.empty:
                st.info("No move recommendations match current filters.")
            else:
                md = move_f[[
                    "Stock_Number", "Make", "Model", "Year", "Store_Name",
                    "Days_In_Stock", "Price",
                    "Move_To_Store", "Move_Days_Faster", "Move_Price_Uplift"
                ]].copy()

                # Keep raw for download
                download_move = md.copy()

                md = md.sort_values("Move_Days_Faster", ascending=False)
                md["Days_In_Stock"] = md["Days_In_Stock"].apply(lambda x: f"{x:.0f}")
                md["Price"] = md["Price"].apply(fmt_money)
                md["Move_Days_Faster"] = md["Move_Days_Faster"].apply(
                    lambda x: f"{x:.0f} days faster"
                    if pd.notna(x) and x > 0 else "—"
                )
                md["Move_Price_Uplift"] = md["Move_Price_Uplift"].apply(
                    lambda x: fmt_money(x) if pd.notna(x) else "—"
                )

                md.rename(columns={
                    "Store_Name": "Current Store",
                    "Days_In_Stock": "Days",
                    "Move_To_Store": "Move To",
                    "Move_Days_Faster": "Turn Improvement",
                    "Move_Price_Uplift": "Net Price Uplift",
                }, inplace=True)

                st.caption(f"Showing {len(md):,} move candidates")
                st.dataframe(md, use_container_width=True, hide_index=True)

                st.download_button(
                    f"📥 Download Move Recommendations ({len(download_move)} units)",
                    data=download_move.to_csv(index=False),
                    file_name="logix_move_recommendations.csv",
                    mime="text/csv",
                    key="dl_move",
                )
