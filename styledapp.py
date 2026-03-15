# app.py
# Intelligent Root Cause Analysis System
# Author: Navinchand Sahu

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import IsolationForest
import os
from urllib.parse import quote, unquote
from db_mysql import fetch_upload_history, save_upload

st.set_page_config(
    page_title="Intelligent RCA System",
    layout="wide",
    page_icon="📊"
)

# =========================
# GLOBAL STYLES
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --surface2:  #1a2236;
    --border:    #1e2d45;
    --accent:    #00d4ff;
    --accent2:   #7c3aed;
    --gold:      #f59e0b;
    --danger:    #ef4444;
    --success:   #10b981;
    --warn:      #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --radius:    12px;
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── App container ── */
.stApp {
    background: var(--bg) !important;
}
.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1280px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ── Header branding ── */
.rca-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.rca-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
.rca-tagline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Navigation radio ── */
div[role="radiogroup"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 6px !important;
    display: flex !important;
    gap: 4px !important;
}
div[role="radiogroup"] label {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
div[role="radiogroup"] label:has(input:checked) {
    background: var(--accent2) !important;
    color: #fff !important;
}
div[role="radiogroup"] label:hover {
    color: var(--text) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem 1.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
}

/* ── Section headings ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
}
h1 { font-size: 2rem !important; color: var(--text) !important; }
h2 { font-size: 1.3rem !important; color: var(--text) !important; }
h3 { font-size: 1.1rem !important; color: var(--text) !important; }

/* Streamlit subheader */
.stMarkdown h3 { color: var(--accent) !important; margin-top: 2rem !important; }

/* ── Chart wrappers ── */
[data-testid="stImage"], .stPyplot > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    margin-top: 0.5rem !important;
}

/* ── Alerts & banners ── */
.stAlert {
    border-radius: var(--radius) !important;
    border-left-width: 3px !important;
    font-size: 0.87rem !important;
}
[data-testid="stAlert"][data-baseweb="notification"] {
    background: var(--surface2) !important;
}

/* ── Info box ── */
.stInfo > div {
    background: rgba(0, 212, 255, 0.07) !important;
    border-left: 3px solid var(--accent) !important;
    color: var(--text) !important;
}
/* ── Success box ── */
.stSuccess > div {
    background: rgba(16, 185, 129, 0.07) !important;
    border-left: 3px solid var(--success) !important;
}
/* ── Warning box ── */
.stWarning > div {
    background: rgba(245, 158, 11, 0.07) !important;
    border-left: 3px solid var(--warn) !important;
}
/* ── Error box ── */
.stError > div {
    background: rgba(239, 68, 68, 0.08) !important;
    border-left: 3px solid var(--danger) !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}
.stDataFrame thead th {
    background: var(--surface2) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px !important;
    color: var(--accent) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Number inputs ── */
[data-testid="stNumberInput"] > div {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] input {
    color: #000000 !important;
    background: #ffffff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
[data-testid="stNumberInput"] label {
    color: var(--muted) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── History table headers ── */
.hist-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.4rem 0;
}
.hist-row {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.hist-row:hover { border-color: var(--accent); }
.hist-link a {
    color: var(--accent) !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    text-decoration: none;
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 3px 10px;
}
.hist-link a:hover { background: var(--accent); color: #000 !important; }

/* ── Chart desc ── */
.chart-desc {
    font-size: 0.85rem;
    color: var(--muted);
    margin: 0.2rem 0 1rem;
    font-style: italic;
}

/* ── Home hero ── */
.hero-card {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem;
    margin: 1rem 0 2rem;
}
.hero-feature {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    font-size: 0.95rem;
    color: var(--text);
}
.hero-feature-icon {
    font-size: 1.3rem;
    min-width: 2rem;
}

/* ── Description page ── */
.desc-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
}
.desc-card h4 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
}
.col-pill {
    display: inline-block;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 99px;
    padding: 4px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent);
    margin: 4px;
}

/* ── Section badge ── */
.section-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent2);
    background: rgba(124, 58, 237, 0.12);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 99px;
    padding: 3px 12px;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# =========================
# MATPLOTLIB THEME
# =========================
CHART_BG    = "#111827"
CHART_FACE  = "#111827"
CHART_GRID  = "#1e2d45"
CHART_TEXT  = "#94a3b8"
ACCENT_CLR  = "#00d4ff"
BAR_PALETTE = ["#00d4ff", "#7c3aed", "#10b981", "#f59e0b", "#ef4444", "#818cf8"]

def apply_chart_theme():
    mpl.rcParams.update({
        "figure.facecolor":  CHART_FACE,
        "axes.facecolor":    CHART_BG,
        "axes.edgecolor":    CHART_GRID,
        "axes.labelcolor":   CHART_TEXT,
        "axes.titlecolor":   "#e2e8f0",
        "axes.titlesize":    10,
        "axes.labelsize":    8,
        "axes.titlepad":     12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        CHART_GRID,
        "grid.linewidth":    0.5,
        "grid.alpha":        0.6,
        "xtick.color":       CHART_TEXT,
        "ytick.color":       CHART_TEXT,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "text.color":        CHART_TEXT,
        "legend.facecolor":  "#1a2236",
        "legend.edgecolor":  CHART_GRID,
        "legend.fontsize":   7,
    })

apply_chart_theme()

def style_axis(ax):
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.title.set_size(10)


def chart_desc(text):
    st.markdown(f'<p class="chart-desc">{text}</p>', unsafe_allow_html=True)


def section_header(icon, title, badge=None):
    badge_html = f'<span class="section-badge">{badge}</span><br>' if badge else ""
    st.markdown(f"""
    {badge_html}
    <h3 style="margin:0 0 0.25rem;">{icon} {title}</h3>
    """, unsafe_allow_html=True)


# =========================
# BRANDED HEADER
# =========================
st.markdown("""
<div class="rca-header">
    <div>
        <div class="rca-logo">RCA<span style="color:#7c3aed;">·</span>IQ</div>
        <div class="rca-tagline">Intelligent Root Cause Analysis System</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# CORE ANALYSIS FUNCTION
# =========================
def run_analysis(df):

    st.markdown('<span class="section-badge">KPI SUMMARY</span>', unsafe_allow_html=True)
    df["Profit"] = df["Revenue"] - df["Cost"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"₹ {df['Revenue'].sum():,.0f}")
    c2.metric("Total Cost",    f"₹ {df['Cost'].sum():,.0f}")
    c3.metric("Total Profit",  f"₹ {df['Profit'].sum():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # PROFIT TREND
    # =========================
    if "Date" in df.columns:
        section_header("📈", "Profit Trend Over Time", "TIME SERIES")
        chart_desc("How total profit changes across different dates.")

        col_a, col_b = st.columns(2)
        w = col_a.number_input("Width", 1, 20, 10, 1, key="trend_w")
        h = col_b.number_input("Height", 1, 15, 3, 1, key="trend_h")

        df["Date"] = pd.to_datetime(df["Date"])
        trend = df.groupby("Date")["Profit"].sum()

        fig, ax = plt.subplots(figsize=(w, h))
        ax.plot(trend.index, trend.values, color=ACCENT_CLR, linewidth=1.8)
        ax.fill_between(trend.index, trend.values, alpha=0.12, color=ACCENT_CLR)
        ax.set_xlabel("Date"); ax.set_ylabel("Profit")
        ax.set_title("Profit Trend")
        style_axis(ax)
        st.pyplot(fig)
        plt.close(fig)

    # =========================
    # PRODUCT PROFIT BAR
    # =========================
    if "Product" in df.columns:
        section_header("📊", "Product-wise Profit", "PRODUCT BREAKDOWN")
        chart_desc("Compares profit generated by each product.")

        col_a, col_b = st.columns(2)
        w = col_a.number_input("Width", 1, 20, 7, 1, key="bar_w")
        h = col_b.number_input("Height", 1, 15, 4, 1, key="bar_h")

        bar = df.groupby("Product")["Profit"].sum()

        fig, ax = plt.subplots(figsize=(w, h))
        colors = [BAR_PALETTE[i % len(BAR_PALETTE)] for i in range(len(bar))]
        bar.plot(kind="bar", ax=ax, color=colors, edgecolor="none")
        ax.set_xlabel("Product"); ax.set_ylabel("Profit")
        ax.set_title("Product-wise Profit")
        ax.tick_params(axis="x", rotation=45)
        style_axis(ax)
        st.pyplot(fig)
        plt.close(fig)

    # =========================
    # FACTOR IMPACT BAR
    # =========================
    section_header("📊", "Factors Impacting Revenue", "CORRELATION ANALYSIS")
    chart_desc("Which numeric factors influence revenue the most.")

    numeric_cols = df.select_dtypes(include=np.number).columns.drop("Revenue", errors="ignore")
    impact = {col: df[col].corr(df["Revenue"]) for col in numeric_cols}
    impact = pd.Series(impact).dropna().sort_values(key=abs, ascending=False)

    col_a, col_b = st.columns(2)
    w = col_a.number_input("Width", 1, 20, 7, 1, key="impact_w")
    h = col_b.number_input("Height", 1, 15, 4, 1, key="impact_h")

    fig, ax = plt.subplots(figsize=(w, h))
    colors = [ACCENT_CLR if v >= 0 else "#ef4444" for v in impact.values]
    impact.plot(kind="bar", ax=ax, color=colors, edgecolor="none")
    ax.set_xlabel("Factor"); ax.set_ylabel("Correlation with Revenue")
    ax.set_title("Revenue Drivers")
    ax.tick_params(axis="x", rotation=45)
    style_axis(ax)
    st.pyplot(fig)
    plt.close(fig)

    # =========================
    # PIE CHART
    # =========================
    if "Product" in df.columns:
        section_header("🥧", "Product Contribution", "SHARE OF PROFIT")
        chart_desc("Percentage contribution of each product to total profit.")

        col_a, col_b = st.columns(2)
        w = col_a.number_input("Width", 1, 15, 5, 1, key="pie_w")
        h = col_b.number_input("Height", 1, 15, 5, 1, key="pie_h")

        data = df.groupby("Product")["Profit"].sum()
        data = data[data > 0]

        fig, ax = plt.subplots(figsize=(w, h))
        wedges, texts, autotexts = ax.pie(
            data, labels=data.index, autopct="%1.1f%%",
            startangle=90, colors=BAR_PALETTE[:len(data)],
            wedgeprops=dict(edgecolor=CHART_BG, linewidth=2)
        )
        for t in autotexts:
            t.set_color("#e2e8f0")
            t.set_fontsize(7)
        ax.set_title("Profit Contribution")
        st.pyplot(fig)
        plt.close(fig)

    # =========================
    # SCATTER
    # =========================
    if {"Region", "Product", "Sales"}.issubset(df.columns):
        section_header("📍", "Product Popularity by Region", "GEO SPREAD")
        chart_desc("Which products are popular in which regions based on sales.")

        col_a, col_b = st.columns(2)
        w = col_a.number_input("Width", 1, 20, 8, 1, key="sc_w")
        h = col_b.number_input("Height", 1, 15, 4, 1, key="sc_h")

        fig, ax = plt.subplots(figsize=(w, h))
        for i, product in enumerate(df["Product"].unique()):
            temp = df[df["Product"] == product]
            ax.scatter(temp["Region"], temp["Sales"],
                       label=product, color=BAR_PALETTE[i % len(BAR_PALETTE)],
                       alpha=0.8, s=40)
        ax.set_xlabel("Region"); ax.set_ylabel("Sales")
        ax.set_title("Product Sales by Region")
        ax.legend(fontsize=6)
        style_axis(ax)
        st.pyplot(fig)
        plt.close(fig)

    # =========================
    # HISTOGRAM
    # =========================
    section_header("📊", "Profit Distribution", "DISTRIBUTION")
    chart_desc("Distribution of profit values across all transactions.")

    col_a, col_b = st.columns(2)
    w = col_a.number_input("Width", 1, 20, 8, 1, key="hist_w")
    h = col_b.number_input("Height", 1, 15, 4, 1, key="hist_h")

    fig, ax = plt.subplots(figsize=(w, h))
    ax.hist(df["Profit"], bins=30, color=ACCENT_CLR, alpha=0.85, edgecolor=CHART_BG)
    ax.set_xlabel("Profit"); ax.set_ylabel("Frequency")
    ax.set_title("Profit Distribution")
    style_axis(ax)
    st.pyplot(fig)
    plt.close(fig)

    # =========================
    # ANOMALY DETECTION
    # =========================
    section_header("🚨", "Anomaly Detection", "ML · ISOLATION FOREST")
    if len(df) > 10:
        iso = IsolationForest(contamination=0.15, random_state=42)
        df["Anomaly"] = iso.fit_predict(df[["Profit"]])
        anomalies = df[df["Anomaly"] == -1]
        if anomalies.empty:
            st.success("No significant anomalies detected.")
        else:
            st.warning(f"{len(anomalies)} anomalous transactions flagged.")
            st.dataframe(anomalies, use_container_width=True)

    # =========================
    # BUSINESS INSIGHTS
    # =========================
    section_header("💡", "Business Insights", "ROOT CAUSE ANALYSIS")

    loss_ratio = (df["Profit"] < 0).mean()

    if loss_ratio > 0.4:
        st.error("⚠️ High loss-making transactions detected across portfolio")
    elif loss_ratio > 0.2:
        st.warning("⚠️ Moderate losses detected — review cost structure")
    else:
        st.success("✅ Business is largely profitable")

    corr = df.select_dtypes(include=np.number).corr()["Profit"].drop("Profit")
    if not corr.empty:
        st.info(f"📈 Strongest Profit Driver: **{corr.idxmax()}**")
        st.info(f"📉 Strongest Loss Driver: **{corr.idxmin()}**")

    if "Product" in df.columns:

        popularity_metric = None
        if "Sales" in df.columns:
            popularity_metric = "Sales"
        elif "Quantity" in df.columns:
            popularity_metric = "Quantity"

        if popularity_metric:
            popularity        = df.groupby("Product")[popularity_metric].sum()
            revenue_by_product = df.groupby("Product")["Revenue"].sum()
            profit_by_product  = df.groupby("Product")["Profit"].sum()
            most_popular  = popularity.idxmax()
            least_popular = popularity.idxmin()

            st.markdown("### 🛒 Customer Popularity Insights")
            st.success(f"⭐ **Most Popular:** {most_popular} · {popularity_metric}: {popularity[most_popular]:,.0f} · Revenue: ₹{revenue_by_product[most_popular]:,.0f} · Profit: ₹{profit_by_product[most_popular]:,.0f}")
            st.warning(f"📉 **Least Popular:** {least_popular} · {popularity_metric}: {popularity[least_popular]:,.0f} · Revenue: ₹{revenue_by_product[least_popular]:,.0f} · Profit: ₹{profit_by_product[least_popular]:,.0f}")

        loss_df = df[df["Profit"] < 0]

        if not loss_df.empty:
            if "Region" in df.columns:
                loss_region = loss_df.groupby("Region")["Profit"].sum().idxmin()
                st.error(f"📍 Highest Loss Region: **{loss_region}**")

            if "Date" in df.columns:
                loss_date = loss_df.groupby(pd.to_datetime(loss_df["Date"]))["Profit"].sum().idxmin()
                st.error(f"📅 Worst Loss Date: **{loss_date.date()}**")

        if "Product" in df.columns:
            cost_imbalance = (
                df.groupby("Product")[["Revenue", "Cost"]].sum()
                .assign(Cost_Ratio=lambda x: x["Cost"] / x["Revenue"])
                .sort_values("Cost_Ratio", ascending=False)
            )
            if not cost_imbalance.empty:
                worst = cost_imbalance.index[0]
                st.warning(f"💸 Cost Inefficiency: **{worst}** (Cost is {cost_imbalance.iloc[0]['Cost_Ratio']:.2f}x Revenue)")

            if {"Sales", "Product"}.issubset(df.columns):
                risky = (
                    df.groupby("Product")[["Sales", "Profit"]].sum()
                    .sort_values("Sales", ascending=False)
                )
                risky = risky[risky["Profit"] < 0]
                if not risky.empty:
                    st.warning(f"⚠️ High Sales but Loss-Making: **{risky.index[0]}**")

            if "Region" in df.columns:
                region_loss_ratio = (
                    df.assign(Is_Loss=df["Profit"] < 0)
                    .groupby("Region")["Is_Loss"].mean()
                    .sort_values(ascending=False)
                )
                worst_lr = region_loss_ratio.index[0]
                st.warning(f"📊 Highest Loss Ratio Region: **{worst_lr}** ({region_loss_ratio.iloc[0]*100:.1f}% loss)")

            if "Anomaly" in df.columns:
                anomaly_prods = (
                    df[df["Anomaly"] == -1].groupby("Product").size().sort_values(ascending=False)
                )
                if not anomaly_prods.empty:
                    st.warning(f"🚨 Most Anomaly-Prone Product: **{anomaly_prods.index[0]}**")

            margin = (
                df.groupby("Product")[["Revenue", "Profit"]].sum()
                .assign(Margin=lambda x: x["Profit"] / x["Revenue"])
                .sort_values("Margin")
            )
            if not margin.empty:
                st.warning(f"📉 Lowest Margin Product: **{margin.index[0]}** ({margin.iloc[0]['Margin']*100:.1f}% margin)")

            if {"Region", "Product"}.issubset(df.columns) and not loss_df.empty:
                combo_loss = loss_df.groupby(["Region", "Product"])["Profit"].sum().sort_values()
                if not combo_loss.empty:
                    region, product = combo_loss.index[0]
                    st.error(f"🔗 Loss Hotspot: **{product}** in **{region}**")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            cost_trend = df.groupby("Date")["Cost"].sum()
            spike = cost_trend[cost_trend > cost_trend.mean() + 2 * cost_trend.std()]
            if not spike.empty:
                st.warning(f"⚡ Sudden Cost Spike on: **{spike.index[0].date()}**")

        if {"Sales", "Revenue"}.issubset(df.columns):
            corr_sr = df["Sales"].corr(df["Revenue"])
            if corr_sr < 0.3:
                st.warning("📉 Revenue weakly correlated with Sales — possible pricing or discount issue")

        if "Date" in df.columns:
            df["Month"] = pd.to_datetime(df["Date"]).dt.month
            monthly_loss = df.groupby("Month")["Profit"].sum().sort_values()
            if not monthly_loss.empty:
                st.warning(f"📆 Worst Performing Month: **{monthly_loss.index[0]}**")


# =========================
# LOAD FROM HISTORY LINK
# =========================
query = st.query_params.get("file")
if query:
    file_name = unquote(query)
    file_path = os.path.join("uploads", file_name)
    if os.path.exists(file_path):
        st.success(f"📂 Loaded dataset: **{file_name}**")
        df = pd.read_csv(file_path) if file_name.endswith(".csv") else pd.read_excel(file_path)
        run_analysis(df)
        st.stop()

# =========================
# NAVIGATION
# =========================
menu = st.radio(
    "Navigation",
    ["🏠 Home", "📘 Description", "📤 Upload Dataset", "🕒 History"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# HOME
# =========================
if menu == "🏠 Home":
    st.markdown("""
    <div class="hero-card">
        <h2 style="font-family:'Syne',sans-serif;font-size:1.6rem;margin:0 0 0.25rem;">
            Intelligent Root Cause Analysis
        </h2>
        <p style="color:#64748b;font-size:0.9rem;margin:0 0 1.5rem;">
            Upload a business dataset and instantly surface what's driving profit and loss.
        </p>
        <div class="hero-feature"><span class="hero-feature-icon">🔍</span> Detect profit & loss anomalies with ML-powered Isolation Forest</div>
        <div class="hero-feature"><span class="hero-feature-icon">🧠</span> Identify root causes across products, regions, and time</div>
        <div class="hero-feature"><span class="hero-feature-icon">📊</span> Interactive trend, distribution, and correlation charts</div>
        <div class="hero-feature"><span class="hero-feature-icon">💡</span> Actionable business insights generated automatically</div>
        <div class="hero-feature"><span class="hero-feature-icon">🕒</span> Full upload history with one-click re-analysis</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("👉 **Navigate using the tabs above to upload a dataset or view past analyses.**")

# =========================
# DESCRIPTION
# =========================
elif menu == "📘 Description":
    st.markdown("""
    <div class="desc-card">
        <h4>⚠️ Data Requirements</h4>
        <p style="font-size:0.9rem;color:#94a3b8;margin-bottom:1rem;">
            Please ensure your uploaded dataset is <strong style="color:#e2e8f0;">clean, structured, and error-free</strong>.
            Missing values, incorrect data types, or inconsistent entries may affect analysis accuracy.
        </p>
        <h4 style="margin-top:1rem;">Mandatory Columns</h4>
        <div>
            <span class="col-pill">Revenue</span>
            <span class="col-pill">Cost</span>
            <span class="col-pill">Product</span>
            <span class="col-pill">Region</span>
            <span class="col-pill">Sales</span>
            <span class="col-pill">Date</span>
        </div>
    </div>
    <div class="desc-card">
        <h4>Supported File Formats</h4>
        <span class="col-pill">.csv</span>
        <span class="col-pill">.xlsx</span>
    </div>
    """, unsafe_allow_html=True)

# =========================
# UPLOAD DATASET
# =========================
elif menu == "📤 Upload Dataset":
    st.markdown('<span class="section-badge">UPLOAD</span>', unsafe_allow_html=True)
    st.markdown("### Upload a Dataset")
    file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"],
                            label_visibility="collapsed")
    if file:
        os.makedirs("uploads", exist_ok=True)
        path = os.path.join("uploads", file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        save_upload(file.name, path)
        st.success(f"✅ **{file.name}** uploaded successfully.")
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        st.markdown("---")
        run_analysis(df)

# =========================
# HISTORY
# =========================
elif menu == "🕒 History":
    st.markdown('<span class="section-badge">AUDIT LOG</span>', unsafe_allow_html=True)
    st.markdown("### Upload History")

    history = fetch_upload_history()

    if not history:
        st.info("No uploaded datasets yet. Go to **Upload Dataset** to get started.")
    else:
        # Header row
        h1, h2, h3, h4 = st.columns([3, 2, 2, 2])
        h1.markdown('<p class="hist-header">Dataset</p>', unsafe_allow_html=True)
        h2.markdown('<p class="hist-header">Date</p>', unsafe_allow_html=True)
        h3.markdown('<p class="hist-header">Time</p>', unsafe_allow_html=True)
        h4.markdown('<p class="hist-header">Action</p>', unsafe_allow_html=True)

        st.markdown("<hr style='margin:0.25rem 0 0.75rem;'>", unsafe_allow_html=True)

        for h in history:
            file = quote(h["file_name"])
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            c1.write(h["file_name"])
            c2.write(str(h["upload_date"]))
            c3.write(str(h["upload_time"])[:8])
            c4.markdown(
                f'<div class="hist-link"><a href="/?file={file}" target="_self">▶ Open</a></div>',
                unsafe_allow_html=True
            )