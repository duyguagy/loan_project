# Loan Approval Prediction — Streamlit App

# Requirements (pip install):
#   streamlit pandas numpy scikit-learn plotly
#
# Run with:
#   streamlit run app.py

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* ---- global ---- */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* ---- header banner ---- */
    .hero {
        background: linear-gradient(135deg, #0f2027 0%, #1a3a4a 50%, #0d3b52 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(56,189,248,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 { color: #f0f9ff; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero p  { color: #94d2f7; font-size: 1rem; margin: 0.4rem 0 0; }

    /* ---- metric cards ---- */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
    .metric-card {
        flex: 1 1 160px;
        background: #1e3a4f;
        border: 1px solid #2d5771;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        text-align: center;
    }
    .metric-card .val { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
    .metric-card .lbl { font-size: 0.78rem; color: #7ec8e3; letter-spacing: .04em; text-transform: uppercase; }

    /* ---- section headers ---- */
    .section-title {
        font-size: 1.1rem; font-weight: 600;
        color: #0ea5e9; text-transform: uppercase;
        letter-spacing: .06em; margin: 1.5rem 0 0.8rem;
        border-left: 4px solid #0ea5e9; padding-left: 0.7rem;
    }

    /* ---- prediction result box ---- */
    .result-approved {
        background: linear-gradient(135deg, #052e16, #064e3b);
        border: 1px solid #22c55e;
        border-radius: 14px; padding: 1.5rem; text-align: center;
    }
    .result-rejected {
        background: linear-gradient(135deg, #3b0f0f, #4c1414);
        border: 1px solid #ef4444;
        border-radius: 14px; padding: 1.5rem; text-align: center;
    }
    .result-approved h2, .result-rejected h2 { font-size: 1.8rem; margin: 0; }
    .result-approved h2 { color: #4ade80; }
    .result-rejected h2 { color: #f87171; }
    .prob-bar-wrap { margin-top: 0.8rem; }
    .prob-label { color: #cbd5e1; font-size: 0.85rem; margin-bottom: 0.3rem; }

    /* ── sidebar ── */
    section[data-testid="stSidebar"] { background: #0b1e2d; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSlider > div > div { background: #1e3a4f !important; }
    .stSelectbox div[data-baseweb="select"] { background: #1e3a4f !important; border-color: #2d5771 !important; }

    /* ── tabs ── */
    .stTabs [data-baseweb="tab-list"] { background: #0f2027; border-radius: 10px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #7ec8e3 !important; font-weight: 500; padding: 0.45rem 1.1rem; }
    .stTabs [aria-selected="true"] { background: #0ea5e9 !important; color: #fff !important; }

    /* ── tables ── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────

@st.cache_data
def load_and_clean(path: str) -> pd.DataFrame:
    """Load dataset, strip whitespace from strings, handle obvious issues."""
    df = pd.read_csv(path)

    # Strip leading/trailing spaces from column names and string values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # loan_id is a row identifier — not a predictive feature; drop it
    if "loan_id" in df.columns:
        df.drop(columns=["loan_id"], inplace=True)

    # Negative asset values exist (data-entry artefacts): clip to 0
    asset_cols = [c for c in df.columns if "assets_value" in c]
    for c in asset_cols:
        df[c] = df[c].clip(lower=0)

    # Drop rows where the target is missing (cannot be imputed meaningfully)
    df.dropna(subset=["loan_status"], inplace=True)

    # For numeric columns with missing values, impute with median
    num_cols = df.select_dtypes(include=np.number).columns
    for c in num_cols:
        if df[c].isna().sum():
            df[c].fillna(df[c].median(), inplace=True)

    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes("object").columns
    for c in cat_cols:
        if df[c].isna().sum():
            df[c].fillna(df[c].mode()[0], inplace=True)

    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    """
    Encode categoricals, scale numerics, train Logistic Regression
    and Random Forest classifiers, return everything needed for inference.
    """
    target = "loan_status"
    feature_cols = [c for c in df.columns if c != target]

    # Encode target: Approved → 1, Rejected → 0
    y = (df[target] == "Approved").astype(int)
    X = df[feature_cols].copy()

    # Encode binary / ordinal categoricals with LabelEncoder
    le_map: dict[str, LabelEncoder] = {}
    cat_feats = X.select_dtypes("object").columns.tolist()
    for c in cat_feats:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        le_map[c] = le

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    results = {
        "RandomForest": {
            "model": rf,
            "train_acc": accuracy_score(y_train, rf.predict(X_train)),
            "test_acc":  accuracy_score(y_test,  rf.predict(X_test)),
            "report":    classification_report(y_test, rf.predict(X_test),
                                               target_names=["Rejected", "Approved"]),
            "cm":        confusion_matrix(y_test, rf.predict(X_test)),
            "roc_auc":   roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
            "fpr_tpr":   roc_curve(y_test, rf.predict_proba(X_test)[:, 1]),
            "feat_imp":  pd.Series(rf.feature_importances_, index=X.columns)
                           .sort_values(ascending=False),
        },
        "LogisticRegression": {
            "model": lr,
            "train_acc": accuracy_score(y_train, lr.predict(X_train)),
            "test_acc":  accuracy_score(y_test,  lr.predict(X_test)),
            "report":    classification_report(y_test, lr.predict(X_test),
                                               target_names=["Rejected", "Approved"]),
            "cm":        confusion_matrix(y_test, lr.predict(X_test)),
            "roc_auc":   roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]),
            "fpr_tpr":   roc_curve(y_test, lr.predict_proba(X_test)[:, 1]),
        },
    }

    return results, scaler, le_map, feature_cols, X.columns.tolist()


# ── Plotly theme ──────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,32,39,0.6)",
    font=dict(color="#94a3b8", family="DM Sans"),
    xaxis=dict(gridcolor="#1e3a4f", linecolor="#1e3a4f"),
    yaxis=dict(gridcolor="#1e3a4f", linecolor="#1e3a4f"),
    margin=dict(l=10, r=10, t=40, b=10),
)
COLOR_APPROVED = "#22c55e"
COLOR_REJECTED = "#ef4444"
PALETTE = px.colors.sequential.Blues[::-1]


# ═══════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Load data ──────────────────────────────────────────────
    DATA_PATH = "loan_approval_dataset.csv"
    try:
        df = load_and_clean(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Dataset not found at `{DATA_PATH}`. "
                 "Place `loan_approval_dataset.csv` in the same directory as `app.py`.")
        st.stop()

    # ── Train models ───────────────────────────────────────────
    with st.spinner("Training models…"):
        results, scaler, le_map, feature_cols, scaled_cols = train_models(df)

    # ── Hero banner ────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <h1>🏦 Loan Approval Predictor</h1>
      <p>Exploratory analysis · Model evaluation · Real-time prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Top-level KPIs ─────────────────────────────────────────
    approved = (df["loan_status"] == "Approved").sum()
    rejected = len(df) - approved
    rf_acc   = results["RandomForest"]["test_acc"]
    lr_acc   = results["LogisticRegression"]["test_acc"]

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, len(df),              "Total Records"),
        (c2, df.shape[1],          "Features"),
        (c3, approved,             "✅ Approved"),
        (c4, rejected,             "❌ Rejected"),
        (c5, f"{rf_acc*100:.1f}%", "RF Accuracy"),
    ]
    for col, val, lbl in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="val">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────
    tab_eda, tab_model, tab_predict = st.tabs(
        ["📊  Exploratory Analysis", "🤖  Model Performance", "🔮  Live Prediction"]
    )

    # ════════════════════════════════════════════════════════════
    #  TAB 1 — EDA
    # ════════════════════════════════════════════════════════════
    with tab_eda:

        # ── Dataset preview ────────────────────────────────────
        st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True, height=260)

        col_info, col_miss = st.columns(2)

        with col_info:
            st.markdown('<div class="section-title">Column Info</div>', unsafe_allow_html=True)
            info_df = pd.DataFrame({
                "dtype":    df.dtypes.astype(str),
                "non-null": df.notnull().sum(),
                "unique":   df.nunique(),
            })
            st.dataframe(info_df, use_container_width=True)

        with col_miss:
            st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
            miss = df.isnull().sum()
            if miss.sum() == 0:
                st.success("✅ No missing values after cleaning.")
            else:
                miss_df = miss[miss > 0].reset_index()
                miss_df.columns = ["Column", "Missing"]
                fig_miss = px.bar(miss_df, x="Column", y="Missing",
                                  color="Missing", color_continuous_scale="Reds")
                fig_miss.update_layout(**DARK_LAYOUT, title="Missing Values per Column")
                st.plotly_chart(fig_miss, use_container_width=True)

        # ── Target distribution ────────────────────────────────
        st.markdown('<div class="section-title">Loan Status Distribution</div>', unsafe_allow_html=True)
        col_pie, col_bar = st.columns(2)

        status_counts = df["loan_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        with col_pie:
            fig_pie = px.pie(
                status_counts, names="Status", values="Count",
                color="Status",
                color_discrete_map={"Approved": COLOR_APPROVED, "Rejected": COLOR_REJECTED},
                hole=0.55,
            )
            fig_pie.update_layout(**DARK_LAYOUT, title="Approval Split")
            fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            fig_bar = px.bar(
                status_counts, x="Status", y="Count",
                color="Status",
                color_discrete_map={"Approved": COLOR_APPROVED, "Rejected": COLOR_REJECTED},
                text="Count",
            )
            fig_bar.update_layout(**DARK_LAYOUT, title="Count by Status", showlegend=False)
            fig_bar.update_traces(textposition="outside", marker_line_width=0)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── CIBIL score & income distributions ────────────────
        st.markdown('<div class="section-title">Key Feature Distributions</div>', unsafe_allow_html=True)
        col_cibil, col_income = st.columns(2)

        with col_cibil:
            fig_cs = px.histogram(
                df, x="cibil_score", color="loan_status",
                color_discrete_map={"Approved": COLOR_APPROVED, "Rejected": COLOR_REJECTED},
                nbins=40, barmode="overlay", opacity=0.75,
            )
            fig_cs.update_layout(**DARK_LAYOUT, title="CIBIL Score Distribution",
                                 legend_title="Status")
            st.plotly_chart(fig_cs, use_container_width=True)

        with col_income:
            fig_inc = px.histogram(
                df, x="income_annum", color="loan_status",
                color_discrete_map={"Approved": COLOR_APPROVED, "Rejected": COLOR_REJECTED},
                nbins=40, barmode="overlay", opacity=0.75,
            )
            fig_inc.update_layout(**DARK_LAYOUT, title="Annual Income Distribution",
                                  legend_title="Status")
            st.plotly_chart(fig_inc, use_container_width=True)

        # ── Loan amount vs CIBIL scatter ───────────────────────
        st.markdown('<div class="section-title">Loan Amount vs CIBIL Score</div>', unsafe_allow_html=True)
        fig_scat = px.scatter(
            df.sample(min(2000, len(df)), random_state=42),
            x="cibil_score", y="loan_amount",
            color="loan_status",
            color_discrete_map={"Approved": COLOR_APPROVED, "Rejected": COLOR_REJECTED},
            opacity=0.55, size_max=6,
            hover_data=["income_annum", "loan_term"],
        )
        fig_scat.update_layout(**DARK_LAYOUT,
                               title="CIBIL Score vs Loan Amount (sample of 2 000)",
                               legend_title="Status")
        st.plotly_chart(fig_scat, use_container_width=True)

        # ── Correlation heatmap ────────────────────────────────
        st.markdown('<div class="section-title">Correlation Heatmap (Numeric Features)</div>',
                    unsafe_allow_html=True)
        num_df = df.select_dtypes(include=np.number)
        corr   = num_df.corr()
        fig_hm = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            aspect="auto",
        )
        fig_hm.update_layout(**DARK_LAYOUT, title="Pearson Correlation Matrix")
        st.plotly_chart(fig_hm, use_container_width=True)

        # ── Box plots: assets by status ────────────────────────
        st.markdown('<div class="section-title">Asset Values by Loan Status</div>',
                    unsafe_allow_html=True)
        asset_cols_plot = [c for c in df.columns if "assets_value" in c]
        fig_box = go.Figure()
        for col in asset_cols_plot:
            for status, color in [("Approved", COLOR_APPROVED), ("Rejected", COLOR_REJECTED)]:
                sub = df[df["loan_status"] == status][col]
                fig_box.add_trace(go.Box(
                    y=sub, name=f"{col.replace('_assets_value','').title()} – {status}",
                    marker_color=color, line_color=color, opacity=0.75,
                    boxmean="sd",
                ))
        fig_box.update_layout(**DARK_LAYOUT, title="Asset Distribution by Approval Status",
                              showlegend=True)
        st.plotly_chart(fig_box, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    #  TAB 2 — MODEL PERFORMANCE
    # ════════════════════════════════════════════════════════════
    with tab_model:

        model_choice = st.radio(
            "Select model to inspect",
            ["RandomForest", "LogisticRegression"],
            horizontal=True,
        )
        res = results[model_choice]

        # ── Accuracy cards ─────────────────────────────────────
        st.markdown('<div class="section-title">Accuracy Metrics</div>', unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""<div class="metric-card">
              <div class="val">{res['train_acc']*100:.2f}%</div>
              <div class="lbl">Train Accuracy</div></div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""<div class="metric-card">
              <div class="val">{res['test_acc']*100:.2f}%</div>
              <div class="lbl">Test Accuracy</div></div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""<div class="metric-card">
              <div class="val">{res['roc_auc']:.4f}</div>
              <div class="lbl">ROC-AUC</div></div>""", unsafe_allow_html=True)

        # ── Classification report ──────────────────────────────
        st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
        st.code(res["report"], language="text")

        # ── Confusion matrix + ROC curve ───────────────────────
        col_cm, col_roc = st.columns(2)

        with col_cm:
            cm = res["cm"]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Rejected", "Approved"], y=["Rejected", "Approved"],
                text_auto=True, color_continuous_scale="Blues",
            )
            fig_cm.update_layout(**DARK_LAYOUT, title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            fpr, tpr, _ = res["fpr_tpr"]
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"AUC = {res['roc_auc']:.4f}",
                line=dict(color="#38bdf8", width=2.5),
            ))
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(dash="dash", color="#475569"))
            fig_roc.update_layout(**DARK_LAYOUT, title="ROC Curve",
                                  xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)

        # ── Feature importance (RF only) ───────────────────────
        if model_choice == "RandomForest":
            st.markdown('<div class="section-title">Feature Importances</div>', unsafe_allow_html=True)
            fi = res["feat_imp"].reset_index()
            fi.columns = ["Feature", "Importance"]
            fig_fi = px.bar(
                fi, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
            )
            fig_fi.update_layout(**DARK_LAYOUT, title="Random Forest Feature Importances",
                                 yaxis=dict(autorange="reversed", **DARK_LAYOUT["yaxis"]))
            st.plotly_chart(fig_fi, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    #  TAB 3 — LIVE PREDICTION
    # ════════════════════════════════════════════════════════════
    with tab_predict:
        st.markdown("### Enter applicant details in the sidebar, then click **Predict**.")

        # ── Sidebar inputs ─────────────────────────────────────
        with st.sidebar:
            st.markdown("## 🔎 Applicant Details")
            st.markdown("---")

            no_of_dependents = st.slider(
                "Number of Dependents", 0, 5, 2,
                help="Number of people financially dependent on the applicant."
            )
            education = st.selectbox(
                "Education Level",
                ["Graduate", "Not Graduate"],
            )
            self_employed = st.selectbox(
                "Self Employed?",
                ["No", "Yes"],
            )
            income_annum = st.number_input(
                "Annual Income (₹)", min_value=0,
                value=5_000_000, step=100_000,
                help="Annual income in Indian Rupees."
            )
            loan_amount = st.number_input(
                "Loan Amount (₹)", min_value=0,
                value=10_000_000, step=500_000,
            )
            loan_term = st.slider(
                "Loan Term (years)", 2, 20, 10, step=2,
            )
            cibil_score = st.slider(
                "CIBIL Score", 300, 900, 650,
                help="Credit score (higher is better)."
            )
            residential_assets_value = st.number_input(
                "Residential Assets (₹)", min_value=0, value=3_000_000, step=100_000,
            )
            commercial_assets_value = st.number_input(
                "Commercial Assets (₹)", min_value=0, value=2_000_000, step=100_000,
            )
            luxury_assets_value = st.number_input(
                "Luxury Assets (₹)", min_value=0, value=5_000_000, step=100_000,
            )
            bank_asset_value = st.number_input(
                "Bank Assets (₹)", min_value=0, value=2_000_000, step=100_000,
            )

            st.markdown("---")
            model_sel = st.selectbox(
                "Prediction Model",
                ["RandomForest", "LogisticRegression"],
            )
            predict_btn = st.button("🔮 Predict Loan Approval", use_container_width=True)

        # ── Run prediction ─────────────────────────────────────
        if predict_btn:
            # Build input dataframe
            input_dict = {
                "no_of_dependents":         [no_of_dependents],
                "education":                [education],
                "self_employed":            [self_employed],
                "income_annum":             [income_annum],
                "loan_amount":              [loan_amount],
                "loan_term":                [loan_term],
                "cibil_score":              [cibil_score],
                "residential_assets_value": [residential_assets_value],
                "commercial_assets_value":  [commercial_assets_value],
                "luxury_assets_value":      [luxury_assets_value],
                "bank_asset_value":         [bank_asset_value],
            }
            input_df = pd.DataFrame(input_dict)

            # Encode categoricals using fitted LabelEncoders
            for col, le in le_map.items():
                if col in input_df.columns:
                    val = input_df[col].iloc[0]
                    if val not in le.classes_:
                        # fallback: pick closest known class
                        val = le.classes_[0]
                    input_df[col] = le.transform([val])

            # Reorder columns to match training order
            input_df = input_df[scaled_cols]

            # Scale
            input_scaled = scaler.transform(input_df)

            clf   = results[model_sel]["model"]
            pred  = clf.predict(input_scaled)[0]
            proba = clf.predict_proba(input_scaled)[0]

            approve_prob = proba[1]
            reject_prob  = proba[0]

            # ── Show result ────────────────────────────────────
            col_res, col_detail = st.columns([1, 1])

            with col_res:
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-approved">
                      <h2>✅ APPROVED</h2>
                      <p style="color:#86efac; margin-top:0.5rem; font-size:1rem;">
                        Confidence: {approve_prob*100:.1f}%
                      </p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-rejected">
                      <h2>❌ REJECTED</h2>
                      <p style="color:#fca5a5; margin-top:0.5rem; font-size:1rem;">
                        Confidence: {reject_prob*100:.1f}%
                      </p>
                    </div>""", unsafe_allow_html=True)

                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=approve_prob * 100,
                    title={"text": "Approval Probability (%)", "font": {"color": "#94a3b8"}},
                    gauge={
                        "axis":  {"range": [0, 100], "tickcolor": "#475569"},
                        "bar":   {"color": COLOR_APPROVED if pred == 1 else COLOR_REJECTED},
                        "steps": [
                            {"range": [0,  40], "color": "#1e1e2e"},
                            {"range": [40, 60], "color": "#1e2a3a"},
                            {"range": [60, 100], "color": "#1a3a2a"},
                        ],
                        "threshold": {
                            "line": {"color": "#38bdf8", "width": 3},
                            "thickness": 0.78, "value": 50,
                        },
                    },
                    number={"font": {"color": "#e2e8f0"}, "suffix": "%"},
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    height=280,
                    margin=dict(l=20, r=20, t=60, b=10),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_detail:
                st.markdown('<div class="section-title">Probability Breakdown</div>',
                            unsafe_allow_html=True)
                prob_df = pd.DataFrame({
                    "Outcome":     ["Rejected", "Approved"],
                    "Probability": [reject_prob, approve_prob],
                })
                fig_prob = px.bar(
                    prob_df, x="Outcome", y="Probability",
                    color="Outcome",
                    color_discrete_map={"Approved": COLOR_APPROVED, "Rejected": COLOR_REJECTED},
                    text=prob_df["Probability"].map("{:.1%}".format),
                )
                fig_prob.update_layout(**DARK_LAYOUT,
                                       yaxis=dict(tickformat=".0%", **DARK_LAYOUT["yaxis"]),
                                       showlegend=False)
                fig_prob.update_traces(textposition="outside", marker_line_width=0)
                st.plotly_chart(fig_prob, use_container_width=True)

                # Summary table
                st.markdown('<div class="section-title">Input Summary</div>', unsafe_allow_html=True)
                summary = pd.DataFrame({
                    "Feature": [
                        "Dependents", "Education", "Self-Employed",
                        "Annual Income", "Loan Amount", "Loan Term",
                        "CIBIL Score", "Residential Assets",
                        "Commercial Assets", "Luxury Assets", "Bank Assets",
                    ],
                    "Value": [
                        no_of_dependents, education, self_employed,
                        f"₹{income_annum:,}", f"₹{loan_amount:,}",
                        f"{loan_term} yrs", cibil_score,
                        f"₹{residential_assets_value:,}",
                        f"₹{commercial_assets_value:,}",
                        f"₹{luxury_assets_value:,}",
                        f"₹{bank_asset_value:,}",
                    ],
                })
                st.dataframe(summary, use_container_width=True, hide_index=True)

        else:
            st.info("👈 Fill in the applicant details in the sidebar and press **Predict**.")

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("""
    <hr style="border-color:#1e3a4f; margin-top:3rem;">
    <p style="text-align:center; color:#475569; font-size:0.82rem;">
        Loan Approval Predictor · Built with Streamlit & scikit-learn ·
        Data: <em>loan_approval_dataset.csv</em>
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()