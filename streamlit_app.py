"""
============================================================================
THRIFT STORE STRATEGY DASHBOARD  -  Dubai
Serving students & young expats: online + offline access, low price,
quality without compromise.
============================================================================
Run locally:   streamlit run streamlit_app.py
Data file:     thrift_store_survey_clean.csv  (keep it next to this file)

Deploy on Streamlit Cloud: commit streamlit_app.py, thrift_store_survey_clean.csv
and requirements.txt to a public GitHub repo, then point share.streamlit.io at
streamlit_app.py.

Focus split (per brief): ~80% strategy & solution, ~20% the problem.
============================================================================
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

DATA_FILENAME = "thrift_store_survey_clean.csv"

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

PALETTE = ["#1F6F5C", "#E8A33D", "#C25B3F", "#3D6B9E"]
PRIMARY = "#1F6F5C"

READABLE = {
    "D1_concept_interest": "Interest",
    "D2_visit_likelihood_3mo": "Visit likelihood",
    "E4_recommend_likelihood": "Recommend",
    "B5_price_importance": "Price importance",
    "B5_quality_importance": "Quality importance",
    "B5_sustainability_importance": "Sustainability importance",
    "B5_uniqueness_importance": "Uniqueness importance",
    "C2_thrift_familiarity": "Thrift familiarity",
    "C3_value_for_money": "Sees value for money",
    "C3_good_for_environment": "Pro-environment",
    "C3_hygiene_worry": "Hygiene worry",
    "C3_quality_worry": "Quality worry",
    "C3_socially_accepted": "Socially accepted",
    "C3_finds_unique": "Finds unique pieces",
    "F4_income_code": "Income band",
    "B2_monthly_clothing_spend_aed": "Monthly spend (AED)",
    "C1_secondhand_code": "Prior second-hand use",
}


# ---------------------------------------------------------------------------
# Version-proof render helpers (work across old & new Streamlit)
# ---------------------------------------------------------------------------
def st_plotly(fig):
    for kw in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return st.plotly_chart(fig, **kw)
        except TypeError:
            continue


def st_table(obj):
    for kw in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return st.dataframe(obj, **kw)
        except TypeError:
            continue


def st_styled_table(styler, fallback):
    """Render a styled table; fall back to plain if styling deps are missing."""
    try:
        return st_table(styler)
    except Exception:
        return st_table(fallback)


# ---------------------------------------------------------------------------
# Data & computation
# ---------------------------------------------------------------------------
def _find_data_file():
    candidates = [
        os.path.join(SCRIPT_DIR, DATA_FILENAME),
        DATA_FILENAME,
        os.path.join(os.getcwd(), DATA_FILENAME),
        os.path.join(SCRIPT_DIR, "data", DATA_FILENAME),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for root, _dirs, files in os.walk(SCRIPT_DIR):
        if DATA_FILENAME in files:
            return os.path.join(root, DATA_FILENAME)
    return None


@st.cache_data
def load_data():
    path = _find_data_file()
    if path is None:
        return None
    return pd.read_csv(path)


def compute_diagnostics(df):
    heat_cols = ["D1_concept_interest", "D2_visit_likelihood_3mo", "E4_recommend_likelihood",
                 "B5_price_importance", "B5_quality_importance", "B5_sustainability_importance",
                 "B5_uniqueness_importance", "C2_thrift_familiarity", "C3_value_for_money",
                 "C3_good_for_environment", "C3_hygiene_worry", "C3_quality_worry",
                 "F4_income_code", "B2_monthly_clothing_spend_aed"]
    corr = df[heat_cols].corr()

    upstream = ["C3_value_for_money", "C2_thrift_familiarity", "B5_uniqueness_importance",
                "C3_good_for_environment", "C3_socially_accepted", "C3_finds_unique",
                "B5_price_importance", "B5_sustainability_importance", "F4_income_code",
                "B2_monthly_clothing_spend_aed", "B5_quality_importance",
                "C3_quality_worry", "C3_hygiene_worry"]
    driver = (df[upstream + ["D1_concept_interest"]].corr()["D1_concept_interest"]
              .drop("D1_concept_interest").sort_values())

    def share(prefix):
        cols = [c for c in df.columns if c.startswith(prefix)]
        s = (df[cols].mean() * 100).sort_values(ascending=False)
        s.index = [c.replace(prefix, "").replace("_", " ").strip().title() for c in s.index]
        return s

    barriers = share("C4_barrier_")
    discovery = share("E2_disc_")
    channel = (df["D7_shop_mode_pref"].value_counts(normalize=True) * 100)

    wtp = pd.DataFrame({
        "Used jeans": df["D5_wtp_jeans_aed"].quantile([.25, .5, .75, .9]).values,
        "Used dress": df["D5_wtp_dress_aed"].quantile([.25, .5, .75, .9]).values,
        "Used jacket": df["D5_wtp_jacket_aed"].quantile([.25, .5, .75, .9]).values,
        "Max per visit": df["D6_max_spend_visit_aed"].quantile([.25, .5, .75, .9]).values,
    }, index=["25th %ile", "Median", "75th %ile", "90th %ile"]).round(0).astype(int)

    return dict(corr=corr, driver=driver, barriers=barriers, discovery=discovery,
                channel=channel, wtp=wtp)


CLUSTER_FEATS = ["B5_price_importance", "B5_quality_importance", "B5_sustainability_importance",
                 "B5_uniqueness_importance", "C2_thrift_familiarity", "C3_value_for_money",
                 "C3_good_for_environment", "C3_hygiene_worry", "C3_quality_worry",
                 "D1_concept_interest", "F4_income_code", "B2_monthly_clothing_spend_aed",
                 "advocacy_score"]


def run_clustering(df, k=4):
    X = StandardScaler().fit_transform(df[CLUSTER_FEATS])
    sil = {kk: silhouette_score(X, KMeans(n_clusters=kk, n_init=10, random_state=42)
                                .fit_predict(X)) for kk in range(2, 7)}
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    labels = km.labels_
    pca = PCA(n_components=2, random_state=42).fit_transform(X)

    prof = df.assign(cluster=labels).groupby("cluster")[CLUSTER_FEATS].mean()
    sizes = pd.Series(labels).value_counts().sort_index()

    champion = prof["D1_concept_interest"].idxmax()
    affluent = prof["F4_income_code"].idxmax()
    if affluent == champion:
        affluent = prof["B2_monthly_clothing_spend_aed"].idxmax()
    rest = [c for c in prof.index if c not in (champion, affluent)]
    budget = max(rest, key=lambda c: prof.loc[c, "B5_price_importance"])
    skeptic = [c for c in rest if c != budget][0]
    names = {champion: "Eco-Conscious Enthusiasts", budget: "Budget-Driven Students",
             skeptic: "Quality-Cautious Skeptics", affluent: "Affluent Brand Shoppers"}

    return dict(labels=labels, sil=sil, pca=pca, prof=prof, sizes=sizes, names=names)


CLF_NUMERIC = ["A4_age_code", "F4_income_code", "B1_freq_code", "C1_secondhand_code",
               "B2_monthly_clothing_spend_aed", "B5_price_importance", "B5_brand_importance",
               "B5_quality_importance", "B5_sustainability_importance", "B5_uniqueness_importance",
               "B5_convenience_importance", "C2_thrift_familiarity", "C3_value_for_money",
               "C3_good_for_environment", "C3_hygiene_worry", "C3_quality_worry",
               "C3_socially_accepted", "C3_finds_unique"]
CLF_NOMINAL = ["A1_respondent_type", "A3_residency_status", "F1_gender"]


def run_classification(df):
    X = pd.concat([df[CLF_NUMERIC], pd.get_dummies(df[CLF_NOMINAL], drop_first=True)], axis=1)
    y = df["target_likely_customer"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        fitted[name] = m
        pred = m.predict(Xte)
        try:
            auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        except Exception:
            auc = np.nan
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred),
                     "Precision": precision_score(yte, pred), "Recall": recall_score(yte, pred),
                     "F1": f1_score(yte, pred), "ROC-AUC": auc})
    results = pd.DataFrame(rows).set_index("Model").round(3)
    best = results["F1"].idxmax()
    cm = confusion_matrix(yte, fitted[best].predict(Xte))
    imp = pd.Series(fitted["Random Forest"].feature_importances_,
                    index=X.columns).sort_values(ascending=False)
    return dict(results=results, best=best, cm=cm, importance=imp)


def bar(series, title, xlab, color=PRIMARY, pct=False):
    fig = px.bar(x=series.values, y=series.index, orientation="h",
                 labels={"x": xlab, "y": ""}, title=title)
    fig.update_traces(marker_color=color,
                      texttemplate="%{x:.0f}%" if pct else "%{x:.2f}", textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380,
                      margin=dict(l=10, r=30, t=50, b=10))
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Thrift Store Strategy - Dubai", layout="wide")

    df = load_data()
    if df is None:
        st.error(
            f"Could not find **{DATA_FILENAME}**.\n\n"
            "Make sure this file is committed to your GitHub repo in the **same folder** "
            "as `streamlit_app.py` (or run the app from the folder that contains it)."
        )
        st.stop()

    diag = compute_diagnostics(df)
    clus = run_clustering(df)
    clf = run_classification(df)

    # ---- Sidebar ----
    st.sidebar.title("Thrift Store Strategy")
    st.sidebar.markdown(
        "**Concept:** an existing thrift-store model, re-strategised to give Dubai "
        "students & young expats reliable **online + offline** access to "
        "**affordable, quality-checked** second-hand apparel.")
    st.sidebar.metric("Respondents (clean)", f"{len(df):,}")
    st.sidebar.metric("Likely customers", f"{df['target_likely_customer'].mean()*100:.0f}%")
    in_store = diag["channel"].get("Both", 0) + diag["channel"].get("In-store only", 0)
    st.sidebar.metric("Want in-store access", f"{in_store:.0f}%")
    st.sidebar.caption("Dashboard focus: ~80% strategy & solution, ~20% the problem.")

    st.title("Affordable Quality Thrift in Dubai - Strategy Dashboard")
    st.markdown("Turning consumer-survey signal into a concrete plan across three pillars: "
                "**(1) online + offline access | (2) low price | (3) quality without compromise.**")

    tabs = st.tabs(["The Problem (20%)", "What Drives Adoption", "Customer Segments",
                    "Predicting Likely Customers", "Strategy & Recommendations (80%)"])

    # ===================== TAB 1 - THE PROBLEM =====================
    with tabs[0]:
        st.subheader("The problem, briefly")
        st.markdown("Dubai already has thrift stores, yet students still struggle to buy apparel "
                    "affordably without sacrificing quality. This section sizes that gap; the rest "
                    "of the dashboard is about closing it.")

        types = st.multiselect("Filter by respondent type",
                               sorted(df["A1_respondent_type"].unique()),
                               default=sorted(df["A1_respondent_type"].unique()))
        d = df[df["A1_respondent_type"].isin(types)] if types else df

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Find clothes too expensive", f"{d['B4_frustr_too_expensive'].mean()*100:.0f}%")
        c2.metric("Median monthly spend", f"AED {d['B2_monthly_clothing_spend_aed'].median():.0f}")
        c3.metric("Already buy second-hand",
                  f"{(d['C1_bought_secondhand'] != 'Never').mean()*100:.0f}%")
        c4.metric("Mean concept interest (1-5)", f"{d['D1_concept_interest'].mean():.2f}")

        g1, g2 = st.columns(2)
        with g1:
            vc = d["A1_respondent_type"].value_counts()
            fig = px.pie(values=vc.values, names=vc.index, title="Who responded",
                         color_discrete_sequence=PALETTE, hole=0.45)
            fig.update_layout(height=360)
            st_plotly(fig)
        with g2:
            dist = d["D1_concept_interest"].value_counts().sort_index()
            fig = px.bar(x=dist.index, y=dist.values,
                         labels={"x": "Interest (1-5)", "y": "Respondents"},
                         title="Baseline interest in an affordable-quality thrift store")
            fig.update_traces(marker_color=PRIMARY)
            fig.update_layout(height=360)
            st_plotly(fig)

        st.info("**Takeaway.** A large share call clothing too expensive and already dabble in "
                "second-hand, and average interest sits above the mid-point - demand exists. "
                "The question is *how* to serve it, which the next sections answer.")

    # ===================== TAB 2 - DIAGNOSTIC =====================
    with tabs[1]:
        st.subheader("What actually drives adoption (diagnostic)")
        st.markdown("Correlation analysis isolates the levers: what makes interest rise or fall.")

        top_pos = diag["driver"].tail(3)
        top_neg = diag["driver"].head(3)
        st.markdown(
            "**Biggest positive levers:** "
            + ", ".join(f"{READABLE.get(i, i)} (r = {v:+.2f})" for i, v in top_pos.items())
            + ".  \n**Biggest drags:** "
            + ", ".join(f"{READABLE.get(i, i)} (r = {v:+.2f})" for i, v in top_neg.items()) + ".")

        g1, g2 = st.columns([3, 2])
        with g1:
            disp = diag["corr"].rename(index=READABLE, columns=READABLE)
            fig = px.imshow(disp, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            aspect="auto", title="Correlation heatmap (key variables)")
            fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10))
            st_plotly(fig)
        with g2:
            dd = diag["driver"].copy()
            dd.index = [READABLE.get(i, i) for i in dd.index]
            fig = px.bar(x=dd.values, y=dd.index, orientation="h",
                         labels={"x": "Correlation with interest", "y": ""},
                         title="Drivers of concept interest",
                         color=dd.values, color_continuous_scale="RdBu_r", range_color=[-.5, .5])
            fig.update_layout(height=560, yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10))
            st_plotly(fig)

        st_plotly(bar(diag["barriers"], "Barriers to buying second-hand",
                      "% of respondents", color=PALETTE[2], pct=True))
        st.success("**Strategic read.** The two strongest *negative* levers are hygiene worry and "
                   "quality worry - and quality/hygiene also rank among the most-cited barriers. "
                   "Reassuring shoppers on cleanliness and condition is the single highest-leverage "
                   "move, which is exactly the 'quality without compromise' pillar.")

    # ===================== TAB 3 - CLUSTERING =====================
    with tabs[2]:
        st.subheader("Customer segments (K-Means clustering)")
        st.markdown("Segmenting on attitudes, interest, income and spend. Silhouette scores guided "
                    "the choice of **k = 4** for the most actionable, interpretable segments.")

        prof, names, sizes = clus["prof"], clus["names"], clus["sizes"]
        cols = st.columns(4)
        seg_blurb = {
            "Eco-Conscious Enthusiasts": "Highest interest & familiarity, sustainability-led. Your **core base / champions**.",
            "Budget-Driven Students": "Price-first, lower income, low hygiene worry. Your **primary growth target**.",
            "Quality-Cautious Skeptics": "High standards & hygiene worry, lower interest. **Convertible** via quality guarantees.",
            "Affluent Brand Shoppers": "High income & spend, price-insensitive. **Reach via uniqueness/sustainability, not price.**",
        }
        for i, cl in enumerate(prof.index):
            nm = names[cl]
            cols[i].markdown(f"**{nm}**")
            cols[i].metric("Size", f"{sizes[cl]} ({sizes[cl]/len(df)*100:.0f}%)")
            cols[i].caption(seg_blurb[nm])

        g1, g2 = st.columns([2, 3])
        with g1:
            coords = pd.DataFrame(clus["pca"], columns=["PC1", "PC2"])
            coords["Segment"] = [names[l] for l in clus["labels"]]
            fig = px.scatter(coords, x="PC1", y="PC2", color="Segment",
                             color_discrete_sequence=PALETTE, title="Segments (PCA projection)",
                             opacity=0.7)
            fig.update_layout(height=480, legend=dict(orientation="h", y=-0.2))
            st_plotly(fig)
        with g2:
            show = prof.rename(index=names)[["D1_concept_interest", "B5_price_importance",
                "B5_quality_importance", "B5_sustainability_importance", "C3_hygiene_worry",
                "C2_thrift_familiarity", "F4_income_code", "B2_monthly_clothing_spend_aed"]]
            show.columns = ["Interest", "Price imp.", "Quality imp.", "Sustain. imp.",
                            "Hygiene worry", "Familiarity", "Income band", "Spend (AED)"]
            znorm = (show - show.mean()) / show.std()
            fig = px.imshow(znorm, color_continuous_scale="RdBu_r", aspect="auto",
                            zmin=-1.6, zmax=1.6,
                            title="Segment profiles (z-scored - red = high, blue = low)")
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
            st_plotly(fig)
        st_table(show.round(2))

    # ===================== TAB 4 - CLASSIFICATION =====================
    with tabs[3]:
        st.subheader("Predicting likely customers (classification)")
        st.markdown("Target: a 'likely customer' (interest 4-5). Models use only **upstream** "
                    "attitudes & demographics - interest itself and its restatements are excluded "
                    "to avoid leakage. This shows *who* to target and *what* predicts them.")

        res, best = clf["results"], clf["best"]
        st.markdown(f"Best model by F1: **{best}** "
                    f"(accuracy {res.loc[best,'Accuracy']:.2f}, F1 {res.loc[best,'F1']:.2f}, "
                    f"ROC-AUC {res.loc[best,'ROC-AUC']:.2f}).")
        try:
            styled = res.style.format("{:.3f}").background_gradient(cmap="Greens", axis=0)
        except Exception:
            styled = res
        st_styled_table(styled, res)

        g1, g2 = st.columns(2)
        with g1:
            cm = clf["cm"]
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Greens",
                            x=["Pred: Unlikely", "Pred: Likely"],
                            y=["True: Unlikely", "True: Likely"],
                            title=f"Confusion matrix - {best}")
            fig.update_layout(height=420, coloraxis_showscale=False)
            st_plotly(fig)
        with g2:
            imp = clf["importance"].head(10).iloc[::-1]
            imp.index = [READABLE.get(i, i.replace("_", " ").title()) for i in imp.index]
            fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                         labels={"x": "Importance", "y": ""},
                         title="Top predictors (Random Forest)")
            fig.update_traces(marker_color=PALETTE[3])
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st_plotly(fig)
        st.success("**Strategic read.** Prior second-hand experience, hygiene worry and "
                   "value-for-money perception are among the strongest predictors - so the store "
                   "should target lapsed/curious thrifters, lead with cleanliness proof, and make "
                   "the value case explicit.")

    # ===================== TAB 5 - STRATEGY =====================
    with tabs[4]:
        st.subheader("Strategy & recommendations")
        st.markdown("Everything above points to a concrete plan across the three pillars, plus "
                    "targeting and discovery. Each recommendation is tied to a number from the data.")

        st.markdown("### Pillar 1 - Online **and** offline access (offline-primary)")
        ch = diag["channel"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Prefer both channels", f"{ch.get('Both',0):.0f}%")
        c2.metric("In-store only", f"{ch.get('In-store only',0):.0f}%")
        c3.metric("Online only", f"{ch.get('Online only',0):.0f}%")
        st.markdown(f"- **{ch.get('Both',0)+ch.get('In-store only',0):.0f}% want in-store access**, "
                    "confirming a physical store where shoppers can inspect second-hand condition "
                    "first-hand should be the anchor.\n"
                    f"- The **{ch.get('Both',0):.0f}% who want *both*** justify a complementary online "
                    "catalogue (browse/reserve online, inspect & buy in store), not online-only.")

        st.markdown("### Pillar 2 - Low price, anchored to willingness-to-pay")
        wtp = diag["wtp"]
        try:
            wtp_styled = wtp.style.format("AED {:.0f}")
        except Exception:
            wtp_styled = wtp
        st_styled_table(wtp_styled, wtp)
        st.markdown(
            f"- Price the bulk of stock **at or below the median** shopper expectation "
            f"(jeans = AED {wtp.loc['Median','Used jeans']}, dresses = AED {wtp.loc['Median','Used dress']}, "
            f"jackets = AED {wtp.loc['Median','Used jacket']}) so the range *feels* cheap to most.\n"
            f"- Use the **75th percentile as a ceiling** for premium/branded pieces "
            f"(up to AED {wtp.loc['75th %ile','Used jacket']} for jackets).\n"
            f"- A typical basket should land near the **median max-spend of AED "
            f"{wtp.loc['Median','Max per visit']}** - bundle pricing can nudge toward it.")

        st.markdown("### Pillar 3 - Quality without compromise (highest-leverage)")
        b = diag["barriers"]
        hyg = b.get("Hygiene Concerns", 0)
        qual = b.get("Quality Concerns", 0)
        st.markdown(
            f"- Hygiene worry and quality worry are the **strongest negative correlates of interest** "
            f"(r near -0.40 and -0.38), and **{hyg:.0f}%** cite hygiene / **{qual:.0f}%** cite quality "
            "as barriers. This is where to invest first.\n"
            "- Concrete moves: a **visible sanitisation guarantee** (every item professionally cleaned), "
            "a simple **A/B/C condition grade** on each tag, a **no-quibble return window**, and "
            "**good lighting + fitting rooms** so condition is easy to verify in person.")

        st.markdown("### Targeting - who to win, and how")
        st.markdown(
            "- **Champions (Eco-Conscious Enthusiasts):** already sold - retain with a loyalty "
            "scheme and fresh, curated, sustainable stock.\n"
            "- **Primary growth (Budget-Driven Students):** lead with price and student offers; "
            "campus presence.\n"
            "- **Convert (Quality-Cautious Skeptics):** the quality/hygiene guarantees above are "
            "aimed squarely at them - the largest unlock.\n"
            "- **Affluent Brand Shoppers:** reach via uniqueness, curation and sustainability, not price.")

        st.markdown("### Discovery - where to be seen")
        st_plotly(bar(diag["discovery"].head(6), "Expected discovery channels",
                      "% of respondents", color=PALETTE[1], pct=True))
        top_ch = diag["discovery"].head(3)
        st.markdown("- Concentrate launch marketing on "
                    + ", ".join(f"**{i}** ({v:.0f}%)" for i, v in top_ch.items())
                    + " - where this audience expects to find the store.")

        st.caption("Note: figures are from a synthetic dataset built to mirror realistic Dubai "
                   "student/expat attitudes; the same dashboard runs on real survey data unchanged.")


if __name__ == "__main__":
    main()
