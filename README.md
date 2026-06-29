# Affordable-Quality Thrift Store — Dubai (Data Analytics Project)

A data-driven strategy for an **existing** thrift-store model, re-designed to give Dubai
students and young expats reliable **online + offline** access to **affordable,
quality-checked** second-hand apparel.

Dashboard focus (per brief): **~80% strategy & solution, ~20% the problem.**

## Files

| File | What it is |
|------|------------|
| `streamlit_app.py` | Interactive dashboard (descriptive, diagnostic, clustering, classification, strategy) |
| `thrift_store_survey_clean.csv` | Cleaned, transformed dataset the app reads |
| `Thrift_Store_Analysis.ipynb` | Colab/Jupyter notebook with the same analysis + written insights (report material) |
| `requirements.txt` | Dependencies |

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

## Deploy free on Streamlit Community Cloud

1. Create a **public GitHub repository** and commit these files together (the CSV
   **must** sit next to `streamlit_app.py` in the repo):
   - `streamlit_app.py`
   - `thrift_store_survey_clean.csv`
   - `requirements.txt`
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. **New app → From existing repo**, choose your repo and branch, and set the
   main file to `streamlit_app.py`.
4. **Deploy.** You get a public URL to share / put in your report.

## Dashboard sections

1. **The Problem (20%)** — price pain, current second-hand behaviour, baseline interest.
2. **What Drives Adoption** — correlation heatmap + driver chart; hygiene/quality worries are the biggest drags on interest.
3. **Customer Segments** — K-Means (k = 4): Eco-Conscious Enthusiasts, Budget-Driven Students, Quality-Cautious Skeptics, Affluent Brand Shoppers.
4. **Predicting Likely Customers** — Logistic Regression / Decision Tree / Random Forest / KNN compared (leakage-free features).
5. **Strategy & Recommendations (80%)** — the three pillars with concrete numbers: channel mix, price points from willingness-to-pay, quality/hygiene guarantees, targeting and discovery.

> The dataset is synthetic, engineered to mirror realistic Dubai student/expat
> attitudes. The app and notebook run unchanged on real survey data.
