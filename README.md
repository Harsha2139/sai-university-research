# Sai University Research Intelligence Dashboard

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard retrieves publication records dynamically from:

https://sai-publications-dashboard.vercel.app/api/publications

## Included features
- Dynamic API ingestion and cleaning
- KPI cards that respond to all filters
- Publication trend
- School contribution
- Indexing status
- SJR quartile distribution
- Document type distribution
- Faculty contribution
- SDG distribution
- Searchable/sortable publication explorer
- DOI/article links
- CSV export
- Automatic insights
- Responsive Streamlit layout
- Methodology / data-quality notes

## Data interpretation
- Scopus/WoS counts come from `Indexing Status`.
- Q1 counts use `SJR Quartile == Q1`; blanks are not inferred.
- Faculty counts use `SaiU Authors`, falling back to `Authors`.
- SDGs use `SaiU SDG Indexing`, falling back to Scopus/WoS SDG fields.
- Missing values remain missing/“Not available”, rather than being guessed.

## Deployment
This project can be deployed on Streamlit Community Cloud by connecting the repository and setting the main file to `app.py`.
