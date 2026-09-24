
import io
import re
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "https://sai-publications-dashboard.vercel.app/api/publications"

st.set_page_config(
    page_title="Sai University | Research Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown("""
<style>
:root { --ink:#172033; --muted:#667085; --line:#E7EAF0; --panel:#FFFFFF; --accent:#4F46E5; }
[data-testid="stAppViewContainer"] { background:#F7F8FC; }
[data-testid="stSidebar"] { background:#111827; }
[data-testid="stSidebar"] * { color:#F9FAFB !important; }
.hero {
  padding: 24px 28px; border:1px solid #E7EAF0; border-radius:18px;
  background:linear-gradient(135deg,#ffffff 0%,#f2f4ff 100%);
  margin-bottom:18px;
}
.hero h1 { margin:0; color:#172033; font-size:34px; }
.hero p { color:#667085; margin:6px 0 0; font-size:15px; }
.kpi {
  background:#fff; border:1px solid #E7EAF0; border-radius:15px;
  padding:15px 17px; min-height:105px;
}
.kpi .label { color:#667085; font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
.kpi .value { color:#172033; font-size:27px; font-weight:750; margin-top:7px; }
.kpi .sub { color:#667085; font-size:12px; margin-top:3px; }
.section-title { color:#172033; font-size:20px; font-weight:700; margin:20px 0 8px; }
.small-note { color:#667085; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ---------- Data ----------
@st.cache_data(ttl=1800, show_spinner="Loading publications from the Publications API…")
def load_data():
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    payload = r.json()
    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Unexpected API response: expected a list under 'data'.")
    df = pd.json_normalize(records)
    return clean_data(df)

def clean_data(df):
    df = df.copy()
    # Normalize column names but preserve original fields in a predictable schema.
    for c in df.columns:
        df[c] = df[c].fillna("").astype(str).str.strip()

    def first_existing(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    year_col = first_existing("Year", "year")
    if year_col:
        df["Year_num"] = pd.to_numeric(df[year_col].str.extract(r"(\d{4})")[0], errors="coerce").astype("Int64")
    else:
        df["Year_num"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    for c in ["School","SaiU Authors","Authors","Document Type","Indexing Status",
              "SJR Quartile","Year Wise Quartile","Publisher","Source title","Title",
              "DOI Link","Article Link","Journal Link","Scopus URL","WoS URL",
              "SaiU SDG Indexing","Achieved SDG (WoS)","Achieved SDG (Scopus)"]:
        if c not in df.columns:
            df[c] = ""

    # A publication can carry multiple SDGs separated by |, commas, or semicolons.
    def split_multi(v):
        if not v:
            return []
        parts = re.split(r"[|,;]+", str(v))
        return sorted(set(p.strip() for p in parts if p.strip() and p.strip().lower() not in {"nan","none","-"}))

    df["SDG_list"] = df["SaiU SDG Indexing"].map(split_multi)
    # Fall back to indexed SDG fields if SaiU SDG Indexing is absent.
    for i in df.index:
        if not df.at[i, "SDG_list"]:
            vals = split_multi(df.at[i, "Achieved SDG (Scopus)"]) + split_multi(df.at[i, "Achieved SDG (WoS)"])
            df.at[i, "SDG_list"] = sorted(set(vals))

    def parse_authors(row):
        # Use SaiU Authors first because it identifies university-affiliated contributors.
        raw = row.get("SaiU Authors","") or ""
        if not raw:
            raw = row.get("Authors","") or ""
        parts = re.split(r"[;,|]+", raw)
        return [p.strip() for p in parts if p.strip()]

    df["Faculty_list"] = df.apply(parse_authors, axis=1)

    def indexing_flags(v):
        s = str(v).lower()
        return ("Scopus" if "scopus" in s else None), ("WoS" if ("wos" in s or "web of science" in s) else None)

    flags = df["Indexing Status"].map(indexing_flags)
    df["Scopus_flag"] = flags.map(lambda x: x[0] is not None)
    df["WoS_flag"] = flags.map(lambda x: x[1] is not None)
    df["Q1_flag"] = df["SJR Quartile"].str.upper().eq("Q1")
    df["Year_label"] = df["Year_num"].astype("string").fillna("Unknown")
    return df

def explode_dimension(df, list_col, value_name):
    x = df[["Title", list_col]].explode(list_col).rename(columns={list_col:value_name})
    return x[x[value_name].notna() & (x[value_name] != "")]

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load the Publications API: {e}")
    st.info("Check your internet connection and try again. The app is intentionally API-driven and does not hard-code publication records.")
    st.stop()

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>Research Intelligence Dashboard</h1>
  <p>Sai University · Publication output, research quality, indexing, faculty contribution and SDG alignment</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar filters ----------
st.sidebar.markdown("## Explore research")
years = sorted([int(y) for y in df["Year_num"].dropna().unique()], reverse=True)
schools = sorted(df["School"].replace("", "Unknown").unique())
authors = sorted(set(a for xs in df["Faculty_list"] for a in xs))
doc_types = sorted(df["Document Type"].replace("", "Unknown").unique())
indexing = sorted(df["Indexing Status"].replace("", "Unknown").unique())
quartiles = sorted(set(q for q in df["SJR Quartile"].replace("", "Not available").unique()))
publishers = sorted(df["Publisher"].replace("", "Unknown").unique())
sdgs = sorted(set(s for xs in df["SDG_list"] for s in xs), key=lambda x: int(x) if x.isdigit() else 999)

selected_years = st.sidebar.multiselect("Year", years, default=years)
selected_schools = st.sidebar.multiselect("School", schools)
selected_authors = st.sidebar.multiselect("Faculty / author", authors)
selected_types = st.sidebar.multiselect("Document type", doc_types)
selected_indexing = st.sidebar.multiselect("Indexing status", indexing)
selected_quartiles = st.sidebar.multiselect("SJR quartile", quartiles)
selected_publishers = st.sidebar.multiselect("Publisher", publishers)
selected_sdgs = st.sidebar.multiselect("SDG", sdgs)

mask = pd.Series(True, index=df.index)
if selected_years:
    mask &= df["Year_num"].isin(selected_years)
if selected_schools:
    mask &= df["School"].replace("", "Unknown").isin(selected_schools)
if selected_authors:
    mask &= df["Faculty_list"].map(lambda xs: any(a in selected_authors for a in xs))
if selected_types:
    mask &= df["Document Type"].replace("", "Unknown").isin(selected_types)
if selected_indexing:
    mask &= df["Indexing Status"].replace("", "Unknown").isin(selected_indexing)
if selected_quartiles:
    mask &= df["SJR Quartile"].replace("", "Not available").isin(selected_quartiles)
if selected_publishers:
    mask &= df["Publisher"].replace("", "Unknown").isin(selected_publishers)
if selected_sdgs:
    mask &= df["SDG_list"].map(lambda xs: any(s in selected_sdgs for s in xs))

f = df.loc[mask].copy()

# ---------- KPIs ----------
total = len(f)
scopus = int(f["Scopus_flag"].sum())
wos = int(f["WoS_flag"].sum())
q1 = int(f["Q1_flag"].sum())
school_count = f["School"].replace("", pd.NA).nunique(dropna=True)
faculty_count = len(set(a for xs in f["Faculty_list"] for a in xs))
sdg_count = len(set(s for xs in f["SDG_list"] for s in xs))
index_rate = (100*sum((f["Scopus_flag"] | f["WoS_flag"])) / total) if total else 0

kpis = [
    ("Total publications", total, "records in current selection"),
    ("Scopus indexed", scopus, f"{scopus/total*100:.1f}% of selection" if total else "—"),
    ("Web of Science", wos, f"{wos/total*100:.1f}% of selection" if total else "—"),
    ("Q1 publications", q1, f"{q1/total*100:.1f}% of selection" if total else "—"),
    ("Schools", school_count, "represented in selection"),
    ("Faculty authors", faculty_count, "SaiU-affiliated authors"),
    ("SDGs represented", sdg_count, "unique SDG numbers"),
    ("Indexed coverage", f"{index_rate:.1f}%", "Scopus or WoS"),
]
cols = st.columns(4)
for i, (label, value, sub) in enumerate(kpis):
    with cols[i % 4]:
        st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)
    if i % 4 == 3 and i != len(kpis)-1:
        st.write("")
st.caption(f"Showing {total:,} publications after filters · API records are cleaned dynamically on every refresh.")

# ---------- Charts ----------
sns.set_theme(style="whitegrid", font_scale=0.9)

def finish_fig(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="section-title">Publication trend</div>', unsafe_allow_html=True)
    trend = f.dropna(subset=["Year_num"]).groupby("Year_num").size().sort_index()
    fig, ax = plt.subplots(figsize=(7.0,3.6))
    if len(trend):
        sns.lineplot(x=trend.index.astype(int), y=trend.values, marker="o", ax=ax)
        ax.set_xlabel("Year"); ax.set_ylabel("Publications"); ax.set_title("")
    else:
        ax.text(.5,.5,"No year data for this selection",ha="center",va="center")
        ax.set_axis_off()
    finish_fig(fig)

with c2:
    st.markdown('<div class="section-title">Publications by school</div>', unsafe_allow_html=True)
    school_counts = f["School"].replace("", "Unknown").value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(7.0,3.6))
    if len(school_counts):
        sns.barplot(x=school_counts.values, y=school_counts.index, ax=ax)
        ax.set_xlabel("Publications"); ax.set_ylabel("")
    else:
        ax.text(.5,.5,"No school data for this selection",ha="center",va="center"); ax.set_axis_off()
    finish_fig(fig)

c3, c4 = st.columns(2)
with c3:
    st.markdown('<div class="section-title">Indexing status</div>', unsafe_allow_html=True)
    idx = f["Indexing Status"].replace("", "Unknown").value_counts()
    fig, ax = plt.subplots(figsize=(7.0,3.6))
    if len(idx):
        ax.pie(idx.values, labels=idx.index, autopct="%1.0f%%", startangle=90)
        ax.set_aspect("equal")
    finish_fig(fig)

with c4:
    st.markdown('<div class="section-title">SJR quartile distribution</div>', unsafe_allow_html=True)
    q = f["SJR Quartile"].replace("", "Not available").value_counts()
    order = [x for x in ["Q1","Q2","Q3","Q4","Not available"] if x in q.index] + [x for x in q.index if x not in {"Q1","Q2","Q3","Q4","Not available"}]
    q = q.reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(7.0,3.6))
    sns.barplot(x=q.index, y=q.values, ax=ax)
    ax.set_xlabel("SJR quartile"); ax.set_ylabel("Publications")
    finish_fig(fig)

c5, c6 = st.columns(2)
with c5:
    st.markdown('<div class="section-title">Document type</div>', unsafe_allow_html=True)
    dt = f["Document Type"].replace("", "Unknown").value_counts()
    fig, ax = plt.subplots(figsize=(7.0,3.6))
    sns.barplot(x=dt.values, y=dt.index, ax=ax)
    ax.set_xlabel("Publications"); ax.set_ylabel("")
    finish_fig(fig)

with c6:
    st.markdown('<div class="section-title">SDG distribution</div>', unsafe_allow_html=True)
    sx = explode_dimension(f, "SDG_list", "SDG")
    sdg_counts = sx["SDG"].value_counts().sort_index(key=lambda s: pd.to_numeric(s, errors="coerce"))
    fig, ax = plt.subplots(figsize=(7.0,3.6))
    if len(sdg_counts):
        sns.barplot(x=sdg_counts.values, y=[f"SDG {x}" for x in sdg_counts.index], ax=ax)
        ax.set_xlabel("Publication records tagged"); ax.set_ylabel("")
    else:
        ax.text(.5,.5,"No SDG data for this selection",ha="center",va="center"); ax.set_axis_off()
    finish_fig(fig)

# ---------- Faculty contribution ----------
st.markdown('<div class="section-title">Faculty contribution</div>', unsafe_allow_html=True)
fx = explode_dimension(f, "Faculty_list", "Faculty")
faculty_counts = fx["Faculty"].value_counts().head(15).sort_values()
fig, ax = plt.subplots(figsize=(11,5))
if len(faculty_counts):
    sns.barplot(x=faculty_counts.values, y=faculty_counts.index, ax=ax)
    ax.set_xlabel("Publications"); ax.set_ylabel("")
else:
    ax.text(.5,.5,"No faculty-author data for this selection",ha="center",va="center"); ax.set_axis_off()
finish_fig(fig)

# ---------- Auto insights ----------
st.markdown('<div class="section-title">Automatically generated insights</div>', unsafe_allow_html=True)
insights = []
if total:
    if len(trend) >= 2:
        first, last = trend.iloc[0], trend.iloc[-1]
        if first:
            pct = (last-first)/first*100
            insights.append(f"Output changed by {pct:+.1f}% between {int(trend.index[0])} and {int(trend.index[-1])} ({int(first)} → {int(last)} publications).")
    if len(school_counts):
        insights.append(f"The largest publication volume in the current selection comes from {school_counts.index[-1]} ({int(school_counts.iloc[-1])} records).")
    insights.append(f"{index_rate:.1f}% of selected publications are marked as Scopus and/or Web of Science indexed.")
    if q1:
        insights.append(f"{q1} selected publications are marked Q1 by the API's SJR Quartile field.")
    if sdg_count:
        top_sdg = sdg_counts.index[0] if len(sdg_counts) else None
        if top_sdg:
            insights.append(f"SDG {top_sdg} is the most frequently represented SDG in the selected records.")
for item in insights:
    st.write("• " + item)

# ---------- Explorer ----------
st.markdown('<div class="section-title">Publication explorer</div>', unsafe_allow_html=True)
search = st.text_input("Search title, author, school, source, publisher or DOI", placeholder="e.g. climate, Abraham, IEEE, 10.1000/…")
sort_by = st.selectbox("Sort by", ["Year (newest)","Year (oldest)","Title","School","SJR Quartile"])
show_n = st.slider("Rows to display", 10, min(200, max(10, len(f))), min(50, max(10, len(f))))

explore = f.copy()
if search:
    hay = (
        explore["Title"] + " " + explore["Authors"] + " " + explore["SaiU Authors"] + " " +
        explore["School"] + " " + explore["Source title"] + " " + explore["Publisher"] + " " +
        explore["DOI"]
    ).str.lower()
    explore = explore[hay.str.contains(search.lower(), regex=False, na=False)]

sort_map = {
    "Year (newest)": ("Year_num", False),
    "Year (oldest)": ("Year_num", True),
    "Title": ("Title", True),
    "School": ("School", True),
    "SJR Quartile": ("SJR Quartile", True),
}
col, asc = sort_map[sort_by]
explore = explore.sort_values(col, ascending=asc, na_position="last")

display_cols = ["Year_num","Title","SaiU Authors","School","Document Type","Indexing Status","SJR Quartile","Publisher","Source title","DOI Link","Article Link"]
display = explore[display_cols].head(show_n).copy().rename(columns={
    "Year_num":"Year","SaiU Authors":"SaiU authors","Source title":"Source"
})
st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "DOI Link": st.column_config.LinkColumn("DOI", display_text="Open DOI"),
        "Article Link": st.column_config.LinkColumn("Article", display_text="Open article"),
    },
)

# Export current filtered records
csv_bytes = explore.drop(columns=["SDG_list","Faculty_list"], errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Export current selection (CSV)",
    data=csv_bytes,
    file_name="sai_research_filtered.csv",
    mime="text/csv",
)

# ---------- Methodology ----------
with st.expander("Data & methodology"):
    st.write("""
    **Data source:** Sai University Publications API.

    **Cleaning:** missing values are treated as unavailable rather than zero; years are parsed from the Year field; SDGs are split from pipe/comma/semicolon-delimited values; faculty contribution uses SaiU Authors when available and falls back to Authors.

    **Indexing:** Scopus and WoS KPIs are derived from the API's Indexing Status field. A record containing both is counted in both KPI totals.

    **Q1:** derived strictly from the API's SJR Quartile field equal to Q1. Records with blank quartile are not inferred.

    **SDGs:** the dashboard prioritizes SaiU SDG Indexing and falls back to Achieved SDG (Scopus/WoS) when needed.

    All KPIs and charts are recalculated after filters are applied. The app fetches the API dynamically and does not hard-code publication records.
    """)
st.caption("Source: Sai University Publications API · Dashboard built with Python, Pandas, Streamlit, Matplotlib and Seaborn.")
