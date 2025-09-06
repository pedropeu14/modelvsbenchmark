
import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import re

st.set_page_config(page_title="Portfolio vs Benchmark — MTD & YTD", layout="wide")
st.title("Portfolio vs Benchmark — MTD & YTD")
st.caption("Build: v2.1 — labels com folga + classes 'deitadas' (sem recálculo)")

# =================== Column mapping (EXPÍCITO) ===================
COL_MTD_BMK = "MTD Benchmark"
COL_MTD_M4  = "MTD M4"
COL_YTD_BMK = "YTD Benchmark"
COL_YTD_M4  = "YTD M4"

# =================== Taxonomia ===================
MACRO_CLASSES = [
    "01. Cash",
    "02. Fixed Income",
    "03. Equities",
    "04. Hedge Funds",
    "05. Commodities",
    "06. Real Estate",
    "07. Cryptocurrencies",
    "08. Asset Allocation",
]

SUBCLASSES_MAP = {
    "02. Fixed Income": [
        "02.Fixed Income 2.2 High Grade",
        "02.Fixed Income 2.3 High Yield",
        "02.Fixed Income 2.5 CoCos",
        "02.Fixed Income 2.4 Emerging Markets Debt",
        "02.Fixed Income 2.4.1 EM Brazil",
        "02.Fixed Income 2.7 Mixed Debt",
        "02.Fixed Income 2.B Distressed Debt",
        "02.Fixed Income 2.1 US Government",
    ],
    "03. Equities": [
        "03.Equities 3.1 US Equity",
        "03.Equities 3.2 US Growth (n)",
        "03.Equities 3.3 European Equty",
        "03.Equities 3.4 Emerging Markets Equity",
        "03.Equities 3.5 World Equity",
    ]
}

# =================== Helpers ===================
@st.cache_data
def load_data(path: str):
    df = pd.read_excel(path, sheet_name=0)
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    required = ["Data","Asset Class",COL_MTD_BMK,COL_MTD_M4,COL_YTD_BMK,COL_YTD_M4]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Colunas ausentes no arquivo: {missing}")
        st.stop()

    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df = df.dropna(subset=["Data"])

    # class Level 1
    ac = df["Asset Class"].astype(str)
    class_l1_code = ac.str.extract(r"^(\d+)\.")[0].fillna("")
    class_l1_name = ac.str.replace(r"^(\d+)\.\s*", "", regex=True) \
                      .str.split(r"\s+\d+\.", n=1, regex=True).str[0].str.strip()
    df["class_l1"] = (class_l1_code + ". " + class_l1_name).str.strip()

    # macro row detection
    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", str(s).lower())
    df["norm_ac"] = df["Asset Class"].apply(normalize)
    df["norm_l1"] = df["class_l1"].apply(normalize)
    df["is_pure_macro"] = df["norm_ac"] == df["norm_l1"]

    # numeric
    for c in ["Weight", COL_MTD_BMK, COL_MTD_M4, COL_YTD_BMK, COL_YTD_M4]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # month fields
    df["period"]    = df["Data"].dt.to_period("M")
    df["month_lbl"] = df["Data"].dt.strftime("%b-%Y")
    df["month_idx"] = df["Data"].dt.year * 12 + df["Data"].dt.month
    return df

def style_pct_df(df, cols):
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = (df2[c] * 100).map("{:.2f}%".format)
    return df2

def month_selectbox(df, label, key):
    months = sorted(df["period"].unique())
    if not months:
        return None
    default_idx = len(months) - 1
    options = [pd.Period(m, freq="M") for m in months]
    labels = [m.strftime("%b-%Y") for m in options]
    sel_lbl = st.selectbox(label, labels, index=default_idx, key=key)
    sel_period = options[labels.index(sel_lbl)]
    # pick the last available date in that month
    sel_date = df.loc[df["period"] == sel_period, "Data"].max()
    return pd.to_datetime(sel_date)

# Aggregate for macros with macro-row preference; else sum subclasses
def agg_macros_lastday(df_day, macros):
    res = []
    for mac in macros:
        rows_macro = df_day[(df_day["class_l1"].str.lower() == mac.lower()) & (df_day["is_pure_macro"])]
        if not rows_macro.empty:
            s = rows_macro[[COL_MTD_BMK, COL_MTD_M4, COL_YTD_BMK, COL_YTD_M4]].sum()
            res.append({"Label": mac, **s.to_dict()})
        else:
            rows_subs = df_day[(df_day["class_l1"].str.lower() == mac.lower()) & (~df_day["is_pure_macro"])]
            if not rows_subs.empty:
                s = rows_subs[[COL_MTD_BMK, COL_MTD_M4, COL_YTD_BMK, COL_YTD_M4]].sum()
                res.append({"Label": mac, **s.to_dict()})
    return pd.DataFrame(res)

# Aggregate subclasses on last day (straight sum)
def agg_subs_lastday(df_day, subs):
    if not subs:
        return pd.DataFrame(columns=["Label", COL_MTD_BMK, COL_MTD_M4, COL_YTD_BMK, COL_YTD_M4])
    rows = df_day[df_day["Asset Class"].isin(subs)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Label", COL_MTD_BMK, COL_MTD_M4, COL_YTD_BMK, COL_YTD_M4])
    agg = (rows.groupby("Asset Class", dropna=False)[[COL_MTD_BMK, COL_MTD_M4, COL_YTD_BMK, COL_YTD_M4]]
                .sum().reset_index().rename(columns={"Asset Class":"Label"}))
    return agg

# =================== Load ===================
DEFAULT_PATH = "planilha base.xlsx"
st.sidebar.header("Fonte de dados")
mode = st.sidebar.radio("Carregar via:", ["Arquivo padrão do app", "Upload"], index=0)
if mode == "Upload":
    up = st.sidebar.file_uploader("Envie seu Excel (.xlsx)")
    if up:
        df = load_data(up)
    else:
        st.stop()
else:
    if not os.path.exists(DEFAULT_PATH):
        st.warning("Arquivo padrão 'planilha base.xlsx' não encontrado. Faça upload.")
        up = st.sidebar.file_uploader("Envie seu Excel (.xlsx)")
        if up:
            df = load_data(up)
        else:
            st.stop()
    else:
        df = load_data(DEFAULT_PATH)

# =================== Date range (mmm-yyyy) ===================
min_p = df["period"].min()
max_p = df["period"].max()
period_range = list(pd.period_range(min_p, max_p, freq="M"))
st.sidebar.write("**Período**")
start_lbl = st.sidebar.selectbox("Início (mmm-yyyy)", [p.strftime("%b-%Y") for p in period_range], index=0)
end_lbl   = st.sidebar.selectbox("Fim (mmm-yyyy)",    [p.strftime("%b-%Y") for p in period_range], index=len(period_range)-1)
start_p = pd.Period(start_lbl, freq="M")
end_p   = pd.Period(end_lbl,   freq="M")
if start_p > end_p:
    start_p, end_p = end_p, start_p
mask = (df["period"] >= start_p) & (df["period"] <= end_p)
df_filt = df.loc[mask].copy()

# =================== Selection widgets (empty by default) ===================
st.sidebar.markdown("---")
st.sidebar.write("**Seleção de classes**")
sel_macros = st.sidebar.multiselect("Macro classes", MACRO_CLASSES, default=[])

# subclasses list (filtered to those present in data)
all_sub_candidates = []
for k, subs in SUBCLASSES_MAP.items():
    all_sub_candidates.extend(subs)
existing_subs = sorted([s for s in all_sub_candidates if s in df_filt["Asset Class"].unique()])
sel_subs = st.sidebar.multiselect("Subclasses", existing_subs, default=[])

# Controls
st.sidebar.markdown("---")
bar_step = st.sidebar.slider("Largura das barras (step)", min_value=12, max_value=64, value=28, step=2)
show_labels_mtd = st.sidebar.checkbox("Mostrar data labels no MTD", value=True)
label_fmt = ".2%"

# =================== Section A: MTD — usar só a ÚLTIMA DATA do mês ===================
st.subheader("A) MTD — Benchmark vs M4 ")
sel_mtd_date = month_selectbox(df_filt, "Selecione o mês (mmm-yyyy) para MTD:", key="mtd_month")

def prepare_mtd_lastday(df_in, macros, subs, sel_date):
    df_day = df_in[df_in["Data"] == sel_date].copy()
    parts = []
    if macros:
        parts.append(agg_macros_lastday(df_day, macros))
    if subs:
        parts.append(agg_subs_lastday(df_day, subs))
    if not parts:
        return pd.DataFrame(columns=["Label", COL_MTD_BMK, COL_MTD_M4, COL_YTD_BMK, COL_YTD_M4])
    out = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    return out

if sel_mtd_date is None:
    st.info("Sem datas disponíveis.")
else:
    mtd_agg = prepare_mtd_lastday(df_filt, sel_macros, sel_subs, sel_mtd_date)
    if mtd_agg.empty:
        st.info("Selecione ao menos uma macro classe e/ou subclasse para visualizar.")
    else:
        tidy = mtd_agg.melt(id_vars=["Label"], value_vars=[COL_MTD_BMK, COL_MTD_M4], var_name="Série", value_name="Valor")
        tidy["Série"] = tidy["Série"].replace({COL_MTD_BMK:"Benchmark", COL_MTD_M4:"Modelo"})

        # Headroom for labels: dynamic Y domain padding
        if not tidy.empty:
            y_min = float(tidy["Valor"].min())
            y_max = float(tidy["Valor"].max())
            dom_lo = min(0.0, y_min)  # keep baseline at 0 when positive
            pad = (y_max - dom_lo) * 0.18 if y_max > dom_lo else 0.02
            dom_hi = y_max + pad
        else:
            dom_lo, dom_hi = 0.0, 1.0

        bars = alt.Chart(tidy).mark_bar().encode(
            x=alt.X("Label:N",
                    title="Classe/Subclasse",
                    sort="-y",
                    axis=alt.Axis(labelAngle=0, labelLimit=1000, labelPadding=8)  # 'deitado' + sem truncar
            ),
            y=alt.Y("Valor:Q",
                    title="MTD",
                    axis=alt.Axis(format=label_fmt),
                    scale=alt.Scale(domain=[dom_lo, dom_hi])  # folga no topo
            ),
            color=alt.Color("Série:N", legend=alt.Legend(title="")),
            xOffset="Série:N",
            tooltip=[alt.Tooltip("Label:N", title="Classe/Subclasse"),
                     alt.Tooltip("Série:N", title="Série"),
                     alt.Tooltip("Valor:Q", title="MTD", format=label_fmt)]
        ).properties(height=340, width=alt.Step(bar_step))

        if show_labels_mtd:
            labels = alt.Chart(tidy).mark_text(dy=-8, baseline="bottom").encode(
                x=alt.X("Label:N"),
                y=alt.Y("Valor:Q", scale=alt.Scale(domain=[dom_lo, dom_hi])),
                detail="Série:N",
                text=alt.Text("Valor:Q", format=label_fmt),
                xOffset="Série:N"
            )
            chart = bars + labels
        else:
            chart = bars

        st.altair_chart(chart, use_container_width=True)

        mtd_agg["MTD Diff (M4 - Bmk)"] = mtd_agg[COL_MTD_M4] - mtd_agg[COL_MTD_BMK]
        st.dataframe(style_pct_df(mtd_agg.copy(), [COL_MTD_BMK, COL_MTD_M4, COL_YTD_BMK, COL_YTD_M4, "MTD Diff (M4 - Bmk)"]))

# =================== Section B: YTD — (permanece igual ao v2.0) ===================
st.subheader("B) YTD — Evolução")

def slice_lastday_per_period(df_in):
    last_by_period = df_in.groupby("period", as_index=False)["Data"].max().rename(columns={"Data":"last_date"})
    out = df_in.merge(last_by_period, on="period", how="left")
    out = out[out["Data"] == out["last_date"]].copy()
    return out

def build_ytd_lines_lastday(df_in, macros, subs):
    df_last = slice_lastday_per_period(df_in)
    frames = []
    if macros:
        for mac in macros:
            macro_rows = df_last[(df_last["class_l1"].str.lower() == mac.lower()) & (df_last["is_pure_macro"])]
            if not macro_rows.empty:
                base = (macro_rows.groupby(["period","month_idx","month_lbl"], dropna=False)[[COL_YTD_BMK, COL_YTD_M4]]
                                .sum().reset_index())
                base["Label"] = mac
            else:
                subs_rows = df_last[(df_last["class_l1"].str.lower() == mac.lower()) & (~df_last["is_pure_macro"])]
                if subs_rows.empty:
                    continue
                base = (subs_rows.groupby(["period","month_idx","month_lbl"], dropna=False)[[COL_YTD_BMK, COL_YTD_M4]]
                                .sum().reset_index())
                base["Label"] = mac
            frames.append(base)
    if subs:
        subs_rows = df_last[df_last["Asset Class"].isin(subs)]
        if not subs_rows.empty:
            base = (subs_rows.groupby(["Asset Class","period","month_idx","month_lbl"], dropna=False)[[COL_YTD_BMK, COL_YTD_M4]]
                           .sum().reset_index().rename(columns={"Asset Class":"Label"}))
            frames.append(base)
    if not frames:
        return pd.DataFrame(columns=["Label","month_lbl","month_idx",COL_YTD_BMK,COL_YTD_M4])
    out = pd.concat(frames, ignore_index=True)
    return out

ytd_lines = build_ytd_lines_lastday(df_filt, sel_macros, sel_subs)
if ytd_lines.empty:
    st.info("Selecione classes para ver a evolução YTD.")
else:
    tidy_ytd = ytd_lines.melt(id_vars=["Label","month_lbl","month_idx"], value_vars=[COL_YTD_BMK, COL_YTD_M4], var_name="Série", value_name="Valor")
    tidy_ytd["Série"] = tidy_ytd["Série"].replace({COL_YTD_BMK:"Benchmark", COL_YTD_M4:"Modelo"})

    lines = alt.Chart(tidy_ytd).mark_line().encode(
        x=alt.X("month_lbl:N", title="Data (mmm-yyyy)", sort=alt.SortField(field="month_idx", order="ascending")),
        y=alt.Y("Valor:Q", title="YTD", axis=alt.Axis(format=".2%")),
        color=alt.Color("Label:N", title="Classe/Subclasse"),
        strokeDash=alt.StrokeDash("Série:N", sort=["Benchmark","Modelo"], title="Série")
    )
    points = alt.Chart(tidy_ytd).mark_point().encode(
        x=alt.X("month_lbl:N", sort=alt.SortField(field="month_idx", order="ascending")),
        y="Valor:Q",
        color="Label:N",
        shape=alt.ShapeValue("circle")
    )
    last_pts = (tidy_ytd.sort_values("month_idx").groupby(["Label","Série"], as_index=False).tail(1))
    labels = alt.Chart(last_pts).mark_text(dx=6, dy=-6).encode(
        x=alt.X("month_lbl:N", sort=alt.SortField(field="month_idx", order="ascending")),
        y="Valor:Q",
        color="Label:N",
        text=alt.Text("Valor:Q", format=".2%")
    )

    chart = lines + points + labels
    st.altair_chart(chart.properties(height=380), use_container_width=True)

st.markdown("---")
st.caption("Ajustes: eixo X 'deitado' no MTD e folga dinâmica no topo para não cortar labels; resto idêntico ao v2.0.")
