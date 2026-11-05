"""Aplikasi Streamlit: peramalan kurs IDR/USD dengan ARIMA dan SVR.

Jalankan dari root repo:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kurs_forecast.data import load_kurs, split_train_test  # noqa: E402
from kurs_forecast.metrics import all_metrics  # noqa: E402
from kurs_forecast.models import fit_arima, fit_svr, forecast_arima, forecast_svr_future, predict_svr  # noqa: E402
from kurs_forecast.risk import ringkasan_risiko  # noqa: E402

st.set_page_config(page_title="Peramalan Kurs IDR/USD", page_icon="📈", layout="wide")
st.title("Peramalan Kurs Rupiah terhadap Dolar AS: ARIMA vs SVR")
st.caption("Implementasi ulang skripsi Immantaka Juniarsa (Statistika, Universitas Airlangga, 2024) dengan pengembangan modul risiko.")


@st.cache_data(show_spinner=False)
def muat_data():
    return load_kurs()


@st.cache_resource(show_spinner="Melatih model ARIMA dan SVR...")
def latih(n_train: int, order: tuple[int, int, int]):
    serie = muat_data()
    tr, te = split_train_test(serie, n_train)
    arima = fit_arima(tr, order)
    # Grid diperkecil agar aplikasi responsif; grid penuh ada di pipeline.
    grid = {"svr__C": [16.0, 128.0, 512.0], "svr__gamma": [0.125, 0.5, 1.0], "svr__epsilon": [0.1, 0.3]}
    svr = fit_svr(tr, kernel="rbf", grid=grid, n_splits=3)
    return tr, te, arima, svr


with st.sidebar:
    st.header("Pengaturan")
    serie = muat_data()
    n_train = st.slider("Jumlah minggu training", min_value=60, max_value=len(serie) - 5, value=84)
    order_label = st.selectbox("Model ARIMA", ["(0,2,1)", "(1,2,0)", "(2,2,0)"], index=0)
    order = tuple(int(c) for c in order_label.strip("()").split(","))
    horizon = st.slider("Horizon peramalan ke depan (minggu)", 1, 12, 4)
    nominal = st.number_input("Nominal posisi (USD) untuk VaR", min_value=1_000, value=100_000, step=10_000)

tr, te, arima, svr = latih(n_train, order)
pred_arima = forecast_arima(arima, len(te))
pred_svr = predict_svr(svr, tr, te)
m_arima, m_svr = all_metrics(te.values, pred_arima), all_metrics(te.values, pred_svr)

c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE ARIMA", f"{m_arima['RMSE']:,.1f}")
c2.metric("MAPE ARIMA", f"{m_arima['MAPE']:.2f} %")
c3.metric("RMSE SVR RBF", f"{m_svr['RMSE']:,.1f}")
c4.metric("MAPE SVR RBF", f"{m_svr['MAPE']:.2f} %")

tab1, tab2, tab3 = st.tabs(["Prediksi data testing", "Peramalan ke depan", "Risiko (VaR)"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tr.index, y=tr.values, name="Training", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=te.index, y=te.values, name="Aktual testing", line=dict(color="#111827"), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=te.index, y=pred_arima, name=f"ARIMA{order}", line=dict(color="#16a34a", dash="dash")))
    fig.add_trace(go.Scatter(x=te.index, y=pred_svr, name="SVR RBF", line=dict(color="#f59e0b", dash="dash")))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Rupiah per USD", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        pd.DataFrame({"tanggal": te.index.date, "aktual": te.values, "ARIMA": np.round(pred_arima, 0), "SVR RBF": np.round(pred_svr, 0)}),
        use_container_width=True,
        hide_index=True,
    )
    st.info(f"Parameter SVR terpilih: {svr.best_params}. Parameter ARIMA: {arima.params}.")

with tab2:
    st.write(
        "Peramalan dilakukan dari titik terakhir data penuh (training + testing). "
        "SVR memakai skema rekursif, sehingga ketidakpastian membesar seiring horizon."
    )
    arima_full = fit_arima(serie, order)
    fc_arima = forecast_arima(arima_full, horizon)
    fc_svr = forecast_svr_future(svr, serie, horizon)
    tanggal = pd.date_range(serie.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="7D")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=serie.index[-30:], y=serie.values[-30:], name="Aktual (30 minggu terakhir)", line=dict(color="#111827")))
    fig2.add_trace(go.Scatter(x=tanggal, y=fc_arima, name="ARIMA", mode="lines+markers", line=dict(color="#16a34a")))
    fig2.add_trace(go.Scatter(x=tanggal, y=fc_svr, name="SVR RBF", mode="lines+markers", line=dict(color="#f59e0b")))
    fig2.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Rupiah per USD", legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(pd.DataFrame({"tanggal": tanggal.date, "ARIMA": np.round(fc_arima, 0), "SVR RBF": np.round(fc_svr, 0)}), hide_index=True)

with tab3:
    r = ringkasan_risiko(serie, nominal_usd=float(nominal))
    k1, k2, k3 = st.columns(3)
    k1.metric("VaR historis 95% (1 minggu)", f"{r['var_historis_pct']:.2f} %", help="Kuantil 95% dari log return mingguan")
    k2.metric("VaR parametrik 95%", f"{r['var_parametrik_pct']:.2f} %")
    k3.metric("Expected Shortfall 95%", f"{r['es_historis_pct']:.2f} %")
    st.write(
        f"Untuk kewajiban **{nominal:,.0f} USD** pada kurs terakhir **Rp{r['kurs_akhir']:,.0f}**, kerugian mingguan terburuk pada "
        f"tingkat keyakinan 95% diperkirakan **Rp{r['var_historis_rp']:,.0f}** (historis) dan rata-rata kerugian saat melewati "
        f"batas itu **Rp{r['es_historis_rp']:,.0f}** (expected shortfall)."
    )
