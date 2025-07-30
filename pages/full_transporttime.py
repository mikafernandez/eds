import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t
import plotly.graph_objects as go
from io import BytesIO

st.title("Vergleich mittlerer Journey-Zeiten pro Tag und System")

# Sheet-Mapping: explizite, unmissverständliche Zuordnung
sheet_map = {
    "Parameter Set 1": [
        "Sheet1", "Sheet3", "Sheet5", "Sheet7", "Sheet9",
        "Sheet11", "Sheet13", "Sheet15", "Sheet17", "Sheet19",
        "Sheet21", "Sheet23", "Sheet25", "Sheet27", "Sheet29"
    ],
    "Parameter Set 0": [
        "Sheet2", "Sheet4", "Sheet6", "Sheet8", "Sheet10",
        "Sheet12", "Sheet14", "Sheet16", "Sheet18", "Sheet20",
        "Sheet22", "Sheet24", "Sheet26", "Sheet28", "Sheet30"
    ]
}

elevators = [
    {"name": f"Elevator {i}", "col_journey": 1 + i * 3}
    for i in range(10)
]

# Farbcodes & Beschriftungen
plot_config = {
    "Parameter Set 1": {"label": "Einfache Steuerung | 10 Aufzüge", "color": "royalblue"},
    "Parameter Set 0": {"label": "Richtungsbasierte Steuerung | 10 Aufzüge", "color": "crimson"}
}
y_labels = [plot_config["Parameter Set 1"]["label"], plot_config["Parameter Set 0"]["label"]]

uploaded_file = st.file_uploader("Excel-Datei mit Simulationsdaten hochladen (.xlsx)", type=["xlsx"])
if uploaded_file:
    results = {"Parameter Set 1": [], "Parameter Set 0": []}
    excel = pd.ExcelFile(uploaded_file)

    with st.expander("🔍 Rohdaten & Zwischenschritte pro Sheet", expanded=False):
        for param_set, sheets in sheet_map.items():
            st.markdown(f"### {plot_config[param_set]['label']}")
            for sheet in sheets:
                elevator_means = []
                if sheet not in excel.sheet_names:
                    st.warning(f"Sheet {sheet} nicht gefunden – wird übersprungen.")
                    continue
                df = pd.read_excel(uploaded_file, sheet_name=sheet, header=None)
                st.write(f"**{sheet}:** DataFrame shape {df.shape}")
                for elevator in elevators:
                    if elevator["col_journey"] < df.shape[1]:
                        journey = pd.to_numeric(df.iloc[2:, elevator["col_journey"]], errors="coerce")
                        journey = journey[~np.isnan(journey)]
                        if len(journey) > 0:
                            elevator_means.append(np.mean(journey))
                            mw_str = f"{np.mean(journey):.2f}"
                        else:
                            elevator_means.append(np.nan)
                            mw_str = "nan"
                        st.write(
                            f"Elevator {elevator['name']}:",
                            journey[:8].tolist(),  # zeige 8 Werte als Vorschau
                            f"(n={len(journey)}, Mittelwert: {mw_str})"
                        )
                # Tagesschnitt aus Aufzügen bilden & speichern
                if len(elevator_means) > 0 and not all(np.isnan(elevator_means)):
                    day_mean = np.nanmean(elevator_means)
                else:
                    day_mean = np.nan
                results[param_set].append({
                    "mean": day_mean,
                    "elevator_means": elevator_means
                })

    # Tabelle: Alle Tagesmittelwerte
    with st.expander("📊 Übersicht: Tagesmittelwerte und Mittelwerte pro Aufzug", expanded=True):
        for param_set in ["Parameter Set 1", "Parameter Set 0"]:
            days = []
            for i, entry in enumerate(results[param_set], 1):
                row = {"Tag": i, "Tagesschnitt": entry["mean"]}
                for j, val in enumerate(entry["elevator_means"]):
                    row[f"Elevator {j}"] = val
                days.append(row)
            df_tage = pd.DataFrame(days)
            st.subheader(plot_config[param_set]["label"])
            st.dataframe(df_tage.style.format("{:.2f}"), use_container_width=True)
            buf = BytesIO()
            df_tage.to_excel(buf, index=False)
            st.download_button(
                f"Excel-Export Tagesmittelwerte: {plot_config[param_set]['label']}",
                data=buf.getvalue(),
                file_name=f"{param_set}_tage_und_aufzuege.xlsx"
            )

    # Statistische Auswertung & Plot
    stats = []
    with st.expander("📈 Statistische Auswertung (Aggregat aller Tage)", expanded=True):
        for param_set in ["Parameter Set 1", "Parameter Set 0"]:
            means = [entry["mean"] for entry in results[param_set] if not np.isnan(entry["mean"])]
            n = len(means)
            mean_val = np.mean(means)
            std_val = np.std(means, ddof=1) if n > 1 else 0
            tval = t.ppf(0.975, n-1) if n > 1 else 0
            se = std_val / np.sqrt(n) if n > 1 else 0
            ci_lower = mean_val - tval * se
            ci_upper = mean_val + tval * se
            stats.append({
                "set": param_set,
                "mean": mean_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "color": plot_config[param_set]["color"],
                "label": plot_config[param_set]["label"],
                "means_per_day": means
            })
            st.markdown(
                f"**{plot_config[param_set]['label']}**: Mittelwert über alle Tage = <span style='color:{plot_config[param_set]['color']};font-weight:bold;'>{mean_val:.2f}</span> s, "
                f"Standardabweichung = `{std_val:.2f}` s, 95%-KI = [`{ci_lower:.2f}`, `{ci_upper:.2f}`], n = {n}",
                unsafe_allow_html=True
            )

    # Plot wie dein Beispiel (horizontale Fehlerbalken)
    fig = go.Figure()
    for i, s in enumerate(stats):
        fig.add_trace(go.Scatter(
            x=[s["mean"]],
            y=[s["label"]],
            mode="markers",
            marker=dict(size=14, color=s["color"]),
            name=s["label"]
        ))
        fig.add_trace(go.Scatter(
            x=[s["ci_lower"], s["ci_upper"]],
            y=[s["label"], s["label"]],
            mode="lines",
            line=dict(color=s["color"], width=6),
            showlegend=False
        ))

    all_means = [s["mean"] for s in stats]
    overall_mean = np.mean(all_means)
    fig.add_vline(x=overall_mean, line_dash="dash", line_color="gray", opacity=0.4)

    fig.update_layout(
        title="Konfidenzintervalle mittlere Journey-Zeit (Tagesmittelwert, über Aufzüge) mit 5%-Signifikanzniveau",
        xaxis_title="Mittlere Journey-Zeit [s]",
        yaxis_title="",
        template="plotly_white",
        margin=dict(l=50, r=30, t=50, b=40),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretationstext
    st.markdown(
        "- **Vorgehen:** Für jeden Tag wird zuerst pro Aufzug der Mittelwert gebildet, dann der Tagesmittelwert über alle Aufzüge. "
        "Erst danach erfolgt die statistische Auswertung dieser Tagesmittelwerte über alle Replications.<br>"
        "- **Balken:** 95%-Konfidenzintervall für den wahren Mittelwert.",
        unsafe_allow_html=True
    )
