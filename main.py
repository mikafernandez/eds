import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tempfile import NamedTemporaryFile
from scipy.stats import t
import io

# --- Eigene Datenverarbeitungsfunktionen (externe Datei) ---
from data_processing import (
    process_uploaded_excel_custom,
    process_utilization_excel_auslastung,
    process_max_waiting_times,
    summarize_hourly_stats,
    summarize_hourly_stats_extended,
    compute_confidence_intervals,
    compute_confidence_intervals_extended
)


def day_matches(entry_day, target_day):
    """
    Hilfsfunktion zum robusten Vergleich von Tagesnummern in den Daten.
    """
    try:
        if isinstance(entry_day, str):
            digits = ''.join(filter(str.isdigit, entry_day))
            if digits and int(digits) == target_day:
                return True
        if str(entry_day) == str(target_day):
            return True
        if entry_day == target_day:
            return True
    except Exception:
        pass
    return False

def plot_day_timeseries(df_day, title):
    """
    Erstellt einen Linienplot für den Tagesverlauf der mittleren Wartezeit.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_day["Stunde"],
        y=df_day["Ø Wartezeit (Sekunden)"],
        mode="lines+markers",
        name="Ø Wartezeit"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Stunde",
        yaxis_title="Ø Wartezeit (Sekunden)",
        template="plotly_white",
        height=260,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def plot_kpi_comparison(title, labels, means, ci_lowers, ci_uppers, farben):
    """
    Visualisiert den Vergleich von KPIs inkl. Fehlerbalken (Konfidenzintervall).
    """
    fig = go.Figure()
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=[label],
            y=[means[i]],
            mode='markers',
            marker=dict(size=18, color=farben[label]),
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ci_uppers[i] - means[i]],
                arrayminus=[means[i] - ci_lowers[i]],
                thickness=4,
                color=farben[label]
            ),
            name=label
        ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Parameter Set", tickvals=labels),
        yaxis=dict(title=""),
        showlegend=False,
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
        template="plotly_white"
    )
    return fig


# ========== 1. Streamlit UI-Konfiguration ==========
st.set_page_config(page_title="AnyLogic Aufzugssteuerung Datenauswertung", layout="wide")
st.title("AnyLogic – Aufzugssteuerung: Datenauswertung")

st.markdown("""
## 📥 Upload  
Bitte lade die Excel-Datei hoch, die aus deinem Simulationslauf mit  
**9 Aufzügen** und **16 Etagen** erzeugt wurde
""")

# Parameter-Eingabe für Auswertung
anzahl_elevator = st.number_input(
    label="🚀 Festlegung der Anzahl der Aufzüge",
    min_value=1, max_value=20, value=9, step=1,
    help="Wähle die Anzahl der Aufzüge aus, die in der Auswertung berücksichtigt werden sollen."
)
st.markdown(f"**Aktuelle Auswahl:** `{anzahl_elevator}` Aufzüge")

anzahl_simulationslaeufe = st.selectbox(
    label="📊 Anzahl der Simulationsläufe wählen",
    options=[5, 10, 15],
    index=0,
    help="Wähle die Anzahl der Simulationsläufe."
)
st.markdown(f"**Aktuelle Auswahl Simulationsläufe:** `{anzahl_simulationslaeufe}`")

uploaded_file = st.file_uploader("Excel-Datei hochladen", type=["xlsx"])


# ========== 2. Hilfsfunktion: Linienplot für Tagesverlauf ==========


# ========== 3. Datei-Upload und Datenverarbeitung ==========
if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    # Datenverarbeitung nur beim ersten Mal oder auf Wunsch
    if st.button("Daten verarbeiten") or "results" not in st.session_state:
        with st.spinner("Analysiere Daten..."):
            results = process_uploaded_excel_custom(temp_file_path, anzahl_simulationslaeufe)
            st.session_state["results"] = results
            auslastung_results = process_utilization_excel_auslastung(temp_file_path, anzahl_elevator, anzahl_simulationslaeufe)
            st.session_state["auslastung_results"] = auslastung_results
            max_results = process_max_waiting_times(temp_file_path, anzahl_simulationslaeufe)
            st.session_state["max_results"] = max_results
        st.success("Verarbeitung abgeschlossen.")

# ========== 4. Tagesverlauf Wartezeiten (je Parameter Set & Tag) ==========
if "results" in st.session_state:
    results = st.session_state["results"]

    st.header("Tagesverläufe (je Parameter Set & Tag)")
    with st.expander("Erklärung der Tagesverläufe", expanded=False):
        for entry in results:
            st.subheader(f"📊 {entry['Parameter Set']} – {entry['Day']}")
            df_day = pd.DataFrame({
                "Stunde": list(range(1, 13)),
                "Ø Wartezeit (Sekunden)": entry["Hourly Means"]
            })
            st.dataframe(df_day.style.format({"Ø Wartezeit (Sekunden)": "{:.2f}"}), use_container_width=True)
            st.markdown(f"**Tagesmittelwert:** `{entry['Daily Mean']:.2f}` Sekunden")
            st.plotly_chart(
                plot_day_timeseries(df_day, f"Tagesverlauf: {entry['Parameter Set']} – {entry['Day']}"),
                use_container_width=True
            )

    # ========== 5. Stundenauswertung & Plot: Aggregierte Stundenwerte ==========
    st.header("Stundenauswertung (Aggregat über alle Tage je Parameter Set)")

    summary_tables = summarize_hourly_stats(results)
    farben = {"Parameter Set 1": "royalblue", "Parameter Set 0": "crimson"}
    fillfarben = {"Parameter Set 1": "rgba(65, 105, 225, 0.18)", "Parameter Set 0": "rgba(220, 20, 60, 0.18)"}

    # Tabellen und gemeinsames Diagramm
    for param_set, table in summary_tables.items():
        st.subheader(f"⏰ {param_set} – Aggregiert über alle Tage")
        st.dataframe(table.style.format({
            "Mittelwert": "{:.2f}",
            "Varianz": "{:.2f}",
            "CI lower (95%)": "{:.2f}",
            "CI upper (95%)": "{:.2f}"
        }), use_container_width=True)

    # Aggregierter Plot für beide Parameter Sets
    fig = go.Figure()
    for param_set, table in summary_tables.items():
        fig.add_trace(go.Scatter(
            x=table["Stunde"],
            y=table["Mittelwert"],
            mode="lines+markers",
            name=f"Mittelwert: {param_set}",
            line=dict(color=farben[param_set], width=3)
        ))
        fig.add_trace(go.Scatter(
            x=table["Stunde"],
            y=table["CI upper (95%)"],
            mode="lines",
            line=dict(width=0, color=farben[param_set]),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=table["Stunde"],
            y=table["CI lower (95%)"],
            mode="lines",
            fill="tonexty",
            fillcolor=fillfarben[param_set],
            line=dict(width=0, color=farben[param_set]),
            name=f"95% CI: {param_set}",
            showlegend=True,
            hoverinfo="skip"
        ))
    fig.update_layout(
        title="Stunden-Mittelwerte & 95%-Konfidenzintervall je Parameter Set",
        xaxis_title="Stunde",
        yaxis_title="Ø Wartezeit (Sekunden)",
        template="plotly_white",
        height=390,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            orientation="v",
            x=1,
            xanchor="left",
            y=0.5,
            yanchor="middle"
        ),
        xaxis=dict(dtick=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ========== 6. Tagesmittelwerte: Konfidenzintervall ==========
    st.header("Konfidenzintervall des Tagesmittelwerts (je Parameter Set)")
    df_day_ci = compute_confidence_intervals(results)
    st.dataframe(df_day_ci.style.format({
        "Mittelwert": "{:.2f}",
        "Varianz": "{:.2f}",
        "CI lower (95%)": "{:.2f}",
        "CI upper (95%)": "{:.2f}"
    }), use_container_width=True)


# ========== 7. Ressourcenauslastung und Transportzeit ==========
if "auslastung_results" in st.session_state:
    auslastung_results = st.session_state["auslastung_results"]

    st.header("Tagesverläufe Ressourcenauslastung, Transportzeit und Personenanzahl (je Parameter Set & Tag)")
    with st.expander("Erklärung der Tagesverläufe", expanded=False):
        for entry in auslastung_results:
            st.subheader(f"📊 {entry['Parameter Set']} – {entry['Day']}")
            df_day = pd.DataFrame({
                "Stunde": list(range(1, 13)),
                "Ø Auslastung (%)": entry["Hourly Util Means (%)"],
                "Ø Transportzeit (s)": entry["Hourly Transport Means (s)"],
                "Ø Personen im Aufzug": entry["Hourly Persons Means"],
            })
            st.dataframe(
                df_day.style.format({
                    "Ø Auslastung (%)": "{:.1f}",
                    "Ø Transportzeit (s)": "{:.2f}",
                    "Ø Personen im Aufzug": "{:.2f}"
                }),
                use_container_width=True
            )
            st.markdown(
                f"**Tagesmittelwerte:** "
                f"Auslastung: `{entry['Daily Util Mean (%)']:.1f}` % &nbsp; | &nbsp; "
                f"Transportzeit: `{entry['Daily Transport Mean (s)']:.2f}` s &nbsp; | &nbsp; "
                f"Personenzahl: `{entry['Daily Persons Mean']:.2f}`",
                unsafe_allow_html=True
            )
            # Plot mit drei y-Achsen
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_day["Stunde"], y=df_day["Ø Auslastung (%)"],
                mode="lines+markers", name="Ø Auslastung (%)", yaxis="y1"
            ))
            fig.add_trace(go.Scatter(
                x=df_day["Stunde"], y=df_day["Ø Transportzeit (s)"],
                mode="lines+markers", name="Ø Transportzeit (s)", yaxis="y2"
            ))
            fig.add_trace(go.Scatter(
                x=df_day["Stunde"], y=df_day["Ø Personen im Aufzug"],
                mode="lines+markers", name="Ø Personen im Aufzug", yaxis="y3"
            ))
            fig.update_layout(
                title=f"Tagesverlauf: {entry['Parameter Set']} – {entry['Day']}",
                xaxis_title="Stunde",
                yaxis=dict(title="Ø Auslastung (%)", side="left"),
                yaxis2=dict(title="Ø Transportzeit (s)", overlaying="y", side="right"),
                yaxis3=dict(title="Ø Personen im Aufzug", overlaying="y", side="right", anchor="free", position=1),
                legend=dict(orientation="h"),
                template="plotly_white",
                height=350,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
    st.header("Auslastung im Zeitverlauf über die Runs hinweg")

    if "auslastung_results" in st.session_state:
        auslastung_results = st.session_state["auslastung_results"]

        # --------- Aggregation über alle Tage pro Parameter-Set ---------
        param_sets = sorted(set([entry["Parameter Set"] for entry in auslastung_results]))
        hours = list(range(1, 13))
        hour_labels = [f"{6 + h - 1}:00–{6 + h}:00" for h in hours]  # "6:00–7:00", ...

        summary_data = []
        for param_set in param_sets:
            # Alle Einträge dieses Parameter-Sets
            entries = [e for e in auslastung_results if e["Parameter Set"] == param_set]
            # Matrix: Zeile=Tag, Spalte=Stunde
            hourly_matrix = np.stack([entry["Hourly Util Means (%)"] for entry in entries])
            means = np.mean(hourly_matrix, axis=0)
            stds = np.std(hourly_matrix, axis=0, ddof=1)
            n = hourly_matrix.shape[0]
            # 95%-KI (zweiseitig, Mittelwert)
            ci_halfwidth = t.ppf(0.975, n - 1) * stds / np.sqrt(n)
            ci_upper = means + ci_halfwidth
            ci_lower = means - ci_halfwidth

            # Speichern für Plot & Export
            summary_data.append(pd.DataFrame({
                "Stunde (Label)": hour_labels,
                "Stunde (Index)": hours,
                "Mittelwert Auslastung (%)": means,
                "KI Min (%)": ci_lower,
                "KI Max (%)": ci_upper,
                "Parameter Set": param_set
            }))

        # --------- Gesamttabelle zum Export & Ansicht ---------
        df_export = pd.concat(summary_data, ignore_index=True)
        with st.expander("📂 Konsolidierte Auslastungsdaten (Export)", expanded=False):
            st.dataframe(df_export.style.format({
                "Mittelwert Auslastung (%)": "{:.2f}",
                "KI Min (%)": "{:.2f}",
                "KI Max (%)": "{:.2f}"
            }), use_container_width=True)

            # Excel-Datei im Speicher erstellen
            excel_buffer = io.BytesIO()
            df_export.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)  # Wichtig: An den Anfang des Puffers setzen

            st.download_button(
                "Excel herunterladen",
                data=excel_buffer,
                file_name="auslastung_konsolidiert.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # --------- Plot (mit KI) ---------

        import plotly.graph_objects as go

        # Label- und Farbmapping nach gewünschter Zuordnung
        label_map = {
            0: "Einfache Steuerung | 10 Aufzüge",  # royalblue
            1: "Richtungsbasierte Steuerung | 10 Aufzüge"  # crimson
        }
        farben = {
            0: "royalblue",  # Einfache Steuerung
            1: "crimson"  # Richtungsbasierte Steuerung
        }
        fillcolors = {
            0: "rgba(65,105,225,0.13)",  # Einfache Steuerung, royalblue soft
            1: "rgba(220,20,60,0.13)"  # Richtungsbasierte, crimson soft
        }

        fig = go.Figure()
        for df in summary_data:
            # Sichere Extraktion des Parametersets (0 oder 1)
            pset_value = df["Parameter Set"].iloc[0]
            try:
                pset = int(str(pset_value).replace("Parameter Set", "").strip())
            except Exception:
                pset = pset_value  # fallback

            kurven_label = label_map.get(pset, f"Parameter Set {pset}")
            ki_label = f"KI: {kurven_label}"

            # Mittelwert-Kurve
            fig.add_trace(go.Scatter(
                x=df["Stunde (Label)"],
                y=df["Mittelwert Auslastung (%)"],
                mode="lines+markers",
                name=kurven_label,
                line=dict(width=2, color=farben.get(pset, "gray")),
                marker=dict(color=farben.get(pset, "gray")),
                showlegend=True
            ))

            # KI-Band
            fig.add_trace(go.Scatter(
                x=pd.concat([df["Stunde (Label)"], df["Stunde (Label)"][::-1]]),
                y=pd.concat([df["KI Max (%)"], df["KI Min (%)"][::-1]]),
                fill="toself",
                fillcolor=fillcolors.get(pset, "rgba(128,128,128,0.12)"),
                line=dict(color="rgba(0,0,0,0)"),
                name=ki_label,
                showlegend=True,
                hoverinfo="skip"
            ))

        fig.update_layout(
            title="Verlauf der Ressourcenauslastung über alle Replikationen",
            xaxis_title="Tageszeit",
            yaxis=dict(
                title="Ø Auslastung [%]",
                rangemode="tozero",
                range=[
                    30,
                    max(df_export["KI Max (%)"]) + 5  # +5% Puffer nach oben, auf Basis aller KI-Maxima!
                ]
            ),
            template="plotly_white",
            legend=dict(
                orientation="v",
                x=1, xanchor="left",
                y=0.5, yanchor="middle"
            ),
            height=410,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ========== 8. Stundenauswertung der Transportzeiten je Parameter Set ==========
    if "auslastung_results" in st.session_state:
        auslastung_results = st.session_state["auslastung_results"]

        # Auswahl des zu vergleichenden Parameter Sets
        parameter_sets = list({entry["Parameter Set"] for entry in auslastung_results})
        selected_set = st.selectbox("Parameter Set auswählen", parameter_sets)

        # Sammle alle Transportzeiten je Stunde für das gewählte Parameter Set
        hours = 12  # Anzahl Stunden (anpassen falls nötig)
        all_hourly_values = [[] for _ in range(hours)]
        for entry in auslastung_results:
            if entry["Parameter Set"] == selected_set:
                for h, values in enumerate(entry["Hourly Transport Values (s)"]):
                    all_hourly_values[h].extend(values)

        # Mittelwerte und 95%-Konfidenzintervalle je Stunde berechnen
        means, cis_lower, cis_upper = [], [], []
        for values in all_hourly_values:
            arr = np.array(values)
            arr = arr[~np.isnan(arr)]
            if len(arr) > 1:
                mean = np.mean(arr)
                std = np.std(arr, ddof=1)
                n = len(arr)
                ci = t.interval(0.95, n - 1, loc=mean, scale=std / np.sqrt(n))
                means.append(mean)
                cis_lower.append(ci[0])
                cis_upper.append(ci[1])
            elif len(arr) == 1:
                means.append(arr[0])
                cis_lower.append(arr[0])
                cis_upper.append(arr[0])
            else:
                means.append(np.nan)
                cis_lower.append(np.nan)
                cis_upper.append(np.nan)

        # Plot: Mittelwert und KI pro Stunde
        fig = go.Figure()
        stunden = list(range(1, hours + 1))
        fig.add_trace(go.Scatter(
            x=stunden, y=means, mode="lines+markers",
            name="Mittelwert Transportzeit", line=dict(width=2)
        ))
        fig.add_trace(go.Scatter(
            x=stunden + stunden[::-1],
            y=cis_upper + cis_lower[::-1],
            fill="toself",
            fillcolor="rgba(70,130,180,0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Konfidenzintervall"
        ))
        fig.update_layout(
            title=f"Mittelwerte und 95%-KIs der Transportzeit pro Stunde ({selected_set})",
            xaxis_title="Stunde",
            yaxis_title="Transportzeit [s]",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # OPTIONAL: Histogramm für einzelne Stunden
        hour_selected = st.slider("Stunde für Verteilungsanalyse auswählen", 1, hours, 1)
        vals = np.array(all_hourly_values[hour_selected - 1])
        vals = vals[~np.isnan(vals)]

        if len(vals) > 0:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=vals,
                nbinsx=min(30, max(5, int(np.sqrt(len(vals))))),
                name=f"Transportzeiten, Stunde {hour_selected}",
                marker=dict(line=dict(width=1, color='DarkSlateGrey'))
            ))
            fig_hist.update_layout(
                title=f"Histogramm der Transportzeiten, Stunde {hour_selected} ({selected_set})",
                xaxis_title="Transportzeit [s]",
                yaxis_title="Anzahl",
                bargap=0.1,
                template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Keine Transportzeiten für diese Stunde vorhanden.")

        # ========== 9. Vergleichs-Histogramm der Transportzeiten beider Parameter Sets ==========
        if "auslastung_results" in st.session_state:
            auslastung_results = st.session_state["auslastung_results"]
            st.header("Vergleichs-Histogramm: Transportzeiten beider Parameter Sets (alle Stunden)")

            # Sammle alle Transportzeiten für beide Parameter Sets
            all_values_set0, all_values_set1 = [], []
            gesamt_set0, gesamt_set1 = 0, 0
            table_data = []
            for entry in auslastung_results:
                param_set = entry["Parameter Set"]
                n_this_day = sum(
                    [np.count_nonzero(~np.isnan(hour_list)) for hour_list in entry["Hourly Transport Values (s)"]])
                if param_set == "Parameter Set 0":
                    for stunden_liste in entry["Hourly Transport Values (s)"]:
                        all_values_set0.extend([v for v in stunden_liste if not np.isnan(v)])
                    gesamt_set0 += n_this_day
                elif param_set == "Parameter Set 1":
                    for stunden_liste in entry["Hourly Transport Values (s)"]:
                        all_values_set1.extend([v for v in stunden_liste if not np.isnan(v)])
                    gesamt_set1 += n_this_day
                table_data.append({
                    "Parameter Set": param_set,
                    "Tag": entry["Day"],
                    "Anzahl Transportzeiten (alle Stunden)": n_this_day
                })

            # Tabellenübersicht: Anzahl Transportzeiten pro Tag und Parameter Set
            table_df = pd.DataFrame(table_data)
            with st.expander("Datenbasis: Anzahl Transportzeiten pro Tag und Parameter Set"):
                st.dataframe(table_df, hide_index=True)
                st.markdown(
                    f"**Summe Transportzeiten:**<br>"
                    f"- Parameter Set 0: <b>{gesamt_set0:,}</b><br>"
                    f"- Parameter Set 1: <b>{gesamt_set1:,}</b>",
                    unsafe_allow_html=True
                )

            # Histogramm mit vergleichender Überlagerung
            if all_values_set0 and all_values_set1:
                combined_min = min(min(all_values_set0), min(all_values_set1))
                combined_max = max(max(all_values_set0), max(all_values_set1))
                bin_width = 5
                bin_start = bin_width * int(np.floor(combined_min / bin_width))
                bin_end = bin_width * int(np.ceil(combined_max / bin_width))
                bin_edges = np.arange(bin_start, bin_end + bin_width, bin_width)

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=all_values_set0,
                    xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=bin_width),
                    name="Richtungsbasierte Steuerung | 10 Aufzüge",
                    opacity=0.8,
                    marker=dict(color='crimson'),
                    histnorm="probability"
                ))
                fig.add_trace(go.Histogram(
                    x=all_values_set1,
                    xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=bin_width),
                    name="Einfache Steuerung | 10 Aufzüge",
                    opacity=0.8,
                    marker=dict(color='royalblue'),
                    histnorm="probability"
                ))
                fig.update_layout(
                    barmode='overlay',
                    bargap=0,
                    title="Histogramm der Transportzeiten über alle Replicationen (5s Intervalle)",
                    xaxis_title="Transportzeit [s]",
                    yaxis_title="Relative Häufigkeit [0,1]",
                    template="plotly_white",
                    height=450,
                    legend=dict(orientation="v", x=1, xanchor="left", y=0.5, yanchor="middle")
                )
                tick_step = 10
                tickvals = np.arange(bin_start, bin_end + tick_step, tick_step)
                fig.update_xaxes(tickvals=tickvals, range=[bin_edges[0], bin_edges[-1]], showgrid=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Für einen oder beide Parameter Sets keine Transportzeiten vorhanden.")

        # ========== 10. Stundenauswertung Transportzeit (Aggregat, beide Parameter Sets) ==========
        if "auslastung_results" in st.session_state:
            auslastung_summary_tables = summarize_hourly_stats_extended(auslastung_results)
            farben = {"Parameter Set 1": "royalblue", "Parameter Set 0": "crimson"}
            fillfarben = {"Parameter Set 1": "rgba(65, 105, 225, 0.12)", "Parameter Set 0": "rgba(220, 20, 60, 0.12)"}
            st.header("Stundenauswertung Transportzeit (Aggregat über 5 Tage, beide Parameter Sets)")
            # Tabellenansicht für beide Parameter Sets
            for param_set, table in auslastung_summary_tables.items():
                st.subheader(f"⏰ {param_set} – Aggregiert über 5 Tage")
                st.dataframe(
                    table.style.format({
                        "Mittelwert Transportzeit": "{:.2f}",
                        "Varianz Transportzeit": "{:.2f}",
                        "CI lower Transportzeit (95%)": "{:.2f}",
                        "CI upper Transportzeit (95%)": "{:.2f}",
                    }),
                    use_container_width=True
                )

            # Mittelwert-Vergleich und Plot
            mean_1 = auslastung_summary_tables["Parameter Set 1"]["Mittelwert Transportzeit"].mean()
            mean_0 = auslastung_summary_tables["Parameter Set 0"]["Mittelwert Transportzeit"].mean()
            st.markdown(
                f"**Durchschnittliche Transportzeit:** "
                f"<span style='background-color:#162720;border-radius:4px;padding:2px 7px;color:royalblue;font-weight:bold;'>Set 1: {mean_1:.2f} s</span> &nbsp;|&nbsp; "
                f"<span style='background-color:#162720;border-radius:4px;padding:2px 7px;color:crimson;font-weight:bold;'>Set 0: {mean_0:.2f} s</span>",
                unsafe_allow_html=True
            )

            # Plot für beide Parameter Sets mit Mittelwert und KI
            all_y = []
            for table in auslastung_summary_tables.values():
                all_y += list(table["Mittelwert Transportzeit"].values)
                all_y += list(table["CI upper Transportzeit (95%)"].values)
                all_y += list(table["CI lower Transportzeit (95%)"].values)
            min_y, max_y = min(all_y), max(all_y)
            puffer = 6
            y_range = [min_y - puffer, max_y + puffer]

            fig = go.Figure()
            for param_set, table in auslastung_summary_tables.items():
                fig.add_trace(go.Scatter(
                    x=list(table["Stunde"]) + list(table["Stunde"])[::-1],
                    y=list(table["CI upper Transportzeit (95%)"]) + list(table["CI lower Transportzeit (95%)"])[::-1],
                    fill="toself",
                    fillcolor=fillfarben[param_set],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    hoverinfo="skip",
                    name=f"95% CI: {param_set}",
                    showlegend=True
                ))
                fig.add_trace(go.Scatter(
                    x=table["Stunde"],
                    y=table["Mittelwert Transportzeit"],
                    mode="lines+markers",
                    name=f"Transportzeit: {param_set}",
                    line=dict(color=farben[param_set], width=3)
                ))
            fig.update_layout(
                title="Stundenverlauf der mittleren Transportzeit (Aggregat, beide Parameter Sets)",
                xaxis_title="Stunde",
                yaxis_title="Mittlere Transportzeit (s)",
                template="plotly_white",
                height=380,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="v", x=1, xanchor="left", y=0.5, yanchor="middle"),
                yaxis=dict(range=y_range),
                xaxis=dict(dtick=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        # ========== 6. Konfidenzintervall des Tagesmittelwerts (je Parameter Set) ==========
        st.header("Konfidenzintervall des Tagesmittelwerts (je Parameter Set)")

        # Tagesbasierte KI-Analyse für Ressourcenauslastung, Transportzeit, max. Transportzeit und Personenanzahl
        auslastung_day_ci = compute_confidence_intervals_extended(auslastung_results)
        st.dataframe(
            auslastung_day_ci.style.format({
                "Mittelwert Auslastung": "{:.1f}",
                "Varianz Auslastung": "{:.2f}",
                "CI lower Auslastung (95%)": "{:.1f}",
                "CI upper Auslastung (95%)": "{:.1f}",
                "Mittelwert Transportzeit": "{:.2f}",
                "Varianz Transportzeit": "{:.2f}",
                "CI lower Transportzeit (95%)": "{:.2f}",
                "CI upper Transportzeit (95%)": "{:.2f}",
                "Mittelwert max. Transportzeit": "{:.2f}",
                "Varianz max. Transportzeit": "{:.2f}",
                "CI lower max. Transportzeit (95%)": "{:.2f}",
                "CI upper max. Transportzeit (95%)": "{:.2f}",
                "Mittelwert Personen": "{:.2f}",
                "Varianz Personen": "{:.2f}",
                "CI lower Personen (95%)": "{:.2f}",
                "CI upper Personen (95%)": "{:.2f}",
            }),
            use_container_width=True
        )

    # ========== 7. Maximale Wartezeiten je Tag und Parameter Set ==========
    st.header("Maximale Wartezeiten je Tag und Parameter Set")

    if "max_results" in st.session_state:
        for entry in st.session_state["max_results"]:
            st.subheader(entry["Parameter Set"])
            st.dataframe(entry["table"], use_container_width=True)
            n = len(entry["max_per_day"])
            tag_labels = [f"Tag {i}" for i in range(1, n + 1)]

            # Visualisierung der Tagesmaxima inkl. Mittelwert und KI
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=tag_labels,
                y=entry["max_per_day"],
                name="Max. Wartezeit pro Tag",
                marker=dict(color="rgba(220, 20, 60, 0.32)")
            ))
            fig.add_trace(go.Scatter(
                x=tag_labels,
                y=[entry["mean"]] * n,
                mode="lines",
                name="Mittelwert",
                line=dict(color="royalblue", dash="dash")
            ))
            ci_lower = [entry["ci"][0]] * n
            ci_upper = [entry["ci"][1]] * n
            fig.add_trace(go.Scatter(
                x=tag_labels + tag_labels[::-1],
                y=ci_upper + ci_lower[::-1],
                fill="toself",
                fillcolor="rgba(65,105,225,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% CI Mittelwert"
            ))
            fig.update_layout(
                title="Maximale Wartezeiten pro Tag",
                xaxis_title="Tag",
                yaxis_title="Max. Wartezeit (s)",
                template="plotly_white",
                height=330,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Wissenschaftliche Fußnote zur Interpretation
            st.markdown("""
            - **Mittelwert:** Erwartungswert des Tagesmaximums.<br>
            - **95%-Konfidenzintervall:** Unsicherheitsbereich des Mittelwerts der Tagesmaxima.<br>
            - **Max der 5 Tage:** Absolutes Maximum in allen 5 Tagen.<br>
            <sub>Hinweis: Aufgrund der geringen Stichprobengröße (n=5) ist das Konfidenzintervall mit Unsicherheit behaftet.</sub>
            """, unsafe_allow_html=True)

        # ========== 8. Zusammenfassung der KPIs: Parameter Set 0 vs. 1 (inkl. 95% KI) ==========
        st.header("Zusammenfassung der KPIs: Parameter Set 0 vs. 1 (inkl. 95% Konfidenzintervall)")

        ordered_labels = ["Parameter Set 0", "Parameter Set 1"]
        farben = {"Parameter Set 1": "royalblue", "Parameter Set 0": "crimson"}

        # --- Transportzeit-Vergleich ---
        kpi_dict = {row["Parameter Set"]: row for _, row in auslastung_day_ci.iterrows()}
        means = [kpi_dict[label]["Mittelwert Transportzeit"] for label in ordered_labels]
        ci_lowers = [kpi_dict[label]["CI lower Transportzeit (95%)"] for label in ordered_labels]
        ci_uppers = [kpi_dict[label]["CI upper Transportzeit (95%)"] for label in ordered_labels]

        st.markdown("#### Mittlere Transportzeit (über alle Tage, 95%-Konfidenzintervall)")
        st.plotly_chart(
            plot_kpi_comparison(
                "Mittlere Transportzeit – Vergleich",
                ordered_labels, means, ci_lowers, ci_uppers, farben
            ),
            use_container_width=True
        )
        st.table(pd.DataFrame({
            "Parameter Set": ordered_labels,
            "Mittelwert": means,
            "95% CI lower": ci_lowers,
            "95% CI upper": ci_uppers
        }))

        # --- Wartezeit-Vergleich ---
        wait_dict = {row["Parameter Set"]: row for _, row in df_day_ci.iterrows()}
        means_wait = [wait_dict[label]["Mittelwert"] for label in ordered_labels]
        ci_lowers_wait = [wait_dict[label]["CI lower (95%)"] for label in ordered_labels]
        ci_uppers_wait = [wait_dict[label]["CI upper (95%)"] for label in ordered_labels]

        st.markdown("#### Mittlere Wartezeit (über alle Tage, 95%-Konfidenzintervall)")
        st.plotly_chart(
            plot_kpi_comparison(
                "Mittlere Wartezeit – Vergleich",
                ordered_labels, means_wait, ci_lowers_wait, ci_uppers_wait, farben
            ),
            use_container_width=True
        )
        st.table(pd.DataFrame({
            "Parameter Set": ordered_labels,
            "Mittelwert": means_wait,
            "95% CI lower": ci_lowers_wait,
            "95% CI upper": ci_uppers_wait
        }))

        # --- Maximale Wartezeit-Vergleich ---
        max_dict = {entry["Parameter Set"]: entry for entry in st.session_state["max_results"]}
        means_max = [max_dict[label]["mean"] for label in ordered_labels]
        ci_lowers_max = [max_dict[label]["ci"][0] for label in ordered_labels]
        ci_uppers_max = [max_dict[label]["ci"][1] for label in ordered_labels]

        st.markdown("#### Maximale Wartezeit (über alle Tage, 95%-Konfidenzintervall)")
        st.plotly_chart(
            plot_kpi_comparison(
                "Maximale Wartezeit – Vergleich",
                ordered_labels, means_max, ci_lowers_max, ci_uppers_max, farben
            ),
            use_container_width=True
        )
        st.table(pd.DataFrame({
            "Parameter Set": ordered_labels,
            "Mittelwert": means_max,
            "95% CI lower": ci_lowers_max,
            "95% CI upper": ci_uppers_max
        }))

    # Nur ausführen, wenn alle nötigen Datenobjekte vorhanden sind
    if (
            "auslastung_summary_tables" in locals()
            and "summary_tables" in locals()
            and "max_results" in st.session_state
            and "auslastung_day_ci" in locals()
            and "df_day_ci" in locals()
    ):
        st.header("Export der finalen Ergebnisdaten (je Parameter Set) für den Vergleich mehrerer Runs")

        for param_set in ordered_labels:
            # 1. Transportzeit stündlich
            df_transport_hourly = auslastung_summary_tables[param_set][[
                "Stunde", "Mittelwert Transportzeit", "CI upper Transportzeit (95%)", "CI lower Transportzeit (95%)"
            ]].rename(columns={
                "Mittelwert Transportzeit": "Transportzeit Mittelwert",
                "CI upper Transportzeit (95%)": "Transportzeit KI max",
                "CI lower Transportzeit (95%)": "Transportzeit KI min"
            })

            # 2. Wartezeit stündlich
            df_wait_hourly = summary_tables[param_set][[
                "Stunde", "Mittelwert", "CI upper (95%)", "CI lower (95%)"
            ]].rename(columns={
                "Mittelwert": "Wartezeit Mittelwert",
                "CI upper (95%)": "Wartezeit KI max",
                "CI lower (95%)": "Wartezeit KI min"
            })

            # 3. Maximale Wartezeit über den Tag (gesamt, nicht stündlich)
            entry_max = next(e for e in st.session_state["max_results"] if e["Parameter Set"] == param_set)
            df_max_wait = pd.DataFrame({
                "Maximale Wartezeit Mittelwert": [entry_max["mean"]],
                "Maximale Wartezeit KI max": [entry_max["ci"][1]],
                "Maximale Wartezeit KI min": [entry_max["ci"][0]]
            })

            # 4. Ressourcenauslastung über den Tag (gesamt, nicht stündlich)
            entry_util = auslastung_day_ci[auslastung_day_ci["Parameter Set"] == param_set].iloc[0]
            df_util = pd.DataFrame({
                "Ressourcenauslastung Mittelwert": [entry_util["Mittelwert Auslastung"]],
                "Ressourcenauslastung KI max": [entry_util["CI upper Auslastung (95%)"]],
                "Ressourcenauslastung KI min": [entry_util["CI lower Auslastung (95%)"]],
            })

            # 5. Tagesmaxima der Transportzeit für das Parameter Set
            daily_max = [
                entry["Daily Transport Max (s)"]
                for entry in auslastung_results
                if entry["Parameter Set"] == param_set
            ]
            # Statistische Auswertung für Tagesmaxima (Tabelle)
            mean_transport = np.mean(daily_max)
            n = len(daily_max)
            std_transport = np.std(daily_max, ddof=1) if n > 1 else 0
            alpha = 0.05
            t_val = t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else 0
            se = std_transport / np.sqrt(n) if n > 1 else 0
            ci_lower = mean_transport - t_val * se
            ci_upper = mean_transport + t_val * se

            df_max_transport = pd.DataFrame({
                "Maximale Transportzeit Mittelwert": [mean_transport],
                "Maximale Transportzeit KI max": [ci_upper],
                "Maximale Transportzeit KI min": [ci_lower]
            })

            # --- Schreibe alle Ergebnisse in eine Excel-Datei mit mehreren Tabellenblättern ---
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_transport_hourly.to_excel(writer, sheet_name="Transportzeit_Std", index=False)
                df_wait_hourly.to_excel(writer, sheet_name="Wartezeit_Std", index=False)
                df_max_wait.to_excel(writer, sheet_name="MaxWartezeit_Tag", index=False)
                df_util.to_excel(writer, sheet_name="Ressourcenauslastung_Tag", index=False)
                df_max_transport.to_excel(writer, sheet_name="TransportMax_Tag", index=False)
            output.seek(0)

            st.download_button(
                label=f"Exportiere Ergebnisse: {param_set}",
                data=output,
                file_name=f"Export_{param_set.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Exportiert alle wichtigen KPIs für dieses Parameter Set in separaten Tabellenblättern."
            )


        # ========== Kompakte Tagweise KPI-Tabelle (Mittelwerte & Maximum) ==========
        ordered_labels = ["Parameter Set 0", "Parameter Set 1"]
        num_days = max(int(''.join(filter(str.isdigit, str(e["Day"])))) for e in auslastung_results)

        # Tagesmaxima sauber auslesen für alle Sets
        max_results_dict = {entry["Parameter Set"]: entry["max_per_day"] for entry in st.session_state["max_results"]}

        rows = []
        row_labels = []

        for day in range(1, num_days + 1):
            # 1. Ressourcenauslastung (Mittelwert)
            label = f"Tag {day} - Ressourcenauslastung (Mittelwert)"
            row = []
            for param_set in ordered_labels:
                entry = next(
                    (e for e in auslastung_results if e['Parameter Set'] == param_set and day_matches(e['Day'], day)),
                    None)
                val = entry.get('Daily Util Mean (%)', np.nan) if entry else np.nan
                row.append(val)
            rows.append(row)
            row_labels.append(label)

            # 2. Wartezeit (Mittelwert)
            label = f"Tag {day} - Wartezeit (Mittelwert)"
            row = []
            for param_set in ordered_labels:
                entry = next((e for e in results if e['Parameter Set'] == param_set and day_matches(e['Day'], day)),
                             None)
                val = entry.get('Daily Mean', np.nan) if entry else np.nan
                row.append(val)
            rows.append(row)
            row_labels.append(label)

            # 3. Wartezeit (Maximum)
            label = f"Tag {day} - Wartezeit (Maximum)"
            row = []
            for param_set in ordered_labels:
                try:
                    val = max_results_dict[param_set][day - 1]
                except Exception:
                    val = np.nan
                row.append(val)
            rows.append(row)
            row_labels.append(label)

            # 4. Transportzeit (Mittelwert)
            label = f"Tag {day} - Transportzeit (Mittelwert)"
            row = []
            for param_set in ordered_labels:
                entry = next(
                    (e for e in auslastung_results if e['Parameter Set'] == param_set and day_matches(e['Day'], day)),
                    None)
                val = entry.get('Daily Transport Mean (s)', np.nan) if entry else np.nan
                row.append(val)
            rows.append(row)
            row_labels.append(label)

        df_compact = pd.DataFrame(rows, columns=ordered_labels, index=row_labels)

        st.header("Tabellarische Übersicht – Tagweise KPIs (Mittelwerte & Tagesmaximum Wartezeit)")
        st.dataframe(df_compact.style.format("{:.2f}"), use_container_width=True)

        # Optional: Excel-Export der Übersicht
        output = io.BytesIO()
        df_compact.to_excel(output, sheet_name="KPIs_Tag_Uebersicht")
        output.seek(0)
        st.download_button(
            label="Tabelle exportieren",
            data=output,
            file_name="TagKPIs_Uebersicht.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

