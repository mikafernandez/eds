import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import io

# --- Seitenkonfiguration und Titel ---
st.set_page_config(page_title="Vergleich von Ergebnissen", layout="wide")
st.title("🔬 Vergleich von Ergebnissen aus mehreren Simulationen")
st.markdown("""
Hier kannst du mehrere Ergebnisdateien (Excel) hochladen, jeweils einen Datensatznamen vergeben und anschließend die KPIs oder Zeitreihen unterschiedlicher Runs vergleichen.
""")

# --- Konsistentes Farbschema für die spätere Visualisierung definieren ---
default_colors = [
    "royalblue", "crimson", "darkorange", "seagreen", "violet", "black",
    "deepskyblue", "goldenrod", "deeppink", "mediumslateblue"
]

# --- Zeitlabels für die Achsenbeschriftung generieren (z.B. 06:00–07:00, ..., 17:00–18:00) ---
time_labels = [f"{6 + i:02d}:00–{7 + i:02d}:00" for i in range(12)]

# --- Hilfsfunktion zur Umwandlung von Farbnamen in RGBA-Farbcodes mit Transparenz für Plotly-Flächen ---
def to_rgba(colorname, alpha=0.13):
    """
    Wandelt einen Farbnamen oder Hexcode in einen RGBA-String für Plotly um.
    Ermöglicht transparente Flächen (z.B. Konfidenzintervall).
    """
    try:
        rgb = mcolors.to_rgb(colorname)
        return f'rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},{alpha})'
    except ValueError:
        if colorname.startswith("#") or colorname.startswith("rgb"):
            return colorname
        raise

# --- Datei-Upload: Mehrere Excel-Dateien einlesen und benennen ---
st.header("1️⃣ Excel-Exports hochladen & benennen")
uploaded_files = st.file_uploader(
    "Wähle eine oder mehrere Export-Dateien im Excel-Format (.xlsx) aus:",
    type=["xlsx"],
    accept_multiple_files=True,
    help="Die Dateien sollten im Export-Format vorliegen (je Parameter Set eine Datei)."
)

datasets = []
if uploaded_files:
    st.markdown("#### Vergib einen Namen für jeden Datensatz:")
    for i, uploaded_file in enumerate(uploaded_files):
        with st.expander(f"Datensatz {i+1}: {uploaded_file.name}", expanded=True):
            default_name = f"Datensatz_{i+1}"
            dataset_name = st.text_input(
                f"Name für Datensatz {i+1} ({uploaded_file.name}):",
                value=default_name,
                key=f"dataset_name_{i}"
            )
            try:
                # Vorschau: Anzeige der ersten Zeilen der hochgeladenen Datei
                df = pd.read_excel(uploaded_file, sheet_name=0)
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.warning(f"Datei konnte nicht gelesen werden: {e}")

            # Farbzuordnung für spätere Plots
            farbe = default_colors[i % len(default_colors)]
            datasets.append({
                "name": dataset_name,
                "filename": uploaded_file.name,
                "file": uploaded_file,
                "color": farbe
            })
    st.success(f"{len(datasets)} Datensätze geladen und benannt.")

# --- Strukturierte Einlesung aller benötigten Datenblätter aus den Dateien ---
data_loaded = []
if datasets:
    for ds in datasets:
        excel = pd.ExcelFile(ds["file"])
        df_transport = pd.read_excel(excel, sheet_name="Transportzeit_Std")
        df_wait = pd.read_excel(excel, sheet_name="Wartezeit_Std")
        df_max = pd.read_excel(excel, sheet_name="MaxWartezeit_Tag")
        df_util = pd.read_excel(excel, sheet_name="Ressourcenauslastung_Tag")
        data_loaded.append({
            "name": ds["name"],
            "color": ds["color"],
            "transport": df_transport,
            "wait": df_wait,
            "max_wait": df_max,
            "util": df_util
        })

# --- Hilfsfunktion: CI-Punktplot für mehrere Datensätze (z.B. Transportzeit, Wartezeit, etc.) ---
def plot_ci_points_multicolor(names, means, ci_mins, ci_maxs, farben, x_title, main_title):
    """
    Erstellt einen horizontalen Punktplot mit Konfidenzintervall-Balken für mehrere Datensätze.
    Jeder Datensatz erhält eine eigene Farbe.
    """
    fig = go.Figure()
    y_pos = list(reversed(range(len(names))))
    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(
            x=[means[i]], y=[y_pos[i]],
            mode="markers",
            marker=dict(size=16, color=farben[i], symbol="circle"),
            error_x=dict(
                type='data',
                symmetric=False,
                array=[ci_maxs[i] - means[i]],
                arrayminus=[means[i] - ci_mins[i]],
                thickness=4,
                width=0,
                color=farben[i]
            ),
            name=name,
            showlegend=False,
            text=[name],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Mittelwert: %{x:.2f}<br>"
                f"KI min: {ci_mins[i]:.2f}<br>"
                f"KI max: {ci_maxs[i]:.2f}<extra></extra>"
            )
        ))
    fig.update_yaxes(
        tickvals=y_pos,
        ticktext=names,
        title=None,
        showgrid=False,
        autorange="reversed"
    )
    fig.update_layout(
        title=main_title,
        xaxis_title=x_title,
        margin=dict(l=120, r=50, t=60, b=40),
        height=80 + 60 * len(names),
        template="simple_white"
    )
    return fig

# ================== VISUALISIERUNG DER KENNZAHLEN (KPIs) ==================
st.header("2️⃣ KPI-Vergleich: Zeitreihen und Intervalle")

if data_loaded:
    # --- Plot 1: Transportzeit im Tagesverlauf (mit Konfidenzintervall) ---
    st.subheader("Transportzeit: Verlauf von Stunde 1 bis 12 (mit Konfidenzintervall)")
    fig = go.Figure()
    y_vals = []
    for ds in data_loaded:
        # Optional: Filter für "10" in Name, je nach Use Case anpassbar
        if "10" in ds["name"]:
            df = ds["transport"]
            y_vals += list(df["Transportzeit Mittelwert"].values)
            y_vals += list(df["Transportzeit KI max"].values)
            y_vals += list(df["Transportzeit KI min"].values)
            # Mittelwert-Linie
            fig.add_trace(go.Scatter(
                x=df["Stunde"],
                y=df["Transportzeit Mittelwert"],
                mode="lines+markers",
                name=ds["name"],
                line=dict(width=2, color=ds["color"]),
                marker=dict(color=ds["color"]),
            ))
            # Schraffiertes KI-Band
            fig.add_trace(go.Scatter(
                x=pd.concat([df["Stunde"], df["Stunde"][::-1]]),
                y=pd.concat([df["Transportzeit KI max"], df["Transportzeit KI min"][::-1]]),
                fill="toself",
                fillcolor=to_rgba(ds["color"], alpha=0.13),
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name=f"KI {ds['name']}",
                showlegend=True
            ))
    # Bereich für die y-Achse bestimmen (Puffer für bessere Lesbarkeit)
    y_range = [min(y_vals) - 5, max(y_vals) + 5] if y_vals else None
    fig.update_layout(
        xaxis=dict(
            title="Tageszeit",
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=time_labels
        ),
        yaxis=dict(
            title="Transportzeit [s]",
            dtick=5,
            range=y_range
        ),
        title="Mittlere Transportzeit pro Person im Tagesverlauf",
        template="plotly_white",
        legend=dict(
            orientation="v",
            x=1, xanchor="left",
            y=0.5, yanchor="middle"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Export der zugrundeliegenden Plott-Daten (Transportzeit) ---
    df_list = []
    for ds in data_loaded:
        if "10" in ds["name"]:
            df_temp = ds["transport"].copy()
            df_temp["Versuch"] = ds["name"]
            df_list.append(df_temp)
    df_export = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    with st.expander("📋 Verwendete Tabellendaten anzeigen und exportieren"):
        if not df_export.empty:
            st.dataframe(df_export, use_container_width=True)
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Plott-Daten')
            st.download_button(
                label="📥 Tabelle als Excel herunterladen",
                data=excel_buffer.getvalue(),
                file_name="PlottDaten_Transportzeit.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("Keine Daten zum Anzeigen/Exportieren vorhanden.")

    # --- Plot 2: Wartezeit im Tagesverlauf (mit Konfidenzintervall) ---
    st.subheader("Wartezeit: Verlauf von Stunde 1 bis 12 (mit Konfidenzintervall)")
    fig2 = go.Figure()
    for ds in data_loaded:
        if "10" in ds["name"]:
            df = ds["wait"]
            # Mittelwert-Linie
            fig2.add_trace(go.Scatter(
                x=df["Stunde"],
                y=df["Wartezeit Mittelwert"],
                mode="lines+markers",
                name=ds["name"],
                line=dict(width=2, color=ds["color"]),
                marker=dict(color=ds["color"])
            ))
            # Schraffiertes KI-Band
            fig2.add_trace(go.Scatter(
                x=pd.concat([df["Stunde"], df["Stunde"][::-1]]),
                y=pd.concat([df["Wartezeit KI max"], df["Wartezeit KI min"][::-1]]),
                fill="toself",
                fillcolor=to_rgba(ds["color"], alpha=0.13),
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name=f"KI {ds['name']}",
                showlegend=True
            ))
    fig2.update_layout(
        xaxis=dict(
            title="Tageszeit",
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=time_labels
        ),
        yaxis_title="Wartezeit [s]",
        title="Mittlere Wartezeit pro Person im Tagesverlauf",
        template="plotly_white",
        legend=dict(
            orientation="v",
            x=1,
            xanchor="left",
            y=0.5,
            yanchor="middle"
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Export der Plott-Daten (Wartezeit) ---
    wait_df_list = []
    for ds in data_loaded:
        if "10" in ds["name"]:
            wait_df = ds["wait"].copy()
            wait_df["Versuch"] = ds["name"]
            wait_df_list.append(wait_df)
    wait_export = pd.concat(wait_df_list, ignore_index=True) if wait_df_list else pd.DataFrame()

    with st.expander("📋 Verwendete Tabellendaten anzeigen und exportieren (Wartezeit)"):
        if not wait_export.empty:
            st.dataframe(wait_export, use_container_width=True)
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                wait_export.to_excel(writer, index=False, sheet_name='Plott-Daten')
            st.download_button(
                label="📥 Tabelle als Excel herunterladen",
                data=excel_buffer.getvalue(),
                file_name="PlottDaten_Wartezeit.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("Keine Daten zum Anzeigen/Exportieren vorhanden.")

    # --- Plot 3: Tages-KIs der mittleren Transportzeit (Punktplot mit CI) ---
    st.subheader("Konfidenzintervalle der mittleren Transportzeit")
    names, means, ci_mins, ci_maxs, farben = [], [], [], [], []
    for ds in data_loaded:
        df = ds["transport"]
        names.append(ds["name"])
        means.append(df["Transportzeit Mittelwert"].mean())
        ci_mins.append(df["Transportzeit KI min"].mean())
        ci_maxs.append(df["Transportzeit KI max"].mean())
        farben.append(ds["color"])
    fig_ci_transport = plot_ci_points_multicolor(
        names, means, ci_mins, ci_maxs, farben,
        x_title="Mittlere Transportzeit [s]",
        main_title="Konfidenzintervalle Mittlere Transportzeit pro Person mit Signifikanzniveau von 5%"
    )
    st.plotly_chart(fig_ci_transport, use_container_width=True)

    # --- Plot 4: Tages-KIs der mittleren Wartezeit ---
    st.subheader("Konfidenzintervalle der mittleren Wartezeit")
    names, means, ci_mins, ci_maxs, farben = [], [], [], [], []
    for ds in data_loaded:
        df = ds["wait"]
        names.append(ds["name"])
        means.append(df["Wartezeit Mittelwert"].mean())
        ci_mins.append(df["Wartezeit KI min"].mean())
        ci_maxs.append(df["Wartezeit KI max"].mean())
        farben.append(ds["color"])
    fig_ci_wait = plot_ci_points_multicolor(
        names, means, ci_mins, ci_maxs, farben,
        x_title="Mittlere Wartezeit [s]",
        main_title="Konfidenzintervalle Mittlere Wartezeit pro Person mit Signifikanzniveau von 5%"
    )
    st.plotly_chart(fig_ci_wait, use_container_width=True)

    # --- Plot 5: Tages-KIs der mittleren Ressourcenauslastung ---
    st.subheader("Konfidenzintervalle der mittleren Auslastung")
    names, means, ci_mins, ci_maxs, farben = [], [], [], [], []
    for ds in data_loaded:
        df = ds["util"]
        names.append(ds["name"])
        means.append(df["Ressourcenauslastung Mittelwert"].mean())
        ci_mins.append(df["Ressourcenauslastung KI min"].mean())
        ci_maxs.append(df["Ressourcenauslastung KI max"].mean())
        farben.append(ds["color"])
    fig_ci_util = plot_ci_points_multicolor(
        names, means, ci_mins, ci_maxs, farben,
        x_title="Ressourcenauslastung [%]",
        main_title="Konfidenzintervalle Mittlere Aufzugsauslastung mit Signifikanzniveau von 5%"
    )
    st.plotly_chart(fig_ci_util, use_container_width=True)

    # --- Plot 6: Tages-KIs der maximalen Wartezeit ---
    st.subheader("Konfidenzintervalle der maximalen Wartezeit")
    names, means, ci_mins, ci_maxs, farben = [], [], [], [], []
    for ds in data_loaded:
        df = ds["max_wait"]
        names.append(ds["name"])
        means.append(df["Maximale Wartezeit Mittelwert"].mean())
        ci_mins.append(df["Maximale Wartezeit KI min"].mean())
        ci_maxs.append(df["Maximale Wartezeit KI max"].mean())
        farben.append(ds["color"])
    fig_ci_maxwait = plot_ci_points_multicolor(
        names, means, ci_mins, ci_maxs, farben,
        x_title="Maximale Wartezeit [s]",
        main_title="Konfidenzintervalle Maximale Wartezeit mit Signifikanzniveau von 5%"
    )
    # Orientierungslinie bei z.B. 180 Sekunden
    fig_ci_maxwait.add_vline(
        x=180,
        line_color="red",
        line_width=2,
        line_dash="dash"
    )
    st.plotly_chart(fig_ci_maxwait, use_container_width=True)
