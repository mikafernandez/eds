import numpy as np
import pandas as pd
from scipy.stats import t
import scipy.stats as stats




# ---------------------------------------------------------------
# Funktion: process_max_waiting_times
# Zweck: Extrahiert und analysiert die maximalen Wartezeiten aus einer Übersichtstabelle je Simulationslauf.
# Die Funktion liest die maximalen Werte aus den jeweiligen Spalten der Excel-Datei,
# berechnet den Mittelwert, das Konfidenzintervall und das absolute Maximum für jeden Parametersatz.
# ---------------------------------------------------------------
def process_max_waiting_times(filepath: str, anzahl_simulationslaeufe: int):
    df = pd.read_excel(filepath, sheet_name="Ubersicht", header=None)

    # Definiert die zugehörigen Spalten für beide Parameterkonfigurationen je nach Simulationsanzahl
    if anzahl_simulationslaeufe == 5:
        param_1_cols = [1, 7, 13, 19, 25]
        param_0_cols = [4, 10, 16, 22, 28]
    elif anzahl_simulationslaeufe == 10:
        param_1_cols = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55]
        param_0_cols = [4, 10, 16, 22, 28, 34, 40, 46, 52, 58]
    elif anzahl_simulationslaeufe == 15:
        param_1_cols = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85]
        param_0_cols = [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88]
    else:
        raise ValueError("Nur 5, 10 oder 15 Simulationsläufe werden unterstützt!")

    rows = list(range(3, 18))  # Betrachtete Zeilen (z. B. Tage 1-15)
    results = []
    for param_label, cols in zip(["Parameter Set 1", "Parameter Set 0"], [param_1_cols, param_0_cols]):
        max_per_day = []
        for day, col in enumerate(cols, 1):
            values = df.iloc[rows, col]
            max_val = np.nanmax(values)
            max_per_day.append(max_val)
        mean = np.mean(max_per_day)
        var = np.var(max_per_day, ddof=1)
        ci = t.interval(0.95, df=len(max_per_day) - 1, loc=mean,
                        scale=np.std(max_per_day, ddof=1) / np.sqrt(len(max_per_day)))
        max_of_max = np.max(max_per_day)
        anzahl = len(max_per_day)
        data = {
            "Tag": [f"Tag {i}" for i in range(1, anzahl + 1)] + ["Mittelwert", "95%-CI", f"Max der {anzahl} Tage"],
            "Max. Wartezeit (s)": [round(val, 2) for val in max_per_day] + [round(mean, 2),
                                                                            f"{ci[0]:.2f} – {ci[1]:.2f}",
                                                                            round(max_of_max, 2)]
        }
        result_df = pd.DataFrame(data)
        results.append({
            "Parameter Set": param_label,
            "table": result_df,
            "max_per_day": max_per_day,
            "mean": mean,
            "ci": ci
        })
    return results


# ---------------------------------------------------------------
# Funktion: process_utilization_excel_auslastung
# Zweck: Extrahiert und aggregiert für jeden Tag und jeden Aufzug Auslastung, Transportzeiten,
# und Personen pro Stunde, inklusive Verteilungs- und Histogrammdaten.
# ---------------------------------------------------------------
def process_utilization_excel_auslastung(file_path: str, elevator_anzahl: int, anzahl_simulationslaeufe: int):
    """
    Liest alle Simulations-Sheets ein und berechnet pro Aufzug und Stunde Mittelwerte der
    Auslastung, Transportzeit und Personenanzahl. Zusätzlich werden pro Stunde die Transportzeitverteilungen
    (für Histogramme) extrahiert.
    """
    # Sheet-Mapping je nach Anzahl Simulationsläufe
    if anzahl_simulationslaeufe == 5:
        sheet_map = {
            "Parameter Set 1": ["Sheet11", "Sheet13", "Sheet15", "Sheet17", "Sheet19"],
            "Parameter Set 0": ["Sheet12", "Sheet14", "Sheet16", "Sheet18", "Sheet20"]
        }
    elif anzahl_simulationslaeufe == 10:
        sheet_map = {
            "Parameter Set 1": ["Sheet23", "Sheet25", "Sheet27", "Sheet29", "Sheet31",
                                "Sheet33", "Sheet35", "Sheet37", "Sheet39", "Sheet41"],
            "Parameter Set 0": ["Sheet24", "Sheet26", "Sheet28", "Sheet30", "Sheet32",
                                "Sheet34", "Sheet36", "Sheet38", "Sheet40", "Sheet42"]
        }
    elif anzahl_simulationslaeufe == 15:
        sheet_map = {
            "Parameter Set 1": ["Sheet33", "Sheet35", "Sheet37", "Sheet39", "Sheet41",
                                "Sheet43", "Sheet45", "Sheet47", "Sheet49", "Sheet51",
                                "Sheet53", "Sheet55", "Sheet57", "Sheet59", "Sheet61"],
            "Parameter Set 0": ["Sheet34", "Sheet36", "Sheet38", "Sheet40", "Sheet42",
                                "Sheet44", "Sheet46", "Sheet48", "Sheet50", "Sheet52",
                                "Sheet54", "Sheet56", "Sheet58", "Sheet60", "Sheet62"]
        }
    else:
        raise ValueError("Nur 5, 10 oder 15 Simulationsläufe unterstützt!")

    hours = 12  # Annahme: 12 Stunden werden je Tag betrachtet
    hour_limits = [(h * 3600, (h + 1) * 3600) for h in range(hours)]
    processed_data = []

    # Index-Offsets pro Aufzug im Sheet
    base_idxs = {
        "idle": (0, 1),
        "transport": (4, 5),
        "person": (8, 9)
    }
    elevator_jump = 11  # Spaltenversatz pro Aufzug im Sheet

    for param_set, sheets in sheet_map.items():
        for day_index, sheet_name in enumerate(sheets, start=1):
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            elevator_hourly_util = [[] for _ in range(elevator_anzahl)]
            elevator_hourly_trans = [[] for _ in range(elevator_anzahl)]
            elevator_hourly_pers = [[] for _ in range(elevator_anzahl)]
            hourly_transport_values = [[] for _ in range(hours)]  # Für Histogramme

            for idx in range(elevator_anzahl):
                offset = idx * elevator_jump
                # Idle-Events extrahieren und auf Stunde aggregieren
                ts_idle, idle_col = offset + base_idxs["idle"][0], offset + base_idxs["idle"][1]
                idle_data = df.iloc[2:, [ts_idle, idle_col]].dropna(how='all')
                idle_data = idle_data.dropna(subset=[ts_idle])
                timestamps_idle = pd.to_numeric(idle_data.iloc[:, 0], errors='coerce')
                idle_events = pd.to_numeric(idle_data.iloc[:, 1], errors='coerce').fillna(0)
                # Transportzeiten
                ts_trans, trans_col = offset + base_idxs["transport"][0], offset + base_idxs["transport"][1]
                trans_data = df.iloc[2:, [ts_trans, trans_col]].dropna(how='all')
                trans_data = trans_data.dropna(subset=[ts_trans])
                timestamps_trans = pd.to_numeric(trans_data.iloc[:, 0], errors='coerce')
                transport_times = pd.to_numeric(trans_data.iloc[:, 1], errors='coerce').fillna(np.nan)
                # Personen
                ts_pers, pers_col = offset + base_idxs["person"][0], offset + base_idxs["person"][1]
                pers_data = df.iloc[2:, [ts_pers, pers_col]].dropna(how='all')
                pers_data = pers_data.dropna(subset=[ts_pers])
                timestamps_pers = pd.to_numeric(pers_data.iloc[:, 0], errors='coerce')
                person_count = pd.to_numeric(pers_data.iloc[:, 1], errors='coerce').fillna(np.nan)

                # Stündliche Aggregation
                for h, (start_sec, end_sec) in enumerate(hour_limits):
                    # Idle-Zeit
                    mask_idle = (timestamps_idle >= start_sec) & (timestamps_idle < end_sec)
                    if mask_idle.any():
                        idle_in_hour = idle_events[mask_idle].sum()
                        util = max(0.0, min(1.0, 1 - idle_in_hour / 3600))
                        elevator_hourly_util[idx].append(util * 100)
                    else:
                        elevator_hourly_util[idx].append(np.nan)
                    # Transportzeit (auch Einzelwerte für Histogramm)
                    mask_trans = (timestamps_trans >= start_sec) & (timestamps_trans < end_sec)
                    if mask_trans.any():
                        mean_trans = transport_times[mask_trans].mean()
                        elevator_hourly_trans[idx].append(mean_trans)
                        hourly_transport_values[h].extend(transport_times[mask_trans].tolist())
                    else:
                        elevator_hourly_trans[idx].append(np.nan)
                    # Personenanzahl
                    mask_pers = (timestamps_pers >= start_sec) & (timestamps_pers < end_sec)
                    if mask_pers.any():
                        mean_pers = person_count[mask_pers].mean()
                        elevator_hourly_pers[idx].append(mean_pers)
                    else:
                        elevator_hourly_pers[idx].append(np.nan)

            # Mittelwerte über alle Aufzüge (je Stunde)
            mean_hourly_util = [np.nanmean([elevator_hourly_util[el][h] for el in range(elevator_anzahl)]) for h in
                                range(hours)]
            mean_hourly_trans = [np.nanmean([elevator_hourly_trans[el][h] for el in range(elevator_anzahl)]) for h in
                                 range(hours)]
            mean_hourly_pers = [np.nanmean([elevator_hourly_pers[el][h] for el in range(elevator_anzahl)]) for h in
                                range(hours)]
            daily_mean_util = np.nanmean(mean_hourly_util)
            daily_mean_trans = np.nanmean(mean_hourly_trans)
            daily_mean_pers = np.nanmean(mean_hourly_pers)

            # Tagesmaximum Transportzeiten (über alle Stunden und Aufzüge)
            all_transport_values = [v for sublist in hourly_transport_values for v in sublist if not np.isnan(v)]
            daily_max_trans = np.nanmax(all_transport_values) if all_transport_values else np.nan

            # Histogramme pro Stunde berechnen (für Visualisierung)
            hourly_histograms = []
            for values in hourly_transport_values:
                arr = np.array(values)
                arr = arr[~np.isnan(arr)]
                if len(arr) > 0:
                    bin_count = min(30, max(5, int(np.ceil(np.sqrt(len(arr))))))
                    hist, bin_edges = np.histogram(arr, bins=bin_count)
                    hourly_histograms.append({"counts": hist.tolist(), "bin_edges": bin_edges.tolist()})
                else:
                    hourly_histograms.append({"counts": [], "bin_edges": []})

            # Ergebnisspeicherung je Tag und Parameterset
            processed_data.append({
                "Parameter Set": param_set,
                "Day": f"Day {day_index}",
                "Hourly Util Means (%)": mean_hourly_util,
                "Daily Util Mean (%)": daily_mean_util,
                "Hourly Transport Means (s)": mean_hourly_trans,
                "Daily Transport Mean (s)": daily_mean_trans,
                "Daily Transport Max (s)": daily_max_trans,
                "Hourly Transport Values (s)": hourly_transport_values,
                "Hourly Transport Histograms": hourly_histograms,
                "Hourly Persons Means": mean_hourly_pers,
                "Daily Persons Mean": daily_mean_pers
            })

    return processed_data


# ---------------------------------------------------------------
# Funktion: summarize_hourly_stats_extended
# Zweck: Aggregiert und berechnet Mittelwerte, Varianzen und 95%-Konfidenzintervalle für
# Auslastung, Transportzeit und Personenanzahl je Stunde und Parameterset.
# ---------------------------------------------------------------
def summarize_hourly_stats_extended(results):
    param_sets = sorted(set(entry["Parameter Set"] for entry in results))
    out = {}
    for param_set in param_sets:
        subset = [e for e in results if e["Parameter Set"] == param_set]
        hours = len(subset[0]["Hourly Util Means (%)"])
        data_util = np.array([e["Hourly Util Means (%)"] for e in subset])
        data_trans = np.array([e["Hourly Transport Means (s)"] for e in subset])
        data_pers = np.array([e["Hourly Persons Means"] for e in subset])
        rows = []
        for h in range(hours):
            vals_util = data_util[:, h]
            vals_trans = data_trans[:, h]
            vals_pers = data_pers[:, h]

            # Statistische Kennwerte + 95%-Konfidenzintervall (t-Verteilung)
            def ci(values):
                n = np.count_nonzero(~np.isnan(values))
                if n <= 1:
                    return (np.nan, np.nan)
                mean = np.nanmean(values)
                sem = np.nanstd(values, ddof=1) / np.sqrt(n)
                return t.interval(0.95, n - 1, loc=mean, scale=sem)

            mean_util, var_util = np.nanmean(vals_util), np.nanvar(vals_util, ddof=1)
            ci_util = ci(vals_util)
            mean_trans, var_trans = np.nanmean(vals_trans), np.nanvar(vals_trans, ddof=1)
            ci_trans = ci(vals_trans)
            mean_pers, var_pers = np.nanmean(vals_pers), np.nanvar(vals_pers, ddof=1)
            ci_pers = ci(vals_pers)
            rows.append({
                "Stunde": h + 1,
                "Mittelwert Auslastung": mean_util,
                "Varianz Auslastung": var_util,
                "CI lower Auslastung (95%)": ci_util[0],
                "CI upper Auslastung (95%)": ci_util[1],
                "Mittelwert Transportzeit": mean_trans,
                "Varianz Transportzeit": var_trans,
                "CI lower Transportzeit (95%)": ci_trans[0],
                "CI upper Transportzeit (95%)": ci_trans[1],
                "Mittelwert Personen": mean_pers,
                "Varianz Personen": var_pers,
                "CI lower Personen (95%)": ci_pers[0],
                "CI upper Personen (95%)": ci_pers[1],
            })
        out[param_set] = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------
# Funktion: compute_confidence_intervals_extended
# Zweck: Berechnet für jede Parameterkonfiguration Mittelwert, Varianz und 95%-Konfidenzintervall
# für Tagesmittelwerte (Auslastung, Transportzeit, Maximum Transportzeit, Personen).
# ---------------------------------------------------------------
def compute_confidence_intervals_extended(results):
    param_sets = sorted(set(entry["Parameter Set"] for entry in results))
    rows = []
    for param_set in param_sets:
        subset = [e for e in results if e["Parameter Set"] == param_set]

        # Kennwerte und KI je Metrik
        def stat(values):
            arr = np.array(values)
            mean = np.nanmean(arr)
            var = np.nanvar(arr, ddof=1)
            n = np.count_nonzero(~np.isnan(arr))
            sem = np.nanstd(arr, ddof=1) / np.sqrt(n) if n > 1 else np.nan
            ci = t.interval(0.95, n - 1, loc=mean, scale=sem) if n > 1 else (np.nan, np.nan)
            return mean, var, ci

        mean_util, var_util, ci_util = stat([e["Daily Util Mean (%)"] for e in subset])
        mean_trans, var_trans, ci_trans = stat([e["Daily Transport Mean (s)"] for e in subset])
        mean_max_trans, var_max_trans, ci_max_trans = stat([e["Daily Transport Max (s)"] for e in subset])
        mean_pers, var_pers, ci_pers = stat([e["Daily Persons Mean"] for e in subset])
        rows.append({
            "Parameter Set": param_set,
            "Mittelwert Auslastung": mean_util,
            "Varianz Auslastung": var_util,
            "CI lower Auslastung (95%)": ci_util[0],
            "CI upper Auslastung (95%)": ci_util[1],
            "Mittelwert Transportzeit": mean_trans,
            "Varianz Transportzeit": var_trans,
            "CI lower Transportzeit (95%)": ci_trans[0],
            "CI upper Transportzeit (95%)": ci_trans[1],
            "Mittelwert max. Transportzeit": mean_max_trans,
            "Varianz max. Transportzeit": var_max_trans,
            "CI lower max. Transportzeit (95%)": ci_max_trans[0],
            "CI upper max. Transportzeit (95%)": ci_max_trans[1],
            "Mittelwert Personen": mean_pers,
            "Varianz Personen": var_pers,
            "CI lower Personen (95%)": ci_pers[0],
            "CI upper Personen (95%)": ci_pers[1],
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------
# Funktion: process_uploaded_excel_custom
# Zweck: Extrahiert für jede Simulation (Sheet) die stündlichen Mittelwerte der Wartezeiten,
# indem aus allen 16 Etagen (je 3-Spalten-Blöcke) je Stunde der Mittelwert berechnet wird.
# ---------------------------------------------------------------
def process_uploaded_excel_custom(file_path: str, anzahl_simulationslaeufe):
    """
    Extrahiert stündliche Mittelwerte der Wartezeit für jeden Tag und Parametersatz.
    Annahme: 16 Floors pro Sheet, pro Floor ein 3-Spalten-Block.
    """
    if anzahl_simulationslaeufe == 5:
        sheet_map = {
            "Parameter Set 1": ["Sheet1", "Sheet3", "Sheet5", "Sheet7", "Sheet9"],
            "Parameter Set 0": ["Sheet2", "Sheet4", "Sheet6", "Sheet8", "Sheet10"]
        }
    elif anzahl_simulationslaeufe == 10:
        sheet_map = {
            "Parameter Set 1": ["Sheet1", "Sheet3", "Sheet5", "Sheet7", "Sheet9",
                                "Sheet11", "Sheet13", "Sheet15", "Sheet17", "Sheet19"],
            "Parameter Set 0": ["Sheet2", "Sheet4", "Sheet6", "Sheet8", "Sheet10",
                                "Sheet12", "Sheet14", "Sheet16", "Sheet18", "Sheet20"]
        }
    elif anzahl_simulationslaeufe == 15:
        sheet_map = {
            "Parameter Set 1": ["Sheet1", "Sheet3", "Sheet5", "Sheet7", "Sheet9",
                                "Sheet11", "Sheet13", "Sheet15", "Sheet17", "Sheet19",
                                "Sheet21", "Sheet23", "Sheet25", "Sheet27", "Sheet29"],
            "Parameter Set 0": ["Sheet2", "Sheet4", "Sheet6", "Sheet8", "Sheet10",
                                "Sheet12", "Sheet14", "Sheet16", "Sheet18", "Sheet20",
                                "Sheet22", "Sheet24", "Sheet26", "Sheet28", "Sheet30"]
        }
    else:
        raise ValueError("Nur 5, 10 oder 15 Simulationsläufe werden unterstützt!")

    # Spalten der 16 Etagen: Jeweils jede 3. Spalte ab Index 1 (Excel-Index, Null-basiert)
    column_indices = [1 + 3 * i for i in range(16)]
    processed_data = []

    for param_set, sheets in sheet_map.items():
        for day_index, sheet_name in enumerate(sheets, start=1):
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            hourly_means = []
            # Zeilen 3 bis 14: Stunden 1–12 (Excel: Zeilenindex 2 bis 13)
            for row_index in range(2, 14):
                values = []
                for col in column_indices:
                    if col < df.shape[1]:
                        values.append(df.iloc[row_index, col])
                mean_val = pd.to_numeric(values, errors="coerce").mean()
                hourly_means.append(mean_val)
            daily_mean = pd.Series(hourly_means).mean(skipna=True)
            processed_data.append({
                "Parameter Set": param_set,
                "Day": f"Day {day_index}",
                "Hourly Means": hourly_means,
                "Daily Mean": daily_mean
            })
    return processed_data

# ---------------------------------------------------------------
# Funktion: compute_confidence_intervals
# Zweck: Berechnet für jeden Parametersatz das 95%-Konfidenzintervall, den Mittelwert und die Varianz der Tagesmittelwerte.
# ---------------------------------------------------------------
def compute_confidence_intervals(results, alpha=0.05):
    """
    Für jeden Parametersatz wird das 95%-Konfidenzintervall sowie Mittelwert und Varianz der Tagesmittelwerte berechnet.
    """
    summary = []
    grouped = {}
    for entry in results:
        param = entry["Parameter Set"]
        grouped.setdefault(param, []).append(entry["Daily Mean"])

    for param, means in grouped.items():
        series = pd.Series(means).dropna()
        n = len(series)
        mean = series.mean()
        var = series.var(ddof=1)
        std_err = series.std(ddof=1) / np.sqrt(n)
        t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci_halfwidth = t_value * std_err
        ci_lower = mean - ci_halfwidth
        ci_upper = mean + ci_halfwidth

        summary.append({
            "Parameter Set": param,
            "n": n,
            "Mittelwert": mean,
            "Varianz": var,
            "CI lower (95%)": ci_lower,
            "CI upper (95%)": ci_upper
        })

    return pd.DataFrame(summary)

# ---------------------------------------------------------------
# Funktion: summarize_hourly_stats
# Zweck: Für jede Parameterkonfiguration werden stündliche Mittelwerte, Varianzen und 95%-Konfidenzintervalle berechnet.
# Die Datenbasis sind die stündlichen Werte aller Tage für jeden Parametersatz.
# ---------------------------------------------------------------
def summarize_hourly_stats(results, alpha=0.05):
    """
    Aggregiert für jede Parameterkonfiguration und jede Stunde:
    - Mittelwert
    - Varianz
    - 95%-Konfidenzintervall (t-Verteilung)
    Gibt DataFrames für beide Parameterkombinationen zurück.
    """
    summary_tables = {}
    for param_set in sorted({entry["Parameter Set"] for entry in results}):
        all_hourly = [entry["Hourly Means"] for entry in results if entry["Parameter Set"] == param_set]
        df = pd.DataFrame(all_hourly)  # Zeilen: Tage, Spalten: Stunden
        df = df.apply(pd.to_numeric, errors='coerce')

        means = df.mean(axis=0)
        variances = df.var(axis=0, ddof=1)
        std_err = df.std(axis=0, ddof=1) / np.sqrt(df.shape[0])
        t_value = stats.t.ppf(1 - alpha / 2, df.shape[0] - 1)
        ci_halfwidth = t_value * std_err
        ci_lower = means - ci_halfwidth
        ci_upper = means + ci_halfwidth

        table = pd.DataFrame({
            "Stunde": list(range(1, 13)),
            "Mittelwert": means,
            "Varianz": variances,
            "CI lower (95%)": ci_lower,
            "CI upper (95%)": ci_upper
        })
        summary_tables[param_set] = table

    return summary_tables
