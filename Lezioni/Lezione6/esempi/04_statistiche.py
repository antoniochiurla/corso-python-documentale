# -*- coding: utf-8 -*-
"""
04_statistiche.py
=================
TERZA FASE: statistiche e DISTILLAZIONE dei dati dal dataset unificato.

Mostra a lezione:
  - statistiche descrittive (describe, quantili, outlier)
  - qualita' del dato per fonte (% validi, confidenza OCR media)
  - aggregazioni con groupby (ore e costi per commessa, mansione, mese)
  - tabelle pivot (operatore x mese)
  - "distillazione": da 135k righe grezze a poche tabelle sintetiche/KPI
  - export dei risultati in Excel multi-foglio (se disponibile) o CSV

Esecuzione (dopo 03_normalizzazione.py):
    python 04_statistiche.py
"""

import os

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# Il dataset unificato e il report stanno nella sottocartella 'output'.
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(DIR_BASE, "output")


def carica_dataset():
    try:
        return pd.read_parquet(os.path.join(OUTPUT, "dataset_unificato.parquet"))
    except Exception:
        df = pd.read_csv(os.path.join(OUTPUT, "dataset_unificato.csv"), parse_dates=["data"])
        return df


def riga(t=""):
    print("\n" + "=" * 78)
    if t:
        print(t)
        print("-" * 78)


# -----------------------------------------------------------------------------
# 1) Panoramica qualita' del dato (per fonte)
# -----------------------------------------------------------------------------
def qualita_per_fonte(df):
    riga("QUALITA' DEL DATO PER FONTE")
    g = df.groupby("fonte", observed=True)
    rep = pd.DataFrame({
        "righe": g.size(),
        "%_validi": 100 * g["valido"].mean(),
        "conf_ocr_media": g["confidenza_ocr"].mean(),
        "%_data_mancante": 100 * g["data"].apply(lambda s: s.isna().mean()),
        "%_commessa_mancante": 100 * g["commessa"].apply(lambda s: s.isna().mean()),
        "%_mansione_non_class": 100 * g["mansione"].apply(
            lambda s: (s == "Non classificato").mean()),
    })
    print(rep.to_string())
    return rep


# -----------------------------------------------------------------------------
# 2) Statistiche descrittive sui campi numerici (solo record validi)
# -----------------------------------------------------------------------------
def descrittive(df):
    riga("STATISTICHE DESCRITTIVE (record validi)")
    val = df[df["valido"]]
    desc = val[["ore", "costo_materiali", "confidenza_ocr"]].describe(
        percentiles=[.05, .25, .5, .75, .95]).T
    print(desc.to_string())

    # individuazione outlier sul costo con metodo IQR
    q1, q3 = val["costo_materiali"].quantile([.25, .75])
    iqr = q3 - q1
    soglia = q3 + 1.5 * iqr
    n_out = (val["costo_materiali"] > soglia).sum()
    print(f"\nOutlier costo_materiali (> {soglia:,.2f} EUR, metodo IQR): "
          f"{n_out:,} record".replace(",", "."))
    return desc


# -----------------------------------------------------------------------------
# 3) Aggregazioni: ore e costi per commessa / mansione / mese
# -----------------------------------------------------------------------------
def aggregazioni(df):
    val = df[df["valido"]].copy()
    val["mese"] = val["data"].dt.to_period("M").astype(str)

    riga("TOP 10 COMMESSE PER ORE TOTALI")
    per_commessa = (val.groupby("commessa", observed=True)
                    .agg(ore_totali=("ore", "sum"),
                         costo_totale=("costo_materiali", "sum"),
                         n_rapportini=("record_id", "count"))
                    .sort_values("ore_totali", ascending=False))
    print(per_commessa.head(10).to_string())

    riga("RIEPILOGO PER MANSIONE")
    per_mansione = (val.groupby("mansione", observed=True)
                    .agg(n=("record_id", "count"),
                         ore_medie=("ore", "mean"),
                         ore_totali=("ore", "sum"),
                         costo_medio=("costo_materiali", "mean"))
                    .sort_values("ore_totali", ascending=False))
    print(per_mansione.to_string())

    riga("ANDAMENTO MENSILE (ore e costo)")
    per_mese = (val.groupby("mese")
                .agg(ore_totali=("ore", "sum"),
                     costo_totale=("costo_materiali", "sum"),
                     n_rapportini=("record_id", "count")))
    print(per_mese.to_string())

    return per_commessa, per_mansione, per_mese


# -----------------------------------------------------------------------------
# 4) Pivot: ore per operatore x mese
# -----------------------------------------------------------------------------
def pivot_operatore_mese(df):
    riga("PIVOT - ORE PER OPERATORE x MESE (primi 8 operatori)")
    val = df[df["valido"]].copy()
    val["mese"] = val["data"].dt.to_period("M").astype(str)
    piv = pd.pivot_table(val, values="ore", index="operatore",
                         columns="mese", aggfunc="sum", fill_value=0)
    print(piv.head(8).to_string())
    return piv


# -----------------------------------------------------------------------------
# 5) Distillazione: KPI sintetici (da 135k righe a una manciata di numeri)
# -----------------------------------------------------------------------------
def distillazione(df):
    riga("DISTILLAZIONE - KPI SINTETICI")
    val = df[df["valido"]]
    kpi = {
        "record_grezzi_totali": len(df),
        "record_validi": int(val.shape[0]),
        "tasso_validita_%": round(100 * val.shape[0] / len(df), 1),
        "operatori_distinti": val["operatore"].nunique(),
        "commesse_distinte": val["commessa"].nunique(),
        "ore_totali": round(val["ore"].sum(), 1),
        "ore_medie_per_rapportino": round(val["ore"].mean(), 2),
        "costo_materiali_totale_EUR": round(val["costo_materiali"].sum(), 2),
        "confidenza_ocr_media": round(val["confidenza_ocr"].mean(), 3),
        "periodo": f"{val['data'].min().date()} -> {val['data'].max().date()}",
    }
    for k, v in kpi.items():
        etichetta = k.replace("_", " ")
        print(f"  {etichetta:32s}: {v}")
    return kpi


# -----------------------------------------------------------------------------
# 6) Export risultati
# -----------------------------------------------------------------------------
def esporta(qual, desc, per_commessa, per_mansione, per_mese, piv, kpi):
    riga("EXPORT RISULTATI")
    kpi_df = pd.DataFrame(list(kpi.items()), columns=["kpi", "valore"])
    os.makedirs(OUTPUT, exist_ok=True)
    try:
        with pd.ExcelWriter(os.path.join(OUTPUT, "report_statistiche.xlsx"),
                            engine="openpyxl") as xl:
            kpi_df.to_excel(xl, sheet_name="KPI", index=False)
            qual.to_excel(xl, sheet_name="Qualita_per_fonte")
            desc.to_excel(xl, sheet_name="Descrittive")
            per_commessa.head(50).to_excel(xl, sheet_name="Top_commesse")
            per_mansione.to_excel(xl, sheet_name="Per_mansione")
            per_mese.to_excel(xl, sheet_name="Per_mese")
            piv.to_excel(xl, sheet_name="Pivot_operatore_mese")
        print("[OK] output/report_statistiche.xlsx (7 fogli)")
    except Exception as e:
        print(f"[!] Excel non disponibile ({e}); esporto CSV separati")
        kpi_df.to_csv(os.path.join(OUTPUT, "kpi.csv"), index=False)
        per_mansione.to_csv(os.path.join(OUTPUT, "per_mansione.csv"))
        per_mese.to_csv(os.path.join(OUTPUT, "per_mese.csv"))
        print("[OK] output/kpi.csv, output/per_mansione.csv, output/per_mese.csv")


def main():
    df = carica_dataset()
    qual = qualita_per_fonte(df)
    desc = descrittive(df)
    per_commessa, per_mansione, per_mese = aggregazioni(df)
    piv = pivot_operatore_mese(df)
    kpi = distillazione(df)
    esporta(qual, desc, per_commessa, per_mansione, per_mese, piv, kpi)
    riga("FINE")


if __name__ == "__main__":
    main()
