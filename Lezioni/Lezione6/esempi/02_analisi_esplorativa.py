# -*- coding: utf-8 -*-
"""
02_analisi_esplorativa.py
=========================
PRIMA FASE del lavoro su dati OCR: capire COSA abbiamo prima di pulire.

Mostra a lezione come:
  1) caricare correttamente CSV con separatori/decimali diversi
  2) caricare JSON "array" annidato e NDJSON (json lines)
  3) ispezionare struttura: shape, dtypes, head, memoria
  4) misurare i valori mancanti / marcatori illeggibili
  5) individuare le ANOMALIE tipiche dell'OCR (caratteri intrusi, formati misti)
  6) capire perche' gli schemi NON sono allineati -> serve normalizzare

Esecuzione (dopo 01_genera_dati.py):
    python 02_analisi_esplorativa.py
"""

import json
import os

import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

# I file dati sono nella sottocartella 'dati' accanto a questo script.
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DATI = os.path.join(DIR_BASE, "dati")


def riga(titolo=""):
    print("\n" + "=" * 78)
    if titolo:
        print(titolo)
        print("-" * 78)


# -----------------------------------------------------------------------------
# 1) Caricamento dei quattro formati
# -----------------------------------------------------------------------------
def carica_alfa():
    """CSV italiano: separatore ';', decimale ','. dtype=str per NON perdere il 'rumore'."""
    # Nota didattica: carichiamo TUTTO come stringa. In fase esplorativa vogliamo
    # vedere i dati cosi' come sono, senza che pandas tenti conversioni che falliscono.
    df = pd.read_csv(os.path.join(DATI, "rapportini_alfa.csv"),
                     sep=";", dtype=str, keep_default_na=False)
    df.attrs["fonte"] = "ALFA (CSV ; , IT)"
    return df


def carica_beta():
    """CSV inglese: separatore ',', decimale '.'."""
    df = pd.read_csv(os.path.join(DATI, "rapportini_beta.csv"),
                     sep=",", dtype=str, keep_default_na=False)
    df.attrs["fonte"] = "BETA (CSV , . EN)"
    return df


def carica_gamma():
    """JSON array con record annidati -> appiattiamo con json_normalize."""
    with open(os.path.join(DATI, "rapportini_gamma.json"), encoding="utf-8") as f:
        data = json.load(f)
    # json_normalize 'appiattisce' operatore.nome, operatore.conf, commessa.codice ...
    df = pd.json_normalize(data, sep=".")
    df.attrs["fonte"] = "GAMMA (JSON annidato)"
    return df


def carica_delta():
    """NDJSON: una riga JSON per record -> lines=True."""
    df = pd.read_json(os.path.join(DATI, "rapportini_delta.ndjson"), lines=True, dtype=str)
    df.attrs["fonte"] = "DELTA (NDJSON + raw_text)"
    return df


# -----------------------------------------------------------------------------
# 2) Ispezione struttura
# -----------------------------------------------------------------------------
def ispeziona(df):
    riga(f"FONTE: {df.attrs.get('fonte','?')}")
    print(f"Righe x Colonne : {df.shape[0]:,} x {df.shape[1]}".replace(",", "."))
    mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memoria (deep)  : {mem:.1f} MB")
    print("\nColonne e dtype:")
    print(df.dtypes.to_string())
    print("\nPrime 3 righe:")
    print(df.head(3).to_string())


# -----------------------------------------------------------------------------
# 3) Qualita': mancanti e marcatori illeggibili
# -----------------------------------------------------------------------------
MARCATORI = {"", "-", "n/d", "###", "ILLEGGIBILE"}


def report_mancanti(df):
    riga(f"VALORI MANCANTI / ILLEGGIBILI - {df.attrs.get('fonte','?')}")
    tot = len(df)
    out = []
    for col in df.columns:
        serie = df[col].astype(str).str.strip()
        n_marcatori = serie.isin(MARCATORI).sum()
        perc = 100 * n_marcatori / tot
        out.append((col, n_marcatori, round(perc, 2)))
    rep = pd.DataFrame(out, columns=["colonna", "n_illeggibili", "perc_%"])
    print(rep.to_string(index=False))


# -----------------------------------------------------------------------------
# 4) Anomalie OCR specifiche
# -----------------------------------------------------------------------------
def anomalie_ocr(df_alfa, df_beta, df_gamma, df_delta):
    riga("ANOMALIE OCR INDIVIDUATE")

    # a) lettere intruse dentro i codici commessa (dovrebbero essere C-AAAA-NNNN)
    import re
    pat_sporco = re.compile(r"C.*[A-Za-z].*\d|[OISBZG]", re.I)
    cod_alfa = df_alfa["Cod. Commessa"].astype(str)
    sporchi = cod_alfa[cod_alfa.str.contains(r"[OISBZ]", regex=True, na=False)]
    print(f"[ALFA] codici commessa con possibili lettere-OCR intruse: "
          f"{len(sporchi):,}".replace(",", "."))
    print("  esempi:", sporchi.head(5).tolist())

    # b) decimali italiani dentro un JSON (ore_totali come '8,0')
    ore_gamma = df_gamma["ore_totali"].astype(str)
    con_virgola = ore_gamma.str.contains(",", na=False).sum()
    print(f"\n[GAMMA] valori 'ore_totali' con virgola decimale (da convertire): "
          f"{con_virgola:,}".replace(",", "."))
    print("  esempi:", ore_gamma.head(5).tolist())

    # c) formati data multipli nello stesso file
    date_alfa = df_alfa["Data"].astype(str)
    f_slash = date_alfa.str.contains("/", na=False).sum()
    f_punto = date_alfa.str.contains(r"\.", na=False).sum()
    print(f"\n[ALFA] date con '/': {f_slash:,}  |  date con '.': {f_punto:,}"
          .replace(",", "."))

    # d) testo OCR grezzo da parsare (DELTA)
    print("\n[DELTA] esempi di raw_text da parsare con regex:")
    for s in df_delta["raw_text"].head(3):
        print("   ", s)

    # e) confidenza OCR bassa -> righe a rischio
    conf = pd.to_numeric(df_delta["ocr_confidence"], errors="coerce")
    basse = (conf < 0.6).sum()
    print(f"\n[DELTA] righe con confidenza OCR < 0.60 (da rivedere): "
          f"{basse:,} su {len(df_delta):,}".replace(",", "."))


# -----------------------------------------------------------------------------
# 5) Confronto schemi: perche' NON possiamo semplicemente concatenare
# -----------------------------------------------------------------------------
def confronto_schemi(dfs):
    riga("CONFRONTO SCHEMI (colonne per fonte)")
    for df in dfs:
        print(f"\n{df.attrs.get('fonte','?')}:")
        print("   ", list(df.columns))
    print("\n=> Nomi colonne, lingua, granularita' (ore separate vs sommate),")
    print("   formati numero/data e annidamento sono DIVERSI: serve un mapping")
    print("   verso uno schema comune (vedi 03_normalizzazione.py).")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    df_alfa = carica_alfa()
    df_beta = carica_beta()
    df_gamma = carica_gamma()
    df_delta = carica_delta()

    for df in (df_alfa, df_beta, df_gamma, df_delta):
        ispeziona(df)

    for df in (df_alfa, df_beta, df_gamma, df_delta):
        report_mancanti(df)

    anomalie_ocr(df_alfa, df_beta, df_gamma, df_delta)
    confronto_schemi([df_alfa, df_beta, df_gamma, df_delta])

    riga("FINE ANALISI ESPLORATIVA")
    print("Prossimo passo: 03_normalizzazione.py")


if __name__ == "__main__":
    main()
