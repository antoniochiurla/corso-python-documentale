# -*- coding: utf-8 -*-
"""
03_normalizzazione.py
=====================
SECONDA FASE: portare i quattro schemi eterogenei a UN UNICO SCHEMA COMUNE.

Schema comune di destinazione (DataFrame finale "unificato"):
    record_id        str     identificativo univoco (fonte + indice)
    fonte            str     ALFA / BETA / GAMMA / DELTA
    data             datetime64
    operatore        str     nome normalizzato
    commessa         str     codice C-AAAA-NNNN
    mansione         str     categoria canonica
    ore              float   ore totali (ord + str)
    costo_materiali  float   EUR
    confidenza_ocr   float   0..1 (1.0 se la fonte non la fornisce)
    valido           bool    supera i controlli di dominio minimi

Mostra a lezione:
  - mapping di colonne diverse -> stesso campo
  - applicazione delle funzioni di utils_ocr (parse numeri/date, normalizzazioni)
  - parsing di TESTO OCR grezzo con regex (fonte DELTA)
  - gestione di granularita' diversa (ALFA ha ore ord+str separate, le altre sommate)
  - concatenazione (pd.concat) in un unico DataFrame
  - de-duplicazione e flag di validita'
  - salvataggio del dataset pulito (parquet + csv) per le statistiche

Esecuzione:
    python 03_normalizzazione.py
"""

import json
import os
import re

import numpy as np
import pandas as pd

import utils_ocr as u

# I file SORGENTE stanno in 'dati'; i RISULTATI prodotti vanno in 'output'.
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DATI = os.path.join(DIR_BASE, "dati")
OUTPUT = os.path.join(DIR_BASE, "output")

COLONNE_COMUNI = [
    "record_id", "fonte", "data", "operatore", "commessa",
    "mansione", "ore", "costo_materiali", "confidenza_ocr", "valido",
]


# -----------------------------------------------------------------------------
# ALFA -> schema comune
# -----------------------------------------------------------------------------
def normalizza_alfa(path=None) -> pd.DataFrame:
    path = path or os.path.join(DATI, "rapportini_alfa.csv")
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    out = pd.DataFrame()
    out["data"] = df["Data"].map(u.parse_data)
    out["operatore"] = df["Dipendente"].map(u.normalizza_nome)
    out["commessa"] = df["Cod. Commessa"].map(u.normalizza_commessa)
    out["mansione"] = df["Attivita"].map(u.normalizza_mansione)
    # granularita' diversa: ALFA tiene ore ordinarie e straordinarie separate -> sommiamo
    ore_ord = df["Ore Ord."].map(u.parse_numero)
    ore_str = df["Ore Str."].map(u.parse_numero)
    out["ore"] = ore_ord.fillna(0) + ore_str.fillna(0)
    out["ore"] = out["ore"].map(u.valida_ore)
    out["costo_materiali"] = df["Costo Mat. (EUR)"].map(u.parse_numero).map(u.valida_costo)
    out["confidenza_ocr"] = 1.0   # ALFA non fornisce confidenza OCR
    out["fonte"] = "ALFA"
    out["record_id"] = ["ALFA-" + str(i) for i in range(len(out))]
    return out


# -----------------------------------------------------------------------------
# BETA -> schema comune
# -----------------------------------------------------------------------------
def normalizza_beta(path=None) -> pd.DataFrame:
    path = path or os.path.join(DATI, "rapportini_beta.csv")
    df = pd.read_csv(path, sep=",", dtype=str, keep_default_na=False)
    out = pd.DataFrame()
    out["data"] = df["date"].map(u.parse_data)
    out["operatore"] = df["worker"].map(u.normalizza_nome)
    out["commessa"] = df["job_code"].map(u.normalizza_commessa)
    out["mansione"] = df["task"].map(u.normalizza_mansione)
    out["ore"] = df["hours"].map(u.parse_numero).map(u.valida_ore)   # gia' sommate
    out["costo_materiali"] = df["material_cost"].map(u.parse_numero).map(u.valida_costo)
    out["confidenza_ocr"] = 1.0
    out["fonte"] = "BETA"
    out["record_id"] = ["BETA-" + str(i) for i in range(len(out))]
    return out


# -----------------------------------------------------------------------------
# GAMMA (JSON annidato) -> schema comune
# -----------------------------------------------------------------------------
def normalizza_gamma(path=None) -> pd.DataFrame:
    path = path or os.path.join(DATI, "rapportini_gamma.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data, sep=".")
    out = pd.DataFrame()
    out["data"] = df["rapporto_data"].map(u.parse_data)
    out["operatore"] = df["operatore.nome"].map(u.normalizza_nome)
    out["commessa"] = df["commessa.codice"].map(u.normalizza_commessa)
    out["mansione"] = df["tipo_lavoro"].map(u.normalizza_mansione)
    out["ore"] = df["ore_totali"].map(u.parse_numero).map(u.valida_ore)
    out["costo_materiali"] = df["materiali_eur"].map(u.parse_numero).map(u.valida_costo)
    # confidenza disponibile a piu' livelli: usiamo la minima tra pagina e campi chiave
    conf = pd.concat([
        pd.to_numeric(df["scansione_conf"], errors="coerce"),
        pd.to_numeric(df["operatore.conf"], errors="coerce"),
        pd.to_numeric(df["commessa.conf"], errors="coerce"),
    ], axis=1).min(axis=1)
    out["confidenza_ocr"] = conf.fillna(1.0)
    out["fonte"] = "GAMMA"
    out["record_id"] = df["id"].astype(str)
    return out


# -----------------------------------------------------------------------------
# DELTA (NDJSON con raw_text) -> schema comune (parsing con REGEX)
# -----------------------------------------------------------------------------
# Il raw_text ha forma: "DATA|NOME|COMMESSA|OREh|MANSIONE|EUR COSTO"
RE_DELTA = re.compile(
    r"^(?P<data>[^|]+)\|(?P<nome>[^|]+)\|(?P<commessa>[^|]+)\|"
    r"(?P<ore>[^|h]+)h?\|(?P<mansione>[^|]+)\|EUR\s*(?P<costo>[\d.,]+)"
)


def _parse_raw(testo):
    """Estrae i campi dal testo OCR grezzo; ritorna dict con None se non combacia."""
    if u.is_nullo(testo):
        return {k: None for k in ("data", "nome", "commessa", "ore", "mansione", "costo")}
    m = RE_DELTA.match(str(testo).strip())
    if not m:
        return {k: None for k in ("data", "nome", "commessa", "ore", "mansione", "costo")}
    return m.groupdict()


def normalizza_delta(path=None) -> pd.DataFrame:
    path = path or os.path.join(DATI, "rapportini_delta.ndjson")
    df = pd.read_json(path, lines=True, dtype=str)
    estratti = df["raw_text"].map(_parse_raw).apply(pd.Series)

    out = pd.DataFrame()
    # preferiamo il campo pre-estratto se presente, altrimenti quello dal raw_text
    data_grezza = df["campo_data"].where(~df["campo_data"].map(u.is_nullo), estratti["data"])
    out["data"] = data_grezza.map(u.parse_data)
    out["operatore"] = estratti["nome"].map(u.normalizza_nome)
    out["commessa"] = estratti["commessa"].map(u.normalizza_commessa)
    out["mansione"] = estratti["mansione"].map(u.normalizza_mansione)
    ore_grezze = df["campo_ore"].where(~df["campo_ore"].map(u.is_nullo), estratti["ore"])
    out["ore"] = ore_grezze.map(u.parse_numero).map(u.valida_ore)
    out["costo_materiali"] = estratti["costo"].map(u.parse_numero).map(u.valida_costo)
    out["confidenza_ocr"] = pd.to_numeric(df["ocr_confidence"], errors="coerce").fillna(1.0)
    out["fonte"] = "DELTA"
    out["record_id"] = df["doc_id"].astype(str)
    return out


# -----------------------------------------------------------------------------
# Validazione finale e unione
# -----------------------------------------------------------------------------
def applica_flag_valido(df: pd.DataFrame) -> pd.DataFrame:
    """Un record e' 'valido' se ha data, operatore, commessa e ore plausibili."""
    df["valido"] = (
        df["data"].notna()
        & df["operatore"].notna()
        & df["commessa"].notna()
        & df["ore"].notna()
        & (df["mansione"] != "Non classificato")
    )
    return df


def unifica() -> pd.DataFrame:
    parti = [
        normalizza_alfa(),
        normalizza_beta(),
        normalizza_gamma(),
        normalizza_delta(),
    ]
    parti = [applica_flag_valido(p)[COLONNE_COMUNI] for p in parti]
    df = pd.concat(parti, ignore_index=True)

    # tipi coerenti
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    for c in ("ore", "costo_materiali", "confidenza_ocr"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # categorie -> risparmio memoria e semantica corretta
    df["fonte"] = df["fonte"].astype("category")
    df["mansione"] = df["mansione"].astype("category")

    # de-duplicazione: stessa data+operatore+commessa+ore = probabile doppia scansione
    prima = len(df)
    df = df.drop_duplicates(subset=["data", "operatore", "commessa", "ore", "costo_materiali"])
    print(f"De-duplicazione: rimosse {prima - len(df):,} righe duplicate".replace(",", "."))
    return df.reset_index(drop=True)


def main():
    print("Normalizzazione verso schema comune...\n")
    df = unifica()

    print(f"\nRecord totali unificati : {len(df):,}".replace(",", "."))
    print(f"Record validi           : {df['valido'].sum():,} "
          f"({100*df['valido'].mean():.1f}%)".replace(",", "."))
    print("\nSchema finale:")
    print(df.dtypes.to_string())
    print("\nAnteprima:")
    print(df.head(6).to_string())

    # salviamo per la fase statistiche (parquet = veloce e con tipi; csv = ispezionabile)
    os.makedirs(OUTPUT, exist_ok=True)
    try:
        df.to_parquet(os.path.join(OUTPUT, "dataset_unificato.parquet"), index=False)
        print("\n[OK] salvato output/dataset_unificato.parquet")
    except Exception as e:   # pyarrow potrebbe non essere installato
        print(f"\n[!] parquet non disponibile ({e}); salvo solo CSV")
    df.to_csv(os.path.join(OUTPUT, "dataset_unificato.csv"), index=False)
    print("[OK] salvato output/dataset_unificato.csv")
    return df


if __name__ == "__main__":
    main()
