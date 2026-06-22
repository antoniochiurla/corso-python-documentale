# -*- coding: utf-8 -*-
"""
utils_ocr.py
============
Funzioni di pulizia riutilizzabili per dati provenienti da OCR di rapportini.
Sono richiamate da 02_analisi_esplorativa.py, 03_normalizzazione.py e 04_statistiche.py.

Filosofia: ogni funzione e' "difensiva". I dati OCR sono sporchi e incoerenti,
quindi non assumiamo mai che un campo sia ben formato: gestiamo il caso peggiore
e, quando non riusciamo a interpretare un valore, restituiamo NaN/None invece di
sollevare un'eccezione che bloccherebbe l'intero batch.
"""

import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1) Marcatori di "cella illeggibile" che l'OCR puo' aver prodotto
# -----------------------------------------------------------------------------
MARCATORI_NULLI = {"", "-", "n/d", "nd", "na", "n.d.", "###", "illeggibile", "null", "none"}


def is_nullo(valore) -> bool:
    """True se il valore va trattato come mancante."""
    if valore is None:
        return True
    if isinstance(valore, float) and np.isnan(valore):
        return True
    return str(valore).strip().lower() in MARCATORI_NULLI


# -----------------------------------------------------------------------------
# 2) Correzione caratteri confusi dall'OCR
# -----------------------------------------------------------------------------
# Mappe SPECIFICHE per contesto: dentro un NUMERO le lettere vanno -> cifre;
# dentro un CODICE commessa sappiamo che la parte numerica deve essere cifre.
LETTERE_VERSO_CIFRE = str.maketrans({
    "O": "0", "o": "0", "l": "1", "I": "1", "i": "1",
    "S": "5", "B": "8", "Z": "2", "G": "6", "g": "9", "A": "4", "T": "7",
})


def pulisci_spazi(s: str) -> str:
    """Normalizza spazi multipli e rumore di bordo cella."""
    if s is None:
        return s
    s = str(s)
    # rimuove caratteri di rumore tipici dei bordi scansione
    s = re.sub(r"[|*]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,")


def normalizza_nome(s) -> str | float:
    """
    Pulisce un nome persona: spazi, correzione cifre->lettere (l'inverso!),
    capitalizzazione coerente. In un NOME, '0' e' quasi sempre una 'O' male letta.
    """
    if is_nullo(s):
        return np.nan
    s = pulisci_spazi(s)
    # nei nomi le CIFRE sono errori: 0->O, 1->I/l, 5->S, 8->B ...
    cifre_verso_lettere = str.maketrans({"0": "o", "1": "i", "5": "s", "8": "b",
                                         "2": "z", "6": "g", "4": "a", "7": "t"})
    # applichiamo solo se il token contiene cifre in mezzo a lettere
    tokens = []
    for t in s.split():
        if re.search(r"[A-Za-z]", t) and re.search(r"\d", t):
            t = t.translate(cifre_verso_lettere)
        tokens.append(t)
    s = " ".join(tokens)
    # togliamo accenti per uniformare le chiavi di join (Esposito vs Espòsito)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.title()


def normalizza_commessa(s) -> str | float:
    """
    Normalizza un codice commessa al formato canonico C-AAAA-NNNN.
    La parte alfabetica iniziale resta lettera; la coda numerica viene forzata a cifre.
    """
    if is_nullo(s):
        return np.nan
    s = pulisci_spazi(s).upper().replace(" ", "")
    # estrae pattern tipo C-2025-0001 anche se sporcato
    m = re.search(r"C[-_]?(\w{4})[-_]?(\w{4})", s)
    if not m:
        return np.nan
    anno = m.group(1).translate(LETTERE_VERSO_CIFRE)
    num = m.group(2).translate(LETTERE_VERSO_CIFRE)
    if not (anno.isdigit() and num.isdigit()):
        return np.nan
    return f"C-{anno}-{num}"


# -----------------------------------------------------------------------------
# 3) Parsing numeri (virgola vs punto, migliaia, lettere intruse)
# -----------------------------------------------------------------------------
def parse_numero(valore) -> float:
    """
    Converte una stringa numerica 'sporca' in float, gestendo:
      - decimale italiano (1.234,56) e inglese (1,234.56 / 1234.56)
      - lettere confuse con cifre (O->0, S->5, ...)
      - simboli valuta e spazi
    Restituisce np.nan se non interpretabile.
    """
    if is_nullo(valore):
        return np.nan
    if isinstance(valore, (int, float)) and not isinstance(valore, bool):
        return float(valore)
    s = str(valore).strip()
    # rimuove valuta e spazi
    s = re.sub(r"(?i)(eur|euro|€|\$)", "", s).strip()
    # corregge lettere->cifre SOLO se la stringa e' "quasi numerica"
    if re.search(r"[A-Za-z]", s) and re.search(r"\d", s):
        s = s.translate(LETTERE_VERSO_CIFRE)
    s = s.replace(" ", "")
    if s in ("", "-", "."):
        return np.nan

    ha_virgola = "," in s
    ha_punto = "." in s
    if ha_virgola and ha_punto:
        # l'ultimo separatore che appare e' quello decimale
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")     # IT: 1.234,56 -> 1234.56
        else:
            s = s.replace(",", "")                        # EN: 1,234.56 -> 1234.56
    elif ha_virgola:
        # solo virgola -> decimale italiano
        s = s.replace(",", ".")
    # solo punto -> gia' decimale inglese, lasciamo
    try:
        return float(s)
    except ValueError:
        return np.nan


# -----------------------------------------------------------------------------
# 4) Parsing date in formati eterogenei
# -----------------------------------------------------------------------------
MESI_IT = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
    "jan": 1, "apr.": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "dec": 12,
}

FORMATI_DATA = [
    "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d",
    "%d.%m.%Y", "%d.%m.%y", "%d/%m/%y",
]


def parse_data(valore):
    """
    Converte una data 'sporca' in pandas.Timestamp.
    Tenta piu' formati espliciti + mesi testuali. Restituisce NaT se impossibile.
    """
    if is_nullo(valore):
        return pd.NaT
    s = str(valore).strip()
    # corregge cifre lette come lettere all'interno della data
    s = s.translate(LETTERE_VERSO_CIFRE) if re.search(r"[A-Za-z]\d|\d[A-Za-z]", s) else s

    # caso "07 Mar 2025" / "07 mar 2025"
    m = re.match(r"(\d{1,2})\s+([A-Za-z]{3,4})\.?\s+(\d{2,4})", s)
    if m:
        giorno, mese_txt, anno = m.groups()
        mese = MESI_IT.get(mese_txt.lower()[:3])
        if mese:
            anno = int(anno)
            anno = anno + 2000 if anno < 100 else anno
            try:
                return pd.Timestamp(int(anno), mese, int(giorno))
            except ValueError:
                return pd.NaT

    for fmt in FORMATI_DATA:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    return pd.NaT


# -----------------------------------------------------------------------------
# 5) Mappatura mansioni "sporche" -> categoria canonica
# -----------------------------------------------------------------------------
# Dizionario di parole-chiave: per ogni canonica, i frammenti che la identificano.
KEYWORDS_MANSIONE = {
    "Scavo": ["scav"],
    "Muratura": ["murat", "murari"],
    "Impianto elettrico": ["elettr"],
    "Impianto idraulico": ["idraul", "idro"],
    "Tinteggiatura": ["tinteg", "pittur", "imbianc"],
    "Manutenzione": ["manuten", "manut"],
    "Trasporto": ["trasp", "tra5p"],
    "Collaudo": ["collaud", "c0llaud", "verific"],
    "Carpenteria": ["carpenter", "carp."],
    "Posa pavimenti": ["pavim", "piastrell", "posa pav"],
}


def normalizza_mansione(s) -> str:
    """Mappa una resa OCR della mansione al valore canonico, o 'Altro/Non classificato'."""
    if is_nullo(s):
        return "Non classificato"
    base = pulisci_spazi(s).lower()
    # normalizza cifre intruse a lettere per il matching (Elettr1co -> elettrico)
    base = base.translate(str.maketrans({"1": "i", "4": "a", "0": "o", "5": "s"}))
    for canonica, chiavi in KEYWORDS_MANSIONE.items():
        if any(k in base for k in chiavi):
            return canonica
    return "Altro"


# -----------------------------------------------------------------------------
# 6) Validazione di dominio (regole di business sui rapportini)
# -----------------------------------------------------------------------------
def valida_ore(x) -> float:
    """Le ore di un rapportino devono stare in [0, 24]. Fuori range -> NaN (sospetto)."""
    if pd.isna(x):
        return np.nan
    return x if 0 <= x <= 24 else np.nan


def valida_costo(x) -> float:
    """Costo materiali non negativo e sotto una soglia di plausibilita'."""
    if pd.isna(x):
        return np.nan
    return x if 0 <= x <= 100000 else np.nan


if __name__ == "__main__":
    # mini auto-test delle funzioni (utile a lezione per mostrare i casi limite)
    assert parse_numero("1.234,56") == 1234.56
    assert parse_numero("1,234.56") == 1234.56
    assert parse_numero("EUR 41,2O") == 41.20      # O finale -> 0
    assert parse_numero("###") != parse_numero("###")  # NaN != NaN
    assert str(parse_data("07 Mar 2025").date()) == "2025-03-07"
    assert str(parse_data("2025-01-15").date()) == "2025-01-15"
    assert normalizza_commessa("c 2025 O001") == "C-2025-0001"
    assert normalizza_mansione("Impianto Elettr1co") == "Impianto elettrico"
    assert normalizza_nome("Laura C0nti") == "Laura Conti"
    print("utils_ocr: tutti i micro-test superati.")
