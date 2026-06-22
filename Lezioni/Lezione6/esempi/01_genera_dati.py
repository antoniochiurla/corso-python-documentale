# -*- coding: utf-8 -*-
"""
01_genera_dati.py
=================
Genera i file SORGENTE del corso: simulano l'output di un OCR applicato a
RAPPORTINI DI LAVORO cartacei scannerizzati da quattro ditte diverse.

L'obiettivo didattico NON e' avere dati puliti: ogni file riproduce i problemi
tipici dell'OCR su moduli cartacei:
  - confusione di caratteri  (O<->0, l/I<->1, S<->5, B<->8, Z<->2, G<->6)
  - separatori decimali misti (virgola vs punto) e migliaia
  - formati data eterogenei  (gg/mm/aaaa, aaaa-mm-gg, gg-mm-aaaa, testo)
  - celle vuote / illeggibili ("", "###", "ILLEGGIBILE")
  - spazi sporchi, maiuscole/minuscole incoerenti, "rumore" a fine cella
  - schemi (nomi colonne) DIVERSI da ditta a ditta -> serve normalizzazione
  - punteggi di confidenza OCR per riga/campo (in due dei quattro file)

Output (cartella corrente):
  rapportini_alfa.csv     CSV  sep=';'  decimali ','  intestazioni IT  date gg/mm/aaaa
  rapportini_beta.csv     CSV  sep=','  decimali '.'  intestazioni EN  date aaaa-mm-gg
  rapportini_gamma.json   JSON array, record annidati + confidenza per campo
  rapportini_delta.ndjson NDJSON, una riga per record, con il TESTO OCR grezzo

Ogni file ha decine di migliaia di righe (default ~35.000).

Esecuzione:
    python 01_genera_dati.py
"""

import csv
import json
import os
import random
from datetime import date, timedelta

# numpy serve solo per generare distribuzioni numeriche realistiche
import numpy as np

# Tutti i file dati vivono nella sottocartella 'dati' accanto a questo script.
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DATI = os.path.join(DIR_BASE, "dati")

# -----------------------------------------------------------------------------
# Parametri generali
# -----------------------------------------------------------------------------
SEED = 42                 # riproducibilita': stessi dati ad ogni esecuzione
N_ALFA = 35000
N_BETA = 38000
N_GAMMA = 32000
N_DELTA = 30000

random.seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Anagrafiche "vere" da cui partiamo prima di sporcare i dati con l'OCR
# -----------------------------------------------------------------------------
NOMI = [
    "Mario Rossi", "Luigi Bianchi", "Giuseppe Verdi", "Anna Esposito",
    "Francesca Russo", "Marco Ferrari", "Paolo Romano", "Chiara Colombo",
    "Stefano Greco", "Laura Conti", "Andrea Ricci", "Giulia Marino",
    "Davide Bruno", "Sara Gallo", "Roberto De Luca", "Elena Costa",
    "Alessandro Giordano", "Martina Mancini", "Federico Rizzo", "Valentina Lombardi",
]

# Codici commessa "puliti": formato C-AAAA-NNNN
COMMESSE = [f"C-2025-{n:04d}" for n in range(1, 61)]

# Mansioni canoniche (categorie obiettivo della normalizzazione)
MANSIONI_CANONICHE = [
    "Scavo", "Muratura", "Impianto elettrico", "Impianto idraulico",
    "Tinteggiatura", "Manutenzione", "Trasporto", "Collaudo",
    "Carpenteria", "Posa pavimenti",
]

# Varianti "sporche" con cui l'OCR / la grafia rendono la stessa mansione.
# Mappare queste varianti al valore canonico e' uno degli esercizi centrali.
VARIANTI_MANSIONE = {
    "Scavo": ["Scavo", "scavo", "SCAVO", "Scav0", "Scavi", "scavo terreno"],
    "Muratura": ["Muratura", "muratura", "MURATURA", "Murat.", "muratur4", "Opere murarie"],
    "Impianto elettrico": ["Impianto elettrico", "Imp. elettrico", "imp elettr.",
                            "Impianto Elettr1co", "ELETTRICO", "imp. elettrico"],
    "Impianto idraulico": ["Impianto idraulico", "Imp. idraulico", "idraulica",
                           "Impianto 1draulico", "IDRAULICO", "imp idraul."],
    "Tinteggiatura": ["Tinteggiatura", "tinteggiatura", "Pittura", "pittura",
                      "Tinteggi4tura", "imbiancatura"],
    "Manutenzione": ["Manutenzione", "manutenzione", "Manut.", "MANUTENZIONE",
                     "Manuten21one", "manut ordinaria"],
    "Trasporto": ["Trasporto", "trasporto", "Trasp.", "TRASPORTO", "Tra5porto",
                  "trasporto materiali"],
    "Collaudo": ["Collaudo", "collaudo", "Coll.", "COLLAUDO", "C0llaudo", "verifica"],
    "Carpenteria": ["Carpenteria", "carpenteria", "Carp.", "CARPENTERIA", "Carpenter1a"],
    "Posa pavimenti": ["Posa pavimenti", "posa pavimenti", "Pavimentazione",
                       "P0sa pavimenti", "posa piastrelle"],
}

MEZZI = ["Furgone", "Escavatore", "Gru", "Autocarro", "Piattaforma", "Nessuno", "Betoniera"]

# -----------------------------------------------------------------------------
# "Motore" di rumore OCR
# -----------------------------------------------------------------------------
# Sostituzioni di caratteri tipiche degli errori di riconoscimento ottico.
SOSTITUZIONI_OCR = {
    "O": "0", "o": "0", "l": "1", "I": "1", "S": "5",
    "B": "8", "Z": "2", "G": "6", "g": "9", "A": "4",
}

def sporca_testo(s, prob=0.06):
    """Applica casualmente sostituzioni OCR a una stringa (carattere per carattere)."""
    out = []
    for ch in s:
        if ch in SOSTITUZIONI_OCR and random.random() < prob:
            out.append(SOSTITUZIONI_OCR[ch])
        else:
            out.append(ch)
    return "".join(out)

def aggiungi_rumore_bordo(s, prob=0.04):
    """Aggiunge spazi o caratteri 'sporchi' tipici dei bordi cella scannerizzati."""
    if random.random() < prob:
        s = s + random.choice([" ", "  ", " .", " ,", " |", "*"])
    if random.random() < prob:
        s = random.choice([" ", "  "]) + s
    return s

def forse_mancante(valore, prob=0.03):
    """Con probabilita' 'prob' restituisce un marcatore di cella illeggibile/vuota."""
    if random.random() < prob:
        return random.choice(["", "", "###", "ILLEGGIBILE", "-", "n/d"])
    return valore

def numero_it(x, decimali=2):
    """Formatta un numero alla 'italiana': virgola decimale, eventuale punto migliaia."""
    s = f"{x:,.{decimali}f}"            # es. 1,234.50  (stile US)
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # -> 1.234,50 (IT)
    return s

def numero_en(x, decimali=2):
    """Formatta un numero alla 'inglese': punto decimale."""
    return f"{x:.{decimali}f}"

# -----------------------------------------------------------------------------
# Generatore di un record "vero" (prima di sporcarlo)
# -----------------------------------------------------------------------------
DATA_INIZIO = date(2025, 1, 1)
DATA_FINE = date(2025, 12, 31)
GIORNI_RANGE = (DATA_FINE - DATA_INIZIO).days

def record_base():
    """Crea un rapportino 'pulito' coerente, poi sara' sporcato per ciascun formato."""
    g = random.randint(0, GIORNI_RANGE)
    d = DATA_INIZIO + timedelta(days=g)
    nome = random.choice(NOMI)
    commessa = random.choice(COMMESSE)
    mansione = random.choice(MANSIONI_CANONICHE)

    # Ore ordinarie 1..9 + eventuali straordinari, con qualche outlier "sospetto"
    ore_ord = round(np.random.choice([4, 6, 7, 8, 8, 8, 9], p=[.05, .1, .15, .4, .1, .1, .1]) +
                    random.choice([0, 0, 0, 0.5]), 1)
    ore_str = round(random.choice([0, 0, 0, 1, 2, 3]) * random.choice([0, 1]), 1)

    # Costo materiali: lognormale -> molti valori piccoli, code lunghe
    costo = round(float(np.random.lognormal(mean=3.4, sigma=1.0)), 2)  # ~ decine/centinaia EUR

    mezzo = random.choice(MEZZI)
    return {
        "data": d,
        "nome": nome,
        "commessa": commessa,
        "mansione_canonica": mansione,
        "ore_ord": ore_ord,
        "ore_str": ore_str,
        "costo": costo,
        "mezzo": mezzo,
    }

def variante_mansione(canonica):
    """Restituisce una resa 'sporca' della mansione canonica."""
    return random.choice(VARIANTI_MANSIONE[canonica])

# -----------------------------------------------------------------------------
# FILE A - Alfa Costruzioni : CSV sep=';' decimali ',' intestazioni IT, date gg/mm/aaaa
# -----------------------------------------------------------------------------
def genera_alfa(path, n):
    intestazioni = ["Data", "Dipendente", "Cod. Commessa", "Ore Ord.", "Ore Str.",
                    "Attivita", "Mezzo", "Costo Mat. (EUR)", "Note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(intestazioni)
        for _ in range(n):
            r = record_base()
            data_str = r["data"].strftime("%d/%m/%Y")
            # talvolta data con anno a 2 cifre o separatore '.'
            if random.random() < 0.05:
                data_str = r["data"].strftime("%d.%m.%y")
            nome = aggiungi_rumore_bordo(sporca_testo(r["nome"]))
            commessa = sporca_testo(r["commessa"], prob=0.1)
            ore_ord = numero_it(r["ore_ord"], 1)
            ore_str = numero_it(r["ore_str"], 1)
            attivita = aggiungi_rumore_bordo(variante_mansione(r["mansione_canonica"]))
            mezzo = r["mezzo"]
            # costo con eventuale separatore migliaia "alla italiana"
            costo = numero_it(r["costo"], 2)
            note = random.choice(["", "", "", "rif. bolla", "doppio turno",
                                  "materiale di scorta", "cliente assente"])
            riga = [
                forse_mancante(data_str),
                forse_mancante(nome),
                forse_mancante(commessa),
                forse_mancante(ore_ord),
                ore_str,
                forse_mancante(attivita),
                mezzo,
                forse_mancante(costo),
                note,
            ]
            w.writerow(riga)
    print(f"  [OK] {path}  ({n} righe)")

# -----------------------------------------------------------------------------
# FILE B - Beta Impianti : CSV sep=',' decimali '.' intestazioni EN, date ISO
# -----------------------------------------------------------------------------
def genera_beta(path, n):
    intestazioni = ["date", "worker", "job_code", "hours", "task", "material_cost", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(intestazioni)
        for _ in range(n):
            r = record_base()
            data_str = r["data"].strftime("%Y-%m-%d")
            if random.random() < 0.04:
                data_str = r["data"].strftime("%Y/%m/%d")
            nome = aggiungi_rumore_bordo(sporca_testo(r["nome"]))
            commessa = sporca_testo(r["commessa"], prob=0.08)
            # qui ore_ord e ore_str sono gia' SOMMATE in un'unica colonna (schema diverso!)
            ore_tot = numero_en(r["ore_ord"] + r["ore_str"], 1)
            task = aggiungi_rumore_bordo(variante_mansione(r["mansione_canonica"]))
            costo = numero_en(r["costo"], 2)
            note = random.choice(["", "", "", "ref. note", "double shift", "spare parts"])
            riga = [
                forse_mancante(data_str),
                forse_mancante(nome),
                forse_mancante(commessa),
                forse_mancante(ore_tot),
                forse_mancante(task),
                forse_mancante(costo),
                note,
            ]
            w.writerow(riga)
    print(f"  [OK] {path}  ({n} righe)")

# -----------------------------------------------------------------------------
# FILE C - Gamma Servizi : JSON array, record ANNIDATI + confidenza per campo
# -----------------------------------------------------------------------------
def genera_gamma(path, n):
    records = []
    for i in range(n):
        r = record_base()
        data_str = r["data"].strftime("%d-%m-%Y")
        if random.random() < 0.06:
            data_str = r["data"].strftime("%d %b %Y")  # es. "07 Mar 2025"
        rec = {
            "id": f"G{i:07d}",
            "rapporto_data": forse_mancante(data_str),
            "operatore": {
                "nome": forse_mancante(sporca_testo(r["nome"])),
                "conf": round(random.uniform(0.55, 0.99), 2),   # confidenza OCR campo
            },
            "commessa": {
                "codice": sporca_testo(r["commessa"], prob=0.1),
                "conf": round(random.uniform(0.5, 0.99), 2),
            },
            # ore come stringa con virgola decimale (italiano) dentro JSON: trappola classica
            "ore_totali": forse_mancante(numero_it(r["ore_ord"] + r["ore_str"], 1)),
            "tipo_lavoro": variante_mansione(r["mansione_canonica"]),
            "materiali_eur": forse_mancante(numero_it(r["costo"], 2)),
            "scansione_conf": round(random.uniform(0.4, 0.99), 2),  # confidenza pagina
        }
        records.append(rec)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=None)
    print(f"  [OK] {path}  ({n} righe)")

# -----------------------------------------------------------------------------
# FILE D - Delta Manutenzioni : NDJSON, una riga JSON per record, con TESTO OCR grezzo
# -----------------------------------------------------------------------------
def genera_delta(path, n):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            r = record_base()
            data_str = r["data"].strftime("%d/%m/%Y")
            nome = sporca_testo(r["nome"])
            ore = numero_it(r["ore_ord"] + r["ore_str"], 1)
            costo = numero_it(r["costo"], 2)
            mansione = variante_mansione(r["mansione_canonica"])
            # 'raw_text': l'intera riga cosi' come 'letta' dall'OCR, da parsare con regex
            raw = f"{data_str}|{nome}|{r['commessa']}|{ore}h|{mansione}|EUR {costo}"
            raw = sporca_testo(raw, prob=0.03)
            rec = {
                "doc_id": f"D-{i:07d}",
                "pagina": random.randint(1, 8),
                "ocr_confidence": round(random.uniform(0.45, 0.99), 3),
                "raw_text": raw,
                # alcuni campi pre-estratti (ma non tutti, e non sempre coerenti col raw)
                "campo_data": forse_mancante(data_str, prob=0.08),
                "campo_ore": forse_mancante(ore, prob=0.12),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [OK] {path}  ({n} righe)")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generazione file sorgente (output OCR simulato)...")
    os.makedirs(DATI, exist_ok=True)
    genera_alfa(os.path.join(DATI, "rapportini_alfa.csv"), N_ALFA)
    genera_beta(os.path.join(DATI, "rapportini_beta.csv"), N_BETA)
    genera_gamma(os.path.join(DATI, "rapportini_gamma.json"), N_GAMMA)
    genera_delta(os.path.join(DATI, "rapportini_delta.ndjson"), N_DELTA)
    tot = N_ALFA + N_BETA + N_GAMMA + N_DELTA
    print(f"Fatto. Totale record generati: {tot:,}".replace(",", "."))
