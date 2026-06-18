"""
Esempio: analisi di un PDF con l'API OpenAI / ChatGPT (Responses API).

Equivalente del programma per Claude, ma con OpenAI:
  1. Legge un PDF locale e lo invia come "input_file" (base64).
  2. Tramite STRUCTURED OUTPUTS (json_schema, strict=True), il modello restituisce
     un JSON GARANTITO conforme allo schema (testo + tabelle): niente parsing
     fragile, niente errori da virgolette nel testo.
  3. Per ogni PDF vengono salvati:
       - {nome}.json          -> risultato completo
       - {nome}.txt           -> solo il testo non strutturato
       - {nome}_tabella_N.csv -> una per ogni tabella trovata
     dove {nome} = nome del PDF senza cartella ne' estensione.

Prerequisiti:
  pip install openai
  export OPENAI_API_KEY="sk-..."   # mai hardcodare la chiave nel codice

Note sul supporto PDF:
  - Serve un modello con capacita' vision (es. gpt-5.5): l'API estrae testo +
    immagine di ogni pagina, ed e' questo che permette di ricostruire le tabelle.
  - Il base64 gonfia i dati di ~33%: per file grandi usare la Files API
    (vedi 'analizza_pdf_grande').
  - Ogni pagina consuma token (testo + immagine): occhio ai costi su PDF lunghi.
"""

import base64
import csv
import json
import os
import sys

from openai import OpenAI

# Modello con vision, necessario per i PDF. Per documenti molto complessi puoi
# valutare un modello superiore; per ridurre i costi, uno piu' piccolo con vision.
MODEL = "gpt-5.5"

PROMPT = (
    "Analizza il documento PDF allegato. Trascrivi tutto il testo e, se sono "
    "presenti tabelle, ricostruiscine la struttura. Rispondi solo con i dati "
    "richiesti dallo schema."
)

# Schema per gli Structured Outputs.
# Vincoli della modalita' strict:
#   - ogni oggetto deve avere "additionalProperties": False
#   - TUTTE le proprieta' devono essere elencate in "required"
#   - i campi opzionali si modellano come nullable (es. ["string", "null"])
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "testo": {
            "type": "string",
            "description": "Trascrizione completa del testo del documento.",
        },
        "tabelle": {
            "type": "array",
            "description": "Tabelle trovate. Lista vuota se non ce ne sono.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "titolo": {
                        "type": ["string", "null"],
                        "description": "Titolo/didascalia della tabella, o null.",
                    },
                    "intestazioni": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Nomi delle colonne.",
                    },
                    "righe": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                        "description": (
                            "Righe della tabella. Ogni riga ha lo stesso numero "
                            "di elementi delle intestazioni."
                        ),
                    },
                },
                "required": ["titolo", "intestazioni", "righe"],
            },
        },
    },
    "required": ["testo", "tabelle"],
}

# Configurazione dell'output strutturato per la Responses API.
TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "risultati_documento",
        "strict": True,
        "schema": SCHEMA,
    }
}


def _estrai_json(response) -> dict:
    """Estrae e parsa il JSON dalla risposta. Gestisce troncamento e rifiuti."""
    if response.status == "incomplete":
        motivo = getattr(response.incomplete_details, "reason", "sconosciuto")
        raise RuntimeError(
            f"Risposta incompleta (motivo: {motivo}). Se e' 'max_output_tokens', "
            f"aumenta max_output_tokens o spezza il PDF."
        )

    for item in response.output:
        if item.type != "message":
            continue
        for parte in item.content:
            if parte.type == "refusal":
                raise RuntimeError(f"Il modello ha rifiutato: {parte.refusal}")
            if parte.type == "output_text":
                # Con strict=True questo testo e' JSON garantito conforme.
                return json.loads(parte.text)

    raise RuntimeError(f"Nessun output testuale nella risposta: {response.output!r}")


def analizza_pdf(percorso: str) -> dict:
    """Invia il PDF (base64) a OpenAI e restituisce il dizionario {testo, tabelle}."""
    client = OpenAI()  # legge OPENAI_API_KEY dall'ambiente

    dimensione_mb = os.path.getsize(percorso) / (1024 * 1024)
    if dimensione_mb > 20:
        print(
            f"[attenzione] Il PDF pesa {dimensione_mb:.1f} MB: con il base64 puoi "
            f"sforare i limiti. Valuta 'analizza_pdf_grande' (Files API).",
            file=sys.stderr,
        )

    with open(percorso, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.responses.create(
        model=MODEL,
        max_output_tokens=8000,  # alza il valore se il documento e' lungo
        text=TEXT_FORMAT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": os.path.basename(percorso),
                        "file_data": f"data:application/pdf;base64,{b64}",
                    },
                    {"type": "input_text", "text": PROMPT},
                ],
            }
        ],
    )
    return _estrai_json(response)


def prefisso_da_pdf(percorso: str) -> str:
    """Ricava il prefisso degli output: nome del PDF senza cartella ed estensione.

    Esempio: '/dati/Fattura 2025.pdf' -> 'Fattura 2025'
    """
    nome_file = os.path.basename(percorso)
    return os.path.splitext(nome_file)[0]


def salva_risultati(risultato: dict, prefisso: str, cartella: str = ".") -> None:
    """Scrive tutti gli output su disco usando 'prefisso' come radice dei nomi.

    Produce:
      - {prefisso}.json          -> risultato completo (testo + tabelle)
      - {prefisso}.txt           -> solo il testo non strutturato
      - {prefisso}_tabella_N.csv -> una per ogni tabella trovata
    """
    os.makedirs(cartella, exist_ok=True)

    percorso_json = os.path.join(cartella, f"{prefisso}.json")
    with open(percorso_json, "w", encoding="utf-8") as f:
        json.dump(risultato, f, ensure_ascii=False, indent=2)
    print(f"JSON   -> {percorso_json}")

    percorso_txt = os.path.join(cartella, f"{prefisso}.txt")
    with open(percorso_txt, "w", encoding="utf-8") as f:
        f.write(risultato.get("testo", ""))
    print(f"Testo  -> {percorso_txt}")

    tabelle = risultato.get("tabelle", [])
    if not tabelle:
        print("Nessuna tabella trovata nel documento.")
        return
    for i, tab in enumerate(tabelle, start=1):
        percorso_csv = os.path.join(cartella, f"{prefisso}_tabella_{i}.csv")
        with open(percorso_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(tab["intestazioni"])
            writer.writerows(tab["righe"])
        titolo = tab.get("titolo") or "(senza titolo)"
        print(f"Tabella {i} [{titolo}] -> {percorso_csv} ({len(tab['righe'])} righe)")


# ---------------------------------------------------------------------------
# Variante per PDF grandi o riusati spesso: Files API.
# Carichi il file UNA volta (purpose='user_data'), ottieni un file_id e lo
# riferisci, senza ri-trasmettere/ri-codificare il PDF a ogni richiesta.
# ---------------------------------------------------------------------------
def analizza_pdf_grande(percorso: str) -> dict:
    client = OpenAI()

    caricato = client.files.create(
        file=open(percorso, "rb"),
        purpose="user_data",
    )

    response = client.responses.create(
        model=MODEL,
        max_output_tokens=8000,
        text=TEXT_FORMAT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": caricato.id},
                    {"type": "input_text", "text": PROMPT},
                ],
            }
        ],
    )
    return _estrai_json(response)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analizza_pdf_chatgpt.py percorso/al/documento.pdf [cartella_output]")
        sys.exit(1)

    percorso_pdf = sys.argv[1]
    cartella_output = sys.argv[2] if len(sys.argv) > 2 else "."

    prefisso = prefisso_da_pdf(percorso_pdf)
    risultato = analizza_pdf(percorso_pdf)
    salva_risultati(risultato, prefisso, cartella_output)