"""
Esempio: analisi di un PDF con l'API Claude (versione con TOOL USE).

Cosa fa:
  1. Legge un PDF locale e lo invia a Claude come "document block".
  2. Tramite TOOL USE, Claude restituisce un oggetto strutturato e GIA' VALIDO
     (testo + tabelle): niente parsing manuale del JSON, niente errori dovuti a
     virgolette o caratteri speciali nel testo.
  3. Per ogni PDF vengono salvati:
       - {nome}.json          -> risultato completo
       - {nome}.txt           -> solo il testo non strutturato
       - {nome}_tabella_N.csv -> una per ogni tabella trovata
     dove {nome} = nome del PDF senza cartella ne' estensione.

Prerequisiti:
  pip install anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."   # mai hardcodare la chiave nel codice

Limiti del supporto PDF nativo:
  - max 100 pagine e 32 MB sull'intero payload della richiesta
  - il base64 gonfia i dati di ~33% (un PDF da ~24 MB sfora il limite)
  - per file grandi o riutilizzati spesso: usare la Files API (vedi in fondo)

References:
    https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python
"""

import base64
import csv
import json
import os
import sys

import anthropic

# Sonnet 4.6 = buon equilibrio costo/capacita per la maggior parte dei documenti.
# Per PDF complessi (tabelle fitte, scansioni difficili) valutare "claude-opus-4-8".
MODEL = "claude-sonnet-4-6"
TOOL_NAME = "registra_risultati_documento"

# Istruzione breve: lo SCHEMA dei dati vive ora nel tool, non nel prompt.
PROMPT = (
    "Analizza il documento PDF allegato. Trascrivi tutto il testo e, se sono "
    "presenti tabelle, ricostruiscine la struttura. Restituisci i risultati "
    f"chiamando il tool '{TOOL_NAME}'."
    "Se riconosci dei check di selezione (es. caselle da spuntare), genera una "
    "tabella con due colonne: 'Elemento' e 'Selezionato' (valore booleano)."
)

# Definizione del tool: lo schema costringe il modello a restituire dati validi.
TOOL = {
    "name": TOOL_NAME,
    "description": (
        "Registra il testo estratto e le tabelle ricostruite dal documento PDF."
    ),
    "input_schema": {
        "type": "object",
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
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "description": (
                                "Righe della tabella. Ogni riga ha lo stesso "
                                "numero di elementi delle intestazioni."
                            ),
                        },
                    },
                    "required": ["intestazioni", "righe"],
                },
            },
        },
        "required": ["testo", "tabelle"],
    },
}


def carica_pdf_base64(percorso: str) -> str:
    """Legge un PDF da disco e lo codifica in base64."""
    dimensione_mb = os.path.getsize(percorso) / (1024 * 1024)
    if dimensione_mb > 24:
        # Oltre ~24 MB il base64 supera il limite di 32 MB del payload.
        print(
            f"[attenzione] Il PDF pesa {dimensione_mb:.1f} MB: vicino/oltre il "
            f"limite. Valuta la Files API (vedi 'analizza_pdf_grande').",
            file=sys.stderr,
        )
    with open(percorso, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _estrai_input_tool(message) -> dict:
    """Recupera l'input del tool dalla risposta. Gestisce il troncamento."""
    if message.stop_reason == "max_tokens":
        raise RuntimeError(
            "Risposta troncata (max_tokens raggiunto): il documento e' troppo "
            "lungo per il valore di max_tokens. Aumenta max_tokens o spezza il PDF."
        )
    for blocco in message.content:
        if blocco.type == "tool_use":
            # blocco.input e' GIA' un dict Python validato secondo lo schema.
            return blocco.input
    raise RuntimeError(
        "Il modello non ha chiamato il tool. stop_reason="
        f"{message.stop_reason}. Contenuto: {message.content!r}"
    )


def analizza_pdf(percorso: str) -> dict:
    """Invia il PDF a Claude e restituisce il dizionario {testo, tabelle}."""
    client = anthropic.Anthropic()  # legge ANTHROPIC_API_KEY dall'ambiente
    pdf_b64 = carica_pdf_base64(percorso)

    message = client.messages.create(
        model=MODEL,
        max_tokens=8000,  # alza il valore se il documento e lungo
        tools=[TOOL],
        # Forza l'uso del tool: il modello DEVE rispondere chiamandolo.
        tool_choice={"type": "tool", "name": TOOL["name"]},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    return _estrai_input_tool(message)


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

    # 1) JSON con il risultato completo
    percorso_json = os.path.join(cartella, f"{prefisso}.json")
    with open(percorso_json, "w", encoding="utf-8") as f:
        json.dump(risultato, f, ensure_ascii=False, indent=2)
    print(f"JSON   -> {percorso_json}")

    # 2) Testo non strutturato
    percorso_txt = os.path.join(cartella, f"{prefisso}.txt")
    with open(percorso_txt, "w", encoding="utf-8") as f:
        f.write(risultato.get("testo", ""))
    print(f"Testo  -> {percorso_txt}")

    # 3) Un CSV per ogni tabella
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
# Variante per PDF grandi o riusati spesso: Files API (consigliata in produzione)
# Carichi il file UNA volta, ottieni un file_id e lo riferisci nelle chiamate,
# senza ri-trasmettere/ri-codificare il PDF a ogni richiesta.
# ---------------------------------------------------------------------------
def analizza_pdf_grande(percorso: str) -> dict:
    client = anthropic.Anthropic()

    with open(percorso, "rb") as f:
        caricato = client.beta.files.upload(
            file=(os.path.basename(percorso), f, "application/pdf")
        )

    message = client.beta.messages.create(
        model=MODEL,
        max_tokens=8000,
        betas=["files-api-2025-04-14"],  # header beta richiesto dalla Files API
        tools=[TOOL],
        tool_choice={"type": "tool", "name": TOOL["name"]},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "file", "file_id": caricato.id},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    return _estrai_input_tool(message)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analizza_pdf_claude.py percorso/al/documento.pdf [cartella_output]")
        sys.exit(1)

    percorso_pdf = sys.argv[1]
    cartella_output = sys.argv[2] if len(sys.argv) > 2 else "."

    prefisso = prefisso_da_pdf(percorso_pdf)
    risultato = analizza_pdf(percorso_pdf)
    salva_risultati(risultato, prefisso, cartella_output)
