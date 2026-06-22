# Corso Python/pandas — Trattamento dati da OCR di rapportini di lavoro

Materiale didattico completo: dati di esempio "sporchi" (output OCR simulato) e
codice per analizzarli, normalizzarli a un formato comune e ricavarne statistiche.

Tema: **rapportini di lavoro** cartacei, scannerizzati e passati all'OCR da
quattro ditte diverse, ciascuna con un proprio modulo e un proprio formato.

## Contenuto della cartella

### Script (da eseguire in ordine)

| File | Cosa mostra |
|------|-------------|
| `01_genera_dati.py` | Genera i 4 file sorgente con errori OCR realistici (~135.000 record totali) |
| `utils_ocr.py` | Funzioni di pulizia riutilizzabili (parsing numeri/date, correzione caratteri, mapping mansioni). Eseguibile da solo per i micro-test |
| `02_analisi_esplorativa.py` | Caricamento dei 4 formati, ispezione struttura, valori mancanti, anomalie OCR, confronto schemi |
| `03_normalizzazione.py` | Mapping dei 4 schemi eterogenei verso un unico schema comune, parsing del testo OCR grezzo con regex, de-duplicazione |
| `04_statistiche.py` | Statistiche descrittive, aggregazioni groupby, pivot, KPI e distillazione, export Excel |

Tutti i file dati stanno nella sottocartella `dati/`. Gli script la individuano
automaticamente (percorso ancorato alla posizione dello script), quindi funzionano
da qualsiasi directory di lavoro.

```
esempi/
├── 01_genera_dati.py
├── 02_analisi_esplorativa.py
├── 03_normalizzazione.py
├── 04_statistiche.py
├── utils_ocr.py
├── README.md
├── dati/                               (dati SORGENTE)
│   ├── rapportini_alfa.csv
│   ├── rapportini_beta.csv
│   ├── rapportini_gamma.json
│   └── rapportini_delta.ndjson
└── output/                             (RISULTATI prodotti dagli script)
    ├── dataset_unificato.parquet / .csv   (creati dalla fase 3)
    └── report_statistiche.xlsx            (creato dalla fase 4)
```

### File dati sorgente (generati da `01_genera_dati.py` in `dati/`)

| File | Formato | Particolarita' didattica |
|------|---------|--------------------------|
| `rapportini_alfa.csv` | CSV `;` decimali `,` | Intestazioni IT, date `gg/mm/aaaa`, ore ordinarie e straordinarie **separate** |
| `rapportini_beta.csv` | CSV `,` decimali `.` | Intestazioni EN, date ISO, ore gia' **sommate** |
| `rapportini_gamma.json` | JSON array annidato | Campi `operatore.nome`, confidenza OCR per campo, numeri come stringa con virgola |
| `rapportini_delta.ndjson` | NDJSON (json lines) | Campo `raw_text` da **parsare con regex**, confidenza per pagina |

### File prodotti dal codice

- `output/dataset_unificato.parquet` / `.csv` — dataset pulito a schema comune (output fase 3)
- `output/report_statistiche.xlsx` — report multi-foglio (output fase 4)

## Come si esegue

```bash
pip install pandas numpy pyarrow openpyxl
python 01_genera_dati.py        # crea i file sorgente
python utils_ocr.py             # (opzionale) verifica le funzioni di pulizia
python 02_analisi_esplorativa.py
python 03_normalizzazione.py    # crea dataset_unificato.*
python 04_statistiche.py        # crea report_statistiche.xlsx
```

## Problemi OCR riprodotti nei dati

- **Caratteri confusi**: `O↔0`, `l/I↔1`, `S↔5`, `B↔8`, `Z↔2`, `G↔6` (es. `Elettr1co`, `C0nti`, `6iuseppe`)
- **Separatori decimali misti**: virgola (IT) vs punto (EN), separatore migliaia
- **Formati data eterogenei**: `gg/mm/aaaa`, `aaaa-mm-gg`, `gg-mm-aaaa`, `gg.mm.aa`, `07 Mar 2025`
- **Celle illeggibili**: `""`, `###`, `ILLEGGIBILE`, `-`, `n/d`
- **Rumore di bordo**: spazi multipli, caratteri `| * .` a fine cella
- **Schemi diversi** tra ditte: nomi colonne, lingua, annidamento, granularita' delle ore
- **Confidenza OCR**: presente in GAMMA (per campo) e DELTA (per pagina); permette di filtrare le righe a rischio

## Schema comune di destinazione

```
record_id | fonte | data | operatore | commessa | mansione |
ore | costo_materiali | confidenza_ocr | valido
```

## Schema del flusso

```
  4 file sorgente (formati diversi)
          |
   [02] analisi esplorativa  --> capisco i problemi
          |
   [03] normalizzazione      --> schema comune unico + de-dup
          |
   [04] statistiche          --> aggregazioni, pivot, KPI, Excel
```

## Spunti per esercizi in aula

1. **Entity resolution / fuzzy matching**: dopo la normalizzazione restano nomi quasi-duplicati
   (es. `Alessandro Giordano` vs `Aiessandro Giordano`) perche' la correzione esatta non basta.
   Risultato: 30 operatori "distinti" invece dei 20 reali. Introdurre `rapidfuzz`/`difflib`
   per consolidarli e mostrare l'impatto sui conteggi.
2. **Filtro per confidenza**: ricalcolare le statistiche tenendo solo i record con
   `confidenza_ocr >= 0.8` e confrontare i risultati.
3. **Recupero righe scartate**: analizzare i record con `valido == False` per capire
   quale campo li ha invalidati e provare regole di recupero piu' aggressive.
4. **Performance**: caricare `rapportini_beta.csv` a `chunksize` ed elaborare a blocchi;
   confrontare tempi e memoria con il caricamento intero.
5. **Validazione date**: individuare date impossibili o fuori periodo e gestirle.
```
```
