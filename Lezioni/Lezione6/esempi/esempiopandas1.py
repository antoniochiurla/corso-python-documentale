import pandas as pd

# 1. Creiamo i dati di base (es. ordini effettuati)
dati_ordini = {
    'ID_Ordine': 648,
    'Data': ['2026-06-15', '2026-06-16', '2026-06-16', '2026-06-17', '2026-06-17'],
    'Prodotto': ['Mouse Wireless', 'Tastiera Meccanica', 'Monitor 27"', 'Mouse Wireless', 'Cavo HDMI'],
    'Quantita': [2, 4, 7, 3, 2],
    'Prezzo_Unitario': [25.50, 79.90, 249.00, 25.50, 12.00]
}

# 2. Generiamo il DataFrame
df = pd.DataFrame(dati_ordini)

# 3. Convertiamo la colonna 'Data' in formato datetime
df['Data'] = pd.to_datetime(df['Data'])

# 4. Calcoliamo il totale per riga (Quantità * Prezzo) usando LaTeX: $Totale = Quantità \times Prezzo$
df['Totale_Riga'] = df['Quantita'] * df['Prezzo_Unitario']

# Visualizziamo il DataFrame
print(df)

