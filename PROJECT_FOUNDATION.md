# PROJECT_FOUNDATION.md
## BESS Tool — MVP 0.1

---

## 1. Stato attuale della cartella

La cartella `bess 0.1/` è pulita: solo CLAUDE.md e README.md.
Il lavoro precedente esiste in `bess-dashboard/` (simulatore.py, dashboard.html, hello.py).
Quel lavoro è un **calcolatore finanziario grezzo**, non un simulatore reale.
Il BESS non viene simulato slot per slot — viene approssimato con formule fisse.
Questa è la differenza fondamentale che il nuovo MVP deve colmare.

---

## 2. Architettura MVP

```
case file JSON
      ↓
profile_builder.py   ← genera profilo annuale 15-min (sintetico o da CSV)
      ↓
bess_engine.py       ← simula slot per slot i 4 scenari
      ↓
economics.py         ← calcola KPI economici e business plan
      ↓
results.json         ← output strutturato
      ↓
dashboard.py         ← Streamlit: grafici + KPI + business plan
```

**Principio chiave**: il motore gira una volta (CLI o da dashboard), produce `results.json`.
La dashboard legge solo quel file — nessun calcolo nella UI.

---

## 3. Struttura file

```
bess 0.1/
├── CLAUDE.md
├── README.md
├── PROJECT_FOUNDATION.md         ← questo file
├── requirements.txt
│
├── cases/
│   └── example_case.json         ← il contratto dati: tutto parte da qui
│
├── engine/
│   ├── __init__.py
│   ├── bess_engine.py            ← simulazione slot per slot (S1-S4)
│   ├── profile_builder.py        ← profilo 35.040 slot/anno
│   └── economics.py              ← payback, NPV, IRR, layer economici
│
├── run_simulation.py             ← CLI: legge case, gira motore, scrive results.json
│
├── results/
│   └── (results_*.json)          ← output generati
│
└── dashboard.py                  ← Streamlit: legge results.json, mostra tutto
```

---

## 4. Il case file standardizzato

`cases/example_case.json` è il contratto dati del sistema.
Contiene:

```json
{
  "site": {
    "name": "Industria Esempio S.r.l.",
    "location": { "lat": 45.5, "lon": 11.5 },
    "consumo_annuo_kwh": 350000,
    "picco_richiesto_kw": 150,
    "tariffa_energia_eur_kwh": 0.20,
    "quota_potenza_eur_kw_mese": 12.0
  },
  "pv": {
    "presente": true,
    "kwp": 80,
    "source": "pvgis"
  },
  "bess": {
    "capacita_nominale_kwh": 200,
    "capacita_utile_kwh": 180,
    "potenza_carica_kw": 100,
    "potenza_scarica_kw": 100,
    "soc_min": 0.05,
    "soc_max": 0.95,
    "efficienza_roundtrip": 0.90,
    "costo_installato_eur_kwh": 255,
    "anni_vita": 15
  },
  "economico": {
    "tasso_sconto": 0.05,
    "anni_analisi": 20,
    "om_rate": 0.01,
    "degradazione_annua": 0.02
  }
}
```

MVP inizia con **profilo sintetico** generato da `consumo_annuo_kwh` + `picco_richiesto_kw`.
Fase 2 aggiunge: upload CSV reale del cliente.

---

## 5. I 4 scenari

| Scenario | Cosa simula |
|---|---|
| S1 | Baseline — no FV, no BESS |
| S2 | Con FV attuale — autoconsumo diretto, no BESS |
| S3 | BESS solo autoconsumo FV |
| S4 | BESS multilayer: autoconsumo + peak shaving + energy shifting |

---

## 6. Output obbligatori

### Tecnici
- Profili settimanali stagionali (4 stagioni × [carico, FV, carica, scarica, prelievo rete, SOC])
- Throughput annuale batteria (kWh)
- Cicli equivalenti annui
- Effetto peak shaving (picco ridotto, kW)
- Effetto autoconsumo (% FV autoconsumata)

### Economici — sempre separati
- Saving FV via BESS (€/anno)
- Saving riduzione quota potenza (€/anno)
- Margine energy shifting (€/anno)
- Saving netto annuo
- Payback semplice
- NPV (tasso 5%, 20 anni)
- IRR

---

## 7. Ordine di costruzione (priorità MVP)

1. **`cases/example_case.json`** — il contratto dati, tutto parte da qui
2. **`engine/profile_builder.py`** — genera il profilo annuale 15-min da input semplici
3. **`engine/bess_engine.py`** — simulazione slot per slot, S1-S4
4. **`engine/economics.py`** — KPI economici e business plan
5. **`run_simulation.py`** — CLI che collega tutto e scrive results.json
6. **`dashboard.py`** — Streamlit che legge results.json e mostra tutto
7. **`requirements.txt`** — dipendenze

---

## 8. Cosa importare dal lavoro precedente

| Componente | Origine | Usare? | Note |
|---|---|---|---|
| Costanti finanziarie | `simulatore.py` | Sì | EFFICIENCY=0.90, DISCOUNT_RATE=0.05, DEGRADATION=0.02 |
| Logica NPV/IRR | `simulatore.py` | Sì | `numpy_financial` — funziona |
| Grafico cash flow (Plotly) | `simulatore.py` | Sì | Struttura riusabile |
| Formula sizing BESS | `simulatore.py` | No | Troppo grezza (`capacity = picco * 2`) |
| dashboard.html | `bess-dashboard/` | No | Statico, abbandonato |
| offerta.html | `bess-dashboard/` | No | Solo riferimento visivo |
| 3 layer economici | Obsidian note 04 | Sì | Struttura corretta da mantenere |
| SOC tracking con quota_fv | Obsidian note 04 | Sì | Importante per evitare doppio conteggio |
| Peak shaving selettivo | Obsidian note 04 | Sì | Solo giorni critici (≥ 85% picco mensile) |
| Carica notturna modulata | Obsidian note 04 | Sì | Solo inverno, solo 00-07 |
| BESSConfig / SiteConfig | Obsidian note 04 | Sì | Struttura dati corretta |

---

## 9. Fuori scope (MVP 0.1)

- Import automatico da PDF bollette
- Fetch PVGIS API (profilo FV da API esterna) — MVP usa stima semplice
- Prezzi MGP orari per arbitraggio reale
- Sizing optimizer automatico multi-taglia
- SaaS / multiutente
- Incentivi statali (Conto Energia, FER X, ecc.)
- Report PDF esportabile

---

## 10. Parametri tecnici BESS obbligatori

Il motore deve sempre usare:
- `capacita_nominale_kwh` e `capacita_utile_kwh` (non la stessa cosa)
- `potenza_carica_kw` e `potenza_scarica_kw`
- `soc_min`, `soc_max` (limiti operativi reali)
- `efficienza_roundtrip` applicata sia in carica che in scarica (√η per ognuno)
- `cicli_equivalenti` calcolati dall'output della simulazione

---

## 11. Tecnologie

| Tool | Uso |
|---|---|
| Python 3.11+ | tutto il backend |
| numpy | profili e calcoli vettoriali |
| numpy_financial | NPV, IRR |
| Streamlit | dashboard |
| Plotly | grafici |

---

## 12. Come avviare il programma (quando sarà pronto)

```bash
# 1. Crea/modifica il case file
nano cases/example_case.json

# 2. Gira la simulazione
python run_simulation.py cases/example_case.json

# 3. Apri la dashboard
streamlit run dashboard.py
```
