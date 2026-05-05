# BESS Tool MVP

Programma locale per simulazione tecnico-economica di sistemi di accumulo BESS in ambito industriale.

## Scopo
Aiutare Carlo / HCE / installatori / clienti industriali a rispondere in modo concreto a questa domanda:

**“Se installo un BESS qui, come posso usarlo, quanto valore genera e quale taglia ha senso?”**

## Missione
Il tool deve:
1. leggere uno scenario energetico aziendale
2. simulare il comportamento di un BESS
3. valutare diverse logiche operative
4. stimare benefici energetici ed economici
5. costruire un business plan comprensibile
6. supportare vendita, proposta e decisione di investimento

## Ambito MVP
In questa fase il tool deve:
- leggere un **case file standardizzato**
- simulare scenari BESS su sito industriale
- mostrare il funzionamento tecnico stagionale
- restituire KPI energetici
- restituire KPI economici
- generare payback, NPV e IRR

## Scenari
- **S1** baseline senza FV
- **S2** stato attuale con FV
- **S3** BESS solo autoconsumo
- **S4** BESS multilayer: autoconsumo + peak shaving + shifting

## Principi chiave
- il tool deve portare a una **decisione**, non solo a grafici
- distinguere sempre tra **taglia teorica** e **taglia commerciale**
- distinguere sempre i layer economici:
  - saving FV via BESS
  - saving riduzione quota potenza
  - margine energy shifting
- il BESS va trattato come **asset tecnico reale**, non come scatola astratta

## Parametri tecnici minimi del BESS
- capacità nominale
- capacità utile
- potenza di carica/scarica
- SOC minimo e massimo
- efficienza round-trip
- cicli equivalenti annui
- limiti operativi reali

## Forma del programma
Pipeline attuale MVP:

**input case file standardizzato -> motore Python -> results.json -> dashboard HTML**

## Fuori scope per ora
- import universale da qualsiasi bolletta / CSV
- catalogo completo di mercato
- motore incentivi completo
- SaaS / multiutente
- sizing automatico avanzato multi-obiettivo completo
- due diligence bancaria completa

## Obiettivo operativo immediato
Chiudere un programma locale usabile entro breve, con:
- un motore funzionante
- una dashboard tecnica
- KPI chiari
- business plan con NPV
