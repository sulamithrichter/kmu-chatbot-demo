# KMU-Chatbot – „Soll & Haben Treuhand Basel"

Ein RAG-basierter Firmen-Chatbot: Kundinnen und Kunden stellen auf der Website
Fragen, der Bot antwortet **ausschliesslich aus den Firmendokumenten** (Preise,
Dienstleistungen, FAQ, Kontakt). Kein erfundenes Wissen, ehrliche „weiss ich
nicht"-Antworten.

> **Hinweis:** „Soll & Haben Treuhand Basel" ist eine **fiktive Firma**.
> Adresse, Kontaktdaten und Personen sind erfunden. Dies ist ein Demo- und
> Portfolioprojekt (zugleich Fallstudie einer Maturaarbeit).

## Was es kann

- Fragen zu Dienstleistungen, Preisen, Öffnungszeiten etc. aus den Dokumenten
  beantworten
- Bei Fragen ausserhalb des Kontexts höflich auf den Kontakt verweisen
- Antworten auf Deutsch, in der Sie-Form, im trockenen Ton der Firma
- **Offline-Fallback:** Ohne API-Key/Guthaben liefert der Bot statt eines
  Fehlers die passende Information direkt aus den Unterlagen

## Architektur (RAG in einem Satz)

```
Frage  →  Retrieval (TF-IDF + Cosine Similarity, reines Python)
       →  relevante Dokumente als Kontext in den Prompt
       →  Claude (Anthropic API) formuliert die Antwort
       →  bei API-Ausfall: Offline-Fallback aus denselben Dokumenten
```

Bewusst **ohne** externe Vektordatenbank, ohne LangChain, ohne Embedding-
Dienst – die Retrieval-Logik ist selbst gebaut und vollständig nachvollziehbar
(siehe `LERNNOTIZEN.md`).

## Tech-Stack

Python · Flask · Anthropic Claude API (`claude-haiku-4-5`) · Vanilla
HTML/CSS/JS · TF-IDF/Cosine Similarity in reinem Python

## Setup

```bash
# 1. Virtuelle Umgebung
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 2. API-Key konfigurieren
cp .env.example .env
#   -> ANTHROPIC_API_KEY in .env eintragen (Key: console.anthropic.com)
#   .env wird NIE committet (steht in .gitignore)

# 3. Starten
./.venv/bin/python app.py
#   -> http://127.0.0.1:5001
```

Ohne gültigen Key/Guthaben startet die App trotzdem und läuft im
**Offline-Modus** (Antworten direkt aus den Dokumenten).

## Projektstruktur

```
app.py                  Flask-Backend (/ und /chat), Anthropic-Aufruf, Fallback
rag.py                  Retrieval: Laden, TF-IDF, Cosine Similarity
hybrid_rag.py           Optional: Hybrid (TF-IDF + Embeddings + RRF + Reranker)
documents/              Firmendokumente (Wissensbasis des Bots)
templates/index.html    Chat-Oberfläche
static/                 style.css, chat.js
experiment_embeddings.py  Vergleich TF-IDF vs. Embeddings (optional, Lernzweck)
DEVLOG.md               Entwicklungstagebuch (Entscheidungen, Stolpersteine)
LERNNOTIZEN.md          Konzept-Vertiefung & Trade-off-Analysen
```

## Bewusste Design-Entscheidungen (Trade-offs als Stärke)

Dieses Projekt dokumentiert seine Grenzen ehrlich statt sie zu verstecken:

- **TF-IDF statt Embeddings:** transparent und selbst gebaut; die
  Wort-statt-Bedeutung-Schwäche wird in `LERNNOTIZEN.md` (Kap. 1–2, 7) mit
  einem eigenen Vergleichsexperiment belegt.
- **Ein Chunk pro Dokument:** für ein kleines Korpus schlägt Vollständigkeit
  feine Präzision (Kap. 6).
- **Graceful Degradation:** API-Ausfall → nützliche Offline-Antwort statt
  Fehlermeldung (Kap. 5).

Die ausführliche Begründung und die Lernmomente stehen in `DEVLOG.md` und
`LERNNOTIZEN.md` – sie sind Teil des Projekts, nicht Beiwerk.

## Hybrid-Modus (optional, fortgeschritten)

Höhere Antwortqualität durch eine Hybrid-Pipeline:
**TF-IDF (exakte Begriffe) + Embeddings (Bedeutung) + Reciprocal Rank
Fusion + Cross-Encoder-Reranker.** Alle Suchschritte laufen **lokal**
(datenschutzkonform); nur die finale Antwort geht an die LLM-API.

```bash
./.venv/bin/pip install -r requirements-hybrid.txt        # schwer (PyTorch)
RETRIEVER=hybrid ./.venv/bin/python app.py                # lokale Modelle
```

Ohne `RETRIEVER=hybrid` läuft der schlanke TF-IDF-Standard (keine schweren
Abhängigkeiten). Fehlen die Hybrid-Pakete, fällt die App automatisch auf
TF-IDF zurück. Konzept & Messungen: `LERNNOTIZEN.md` Kap. 8.

## Optionales Experiment

```bash
./.venv/bin/pip install -r requirements-experiment.txt   # schwer (PyTorch)
./.venv/bin/python experiment_embeddings.py              # lokal, kostenlos
```
Vergleicht TF-IDF mit einem lokalen Embedding-Modell auf denselben Fragen.

## Sicherheit

Der API-Key gehört **ausschliesslich** in die (gitignorierte) `.env` – nie in
den Code, nie in `.env.example`, nie ins öffentliche Repo.

---

Erstellt von Sulamith Richter · [sulamithrichter.ch](https://sulamithrichter.ch)
· Demo-/Portfolioprojekt
