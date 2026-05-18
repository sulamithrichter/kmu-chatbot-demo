---
title: KMU-Chatbot Demo
emoji: 💬
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

<!-- Der YAML-Block oben konfiguriert den Hugging-Face-Space (SDK Docker,
     Port 7860). GitHub rendert ihn nicht – der Rest dieser Datei ist die
     normale Projekt-README. Details: DEPLOY.md / LERNNOTIZEN Kap. 10. -->

# KMU-Chatbot – „Soll & Haben Treuhand Basel"

Ein RAG-basierter Firmen-Chatbot: Kundinnen und Kunden stellen auf der Website
Fragen, der Bot antwortet **ausschliesslich aus den Firmendokumenten** (Preise,
Dienstleistungen, FAQ, Kontakt). Kein erfundenes Wissen, ehrliche „weiss ich
nicht"-Antworten.

**Live-Demo:** auf Hugging Face Spaces deployt (ein Docker-Container lädt
**beide** Retriever). Im Chat lässt sich pro Frage zwischen **`tf-idf`** und
**`hybrid`** umschalten – so kann man beide Verfahren direkt vergleichen.
Einbettung/Verlinkung auf [sulamithrichter.ch](https://sulamithrichter.ch).
Deployment-Schritte: siehe `DEPLOY.md`.

> **Hinweis:** „Soll & Haben Treuhand Basel" ist eine **fiktive Firma**.
> Adresse, Kontaktdaten und Personen sind erfunden. Dies ist ein Demo- und
> Portfolioprojekt.

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

Im Frontend wählt ein Umschalter pro Frage das Retrieval-Verfahren
(`tfidf` | `hybrid`); ist `hybrid` nicht geladen, fällt die App ehrlich
gekennzeichnet auf `tfidf` zurück.

## Tech-Stack

Python · Flask · Anthropic Claude API (`claude-haiku-4-5`) · Vanilla
HTML/CSS/JS · TF-IDF/Cosine Similarity in reinem Python · Hybrid: lokale
Embeddings + Cross-Encoder-Reranker · Deployment: Docker / Hugging Face Spaces

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
Dockerfile              Baut den Live-Demo-Container (backt die Modelle ein)
.dockerignore           Hält .git/.venv/.env aus dem Image
DEPLOY.md               Deployment-Schritte (Hugging Face Spaces)
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
./.venv/bin/pip install -r requirements-hybrid.txt   # schwer (PyTorch)
RETRIEVER=hybrid ./.venv/bin/python app.py
```

**Erststart:** Beim ersten Mal werden zwei lokale Modelle (~mehrere hundert
MB) **einmalig** vom Hugging-Face-Hub geladen – je nach Verbindung einige
Minuten (Meldung „Lade Hybrid-Modelle…", **nicht abbrechen**). Ab dem
zweiten Start erkennt die App die gecachten Modelle automatisch, arbeitet
offline und startet in **wenigen Sekunden**.

Fehlen die Hybrid-Pakete, fällt die App automatisch auf TF-IDF zurück. Für
schnelles lokales TF-IDF-Arbeiten ohne den schweren Modell-Ladevorgang:
`ENABLE_HYBRID=0 ./.venv/bin/python app.py`. Konzept & Messungen:
`LERNNOTIZEN.md` Kap. 8; Debugging-Lektion (HF-Hub-Hang) Kap. 9.

In der **Live-Demo** lädt ein einzelner Container beide Retriever gleichzeitig
(genug RAM auf Hugging Face Spaces); der Umschalter im Chat wählt pro Frage.
Warum Hugging Face statt eines 512-MB-Gratis-Tiers, und wie das Deployment
funktioniert: `LERNNOTIZEN.md` Kap. 10 und `DEPLOY.md`.

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
