# DEVLOG – KMU-Chatbot "Soll & Haben Treuhand Basel"

Entwicklungstagebuch. Pro Session: Was gebaut wurde, welche Entscheidungen warum getroffen wurden, Lernmomente/Stolpersteine, offene Punkte.

---

## Session 1 – 2026-05-15

### Was gebaut wurde
- Projektstruktur angelegt: `documents/`, `static/`, `templates/`
- Vier Firmendokumente erstellt (Dummy-Inhalt, Schweizer Treuhand-Kontext):
  - `dienstleistungen.md`, `preise.md`, `faq.md`, `kontakt.md`
- `requirements.txt` (Flask, anthropic, python-dotenv)
- `.env.example` als Vorlage für den API-Key
- `.gitignore` (schützt die echte `.env` vor dem öffentlichen Repo)

### Entscheidungen & Begründungen
- **Firmenname "Soll & Haben Treuhand Basel":** Klingt wie eine echte, seriöse
  Kanzlei; der Witz (Buchhaltungs-Fachbegriff als Firmenname) erschliesst sich
  erst auf den zweiten Blick. Passt zum gewünschten "Hauch Humor".
- **Humor dezent & trocken, im Stil von sulamithrichter.ch:** Understatement
  und Selbstironie in einzelnen Sätzen, nicht in der ganzen Firma – so bleibt
  die Demo glaubwürdig und ist als echtes Portfoliostück nutzbar.
- **Anrede "Sie" in den Dokumenten:** Eine Schweizer Treuhandfirma siezt ihre
  KMU-Kunden. Realistischer für die Demo als das "Du" der persönlichen Website.
- **Inhalt fachlich korrekt gehalten:** Echte CH-Schwellenwerte (MWST-Pflicht
  ab CHF 100'000 Umsatz, doppelte Buchhaltung ab CHF 500'000, 10 Jahre
  Belegaufbewahrung nach Art. 958f OR). So lernt man am Demo auch etwas Echtes.
- **Dokumente klar strukturiert (kurze Abschnitte, klare Titel):** Wichtig fürs
  spätere "Chunking" im RAG – darüber sprechen wir in Schritt 2.
- **API-Key bewusst aufgeschoben:** Struktur, Dokumente, RAG-Logik und Frontend
  brauchen keinen Key. Nur der Live-Test braucht ihn. Wird in Schritt 5 geklärt.

### Lernmomente / Stolpersteine
- `.gitignore` ist hier kein Detail, sondern Pflicht: Das Repo wird public, ein
  versehentlich committeter API-Key wäre öffentlich. Erste Sicherheitsregel.

### Offene Punkte
- [ ] Anthropic API-Key besorgen (console.anthropic.com, $5 Gratisguthaben)
- [ ] GitHub-Repo anlegen (github.com/sulamithrichter/kmu-chatbot-demo)

---

## Session 2 – 2026-05-15

### Was gebaut wurde
- RAG-Konzept erklärt (Text, kein Code): Grundproblem, Pipeline, Chunking,
  Vektoren & Cosine Similarity, Embeddings vs. TF-IDF, keine Vektor-DB nötig
- `LERNNOTIZEN.md` angelegt – Begleitheft für vertiefende Konzepte/Alternativen
  (Kapitel 1: Embeddings vs. TF-IDF, lokal vs. API, Datenschutz bei Treuhand)
- `rag.py` implementiert (TF-IDF + Cosine Similarity, reines Python) und mit
  5 Testfragen per Selbsttest verifiziert – alle Treffer korrekt
- LERNNOTIZEN Kapitel 2 ergänzt: reale Schwächen aus dem Selbsttest

### Entscheidungen & Begründungen
- **Retrieval-Verfahren: TF-IDF + Cosine Similarity in reinem Python.**
  Begründung: erfüllt "alles selbst gebaut, keine externen Dienste, voll
  verstehbar"; ideal fürs Lernziel (Mathematik selbst erklärbar);
  für 4 kleine Dokumente qualitativ ausreichend. Architektur ist identisch
  zu Embedding-RAG – nur der Vektorisierungs-Schritt unterscheidet sich.
- **Bewusst gegen Embeddings entschieden** (für *dieses* Demo): Anthropic hat
  keine Embedding-API; Alternativen wären externer Dienst (widerspricht
  Vorgaben) oder schweres lokales ML-Modell (widerspricht "einfach/verstehbar").
- **Lernstoff in separater Datei statt im DEVLOG:** Trennung von "Was gebaut"
  (DEVLOG) und "Konzept-Vertiefung/Alternativen" (LERNNOTIZEN.md), auf Wunsch.

### Lernmomente / Stolpersteine
- Erkenntnis fürs Verständnis: RAG-Pipeline ist immer gleich (chunk →
  vektorisieren → Cosine Similarity → Prompt). TF-IDF vs. Embeddings tauschen
  *nur* den Vektorisierungs-Schritt. Das entzaubert "Embeddings" als Mysterium.
- Selbsttest deckte reale TF-IDF-Grenzen auf (Details in LERNNOTIZEN Kap. 2):
  (1) Vokabular-Lücke "offen" vs. "Öffnungszeiten" → Dokument in Nutzer-Worten
  formuliert; (2) deutsches Kompositum "Mehrwertsteuer" vs. "...pflichtig" ohne
  Stemming; (3) Stoppwortliste ist korpus-spezifisch, mit echten Fragen
  validieren statt nach Bauchgefühl.
- Wichtigste Einsicht: Retrieval muss nicht perfekt sortieren – es muss den
  richtigen Chunk nur unter die Top-k bringen; das LLM urteilt fein nach.

### Offene Punkte
- [ ] Anthropic API-Key besorgen
- [ ] GitHub-Repo anlegen

---

## Session 3 – 2026-05-16

### Was gebaut wurde
- Konzept-Erklärung Flask + Anthropic Messages-API (Text); LERNNOTIZEN Kap. 3
- `app.py`: Flask-Backend mit `/` (Frontend später) und `/chat` (POST):
  RAG → Prompt → Claude → JSON. RAGIndex einmal beim Start.
- venv `.venv/` angelegt, requirements installiert
- Endpoints lokal mit curl getestet (ohne API-Key, graceful)

### Entscheidungen & Begründungen
- **Modell `claude-haiku-4-5`** (über Anthropic-Doku verifiziert, nicht
  geraten): günstigstes aktuelles Modell ($1/$5 pro Mio Tok), für RAG-Q&A
  ausreichend. Als eine Konstante gehalten (1-Zeilen-Wechsel). ~$0.0025/Frage
  → die $5 reichen für ~2000 Fragen.
- **System-Prompt** mit harter Wissensgrenze ("nur aus Kontext, sonst ehrlich
  weiss ich nicht") + Sie-Form + dezenter Ton. Kontext-Chunks in die
  user-Nachricht (System = Verhalten, Message = Daten).
- **Graceful degradation:** kein Key → Server läuft trotzdem, /chat meldet es
  freundlich. Erlaubt Testen der ganzen Verkabelung ohne Key.
- **Port 5001 statt 5000** (konfigurierbar via PORT-Env).

### Lernmomente / Stolpersteine
- `Anthropic()` wirft ohne Key sofort eine Exception → Client nur bauen, wenn
  Key vorhanden (`... if API_KEY else None`), sonst startet der Server nicht.
- `print()` wird beim Umleiten in eine Datei gepuffert → `flush=True`, sonst
  ist der Dev-Log unsichtbar.
- **macOS-Falle:** Port 5000 ist vom AirPlay-Empfänger ("ControlCenter")
  belegt. Symptom: leere curl-Antworten, Flask bindet nie. Lösung: Port 5001.
- Flask `debug=True` startet zwei Prozesse (Reloader); `kill` des Eltern-
  prozesses lässt einen verwaisten Server zurück. Für automatisiertes Testen
  besser `use_reloader=False`.

### Offene Punkte
- [ ] Schritt 4: Frontend (index.html, style.css, chat.js)
- [ ] Anthropic API-Key besorgen (für Live-Test in Schritt 5)
- [ ] GitHub-Repo anlegen

---

## Session 4 – 2026-05-16

### Was gebaut wurde
- Frontend: `templates/index.html`, `static/style.css`, `static/chat.js`
- Verkabelung end-to-end getestet (Port 5001): Seite, statische Dateien,
  /chat, RAG-Log – alles grün (ohne Live-Key, graceful)
- LERNNOTIZEN Kapitel 4: Secrets-Hygiene

### Entscheidungen & Begründungen
- Vanilla HTML/CSS/JS (kein Framework), wie im Projektrahmen vorgegeben.
- Antworten via `textContent` statt `innerHTML` → kein HTML-Injection-Risiko.
- Eingabe als `<form>` → Enter-Taste sendet ohne Extra-Code.
- Ruhige Treuhand-Palette + dezente Tagline (Marke „Soll & Haben").

### Lernmomente / Stolpersteine
- **Security-Vorfall (eingegrenzt, behoben):** Der echte API-Key wurde in
  `.env.example` eingetragen. Diese Datei ist NICHT gitignored (nur `.env`),
  wäre also bei einem Push ins public Repo öffentlich gewesen. Behebung:
  Key aus `.env.example` entfernt (Platzhalter), echte `.env` (gitignored)
  angelegt. Kein Git-Repo existierte → nichts gepusht, Schaden eingegrenzt.
  Profi-Konsequenz: Key wird rotiert (alter widerrufen, neuer erstellt), weil
  ein Secret, das je in einer commit-bestimmten Datei stand, als kompromittiert
  gilt. Mentales Modell: `.env.example` = öffentliche leere Vorlage,
  `.env` = privates Geheimnis. Gleicher Name, gegensätzlicher Zweck.

### Offene Punkte
- [ ] **Sulamith: alten Key in console.anthropic.com widerrufen, neuen Key
      erstellen, in `.env` eintragen** (nicht in `.env.example`!)
- [ ] Schritt 5: Live-Test mit neuem Key + README finalisieren
- [ ] GitHub-Repo anlegen (vorher prüfen: `.env` NICHT enthalten)

---

## Session 5 – 2026-05-16

### Was gebaut wurde
- **Offline-Fallback** in `app.py`: schlägt der Live-Call fehl (kein Key,
  kein Guthaben, Netzwerk), wird die Antwort aus den Top-Chunks gebaut
  (`antwort_aus_chunks`, `_klartext`), JSON-Feld `mode: live|offline`.
- Voll getestet: Treffer→Auszug, kein Treffer→höfliche Absage, leer→400,
  Live-Pfad nachweislich ausgeführt (echter API-Fehler im Log).

### Entscheidungen & Begründungen
- **Fallback statt 502-Fehlermeldung:** Portfolio-Demo funktioniert auch mit
  0 Credits. Graceful Degradation eine Stufe weiter, ehrlich gekennzeichnet.
- **`load_dotenv(override=True)`:** projekteigene `.env` hat Vorrang.
- **Account-Status:** Anthropic meldet „credit balance too low". Das
  erwartete Gratisguthaben ist nicht (mehr) verfügbar. Code ist korrekt &
  live-bereit; offen ist nur der Credit-Kauf durch Sulamith.

### Lernmomente / Stolpersteine
- **`load_dotenv()` überschreibt vorhandene Env-Variablen NICHT** (Default
  `override=False`). Eine leere/veraltete `ANTHROPIC_API_KEY`-Variable im
  Shell-Profil hebelte die `.env` still aus → `client=None`, Live-Pfad nie
  versucht. Fix: `override=True`. Lehre: bei „App liest .env nicht" immer
  prüfen, ob die Variable schon (leer) in der Umgebung steht.
- 401 (invalid key) vs. 400 (credit balance) unterscheiden: Letzteres
  beweist, dass Auth & Integration funktionieren – nur das Konto ist leer.
- Offline-Modus macht die Retrieval-Qualität direkt sichtbar: bei „Was
  kostet die Buchhaltung?" rankt durch das Stoppwort „was" der Erstgespräch-
  Chunk zu hoch. Im Live-Modus weniger kritisch (Claude bekommt Top-3 +
  synthetisiert). Bekannte TF-IDF-Grenze, Feinschliff-Kandidat.

### Offene Punkte
- [ ] Sulamith: Credits kaufen (console.anthropic.com → Plans & Billing)
- [ ] Live-Test im Browser (http://127.0.0.1:5001), Antworten/Ton prüfen
- [ ] README schreiben (Setup/Start/Architektur fürs Portfolio)
- [ ] Optionaler Feinschliff: Retrieval bei Kurzfragen, favicon-404
- [ ] GitHub-Repo anlegen (vorher: `.env` wirklich draussen?)

---

## Session 6 – 2026-05-16

### Was gebaut wurde
- Live-Modus bestätigt funktionsfähig (Sulamith hat Credits gekauft, Bot
  antwortet echt, Ton/System-Prompt sehr gut).
- **Chunking umgestellt: 1 Chunk = 1 Dokument** (`lade_chunks`,
  `_in_abschnitte_splitten` entfernt). Offline-Fallback gibt jetzt das
  beste *ganze* Dokument aus (`antwort_aus_chunks`, Cap 3000).
- Kleine Wortschatz-Ergänzung in `dienstleistungen.md` (Kap.-2-Technik).

### Entscheidungen & Begründungen
- Grobes Chunking, weil Live-Test zeigte: feiner Schnitt zersplitterte die
  Leistungsliste → „Was bietet ihr an?" bekam nur den Einleitungs-Chunk
  ohne Inhalt. Nach Fix: Frage trifft das ganze `dienstleistungen.md`. ✅
- **Hand-Tuning bewusst gestoppt:** „Was macht ihr?" (Konjugation
  `macht`≠`machen`) und „Welche Leistungen…" bleiben als dokumentierte
  Negativbeispiele. Weiter am Korpus schrauben würde die zu zeigende
  Schwäche verschleiern. Semantik ist Aufgabe von Experiment (B).

### Lernmomente / Stolpersteine
- Zwei empirisch belegte Tradeoffs der Chunk-Granularität:
  (1) zu fein → Kontext zersplittert (Vollständigkeit leidet);
  (2) zu grob → seltene Einzelwörter verwässern im grossen Vektor
  (Präzision leidet, z.B. „Leistungen" einmal in ganzem Dokument).
- Konjugation/Plural (`macht`/`machen`, `Angebot`/`Angebote`) ist für
  TF-IDF ohne Stemming unüberwindbar → stärkstes Argument für Embeddings.

### Offene Punkte
- [ ] (B) Embedding-Vergleichs-Experiment (eigenes Skript, lokales Modell)
- [ ] README schreiben; GitHub-Repo (mit `.env`-Check)

---

## Session 7 – 2026-05-16

### Was gebaut wurde
- `experiment_embeddings.py` + `requirements-experiment.txt` (separat,
  schwere Abhängigkeit nur fürs Experiment; `sentence-transformers` 5.5.0
  ins `.venv` installiert). Vergleich TF-IDF vs. lokales Embedding-Modell.
- LERNNOTIZEN Kapitel 7: Ergebnistabelle + Interpretation.

### Entscheidungen & Begründungen
- Lokales Modell statt API: kein Geld, kein externer Dienst, datenschutz-
  konform – passend zum Treuhand-Anwendungsfall.
- Vergleich auf identischen Chunks (rag.lade_chunks) für Fairness.

### Lernmomente / Stolpersteine
- Embeddings lösen die semantische Lücke (z.B. „Angebote für Angestellte"
  0.00→0.37), sind aber fehlbar: „Wann offen?" wurde mit Embeddings
  *falsch* (faq statt kontakt), während TF-IDF korrekt war.
- Kontroll-Frage (WM 2022) erhielt Embedding-Score 0.22 ≈ schwache gültige
  Frage 0.21 → auch Embeddings brauchen eine Schwelle, deren Wahl ist sogar
  subtiler. Sauber belegtes Gegenargument gegen „Embeddings = magisch".
- Praxisnuance: TF-IDF rankte „Welche Leistungen…" zwar top, aber 0.04 <
  min_score 0.05 → in der App trotzdem Fehlschlag.

### Offene Punkte
- [ ] README (Setup/Architektur/Trade-offs fürs Portfolio)
- [ ] GitHub-Repo anlegen – PFLICHT-Check vorher: `.env` NICHT enthalten

---

## Session 8 – 2026-05-16 (Abschluss)

### Was gebaut wurde
- `README.md` (portfolio-tauglich: Pitch, Architektur, Setup, Trade-offs als
  Stärke, Fiktiv-/Sicherheitshinweis)
- favicon-404 behoben (Route `/favicon.ico` → 204, kein Binärasset)
- `git init` + Sicherheits-Check: Dry-run beweist, dass `.env` und `.venv`
  NICHT eingecheckt werden (17 korrekte Dateien). `.env` via .gitignore Z.2.

### Status
**Projekt inhaltlich fertig.** Demo läuft live, Offline-Fallback,
Embedding-Experiment dokumentiert. Repo lokal initialisiert, push-sicher.

### Offene Punkte (in Sulamiths Hand)
- [ ] `git commit` + GitHub-Repo erstellen + `git push` (Befehle s. Chat)
- [ ] Nach erstem Push auf github.com gegenprüfen: `.env` wirklich nicht da
- [ ] Optional: System-Prompt/Ton anhand weiterer Live-Antworten feintunen
- [ ] Optional: Deployment (Vercel/Railway) – eigene spätere Session

---

## Session 9 – 2026-05-16 (Hybrid-Retrieval-Erweiterung)

Idee von Sulamith: Qualität via Hybrid-Suche steigern, TF-IDF-Idee bleibt
drin, wird erweitert.

### Was gebaut wurde
- `hybrid_rag.py` (stark kommentiert): A) TF-IDF (rag.RAGIndex wiederverwendet)
  + B) lokale Embeddings + C) Reciprocal Rank Fusion + D) Cross-Encoder-
  Reranker. Gleiche Methode wie RAGIndex -> in app.py austauschbar.
- `app.py`: Schalter `RETRIEVER` (tfidf|hybrid) mit sauberem Fallback auf
  tfidf, wenn schwere Deps fehlen. Standard bleibt tfidf.
- `requirements-hybrid.txt` (sentence-transformers, nur Hybrid-Pfad).
- Switch verifiziert (Import-Test, kein API-Call).

### Entscheidungen & Begründungen
- Lexikalischer Arm = bestehendes TF-IDF (Sulamiths Wahl): „Idee bleibt drin".
- **RRF statt Score-Addition:** TF-IDF- und Embedding-Scores liegen auf
  unvergleichbaren Skalen (in Kap. 7 gemessen). RRF kombiniert über RÄNGE,
  kalibrierungsfrei.
- **Cross-Encoder-Reranker** nur auf den Top-10 der Fusion (präzise, aber
  langsam -> nicht auf alle anwenden).
- **Schwelle pro Kandidat** statt nur auf den Bestwert: LLM bekommt nur
  wirklich relevanten Kontext (günstiger, fokussierter).
- Default tfidf, damit requirements.txt schlank bleibt; Hybrid opt-in.

### Lernmomente / Stolpersteine
- **Cross-Encoder-Scores sind UNGEEICHTE Logits** – auch korrekte Treffer oft
  negativ. `RERANK_MIN=0.0` filterte erst gültige Fragen weg. Schwelle
  datenbasiert kalibriert: gültige Fragen +1.2…-3.6, ausserhalb -8.6 ->
  -5.0 in der Lücke. (Verstärkt die Kap.-7-Lektion: Schwellen sind heikel.)
- Hybrid+Reranker löst „Was macht ihr?" – das hatten TF-IDF UND Embeddings
  je allein NICHT geschafft. Stärkstes Argument fürs Hybrid-Konzept.
- „Habt ihr Angebote für Angestellte?": Reranker-Gleichstand kontakt vs.
  dienstleistungen (-3.621 vs -3.633). top_k=3 mildert das (beide im Kontext).
- Datenschutz präzise: A–D laufen LOKAL; einzig der LLM-Schritt ist extern.

### Ergebnis
Bestes Resultat aller Verfahren: 6/7 Testfragen korrektes Dokument, 1
ausserhalb korrekt abgelehnt. Preis: zwei lokale Modelle + kalibrierte
Schwelle + mehr Komplexität.

### Offene Punkte
- [ ] (wie zuvor) git commit/push durch Sulamith; optional Tontuning/Deploy

---

## Session 10 – 2026-05-16 (Performance & ehrliche Kommunikation)

Zwei kritische Rückfragen von Sulamith.

### Was gebaut/geändert wurde
- `app.py`: `app.run(debug=True, use_reloader=False, port=PORT)`. Der
  Flask-Reloader startet die App in ZWEI Prozessen -> im Hybrid-Modus
  wurden Embedding- + Reranker-Modell DOPPELT geladen. Ohne Reloader
  laden sie genau einmal -> deutlich schnellerer Start.
- Portfolio (`sulamithrichter/portfolio`): Chatbot-Beschreibung ehrlich
  präzisiert (Retrieval selbst gebaut/lokal; Antwort via Claude-API),
  committet & gepusht (531b3de..67943a5).

### Lernmomente
- **Cold-Start von lokalen Modellen** ist der Hauptengpass des Hybrid-
  Modus (PyTorch-Import + 2 Modelle in RAM), NICHT die Embedding-Rechnung.
  Cloud-Embeddings würden das nicht lösen und den Datenschutz opfern ->
  lokal optimieren ist richtig (1. Reloader aus; weitere Optionen: Warm-up,
  weniger Rerank-Kandidaten, Apple-MPS).
- **Ehrlichkeit in der Aussendarstellung:** "keine Cloud-Bausteine" war
  ungenau, weil die LLM-Antwort über die Claude-API (Cloud) läuft und der
  Hybrid vortrainierte Modelle nutzt. Präzise Formulierung ist glaub-
  würdiger als ein zu starkes Versprechen. (Methodik-Punkt für die Arbeit.)

### Offene Punkte
- [ ] Entscheidung: app.py-Fix auch ins öffentliche kmu-chatbot-demo-Repo
      pushen? (lokal gefixt, Remote noch nicht)
- [ ] Optional: weitere Speed-Fixes (Warm-up/Kandidaten/MPS), Deployment

---

## Session 11 – 2026-05-16 (Hybrid-Hang gelöst: HF-Hub offline)

### Symptom
Hybrid-Start „hängt" minuten- bis 7+ minutenlang; tfidf sofort.

### Diagnose (instrumentiert, auf Sulamiths Maschine)
- Modellgewichte laden in <1 s aus dem lokalen Cache.
- 7+ Min = **reines Netz-Warten**: `sentence-transformers`/`huggingface_hub`
  kontaktieren beim Import/Laden den HF-Hub (Update-Check). Unauthentifiziert
  rate-limitiert -> lange Retries -> scheinbarer „Hang".
- Mit erzwungenem Offline: gesamter Hybrid-Start ~9–48 s (davon der Grossteil
  einmaliger Import von transformers+torch), **kein Hang**.

### Fix
`os.environ.setdefault("HF_HUB_OFFLINE","1")` + `TRANSFORMERS_OFFLINE`
**ganz oben in app.py, VOR allen Imports** (huggingface_hub liest die
Variable beim Import in eine Konstante -> später gesetzt = wirkungslos;
darum scheiterte der erste Versuch in hybrid_rag.py). Verifiziert ohne
jegliche Shell-Variable: `HybridRetriever in 9.4s`, `HF_HUB_OFFLINE=1`.

### Lernmoment
„Hängt" ≠ „rechnet". Erst instrumentieren (Stacktrace/Timing) statt raten.
Schwere ML-Libs machen beim Import/Load Netz-Calls; für reproduzierbaren,
schnellen Start: Offline erzwingen, und Konfig früh genug setzen
(Reihenfolge der Imports zählt). Modelle sind nach 1× Download lokal.

### Nachtrag: robust für fremde Klone (Sulamiths Einwand)
Unbedingtes `HF_HUB_OFFLINE=1` würde einen FRISCHEN Klon (Modelle noch
nicht lokal) scheitern lassen – offline verbietet den nötigen Erst-
Download. Lösung: **cache-abhängiger Schalter** in hybrid_rag.py
(reiner Dateisystem-Check vor dem sentence-transformers-Import):
- Modelle gecacht  -> Offline (Start ~9 s, kein Hub-Hang)
- Modelle fehlen   -> online lassen + Hinweis, einmaliger Download,
  danach ab nächstem Start automatisch der schnelle Offline-Pfad.
Verifiziert: neuer Code, ohne Shell-Var, `HybridRetriever in 8.9s`,
HF_HUB_OFFLINE automatisch=1. Logik gegen leeren Cache gegengetestet.
Lehre: „offline für Speed" muss mit „funktioniert beim Erstnutzer"
versöhnt werden -> Konfiguration *bedingt* statt absolut.

### Offene Punkte
- [ ] app.py + hybrid_rag.py-Fix ins öffentliche Repo pushen? (lokal gefixt;
      öffentliches Repo gibt fremden Klonen sonst den langsamen Hang-Stand)
- [ ] Optional: Lib-Import (~40 s kalt) bleibt; bewusst belassen.
