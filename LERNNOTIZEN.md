# Lernnotizen – KMU-Chatbot

Vertiefendes Begleitheft zum Projekt. Hier stehen Konzepte, Alternativen und
Abwägungen, die *über* die konkrete Implementierung hinausgehen – gedacht als
Material für die Maturaarbeit. Das DEVLOG protokolliert, *was* gebaut wurde;
hier steht, *warum so* und *was die Alternativen wären*.

---

# Kapitel 1: Embeddings vs. TF-IDF (und lokal vs. API)

**Kontext:** Für die Implementierung haben wir TF-IDF gewählt (transparent,
selbst gebaut, voll verstehbar). Hier die Vertiefung zum Embedding-Weg, den wir
*nicht* implementieren, aber verstehen wollen.

## 1.1 Was ist ein Embedding genau?

Ein **Embedding** ist ein Vektor fester Länge (z.B. 384, 768 oder 1536 Zahlen),
den ein **neuronales Modell** aus einem Text erzeugt. Das Modell wurde auf
riesigen Textmengen trainiert und hat dabei gelernt, Texte so in Zahlen zu
übersetzen, dass **Bedeutungsnähe zu räumlicher Nähe** wird:

- „Was kostet die Buchhaltung?" und „Wie hoch ist der Preis fürs Bücherführen?"
  ergeben *unterschiedliche* TF-IDF-Vektoren (kaum gemeinsame Wörter), aber
  *ähnliche* Embedding-Vektoren (gleiche Bedeutung).
- Das nennt man **semantische** Suche – gesucht wird nach Sinn, nicht nach
  Wortübereinstimmung.

Wichtig fürs Verständnis: **Die Architektur bleibt exakt gleich wie bei
TF-IDF.** Pipeline ist immer:

```
Dokumente → Chunks → Vektoren → Cosine Similarity → Top-Treffer in den Prompt
```

Nur der eine Schritt „→ Vektoren" wechselt: statt Wörter zu zählen (TF-IDF)
fragt man ein Embedding-Modell. **Chunking und Cosine Similarity ändern sich
nicht.** Genau deshalb überträgt sich das TF-IDF-Wissen 1:1.

### Skizze (Pseudocode, kein echter Projektcode)

TF-IDF (unser Weg):
```
vektor = tfidf_vektorisieren(text)          # reines Python, lokal, gratis
```

Embedding-Weg, lokales Modell:
```
from sentence_transformers import SentenceTransformer
modell = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
vektor = modell.encode(text)                # läuft auf deinem Rechner
```

Embedding-Weg, API (Beispiel Voyage AI – von Anthropic empfohlen):
```
vektor = voyage_client.embed(text, model="voyage-3").embeddings[0]   # Netzwerkaufruf, kostet pro Token
```

Der Rest des Programms (Cosine Similarity, Top-3, Prompt bauen) bleibt
identisch. Das ist die wichtigste Erkenntnis dieses Kapitels.

## 1.2 TF-IDF vs. Embeddings – detaillierte Abwägung

| Kriterium | TF-IDF (unser Weg) | Embeddings |
|---|---|---|
| Findet Synonyme/Umschreibungen | Nein – nur Wortüberlappung | Ja – Bedeutung zählt |
| Mehrsprachigkeit (Frage DE, Doku FR) | Schwach | Stark (bei mehrsprachigem Modell) |
| Transparenz / erklärbar | Sehr hoch – jede Zahl nachvollziehbar | Black Box – Vektor ist nicht interpretierbar |
| Abhängigkeiten | Keine (reines Python) | ML-Bibliothek **oder** externer Dienst |
| Kosten | Null | API: pro Token; lokal: einmalig Rechenleistung |
| Qualität bei Fachjargon / exakten Begriffen | Stark („MWST", „AHV" matchen exakt) | Gut, aber Fachbegriffe manchmal verwässert |
| Eignung kleiner Korpus (Demo) | Sehr gut | Gut, aber Overkill |
| Eignung grosser Korpus (Produktion) | Wird ungenau | Klar besser |
| Aufwand bei Doku-Änderung | Trivial (neu zählen) | Neu einbetten + Vektoren speichern |
| Eignung Maturaarbeit (selbst erklären) | Ideal | Schwer, weil neuronale Black Box |

**Fazit der Abwägung:** Für ein kleines, gut strukturiertes Demo mit dem Ziel
„ich verstehe und erkläre jede Zeile" ist TF-IDF überlegen. Embeddings gewinnen,
sobald (a) der Korpus gross wird, (b) Nutzer frei formulieren und Synonyme
nutzen, oder (c) mehrere Sprachen im Spiel sind.

## 1.3 Lokales Modell vs. Embedding-API

Wenn man sich *für* Embeddings entscheidet, kommt die nächste Abwägung: das
Modell selbst betreiben oder einen Dienst aufrufen?

| Kriterium | Lokales Modell (z.B. `sentence-transformers`) | Embedding-API (z.B. Voyage AI, OpenAI, Cohere) |
|---|---|---|
| Datenschutz | **Daten bleiben auf dem Rechner** | Text wird an Anbieter gesendet |
| Kosten | Einmalig: Rechenleistung/RAM | Laufend: pro Token, aber sehr günstig |
| Setup | Schwere Abhängigkeit (PyTorch, Modell ~80–500 MB) | Nur API-Key + paar Zeilen |
| Qualität | Gut, je nach Modell | In der Regel Spitzenklasse |
| Offline-fähig | Ja | Nein (Internet nötig) |
| Geschwindigkeit | Ohne GPU langsamer | Schnell, skaliert beim Anbieter |
| Wartung | Du pflegst Modell/Umgebung | Anbieter pflegt |
| Abhängigkeit von Dritten | Keine | Vendor-Lock-in, Preis-/API-Änderungen |

### Der Datenschutz-Punkt – relevant für eine Treuhandfirma

Besonders lehrreich für *deine* Fallstudie: Eine Treuhandfirma unterliegt einer
**Geheimhaltungspflicht** und verarbeitet sensible Finanzdaten. Würde man
*kundenspezifische* Dokumente über eine externe Embedding-API einbetten,
verlassen diese Daten das Haus – ein echtes Compliance-Thema.

Nuance (auch das gehört in eine gute Arbeit): In *diesem* Projekt sind die
Dokumente bewusst **öffentliche** Firmeninfos (Preise, FAQ, Öffnungszeiten) –
da wäre eine API unkritisch. Heikel würde es erst, wenn man Mandantsakten oder
die *Fragen der Nutzer* (die Geschäftsgeheimnisse enthalten können) an einen
Dritten schickt. Genau dort spielt das lokale Modell seinen Vorteil aus.

## 1.4 Entscheidungs-Heuristik (zum Mitnehmen)

- Kleiner, statischer, gut strukturierter Korpus + Lernziel → **TF-IDF**.
- Grösserer Korpus, freie Nutzerformulierung, Synonyme → **Embeddings**.
- Embeddings + sensible Daten / Offline / kein laufendes Budget → **lokal**.
- Embeddings + beste Qualität, wenig Setup, Daten unkritisch → **API**.
- Profi-Setup kombiniert beides (Stichworte zum Weiterlesen: *Hybrid Search*
  aus BM25 + Embeddings, danach *Reranking*) – bewusst ausserhalb des Demos.

---

# Kapitel 2: Was der TF-IDF-Test über die Grenzen verriet (Praxisbefund)

Beim ersten Selbsttest von `rag.py` (5 Testfragen) traten genau die in
Kapitel 1 theoretisch beschriebenen Schwächen real auf. Echte Evidenz aus
dem eigenen Projekt – wertvoller als jedes Lehrbuchzitat.

## 2.1 Vokabular-Lücke: „offen" vs. „Öffnungszeiten"

Die Frage *„Wann habt ihr offen?"* fand zuerst **nicht** die Öffnungszeiten,
weil die Frage das Wort „offen" nutzt, das Dokument aber nur „Öffnungszeiten".
Keine Wortüberlappung → TF-IDF ist blind. Embeddings hätten das über die
Bedeutung gelöst.

**Pragmatische Lösung (angewandt):** Den Öffnungszeiten-Abschnitt um die Wörter
„offen / geöffnet / erreichbar" ergänzt. **Verallgemeinerte Lektion:** Bei
keyword-basiertem Retrieval ist *Auffindbarkeit eine Eigenschaft der
Dokumentation*. Man schreibt die Texte in den Worten, die Nutzer benutzen –
nicht nur im internen Fachjargon. Dieselbe Lektion gilt abgeschwächt sogar
für Embeddings (auch dort hilft natürliche, redundante Formulierung).

## 2.2 Deutsche Komposita & fehlendes Stemming

*„Mehrwert**steuer** zahlen"* (Frage) vs. *„mehrwertsteuer**pflichtig**"*
(Dokument): für reine Wortsuche zwei **verschiedene** Tokens. Es gibt kein
*Stemming* (Wörter auf den Wortstamm zurückführen) und keine Kompositazerlegung.
Das ist eine spezifisch deutsche Hürde (Sprache baut lange zusammengesetzte
Wörter).

**Nächste Stufe (bewusst NICHT im Demo):** Stemming/Lemmatisierung (z.B.
Snowball-Stemmer, spaCy) oder Kompositazerlegung würden „mehrwertsteuer-
pflichtig" auf „mehrwertsteuer" + „pflichtig" zurückführen. Embeddings umgehen
das Problem ganz, weil sie auf Bedeutung statt Wortform arbeiten. Bewusst
ausgelassen, weil es zusätzliche Abhängigkeiten/Komplexität bringt und das
Lernziel „jede Zeile verstehen" verwässern würde.

## 2.3 Stoppwort-Abwägung ist empirisch

Erst wurden Fragewörter (was, wann, wo …) pauschal zu Stoppwörtern erklärt.
Das beseitigte Falschtreffer (Frage *„Hauptstadt von Frankreich?"* matchte
vorher „**Was** wir nicht machen" nur über „was"), **löschte aber auch echtes
Signal**: „wann" steht in sinnvollen Überschriften wie *„Ab **wann** bin ich
mehrwertsteuerpflichtig?"*. Lösung: „was" raus (reines Füllwort), „wann/wo"
bleiben suchbar.

**Lektion:** Stoppwortlisten sind **korpus-spezifisch** und gehören mit echten
Testfragen validiert, nicht nach Bauchgefühl gesetzt.

## 2.4 Warum „unscharfes" Retrieval trotzdem reicht

Auf Platz 2/3 tauchen teils thematisch schwache Chunks auf (z.B. „Sind meine
Daten sicher?" matcht *„zahlen"*). Das ist unkritisch, weil das **Sprachmodell
in Schritt 3 die Sicherheitsschicht ist**: Es bekommt die Top-Treffer plus die
strikte Anweisung, *nur* aus tatsächlich relevantem Kontext zu antworten und
sonst „weiss ich nicht" zu sagen. **Retrieval muss nicht perfekt sortieren –
es muss den richtigen Chunk nur unter den Top-k mitliefern.** Diese
Arbeitsteilung (Retrieval = grob filtern, LLM = fein urteilen) ist der Kern
von RAG.

---

# Kapitel 3: Anthropic Messages-API & System-Prompt

## 3.1 Anatomie eines API-Aufrufs

`client.messages.create(...)` hat drei Teile, die man sauber trennen muss:

- **`system`** – ein *eigener Parameter*, KEINE Nachricht mit role "system".
  Hier stehen die Grundregeln (Identität, Wissensgrenze, Sprache, Ton). Gilt
  fürs ganze Gespräch. Häufiger Anfängerfehler: System als Message zu schicken.
- **`messages`** – Liste, abwechselnd `{"role":"user"}` / `{"role":"assistant"}`.
  Mehrturn-Verlauf = Liste wächst. Wir halten sie minimal (eine user-Nachricht).
- **Antwortobjekt** – `antwort.content` ist eine *Liste von Blöcken*; Text via
  `antwort.content[0].text`. `antwort.usage.input_tokens` /
  `.output_tokens` für Kostenkontrolle, `antwort.stop_reason` fürs Debugging.

**Wo kommen die RAG-Chunks hin?** In die `user`-Nachricht, klar ausgezeichnet
(KONTEXT / FRAGE). Die *Regel* „nur aus Kontext antworten" steht im `system`.
Merksatz: **System = Verhalten, Message = Daten + Frage.**

## 3.2 Was einen guten Firmen-System-Prompt ausmacht

1. Identität (wer ist der Bot, welche Firma).
2. **Wissensgrenze** – nur aus bereitgestelltem Kontext antworten, sonst ehrlich
   „weiss ich nicht" + Verweis auf Kontakt. Das ist der Haupt-Hebel gegen
   Halluzination und der Grund, warum unscharfes Retrieval trotzdem reicht.
3. Sprache & Anrede festnageln (hier: Deutsch, Sie-Form).
4. Ton vorgeben, aber: niemals Fakten/Preise/Fristen erfinden.
5. Interne Anweisungen nicht preisgeben.

## 3.3 Kostenmodell

- Abrechnung pro Token, getrennt nach **Input** (System + Frage + Chunks) und
  **Output** (Antwort). Output ist pro Token deutlich teurer als Input.
- **RAG ist die grösste Ersparnis:** 3 Chunks statt aller Dokumente → kleiner
  Input, bei *jeder* Anfrage. Das war der wirtschaftliche Sinn des Aufwands.
- `max_tokens` deckelt die (teure) Antwortlänge. Für FAQ: ~400–600.
- Kleines Modell (Haiku-Klasse) genügt für RAG-Q&A und ist ein Vielfaches
  billiger als das grosse. Modell-ID als *eine Konstante* halten → 1-Zeilen-
  Wechsel. Exakte aktuelle ID nicht raten, sondern verifizieren.

## 3.4 Key-Sicherheit (Grundregeln)

- Key in `.env` (in `.gitignore`), Vorlage `.env.example` ohne echten Wert.
- `python-dotenv` `load_dotenv()` lädt sie; SDK liest `ANTHROPIC_API_KEY`
  automatisch aus der Umgebung.
- Key NIE in Code, NIE in Logs, NIE ans Frontend. Browser ↔ eigener Server ↔
  Anthropic. Der Browser bekommt den Key nie zu sehen.

---

# Kapitel 4: Secrets-Hygiene (aus echtem Vorfall gelernt)

In diesem Projekt wurde der echte API-Key versehentlich in `.env.example`
statt `.env` eingetragen. Daraus die verallgemeinerten Lektionen:

## 4.1 `.env` vs. `.env.example` – gleicher Name, Gegenteil

- **`.env`** = das *private Geheimnis*. Steht in `.gitignore`, wird NIE
  committet. Enthält den echten Key.
- **`.env.example`** = die *öffentliche Vorlage*. Wird absichtlich committet,
  damit andere wissen, *welche* Variablen es gibt – aber mit **leeren** Werten.
- Verwechslung ist gefährlich, weil nur `.env` ignoriert wird. Ein Key in
  `.env.example` landet bei `git push` öffentlich.

## 4.2 Warum „rotieren" nach Exposition?

Ein Secret, das je an einem unsicheren Ort war (commit-bestimmte Datei, Chat,
Log, Screenshot), gilt als **kompromittiert** – egal ob „wirklich" jemand
zugegriffen hat. Gegenmassnahme: alten Key widerrufen, neuen erzeugen
(*Rotation*). Kosten: ein paar Klicks. Kosten eines geleakten Bezahl-Keys:
potenziell fremde API-Nutzung auf deine Rechnung. Asymmetrie → immer rotieren.

## 4.3 Grundregeln (zum Mitnehmen für Freelance/Maturaarbeit)

- Secrets nie in Code, nie in `.example`-Dateien, nie in Logs, nie in den
  Chat. Nur in der gitignorten `.env` (oder einem Secret-Manager).
- `.gitignore` VOR dem ersten Commit aufsetzen (haben wir).
- Vor `git push` bei public Repos prüfen: ist `.env` wirklich draussen?
  (`git status`, `git check-ignore .env`).
- Server gibt Key-/Fehlerdetails NIE ans Frontend (siehe `app.py`: generische
  Fehlermeldung, Details nur in den Server-Log).

---

# Kapitel 5: Graceful Degradation & der Offline-Fallback

## 5.1 Stufen der Ausfallsicherheit

Ein System kann auf einen Fehler unterschiedlich „würdevoll" reagieren:

1. **Crash** – Server stürzt ab. Schlechteste Stufe.
2. **Fehlermeldung** – „Es gab ein Problem." Ehrlich, aber nutzlos.
3. **Reduzierte Funktion** – liefert *etwas Nützliches* trotz Ausfall.

Wir sind von Stufe 2 (502 + generische Meldung) auf Stufe 3 gegangen:
Fällt der KI-Call aus (kein Key, **kein Guthaben**, Netzwerk), baut der
Server die Antwort direkt aus den per RAG gefundenen Chunks – klar als
„Offline-Modus" gekennzeichnet. Die Portfolio-Demo bleibt so selbst mit 0
Credits benutzbar.

## 5.2 Bewusster Tradeoff: Offline-Modus zeigt das Retrieval ungeschminkt

Im Live-Modus bekommt Claude die Top-3-Chunks und *formuliert* daraus eine
saubere Antwort – kleine Retrieval-Ungenauigkeiten fallen kaum auf. Im
Offline-Modus zeigen wir die Chunks fast roh: Schwächen der TF-IDF-Suche
(siehe Kap. 2) werden direkt sichtbar (z.B. bei sehr kurzen Fragen). Das ist
kein Bug, sondern die ehrliche Konsequenz daraus, dass im Fallback die
„Veredelungsschicht" (das LLM) fehlt. Gute Diskussion für die Maturaarbeit:
*Wer macht im RAG eigentlich welche Arbeit – und was passiert, wenn ein
Teil ausfällt?*

## 5.3 Konfig-Lektion am Rande (Details im DEVLOG, Session 5)

`load_dotenv()` überschreibt **vorhandene** Umgebungsvariablen nicht. Eine
leere Variable aus dem Shell-Profil kann eine korrekte `.env` still
aushebeln. Merke: „App liest .env nicht" → zuerst prüfen, ob die Variable
schon (leer) in der Umgebung steht. Lösung hier: `load_dotenv(override=True)`.

---

# Kapitel 6: Chunk-Granularität – der unterschätzte Hebel

Aus einem realen Live-Fehler gelernt: „Was bietet ihr für Dienstleistungen
an?" lieferte eine schlechte Antwort – nicht wegen Ton oder Prompt, sondern
weil das **Chunking** zu fein war.

## 6.1 Das Dreieck: Präzision ↔ Vollständigkeit ↔ Kosten

- **Zu fein** (jede `##` ein Chunk): Treffer sind chirurgisch genau, aber
  zusammengehörender Kontext zerfällt. Die Leistungsliste war über viele
  Mini-Chunks verstreut; der einzige „dienstleistungs-ähnliche" Chunk war die
  *inhaltsleere* Einleitung → unvollständige Antwort.
- **Zu grob** (ein Dokument = ein Chunk): Kontext bleibt komplett zusammen
  (Vollständigkeit ✓), aber **seltene Einzelwörter verwässern**: „Leistungen"
  kommt einmal in einem grossen Dokument vor → winziges TF-IDF-Gewicht →
  „Welche Leistungen bietet ihr?" fällt unter `min_score`.
- **Kosten:** grössere Chunks = mehr Tokens pro Anfrage. Bei Haiku hier
  vernachlässigbar, bei grossem Korpus/teurem Modell relevant.

**Entscheidung für dieses Projekt:** ein Chunk pro Dokument. Für ein kleines,
klar gegliedertes Korpus überwiegt Vollständigkeit. Der Preis (verwässerte
Einzelwörter) ist bewusst akzeptiert und dokumentiert.

## 6.2 Warum das Embeddings motiviert

Beide TF-IDF-Restlücken sind *Bedeutungs*-Probleme, keine Tuning-Probleme:
„macht" vs. „machen", „Leistungen" als Synonym für die aufgezählten Dienste.
Kein Chunking und kein Wortschatz-Trick löst das grundsätzlich – nur ein
Verfahren, das *Bedeutung* statt *Wortform* vergleicht. Genau das prüfen wir
in Experiment (B) mit echten, selbst erzeugten Vergleichsdaten.

Methodischer Merksatz für die Fallstudie: **Eine Schwäche, die man sauber
belegt, ist wertvoller als eine, die man wegtuned.**

---

# Kapitel 7: Experiment TF-IDF vs. Embeddings – die Ergebnisse

Skript: `experiment_embeddings.py` (lokales Modell
`paraphrase-multilingual-MiniLM-L12-v2`, kein API-Call). Gleiche Chunks wie
die App. „OK/XX" = erwartetes Dokument getroffen / verfehlt; Score = bester
Cosine-Wert (TF-IDF ohne min_score-Filter, fairer Vergleich).

```
Frage                                   | TF-IDF                | Embedding             | erwartet
Was kostet die Buchhaltung?             | preise.md OK   (0.04) | preise.md OK   (0.49) | preise.md
Wann habt ihr offen?                    | kontakt.md OK  (0.08) | faq.md XX      (0.25) | kontakt.md
Was bietet ihr für Dienstleistungen an? | dienstl.  OK   (0.09) | dienstl.  OK   (0.47) | dienstleistungen.md
Was macht ihr?                          | —  XX          (0.00) | kontakt.md XX  (0.21) | dienstleistungen.md
Welche Leistungen bietet ihr?           | dienstl.  OK   (0.04) | dienstl.  OK   (0.35) | dienstleistungen.md
Habt ihr Angebote für Angestellte?      | —  XX          (0.00) | dienstl.  OK   (0.37) | dienstleistungen.md
Wer gewann die Fussball-WM 2022?        | —              (0.00) | faq.md         (0.22) | ausserhalb
```

## 7.1 Wo Embeddings klar gewinnen
- „Habt ihr Angebote für Angestellte?": TF-IDF 0.00 (kein gemeinsames Wort),
  Embedding trifft `dienstleistungen.md` (0.37). Reiner Bedeutungs-Treffer.
- Konfidenz: korrekte Embedding-Treffer 0.35–0.49 vs. TF-IDF 0.04–0.09.
- „Welche Leistungen bietet ihr?": TF-IDF rankt zwar richtig, aber **0.04 <
  App-Schwelle 0.05** → in der echten App ein Fehlschlag. Embedding 0.35 robust.

## 7.2 Wo Embeddings NICHT helfen (Gegenevidenz, wichtig)
- „Wann habt ihr offen?": TF-IDF korrekt, Embedding **falsch** (faq.md) –
  Embeddings führten einen *neuen* Fehler ein.
- „Was macht ihr?": beide scheitern.
- Kontrolle „WM 2022": Embedding 0.22 ≈ schwache gültige Frage „Was macht
  ihr?" 0.21. Das Grundrauschen irrelevanter Fragen überlappt schwache
  echte Signale → Schwellenwahl ist bei Embeddings *subtiler*, nicht trivialer.

## 7.3 Schlussfolgerung (für die Maturaarbeit)
Embeddings sind im Schnitt besser (lösen semantische Lücken, klarere
Score-Trennung), aber **kein Allheilmittel**: fehlbar, brauchen weiterhin
eine – subtilere – Relevanzschwelle, und die Infrastruktur-/Datenschutz-
Kosten aus Kap. 1 bleiben. Für dieses kleine, statische, einsprachige Demo
ist selbst gebautes TF-IDF + gutes Chunking vertretbar und vollständig
verstanden. Für ein grosses, freisprachiges Produktivsystem wären Embeddings
die richtige Wahl – mit empirisch getunter Schwelle und bewusst akzeptierten
Kosten. *Kernaussage: Die Wahl des Retrieval-Verfahrens ist eine begründete
Abwägung, kein Dogma.*

---

# Kapitel 8: Hybrid Search & Reranking

Eigene Idee zur Qualitätssteigerung: lexikalische + semantische Suche
kombinieren und fein nachsortieren. Implementiert in `hybrid_rag.py`,
zuschaltbar via `RETRIEVER=hybrid` (Standard bleibt reines TF-IDF).

## 8.1 Warum überhaupt hybrid?
Die beiden Verfahren sind **komplementär**:
- Lexikalisch (TF-IDF) ist stark bei exakten Begriffen/Fachjargon
  („MWST", „Art. 958f"), versagt bei Synonymen/Konjugation.
- Semantisch (Embeddings) versteht Bedeutung, ist aber bei seltenen
  exakten Begriffen und manchmal überraschend daneben („Wann offen?" ->
  faq statt kontakt, Kap. 7).
Zusammen decken sie die Schwächen des jeweils anderen ab.

## 8.2 Fusion: Reciprocal Rank Fusion (RRF)
Problem: TF-IDF-Scores (~0.04–0.09) und Embedding-Scores (~0.2–0.5) liegen
auf **unvergleichbaren Skalen** (in Kap. 7 selbst gemessen) – Addieren wäre
unsinnig. RRF nutzt nur die **Reihenfolge**:

  RRF-Score(Dok) = Σ über alle Listen  1 / (k + rang)     (rang 0-basiert, k≈60)

Kalibrierungsfrei, robust, ~10 Zeilen. Ein Dokument, das in beiden Listen
weit oben steht, gewinnt.

## 8.3 Reranker: Cross-Encoder vs. Bi-Encoder
- **Bi-Encoder** (= das Embedding-Modell): kodiert Frage und Dokument
  *getrennt*, vergleicht dann die Vektoren. Schnell (Dokumente vorab
  einbettbar), aber grob.
- **Cross-Encoder** (Reranker): steckt Frage UND Dokument *gemeinsam* durch
  das Modell und gibt einen Relevanzwert aus. Viel präziser, aber langsam
  (jede Frage-Dokument-Kombi neu) -> nur auf die ~10 Fusions-Kandidaten.
Arbeitsteilung: grob+schnell filtern (A/B/C), dann fein+langsam sortieren (D).

## 8.4 Die Kalibrierungs-Lektion (wichtig!)
Cross-Encoder geben **ungeeichte Logits** aus – auch korrekte Treffer sind
oft negativ. Erste Schwelle `0.0` filterte gültige Fragen weg. Lösung:
Schwelle **aus Daten** ableiten. Gemessene beste Scores:

```
gültige Fragen:        +1.22, +0.28, -0.27, -1.90, -3.20, -3.62
klar ausserhalb (WM):  -8.63
```

Lücke zwischen -3.6 und -8.6 -> Schwelle **-5.0**. **Warnung:** auf winzigem
Test-Set kalibriert -> illustrativ, nicht produktiv robust. Echt: separates
Validierungs-Set. (Gleiche Lehre wie Kap. 7, jetzt am Reranker: „Schwellen
sind heikel und müssen empirisch bestimmt werden.")

## 8.5 Ergebnis-Vergleich & Schluss

| Frage | TF-IDF | Embedding | Hybrid+Rerank |
|---|---|---|---|
| „Was macht ihr?" | ✗ | ✗ (kontakt) | ✓ dienstleistungen |
| „Welche Leistungen?" | ✗ (<Schwelle) | ✓ | ✓ |
| „Angebote Angestellte?" | ✗ (0.00) | ✓ | ~ (Gleichstand, ok) |
| „Wann offen?" | ✓ | ✗ (faq) | ✓ |
| WM 2022 (ausserhalb) | ✓ leer | ✗ (0.22) | ✓ leer |

Hybrid+Reranker ist durchgängig am besten – löst sogar die für *beide*
Einzelverfahren unlösbare Frage. **Preis:** zwei lokale Modelle, mehr
Komplexität, eine kalibrierte Schwelle. Datenschutz: Suche (A–D) komplett
lokal; einzig die Antwortgenerierung (LLM) bleibt extern.

**Methoden-Merksatz:** Gute Suche ist eine Pipeline aus billig-grob-filtern
und teuer-fein-sortieren – kein einzelner magischer Algorithmus.

---

*Nächste Kapitel folgen (z.B. Deployment), falls das Projekt weiterwächst.*
