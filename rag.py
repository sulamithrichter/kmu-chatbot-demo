"""
rag.py – Retrieval-Logik für den KMU-Chatbot.

Aufgabe: Aus den Firmendokumenten (Ordner documents/) die wenigen Textstellen
finden, die zu einer Nutzerfrage am besten passen. Diese Stellen werden in
app.py zusammen mit der Frage an Claude geschickt.

Verfahren: TF-IDF + Cosine Similarity, bewusst in reinem Python (kein externer
Dienst, keine ML-Bibliothek), damit jede Zeile nachvollziehbar ist.
Die Konzepte stehen ausführlich in LERNNOTIZEN.md.
"""

import os
import re
import math
from collections import Counter

DOKUMENTE_ORDNER = "documents"

# Sehr häufige deutsche Füllwörter ohne inhaltlichen Wert. IDF würde die meisten
# davon ohnehin gegen 0 drücken – die Liste macht das nur explizit und sauber.
STOPPWOERTER = {
    "der", "die", "das", "und", "oder", "aber", "ist", "sind", "war", "waren",
    "ein", "eine", "einen", "einem", "einer", "eines", "den", "dem", "des",
    "im", "in", "an", "am", "auf", "für", "mit", "von", "vom", "zu", "zur",
    "zum", "bei", "aus", "auch", "als", "wie", "wir", "sie", "es", "ihr",
    "ihre", "ihren", "ihrem", "sich", "nicht", "nur", "noch", "schon", "so",
    "dass", "wenn", "weil", "man", "haben", "hat", "wird", "werden", "kann",
    "können", "sehr", "mehr", "über", "unter", "nach", "vor", "durch",
    # Frage- und Füllwörter: tragen bei reiner Wortsuche keine Bedeutung,
    # erzeugen aber Falschtreffer (z.B. "was" matcht jede "Was ..."-Überschrift).
    "was", "wer", "wen", "wem", "warum", "wieso", "weshalb",
    "welche", "welcher", "welches", "wessen", "gibt", "habt", "hast",
    "euch", "eure", "euer", "eurem", "euren", "denn", "mal", "bitte", "dann",
}


# --- Baustein 1: Dokumente laden und in Chunks zerschneiden -------------------

def lade_chunks(ordner=DOKUMENTE_ORDNER):
    """Liest alle .md-Dateien. Strategie: EIN Chunk pro Dokument.

    Begründung (Details: LERNNOTIZEN Kap. 6 / DEVLOG Session 6):
    Mit feinem Schnitt an jeder "## "-Überschrift zersplitterte z.B. die
    Leistungsliste. Die Frage "Was bietet ihr für Dienstleistungen an?" fand
    dann nur den wort-, aber nicht inhaltsreichen Einleitungs-Chunk → schlechte
    Antwort. Für ein kleines, klar gegliedertes Korpus schlägt Vollständigkeit
    (ganzes Dokument bleibt zusammen) die chirurgische Präzision; die Mehrkosten
    an Tokens sind bei Haiku vernachlässigbar.
    """
    chunks = []
    for dateiname in sorted(os.listdir(ordner)):
        if not dateiname.endswith(".md"):
            continue
        pfad = os.path.join(ordner, dateiname)
        with open(pfad, encoding="utf-8") as f:
            inhalt = f.read().strip()
        if inhalt:
            chunks.append({"quelle": dateiname, "text": inhalt})
    return chunks


# --- Baustein 2: Tokenisieren ------------------------------------------------

def tokenisieren(text):
    """Text -> Liste kleingeschriebener Wörter, ohne Satzzeichen/Markdown,
    ohne Stoppwörter. \\w+ erfasst auch Umlaute und ß korrekt (Unicode).
    """
    woerter = re.findall(r"\w+", text.lower())
    return [w for w in woerter if w not in STOPPWOERTER]


# --- Bausteine 3 & 4: TF-IDF-Index + Suche -----------------------------------

class RAGIndex:
    """Hält die Chunks, die IDF-Werte und die fertigen Chunk-Vektoren.

    Einmal beim Start gebaut, danach beantwortet finde_relevante_chunks()
    jede Frage in Millisekunden (nur eine Schleife über wenige Chunks –
    genau deshalb brauchen wir keine Vektordatenbank).
    """

    def __init__(self, ordner=DOKUMENTE_ORDNER):
        self.chunks = lade_chunks(ordner)
        if not self.chunks:
            raise ValueError(f"Keine .md-Dokumente in '{ordner}' gefunden.")
        self._index_bauen()

    def _index_bauen(self):
        anzahl_chunks = len(self.chunks)

        # Document Frequency: in wie vielen Chunks kommt ein Wort vor?
        df = Counter()
        chunk_tokens = []
        for chunk in self.chunks:
            tokens = tokenisieren(chunk["text"])
            chunk_tokens.append(tokens)
            for wort in set(tokens):          # set: pro Chunk nur einmal zählen
                df[wort] += 1

        # IDF: seltene Wörter -> hoher Wert; Wort in allen Chunks -> log(1)=0
        self.idf = {
            wort: math.log(anzahl_chunks / df_wort)
            for wort, df_wort in df.items()
        }

        # Jeden Chunk in seinen (normierten) TF-IDF-Vektor umwandeln
        self.chunk_vektoren = [self._vektor(t) for t in chunk_tokens]

    def _vektor(self, tokens):
        """tokens -> normierter TF-IDF-Vektor als dict {wort: gewicht}.

        Sparse dict statt langer Liste: wir speichern nur Wörter, die wirklich
        vorkommen. Das spart Platz und die Cosine-Berechnung wird einfacher.
        """
        tf = Counter(tokens)                  # TF = Häufigkeit in diesem Text
        vektor = {}
        for wort, anzahl in tf.items():
            idf = self.idf.get(wort)
            # Wörter, die im ganzen Korpus nie vorkamen (idf is None) oder die
            # überall stehen (idf == 0), tragen keine Unterscheidungskraft.
            if idf:
                vektor[wort] = anzahl * idf

        # L2-Normierung: Vektor auf Länge 1 bringen. Danach ist die Cosine
        # Similarity nur noch das Skalarprodukt -> Länge des Textes egal,
        # nur die Richtung (= das Thema) zählt.
        laenge = math.sqrt(sum(g * g for g in vektor.values()))
        if laenge > 0:
            vektor = {w: g / laenge for w, g in vektor.items()}
        return vektor

    def finde_relevante_chunks(self, frage, top_k=3, min_score=0.05):
        """Gibt die top_k zur Frage passendsten Chunks zurück.

        Rückgabe: Liste von (score, chunk-dict), absteigend nach score.
        Chunks unter min_score gelten als "nicht wirklich relevant" und
        fallen weg – so kann app.py erkennen, wenn es keine Antwort gibt.
        """
        frage_vektor = self._vektor(tokenisieren(frage))
        if not frage_vektor:
            return []

        treffer = []
        for chunk, chunk_vektor in zip(self.chunks, self.chunk_vektoren):
            # Beide Vektoren sind normiert -> Skalarprodukt = Cosine Similarity
            score = sum(
                gewicht * chunk_vektor.get(wort, 0.0)
                for wort, gewicht in frage_vektor.items()
            )
            if score >= min_score:
                treffer.append((score, chunk))

        treffer.sort(key=lambda paar: paar[0], reverse=True)
        return treffer[:top_k]


# --- Selbsttest: python rag.py  (braucht KEINEN API-Key) ---------------------

if __name__ == "__main__":
    index = RAGIndex()
    print(f"{len(index.chunks)} Chunks aus documents/ geladen.\n")

    testfragen = [
        "Was kostet die Buchhaltung im Monat?",
        "Wann habt ihr offen?",
        "Ab wann muss ich Mehrwertsteuer zahlen?",
        "Macht ihr auch meine private Steuererklärung?",
        "Was ist die Hauptstadt von Frankreich?",   # bewusst ausserhalb
    ]

    for frage in testfragen:
        print(f"FRAGE: {frage}")
        treffer = index.finde_relevante_chunks(frage)
        if not treffer:
            print("  -> kein relevanter Chunk (Bot würde 'weiss ich nicht' sagen)\n")
            continue
        for score, chunk in treffer:
            titel = chunk["text"].splitlines()[0]
            print(f"  [{score:.3f}] {chunk['quelle']} :: {titel}")
        print()
