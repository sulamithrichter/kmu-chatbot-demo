"""
experiment_embeddings.py – Vergleich TF-IDF vs. Embeddings.

ISOLIERT vom Hauptcode: app.py bleibt reines, selbst gebautes TF-IDF.
Dieses Skript ist nur Lern-/Vergleichsmaterial für die Maturaarbeit.

Kein API-Aufruf, keine Kosten: Das Embedding-Modell läuft komplett LOKAL
(sentence-transformers). Genau dieselben Chunks wie die App (rag.lade_chunks)
-> fairer Vergleich. Ziel: schwarz auf weiss zeigen, bei welchen Fragen
TF-IDF versagt und Embeddings (semantische Suche) treffen.

Lauf:  ./.venv/bin/python experiment_embeddings.py
"""

from rag import RAGIndex, lade_chunks
from sentence_transformers import SentenceTransformer, util

MODELL = "paraphrase-multilingual-MiniLM-L12-v2"   # mehrsprachig, klein

# (Frage, erwartetes Dokument). None = Frage liegt absichtlich ausserhalb.
TESTFRAGEN = [
    ("Was kostet die Buchhaltung?",              "preise.md"),          # TF-IDF ok
    ("Wann habt ihr offen?",                     "kontakt.md"),          # TF-IDF ok
    ("Was bietet ihr für Dienstleistungen an?",  "dienstleistungen.md"), # nach Fix ok
    ("Was macht ihr?",                           "dienstleistungen.md"), # TF-IDF FAIL
    ("Welche Leistungen bietet ihr?",            "dienstleistungen.md"), # TF-IDF FAIL
    ("Habt ihr Angebote für Angestellte?",       "dienstleistungen.md"), # TF-IDF FAIL
    ("Wer gewann die Fussball-WM 2022?",         None),                  # Kontrolle
]


def tfidf_bester(ix, frage):
    """Bester TF-IDF-Treffer OHNE min_score-Filter (für fairen Vergleich)."""
    treffer = ix.finde_relevante_chunks(frage, top_k=1, min_score=0.0)
    if not treffer:                       # kein gemeinsames Wort -> kein Vektor
        return ("—", 0.0)
    score, chunk = treffer[0]
    return (chunk["quelle"], score)


def main():
    ix = RAGIndex()
    chunks = lade_chunks()
    quellen = [c["quelle"] for c in chunks]

    print(f"Lade lokales Embedding-Modell '{MODELL}'")
    print("(beim ersten Lauf einmaliger Download ~470 MB, kein API-Call)…\n")
    modell = SentenceTransformer(MODELL)

    # Alle Chunks einmal einbetten (normalisiert -> Cosine = Skalarprodukt)
    chunk_emb = modell.encode(
        [c["text"] for c in chunks],
        convert_to_tensor=True, normalize_embeddings=True,
    )

    kopf = f"{'Frage':42s} | {'TF-IDF':27s} | {'Embedding':27s} | erwartet"
    print(kopf)
    print("-" * len(kopf))

    for frage, erwartet in TESTFRAGEN:
        # TF-IDF
        tq, ts = tfidf_bester(ix, frage)

        # Embedding
        q_emb = modell.encode(frage, convert_to_tensor=True,
                              normalize_embeddings=True)
        sims = util.cos_sim(q_emb, chunk_emb)[0]
        idx = int(sims.argmax())
        eq, es = quellen[idx], float(sims[idx])

        def mark(quelle):
            if erwartet is None:
                return quelle
            return f"{quelle} {'OK' if quelle == erwartet else 'XX'}"

        print(f"{frage:42s} | {mark(tq):21s}({ts:4.2f}) "
              f"| {mark(eq):21s}({es:4.2f}) | {erwartet or 'ausserhalb'}")

    print("\nWichtig: Auch bei der Kontroll-Frage liefert das Embedding einen")
    print("'besten' Treffer mit Score > 0. Auch Embeddings brauchen also eine")
    print("Relevanz-Schwelle – sie sind nicht magisch. (Diskussion: LERNNOTIZEN.)")


if __name__ == "__main__":
    main()
