"""
hybrid_rag.py – Hybrid Retrieval.

Idee (von Sulamith vorgeschlagen): die Stärken zweier Suchverfahren
kombinieren und das Ergebnis fein nachsortieren.

  Frage
   ├─ A) LEXIKALISCH : rag.RAGIndex (TF-IDF + Cosine)   -> Rangliste L
   ├─ B) SEMANTISCH  : lokales Embedding-Modell + Cosine -> Rangliste S
   ├─ C) FUSION      : Reciprocal Rank Fusion (RRF)      -> Kandidaten K
   └─ D) RERANKING   : Cross-Encoder bewertet (Frage,K_i)-> finale Top-k

Datenschutz: A, B, C, D laufen alle LOKAL. Kein Dokument/keine Frage
verlässt den Rechner für die Suche. (Nur der spätere LLM-Aufruf in app.py
geht an die Anthropic-API – das ist der einzige externe Schritt.)

`rag.py` (reines TF-IDF) bleibt unverändert und weiter nutzbar; app.py
lädt BEIDE Retriever und wählt pro Anfrage per Frontend-Umschalter
(RETRIEVER setzt nur den Default). Fehlt eine schwere Abhängigkeit
(sentence-transformers), fällt app.py sauber auf TF-IDF zurück. Schwere
Abhängigkeiten: siehe requirements-hybrid.txt.
"""

import os
from pathlib import Path

from rag import RAGIndex

# --- HF-Hub: cache-abhängiger Offline-Schalter (VOR sentence-transformers!) --
#
# Problem: huggingface_hub fragt beim Laden den HF-Hub nach Modell-Updates.
# Unauthentifiziert ist das rate-limitiert -> bei gecachten Modellen "hängt"
# der Start minutenlang. Lösung: sind die Modelle schon lokal -> Offline
# erzwingen (Sekunden statt Minuten). Sind sie NICHT da (frischer Klon) ->
# online lassen, damit sie EINMALIG heruntergeladen werden. So läuft das
# Projekt bei jedem fehlerfrei – ohne manuelle Schritte.
# WICHTIG: muss VOR dem sentence-transformers-Import stehen, weil
# huggingface_hub HF_HUB_OFFLINE beim Import in eine Konstante einliest.

# Diese IDs müssen zu EMBED_MODELL / RERANK_MODELL unten passen.
_BENOETIGTE_MODELLE = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
]


def _hf_cache_dir():
    """Pfad zum HF-Hub-Cache (respektiert HF_HUB_CACHE / HF_HOME)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _ist_gecacht(repo_id):
    """True, wenn das Modell schon lokal im HF-Cache liegt (reiner
    Dateisystem-Check, KEIN Netzaufruf, KEIN schwerer Import)."""
    ordner = _hf_cache_dir() / ("models--" + repo_id.replace("/", "--"))
    snapshots = ordner / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


if all(_ist_gecacht(r) for r in _BENOETIGTE_MODELLE):
    # Modelle liegen lokal -> Offline: kein rate-limitierter Hub-Check.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
else:
    # Frischer Klon: Modelle fehlen -> ONLINE lassen für den einmaligen
    # Download (mehrere hundert MB, kann ein paar Minuten dauern). Ab dem
    # nächsten Start greift automatisch der schnelle Offline-Pfad.
    print("[Hybrid] Modelle noch nicht lokal vorhanden – einmaliger "
          "Download vom Hugging-Face-Hub (mehrere hundert MB, kann einige "
          "Minuten dauern). Ab dem nächsten Start lädt es in Sekunden.",
          flush=True)

# sentence-transformers liefert beides:
#  - SentenceTransformer = Bi-Encoder  -> Embeddings (Schritt B)
#  - CrossEncoder        = Cross-Encoder -> Reranker  (Schritt D)
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# Mehrsprachiges Embedding-Modell (klein, schon aus dem Experiment bekannt).
EMBED_MODELL = "paraphrase-multilingual-MiniLM-L12-v2"

# Mehrsprachiger Cross-Encoder-Reranker (MS MARCO). Ein Cross-Encoder liest
# Frage UND Kandidatentext GEMEINSAM durch das Modell und gibt EINE
# Relevanzzahl aus -> viel präziser als die (unabhängige) Embedding-
# Ähnlichkeit, aber langsamer. Darum nur auf wenige Kandidaten anwenden.
RERANK_MODELL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

RRF_K = 60          # Standard-Konstante aus der RRF-Literatur
KANDIDATEN = 10     # so viele Fusions-Treffer gehen in den Reranker
# Mindest-Relevanz des besten Reranker-Treffers. Liegt der beste Kandidat
# darunter, gilt die Frage als "nicht beantwortbar" -> leere Liste, damit
# der Bot ehrlich "weiss ich nicht" sagt. Wert empirisch gesetzt (siehe
# Selbsttest unten / LERNNOTIZEN Kap. 8).
# Datenbasiert kalibriert (Selbsttest, siehe LERNNOTIZEN Kap. 8): Der
# Cross-Encoder gibt UNGEEICHTE Logits aus – auch korrekte Treffer sind oft
# negativ. Gemessen: gültige Fragen bester Score +1.2 bis -3.6; klar
# ausserhalb ("WM 2022") -8.6. -5.0 liegt sauber in der Lücke dazwischen.
# WICHTIG: auf kleinem Test-Set kalibriert -> illustrativ, nicht produktiv
# robust. Echt: auf separatem Validierungs-Set tunen.
RERANK_MIN = -5.0


class HybridRetriever:
    """Gleiche öffentliche Methode wie RAGIndex (finde_relevante_chunks),
    damit app.py beide Retriever austauschbar verwenden kann."""

    def __init__(self):
        # --- A) Lexikalischer Arm = unser bestehender TF-IDF-Index --------
        # Wird wiederverwendet, NICHT neu gebaut: "die ursprüngliche Idee
        # bleibt wörtlich drin".
        self.tfidf = RAGIndex()
        self.chunks = self.tfidf.chunks          # identische Chunks für B)

        # --- B) Semantischer Arm: Chunks einmalig lokal einbetten ---------
        # normalize_embeddings=True -> Cosine = Skalarprodukt (wie in rag.py).
        self.embed = SentenceTransformer(EMBED_MODELL)
        self.chunk_emb = self.embed.encode(
            [c["text"] for c in self.chunks],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        # --- D) Reranker-Modell laden ------------------------------------
        self.reranker = CrossEncoder(RERANK_MODELL)

    # ---- A) lexikalische Rangliste (Chunk-Indizes in Rangfolge) ---------
    def _lexikalisch(self, frage, n):
        # min_score=0.0: hier wollen wir eine RANGLISTE, noch nicht filtern.
        treffer = self.tfidf.finde_relevante_chunks(
            frage, top_k=n, min_score=0.0
        )
        return [self.chunks.index(chunk) for _, chunk in treffer]

    # ---- B) semantische Rangliste (Chunk-Indizes in Rangfolge) ----------
    def _semantisch(self, frage, n):
        q = self.embed.encode(
            frage, convert_to_tensor=True, normalize_embeddings=True
        )
        sims = util.cos_sim(q, self.chunk_emb)[0]      # je 1 Wert pro Chunk
        return sims.argsort(descending=True)[:n].tolist()

    # ---- C) Reciprocal Rank Fusion --------------------------------------
    @staticmethod
    def _rrf(ranglisten, k=RRF_K):
        """Kombiniert mehrere Ranglisten über RÄNGE statt Scores.

        Warum nicht Scores addieren? TF-IDF (~0.04–0.09) und Embeddings
        (~0.2–0.5) liegen auf völlig verschiedenen Skalen (im Experiment
        gemessen, LERNNOTIZEN Kap. 7) – addieren wäre Unsinn. RRF braucht
        nur die Reihenfolge: Score = Σ 1/(k + rang), rang 0-basiert.
        """
        punkte = {}
        for liste in ranglisten:
            for rang, doc_idx in enumerate(liste):
                punkte[doc_idx] = punkte.get(doc_idx, 0.0) + 1.0 / (k + rang)
        return [i for i, _ in sorted(
            punkte.items(), key=lambda p: p[1], reverse=True)]

    # ---- öffentliche API: identische Signatur wie RAGIndex --------------
    def finde_relevante_chunks(self, frage, top_k=3, min_score=None):
        # min_score wird nur fuer Signatur-Kompatibilität akzeptiert; der
        # Hybrid filtert ueber den Reranker (RERANK_MIN), nicht ueber min_score.
        lex = self._lexikalisch(frage, KANDIDATEN)     # A
        sem = self._semantisch(frage, KANDIDATEN)      # B
        fusion = self._rrf([lex, sem])[:KANDIDATEN]    # C
        if not fusion:
            return []

        # D) Cross-Encoder bewertet jeden Kandidaten ZUSAMMEN mit der Frage.
        paare = [(frage, self.chunks[i]["text"]) for i in fusion]
        scores = self.reranker.predict(paare)

        rangiert = sorted(
            zip(scores, fusion), key=lambda p: p[0], reverse=True
        )

        # JEDEN Kandidaten einzeln gegen die Schwelle pruefen -> das LLM
        # bekommt nur wirklich relevanten Kontext (kein Fuelltext, guenstiger).
        # Nichts ueber der Schwelle -> leere Liste -> Bot sagt "weiss nicht".
        relevant = [
            (float(s), self.chunks[i])
            for s, i in rangiert
            if float(s) >= RERANK_MIN
        ]
        # Gleiche Rückgabeform wie RAGIndex: [(score, chunk_dict), ...]
        return relevant[:top_k]


if __name__ == "__main__":
    # Selbsttest – KEIN API-Aufruf, KEINE Kosten (alles lokal).
    hr = HybridRetriever()
    fragen = [
        "Was kostet die Buchhaltung?",
        "Wann habt ihr offen?",
        "Was bietet ihr für Dienstleistungen an?",
        "Was macht ihr?",                       # TF-IDF allein: Fehlschlag
        "Welche Leistungen bietet ihr?",        # TF-IDF allein: schwach
        "Habt ihr Angebote für Angestellte?",   # TF-IDF allein: 0.00
        "Wer gewann die Fussball-WM 2022?",     # ausserhalb -> idealerweise []
    ]
    for f in fragen:
        print(f"\nFRAGE: {f}")
        treffer = hr.finde_relevante_chunks(f)
        if not treffer:
            print("  -> kein Treffer (Bot sagt 'weiss ich nicht')")
        for s, c in treffer:
            print(f"  [{s:+.3f}] {c['quelle']} :: {c['text'].splitlines()[0]}")
