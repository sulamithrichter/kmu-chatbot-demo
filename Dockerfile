# Dockerfile – baut den KMU-Chatbot als Hugging-Face-Space (SDK: docker).
#
# Ein EINZIGER Prozess lädt beide Retriever (tfidf + hybrid); der Umschalter
# im Frontend wählt pro Anfrage. Die schweren Hybrid-Modelle werden schon
# hier im Build heruntergeladen ("ins Image gebacken") -> beim ersten Aufruf
# muss nichts mehr geladen werden (vgl. LERNNOTIZEN Kap. 10 & 9.1b).
#
# Aufbau folgt dem OFFIZIELLEN HF-Docker-Muster (verifiziert über die HF-Doku
# via Connector): Container läuft als User mit UID 1000. Den User FRÜH anlegen
# und VOR allen COPY/Downloads dorthin wechseln, plus `COPY --chown=user`.
# Ein nachträgliches `chown -R` würde laut HF-Doku alle (grossen!) Modell-
# Dateien in eine neue Layer DUPLIZIEREN -> doppelt so grosses Image.

FROM python:3.12-slim

# 1) User zuerst – danach alles als dieser User.
RUN useradd -m -u 1000 user
USER user

# HF_HOME: derselbe Cache-Pfad für Build (Download) UND Laufzeit, im
#   beschreibbaren Home des Users. Liegen die Modelle hier, schaltet
#   hybrid_rag.py automatisch offline (cache-abhängiger Schalter,
#   LERNNOTIZEN Kap. 9.1b) -> kein Hub-Hang, schneller Start.
# PORT 7860: HF-Standard (muss zu app_port in README.md passen).
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860 \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# 2) Erst nur die Abhängigkeitslisten kopieren: solange sie sich nicht
#    ändern, nutzt Docker den gecachten pip-Layer auch bei Code-Änderungen
#    wieder. requirements.txt bleibt schlank (nur tfidf, so dokumentiert);
#    die schweren Hybrid-Pakete kommen aus dem bestehenden Hybrid-File.
COPY --chown=user requirements.txt requirements-hybrid.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements-hybrid.txt

# 3) Restlichen Code + Dokumente (als user, kein nachträgliches chown).
COPY --chown=user . .

# 4) Beide Hybrid-Modelle EINMAL im Build laden (ins Image backen), als user
#    -> Cache-Dateien gehören user, zur Laufzeit lesbar. IDs müssen zu
#    EMBED_MODELL / RERANK_MODELL in hybrid_rag.py passen (auf dem HF-Hub
#    verifiziert: beide Repos existieren, sentence-transformers, ~118M Par.).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

EXPOSE 7860
CMD ["python", "app.py"]
