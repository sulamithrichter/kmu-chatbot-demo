# Deployment – Live-Demo auf Hugging Face Spaces

Die Demo läuft als **ein** Docker-Container auf Hugging Face Spaces. Ein
Prozess lädt beide Retriever (tfidf + hybrid); der Umschalter im Chat wählt
pro Frage. Warum Hugging Face und nicht ein 512-MB-Gratis-Tier: siehe
`LERNNOTIZEN.md` Kapitel 10.

## Schritte

1. **Space anlegen** – auf <https://huggingface.co/new-space>:
   - Owner: dein Account · Space-Name: z. B. `kmu-chatbot-demo`
   - **SDK: Docker** (nicht Gradio/Streamlit) · Hardware: *CPU basic* (gratis)
   - Sichtbarkeit: Public

2. **Code in den Space pushen** – der Space ist ein eigenes Git-Repo.
   Im lokalen Projektordner:
   ```bash
   git remote add space https://huggingface.co/spaces/<DEIN-USER>/kmu-chatbot-demo
   git push space main
   ```
   (Beim Push nach Username + einem Hugging-Face *Access-Token* fragen lassen:
   huggingface.co → Settings → Access Tokens → Token mit *write*.)
   Alternative ohne Token-Push: im Space die Option „sync with GitHub repo".

3. **Secret setzen (das musst du von Hand machen)** – im Space:
   **Settings → Variables and secrets → New secret**
   - Name: `ANTHROPIC_API_KEY`
   - Value: dein echter Key von <https://console.anthropic.com>

   Der Key gehört **ausschliesslich** hierhin – nie in den Code, nie ins Repo,
   nie in `.env.example` (gleiche Regel wie lokal, nur in der Cloud;
   `LERNNOTIZEN.md` Kap. 4).

4. **Build abwarten** – der erste Build dauert einige Minuten (lädt PyTorch
   und backt die beiden Hybrid-Modelle ins Image). Danach ist die Demo unter
   `https://<DEIN-USER>-kmu-chatbot-demo.hf.space` erreichbar.

5. **Auf die Website bringen** – auf `sulamithrichter.ch` entweder per
   `<iframe src="https://<DEIN-USER>-kmu-chatbot-demo.hf.space">` einbetten
   oder als Button „Demo öffnen" verlinken.

## Manuell zu setzende Umgebungsvariablen

| Variable            | Wo                         | Pflicht | Zweck                          |
|---------------------|----------------------------|---------|--------------------------------|
| `ANTHROPIC_API_KEY` | Space → Settings → Secrets | ja      | Live-Antworten via Claude-API  |

Ohne den Key startet die Demo trotzdem und läuft im **Offline-Modus**
(Antworten direkt aus den Dokumenten) – gleiche Graceful Degradation wie lokal.

Optional: `ENABLE_HYBRID=0` (als *Variable*, nicht Secret) deaktiviert den
schweren Hybrid-Arm, falls man bewusst nur tfidf ausliefern will.

## Hinweise

- **Kaltstart:** Der Gratis-Space pausiert nach längerer Inaktivität und
  startet beim nächsten Besuch neu (einige Sekunden). Die Modelle sind im
  Image gebacken → kein erneuter Download.
- **Aktualisieren:** Code ändern → `git push space main` → der Space baut neu.
