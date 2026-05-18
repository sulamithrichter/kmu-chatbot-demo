// chat.js – verbindet das Eingabefeld mit dem /chat-Endpoint.
// Ablauf: Frage zeigen -> "tippt…" -> fetch POST /chat -> Antwort zeigen.

const form  = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const messages = document.getElementById("messages");

// Aktuell gewähltes Suchverfahren aus dem Umschalter (Default tfidf).
function selectedRetriever() {
  const el = document.querySelector('input[name="retriever"]:checked');
  return el ? el.value : "tfidf";
}

// Eine Sprechblase einfügen. WICHTIG: textContent, nicht innerHTML
// -> der Text wird NIE als HTML interpretiert (kein Injection-Risiko).
function addMessage(text, sender) {
  const wrap = document.createElement("div");
  wrap.className = `message ${sender}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrap.appendChild(bubble);
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;   // immer ans Ende scrollen
  return wrap;
}

// Das <form>-submit fängt Klick UND Enter-Taste ab.
form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const frage = input.value.trim();
  if (!frage) return;

  addMessage(frage, "user");
  input.value = "";

  // Eingabe sperren, solange wir auf die Antwort warten.
  input.disabled = true;
  sendBtn.disabled = true;

  // "tippt…"-Platzhalter, den wir gleich durch die echte Antwort ersetzen.
  const typing = addMessage("tippt…", "bot");
  typing.querySelector(".bubble").classList.add("typing");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: frage, retriever: selectedRetriever() }),
    });

    const data = await res.json();
    typing.remove();
    // Auch bei HTTP-Fehlern (400/502) liefert das Backend ein {reply}.
    const botMsg = addMessage(data.reply || "Es kam keine Antwort zurück.", "bot");

    // Kleines Label: welcher Retriever hat WIRKLICH geantwortet (zeigt auch
    // den ehrlichen hybrid->tfidf-Fallback) und ob live oder offline.
    if (data.retriever) {
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      meta.textContent = `via ${data.retriever} · ${data.mode || "?"}`;
      botMsg.appendChild(meta);
      messages.scrollTop = messages.scrollHeight;
    }
  } catch (err) {
    // Netzwerkfehler / Server nicht erreichbar.
    typing.remove();
    addMessage(
      "Verbindung zum Server fehlgeschlagen. Läuft das Backend?",
      "bot"
    );
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
});
