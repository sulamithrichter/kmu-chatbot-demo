// chat.js – verbindet das Eingabefeld mit dem /chat-Endpoint.
// Ablauf: Frage zeigen -> "tippt…" -> fetch POST /chat -> Antwort zeigen.

const form  = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const messages = document.getElementById("messages");

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
      body: JSON.stringify({ message: frage }),
    });

    const data = await res.json();
    typing.remove();
    // Auch bei HTTP-Fehlern (400/502) liefert das Backend ein {reply}.
    addMessage(data.reply || "Es kam keine Antwort zurück.", "bot");
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
