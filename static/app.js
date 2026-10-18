const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const medicationInput = document.querySelector("#medications");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const welcomeMarkup = welcome.outerHTML;
const sendButton = document.querySelector("#send-button");

function scrollToLatest() {
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(role, text, timestamp = new Date()) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "YO" : "CE";

  const body = document.createElement("div");
  body.className = "message-body";

  const meta = document.createElement("div");
  meta.className = "message-meta";
  const name = document.createElement("strong");
  name.textContent = role === "user" ? "You" : "Care Evidence";
  const time = document.createElement("span");
  time.textContent = timestamp.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
  meta.append(name, time);

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  body.append(meta, bubble);
  article.append(avatar, body);
  messages.append(article);
  scrollToLatest();
  return body;
}

function addDetails(parent, title, content) {
  const details = document.createElement("details");
  details.className = "message-details";
  const summary = document.createElement("summary");
  summary.textContent = title;
  details.append(summary);

  const paragraphs = typeof content === "string" ? [content] : content;
  for (const item of paragraphs) {
    const paragraph = document.createElement("p");
    paragraph.textContent = item;
    details.append(paragraph);
  }
  parent.append(details);
}

function addAssistantResponse(parent, data) {
  if (data.review_required) {
    const badge = document.createElement("div");
    badge.className = "review-badge";
    badge.textContent = "CLINICIAN REVIEW RECOMMENDED";
    parent.append(badge);
  }

  const answer = document.createElement("div");
  answer.className = "message-bubble";
  answer.textContent = data.answer;
  parent.append(answer);

  if (data.triage) {
    addDetails(
      parent,
      `Triage · ${data.triage.severity} · ${data.triage.category}`,
      data.triage.rationale,
    );
  }

  if (data.interaction) {
    addDetails(parent, "Medication-label screen", data.interaction.summary);
    for (const finding of data.interaction.findings ?? []) {
      addDetails(
        parent,
        `Potential interaction · ${finding.medications.join(", ")}`,
        `"${finding.exact_quote}" [${finding.source_id}]`,
      );
    }
  }

  if (data.sources?.length) {
    const sourceTexts = data.sources.map(
      (source) => `[${source.id}] ${source.title}\n${source.excerpt}`,
    );
    addDetails(
      parent,
      `Retrieved evidence · ${data.sources.length} sources`,
      sourceTexts,
    );
    for (const source of data.sources) {
      const link = document.createElement("a");
      link.className = "source-link";
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = `[${source.id}] ${source.title}`;
      parent.append(link);
    }
  }

  if (data.drug_labels?.length) {
    const labels = data.drug_labels.map(
      (label) => `[${label.id}] ${label.medication}: ${label.status}`,
    );
    addDetails(parent, "OpenFDA label lookups", labels);
  }

  if (data.safety_notes?.length) {
    addDetails(parent, "Safety and availability notes", data.safety_notes);
  }

  if (data.disclaimer) {
    const note = document.createElement("p");
    note.className = "composer-disclaimer";
    note.textContent = data.disclaimer;
    parent.append(note);
  }
  scrollToLatest();
}

function showTyping() {
  const article = document.createElement("article");
  article.className = "message";
  article.id = "typing-indicator";

  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.textContent = "CE";

  const body = document.createElement("div");
  body.className = "message-body";
  const meta = document.createElement("div");
  meta.className = "message-meta";
  const name = document.createElement("strong");
  name.textContent = "Care Evidence";
  meta.append(name);

  const dots = document.createElement("div");
  dots.className = "typing-indicator";
  dots.setAttribute("aria-label", "Assistant is preparing a response");
  for (let index = 0; index < 3; index += 1) {
    dots.append(document.createElement("span"));
  }
  body.append(meta, dots);
  article.append(avatar, body);
  messages.append(article);
  scrollToLatest();
}

async function sendQuestion(question, medications) {
  if (!question.trim() || sendButton.disabled) return;
  document.querySelector("#welcome")?.remove();
  addMessage("user", question.trim());
  questionInput.value = "";
  medicationInput.value = "";
  questionInput.style.height = "auto";
  sendButton.disabled = true;
  questionInput.disabled = true;
  showTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question.trim(), medications }),
    });
    const data = await response.json();
    document.querySelector("#typing-indicator")?.remove();
    if (!response.ok) {
      throw new Error(data.error || "The assistant could not complete the request.");
    }
    const assistantBody = addMessage("assistant", "");
    assistantBody.querySelector(".message-bubble")?.remove();
    addAssistantResponse(assistantBody, data);
  } catch (error) {
    document.querySelector("#typing-indicator")?.remove();
    const body = addMessage(
      "assistant",
      error instanceof Error ? error.message : "An unexpected error occurred.",
    );
    body.querySelector(".message-bubble")?.classList.add("error-note");
  } finally {
    sendButton.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
}

function bindSuggestions() {
  document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
      questionInput.value = button.dataset.prompt ?? "";
      questionInput.focus();
      questionInput.dispatchEvent(new Event("input"));
    });
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const medications = medicationInput.value
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean);
  sendQuestion(questionInput.value, medications);
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 150)}px`;
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelector("#new-chat").addEventListener("click", () => {
  messages.innerHTML = welcomeMarkup;
  questionInput.value = "";
  medicationInput.value = "";
  questionInput.disabled = false;
  sendButton.disabled = false;
  bindSuggestions();
  questionInput.focus();
});

bindSuggestions();
