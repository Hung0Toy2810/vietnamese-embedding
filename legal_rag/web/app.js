const $ = (id) => document.getElementById(id);
const state = {
  documents: [],
  selected: null,
  chats: new Map(),
  uploading: false,
};
const icon = (name) =>
  `<svg aria-hidden="true"><use href="#icon-${name}"/></svg>`;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "The request could not be completed. Please try again.",
    );
  }
  return response.json();
}

function conversation(id = state.selected) {
  if (!state.chats.has(id))
    state.chats.set(id, { messages: [], pending: null, draft: "" });
  return state.chats.get(id);
}

function selectedDocument() {
  return state.documents.find((doc) => doc.id === state.selected);
}
function size(bytes) {
  return bytes < 1024 * 1024
    ? `${Math.ceil(bytes / 1024)} KB`
    : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function renderDocuments() {
  $("document-count").textContent = state.documents.length;
  $("document-list").replaceChildren();
  for (const doc of state.documents) {
    const button = document.createElement("button");
    button.className = `document-item${doc.id === state.selected ? " active" : ""}`;
    button.setAttribute("aria-pressed", String(doc.id === state.selected));
    button.title = doc.name;
    button.innerHTML = `<span class="pdf-icon">${icon("file")}</span><span class="document-details"><span class="document-name"></span><span class="document-meta"><span class="small-dot"></span><span></span></span><span class="document-status"></span></span>`;
    button.querySelector(".document-name").textContent = doc.name;
    button.querySelector(".document-meta").classList.add(doc.status);
    button.querySelector(".document-meta span:last-child").textContent =
      `${doc.pages ? doc.pages + " pages" : doc.suffix.slice(1).toUpperCase()} · ${size(doc.size)}`;
    button.querySelector(".document-status").textContent =
      doc.status === "ready" ? doc.message : doc.message;
    button.onclick = () => selectDocument(doc.id);
    $("document-list").append(button);
  }
  if (!state.documents.length) {
    const empty = document.createElement("p");
    empty.className = "no-documents";
    empty.textContent = "No documents yet. Upload a document to start.";
    $("document-list").append(empty);
  }
}

function openPdf(url, title, page = 1, info = "") {
  $("mindmap-viewer").hidden = true;
  $("return-pdf").hidden = true;
  $("viewer-empty").hidden = true;
  $("pdf-viewer").hidden = false;
  $("pdf-viewer").removeAttribute("sandbox");
  $("pdf-viewer").src = `${url}#page=${page}&view=FitH`;
  $("viewer-title").textContent = title;
  $("viewer-info").textContent = info || `Page ${page}`;
  $("open-pdf").hidden = false;
  $("open-pdf").href = `${url}#page=${page}`;
}

function selectDocument(id) {
  if (state.selected) conversation().draft = $("question").value;
  state.selected = id;
  const doc = selectedDocument();
  if (doc.suffix === ".pdf")
    openPdf(
      `/api/files/${id}`,
      doc.name,
      1,
      `${doc.pages ? doc.pages + " pages" : doc.suffix.slice(1).toUpperCase()} · ${size(doc.size)}`,
    );
  else openPreview(doc);
  $("question").value = conversation().draft;
  renderDocuments();
  renderChat();
}

function renderAnswerText(element, text, evidence) {
  element.innerHTML = DOMPurify.sanitize(marked.parse(text, { breaks: true }), {
    ALLOWED_TAGS: [
      "p",
      "strong",
      "em",
      "ul",
      "ol",
      "li",
      "table",
      "thead",
      "tbody",
      "tr",
      "th",
      "td",
      "br",
      "blockquote",
      "code",
      "pre",
      "h2",
      "h3",
      "h4",
    ],
    ALLOWED_ATTR: [],
  });
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    if (!/\[Source \d+\]/.test(node.textContent)) continue;
    const fragment = document.createDocumentFragment();
    for (const part of node.textContent.split(/(\[Source \d+\])/g)) {
      const match = part.match(/^\[Source (\d+)\]$/);
      const source =
        match && evidence.find((item) => item.source === Number(match[1]));
      if (source) {
        const button = document.createElement("button");
        button.className = "citation-inline";
        button.textContent = part;
        button.title = "Open cited source";
        button.onclick = () => openEvidence(source);
        fragment.append(button);
      } else fragment.append(document.createTextNode(part));
    }
    node.replaceWith(fragment);
  }
}

function renderChat() {
  const doc = selectedDocument();
  const chat = conversation();
  $("chat-context").textContent = doc ? doc.name : "Select a document to start";
  $("chat-empty").hidden = chat.messages.length > 0;
  $("messages").replaceChildren();
  for (const message of chat.messages) {
    const block = document.createElement("div");
    block.className = `message ${message.role}${message.error ? " error" : ""}`;
    if (message.role === "user") block.textContent = message.text;
    else {
      const label = document.createElement("div");
      label.className = "message-label";
      label.innerHTML = `${icon("chat")}<span>DOCUMENT ASSISTANT</span>`;
      const body = document.createElement("div");
      body.className = "answer-text";
      renderAnswerText(body, message.text, message.evidence || []);
      block.append(label, body);
      if (message.evidence?.length) {
        const details = document.createElement("details");
        details.className = "sources";
        const summary = document.createElement("summary");
        summary.textContent = `View citations (${message.evidence.length})`;
        details.append(summary);
        for (const source of message.evidence) {
          const card = document.createElement("button");
          card.className = "source-card";
          card.innerHTML =
            '<span class="source-card-title"></span><span class="source-card-quote"></span><span class="source-page"></span>';
          card.querySelector(".source-card-title").textContent =
            `[Source ${source.source}] ${source.title}`;
          card.querySelector(".source-card-quote").textContent = source.quote;
          card.querySelector(".source-page").textContent = source.pdf_url
            ? `Open PDF · Page ${source.page} ↗`
            : "Open web source ↗";
          card.onclick = () => openEvidence(source);
          details.append(card);
        }
        block.append(details);
      }
    }
    if (message.follow_up?.length) {
      const questions = document.createElement("div");
      questions.className = "follow-ups";
      questions.textContent =
        "To clarify your question:\n" +
        message.follow_up.map((text) => `• ${text}`).join("\n");
      block.append(questions);
    }
    if (message.mindmap) {
      const button = document.createElement("button");
      button.className = "map-button";
      button.textContent = "View source graph";
      button.onclick = () => showMindmap(message.mindmap);
      block.append(button);
    }
    if (message.steps?.length) {
      const details = document.createElement("details");
      details.className = "sources agent-steps";
      const summary = document.createElement("summary");
      summary.textContent = `Retrieval steps (${message.steps.length})`;
      details.append(summary);
      for (const step of message.steps) {
        const line = document.createElement("div");
        line.textContent = `${step.tool}: ${step.input}`;
        details.append(line);
      }
      block.append(details);
    }
    $("messages").append(block);
  }
  if (chat.pending) {
    const pending = document.createElement("div");
    pending.className = "pending";
    pending.innerHTML =
      '<span class="spinner"></span><span><span class="pending-message"></span><span class="pending-time"></span></span>';
    pending.querySelector(".pending-message").textContent =
      chat.pending.message;
    pending.querySelector(".pending-time").textContent = elapsed(
      chat.pending.start,
    );
    $("messages").append(pending);
  }
  const ready = Boolean(doc?.active_version);
  $("index-document").hidden =
    !doc || ["queued", "processing"].includes(doc.status);
  $("approve-document").hidden = !doc?.pending_version;
  $("show-document-graph").hidden = !doc?.active_version;
  $("question").disabled = !ready || Boolean(chat.pending);
  $("send-button").disabled =
    !ready || Boolean(chat.pending) || !$("question").value.trim();
  $("new-chat").disabled = Boolean(chat.pending) || !chat.messages.length;
  $("document-notice").hidden =
    !doc || (ready && !doc.pending_version && !doc.warnings?.length);
  $("document-notice").textContent = doc
    ? [doc.message, ...(doc.warnings || [])].join("\n")
    : "";
  $("suggestions").replaceChildren();
  if (doc?.active_version) {
    for (const text of [
      "Summarize the key points of this document.",
      "What qualifications or exceptions are stated?",
    ]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "suggestion";
      button.textContent = text;
      button.disabled = !ready;
      button.onclick = () => {
        $("question").value = text;
        $("question").dispatchEvent(new Event("input"));
        $("question").focus();
      };
      $("suggestions").append(button);
    }
  }
}

function elapsed(start) {
  const seconds = Math.floor((Date.now() - start) / 1000);
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

async function refreshDocuments() {
  const previous = JSON.stringify(selectedDocument());
  state.documents = await api("/api/documents");
  renderDocuments();
  if (!state.selected && state.documents.length)
    selectDocument(state.documents[0].id);
  else if (previous !== JSON.stringify(selectedDocument())) renderChat();
}

async function uploadFiles(files) {
  if (!files.length || state.uploading) return;
  state.uploading = true;
  $("upload-button").disabled = true;
  $("upload-message").hidden = false;
  try {
    for (const file of files) {
      if (file.size > 25 * 1024 * 1024)
        throw new Error(`${file.name}: exceeds 25 MB.`);
      $("upload-message").textContent = `Uploading ${file.name}…`;
      const body = new FormData();
      body.append("file", file);
      const result = await api("/api/documents", { method: "POST", body });
      await refreshDocuments();
      selectDocument(result.document.id);
    }
    $("upload-message").textContent =
      "Uploaded. Semantic analysis and indexing are queued.";
  } catch (error) {
    $("upload-message").textContent = error.message;
  } finally {
    state.uploading = false;
    $("upload-button").disabled = false;
    $("file-input").value = "";
  }
}

$("upload-button").onclick = () => $("file-input").click();
$("file-input").onchange = (event) => uploadFiles([...event.target.files]);
let dragDepth = 0;
$("drop-area").addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth++;
  $("drop-area").classList.add("dragging");
});
$("drop-area").addEventListener("dragover", (event) => event.preventDefault());
$("drop-area").addEventListener("dragleave", () => {
  if (--dragDepth <= 0) $("drop-area").classList.remove("dragging");
});
$("drop-area").addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  $("drop-area").classList.remove("dragging");
  uploadFiles([...event.dataTransfer.files]);
});
$("question").addEventListener("input", () => {
  conversation().draft = $("question").value;
  $("send-button").disabled =
    !selectedDocument() ||
    Boolean(conversation().pending) ||
    !$("question").value.trim();
});
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!$("send-button").disabled) $("chat-form").requestSubmit();
  }
});
$("new-chat").onclick = () => {
  state.chats.set(state.selected, { messages: [], pending: null, draft: "" });
  $("question").value = "";
  renderChat();
};

$("chat-form").onsubmit = async (event) => {
  event.preventDefault();
  const doc = selectedDocument();
  const chat = conversation();
  const question = $("question").value.trim();
  if (!doc || !doc.active_version || !question || chat.pending) return;
  chat.messages.push({ role: "user", text: question });
  chat.draft = "";
  $("question").value = "";
  chat.pending = { message: "Sending question", start: Date.now() };
  renderChat();
  $("chat-scroll").scrollTop = $("chat-scroll").scrollHeight;
  try {
    const response = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document_id: doc.id,
        question,
        use_web: $("use-web").checked,
        history: chat.messages
          .slice(0, -1)
          .filter((m) => m.role === "user" || m.follow_up?.length)
          .slice(-4)
          .map((m) => ({
            role: m.role,
            content: (m.role === "user"
              ? m.text
              : m.follow_up.join("\n")
            ).slice(0, 2000),
          })),
      }),
    });
    let job;
    do {
      job = await api(`/api/jobs/${response.job_id}`);
      chat.pending.message = job.message;
      if (state.selected === doc.id) {
        const label = document.querySelector(".pending-message");
        if (label) label.textContent = job.message;
      }
      if (!["done", "error"].includes(job.status))
        await new Promise((resolve) => setTimeout(resolve, 1500));
    } while (!["done", "error"].includes(job.status));
    if (job.status === "error") throw new Error(job.message);
    chat.messages.push({
      role: "assistant",
      text: job.result.answer,
      evidence: job.result.evidence,
      mindmap: job.result.mindmap,
      follow_up: job.result.follow_up,
      steps: job.result.steps,
    });
  } catch (error) {
    chat.messages.push({ role: "assistant", text: error.message, error: true });
  } finally {
    chat.pending = null;
    if (state.selected === doc.id) {
      renderChat();
      $("chat-scroll").scrollTop = $("chat-scroll").scrollHeight;
      $("question").focus();
    }
  }
};

setInterval(() => {
  const pending = conversation().pending;
  if (pending) {
    const timer = document.querySelector(".pending-time");
    if (timer) timer.textContent = elapsed(pending.start);
  }
  if (
    state.documents.some((doc) => ["queued", "processing"].includes(doc.status))
  )
    refreshDocuments().catch(() => {});
}, 2000);
refreshDocuments().catch((error) => {
  $("upload-message").hidden = false;
  $("upload-message").textContent = error.message;
});

// Render stored semantic relations; selecting a node opens its source.
let graph = null;
function openEvidence(source) {
  if (source.pdf_url)
    openPdf(source.pdf_url, source.title || source.label, source.page);
  else if (source.url?.startsWith("https://"))
    window.open(source.url, "_blank", "noopener,noreferrer");
  else if (source.preview_url)
    openPreviewUrl(source.preview_url, source.title || source.label);
}
function showMindmap(map) {
  $("pdf-viewer").hidden = true;
  $("viewer-empty").hidden = true;
  $("mindmap-viewer").hidden = false;
  $("return-pdf").hidden = false;
  $("viewer-title").textContent = "Semantic source graph";
  $("viewer-info").textContent =
    "Select a node to open its source · Scroll to zoom";
  if (graph) graph.destroy();
  graph = cytoscape({
    container: $("mindmap-viewer"),
    elements: [
      ...map.nodes.map((node) => ({ data: node })),
      ...map.edges.map((edge, i) => ({ data: { id: `edge-${i}`, ...edge } })),
    ],
    style: [
      {
        selector: "node",
        style: {
          label: "data(label)",
          "background-color": "#eaf2ff",
          color: "#245397",
          "border-color": "#a8c7f0",
          "border-width": 1,
          shape: "roundrectangle",
          width: 145,
          height: 64,
          "text-wrap": "wrap",
          "text-max-width": 130,
          "text-valign": "center",
          "font-size": 11,
        },
      },
      {
        selector: "edge",
        style: {
          label: "data(label)",
          width: 1,
          "line-color": "#b4c8e1",
          "target-arrow-color": "#b4c8e1",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          "font-size": 9,
          color: "#71849b",
        },
      },
    ],
    layout: {
      name: "breadthfirst",
      directed: true,
      spacingFactor: 1.2,
      padding: 30,
    },
    minZoom: 0.15,
    maxZoom: 2.5,
  });
  graph.on("tap", "node", (event) => openEvidence(event.target.data()));
}
$("return-pdf").onclick = () => {
  $("mindmap-viewer").hidden = true;
  $("pdf-viewer").hidden = false;
  $("return-pdf").hidden = true;
  $("viewer-title").textContent = selectedDocument()?.name || "PDF";
  $("viewer-info").textContent = "Current document";
};
new ResizeObserver(() => {
  if (graph && !$("mindmap-viewer").hidden) graph.resize();
}).observe($("mindmap-viewer"));

function openPreview(doc) {
  $("mindmap-viewer").hidden = true;
  $("viewer-empty").hidden = true;
  $("pdf-viewer").hidden = false;
  $("return-pdf").hidden = true;
  $("pdf-viewer").setAttribute("sandbox", "allow-same-origin");
  $("pdf-viewer").src = `/api/preview/${doc.id}`;
  $("viewer-title").textContent = doc.name;
  $("viewer-info").textContent = "Extracted source text";
  $("open-pdf").hidden = false;
  $("open-pdf").href = `/api/files/${doc.id}`;
}
$("index-document").onclick = async () => {
  const doc = selectedDocument();
  try {
    await api(`/api/documents/${doc.id}/index`, { method: "POST" });
    await refreshDocuments();
  } catch (error) {
    $("document-notice").hidden = false;
    $("document-notice").textContent = error.message;
  }
};
$("approve-document").onclick = async () => {
  const doc = selectedDocument();
  try {
    await api(`/api/documents/${doc.id}/approve`, { method: "POST" });
    await refreshDocuments();
  } catch (error) {
    $("document-notice").textContent = error.message;
  }
};
$("show-document-graph").onclick = async () => {
  try {
    const doc = selectedDocument();
    const result = await api(`/api/documents/${doc.id}/graph`);
    const semantic = result.graph;
    showMindmap({
      nodes: semantic.nodes.map((node) => ({
        id: node.id,
        label: node.text.slice(0, 90),
        page: node.spans.find((span) => span.page)?.page || 1,
        pdf_url: doc.suffix === ".pdf" ? `/api/files/${doc.id}` : null,
        preview_url: `/api/preview/${doc.id}`,
      })),
      edges: semantic.edges
        .filter((edge) => edge.semantic ?? edge.grouping)
        .map((edge) => ({
          source: edge.source,
          target: edge.target,
          label: edge.kind,
        })),
    });
    $("viewer-info").textContent = result.plan.rationale;
  } catch (error) {
    $("document-notice").hidden = false;
    $("document-notice").textContent = error.message;
  }
};

function openPreviewUrl(url, title) {
  $("mindmap-viewer").hidden = true;
  $("viewer-empty").hidden = true;
  $("pdf-viewer").hidden = false;
  $("return-pdf").hidden = false;
  $("pdf-viewer").setAttribute("sandbox", "allow-same-origin");
  $("pdf-viewer").src = url;
  $("viewer-title").textContent = title;
  $("viewer-info").textContent = "Cited source text";
}
