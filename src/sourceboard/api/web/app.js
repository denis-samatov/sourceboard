// SourceBoard web client. All user and file content is rendered with textContent.

const ROLE_LABELS = {
  supports: "Подтверждает",
  limits: "Ограничивает",
  contradicts: "Противоречит",
  context: "Контекст",
};
const ROLE_SHORT = { supports: "Подтв.", limits: "Огранич.", contradicts: "Противор.", context: "Контекст" };
const MODALITY_LABELS = { text: "текст", image: "изображение", video: "кадр", audio: "звук" };
const KIND_LABELS = { document: "документ", image: "изображение", audio: "аудио", video: "видео", transcript: "транскрипт" };
const STATUS_LABELS = { indexed: "проиндексирован", indexing: "индексируется", pending: "в очереди", error: "ошибка", missing: "оригинал недоступен" };

const app = {
  state: null,
  theses: [],
  activeThesisId: readStored("activeThesis"),
  results: [],
  lastSearch: null,
  wasRunning: false,
};

// ------------------------------------------------------------------ helpers

function readStored(key) {
  try { const v = localStorage.getItem(key); return v ? Number(v) : null; } catch { return null; }
}
function store(key, value) {
  try { value == null ? localStorage.removeItem(key) : localStorage.setItem(key, String(value)); } catch { /* private mode */ }
}

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

async function api(path, options = {}) {
  const init = { ...options };
  if (init.body && !(init.body instanceof FormData)) {
    init.headers = { "Content-Type": "application/json", ...(init.headers || {}) };
    init.body = JSON.stringify(init.body);
  }
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch { /* not json */ }
    throw new Error(detail);
  }
  return response.json();
}

let toastTimer;
function toast(message) {
  document.querySelector(".toast")?.remove();
  const el = h("div", { class: "toast", role: "status", text: message });
  document.body.append(el);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.remove(), 3500);
}

function fmtTime(seconds) {
  if (seconds == null) return "";
  const s = Math.max(0, Math.floor(seconds));
  const hh = Math.floor(s / 3600);
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return hh ? `${hh}:${mm}:${ss}` : `${mm}:${ss}`;
}

function locatorText(item) {
  const parts = [];
  if (item.page != null) parts.push(`стр. ${item.page}`);
  if (item.t_start != null) parts.push(`${fmtTime(item.t_start)}–${fmtTime(item.t_end)}`);
  return parts.join(" · ");
}

function highlight(text, query) {
  // Mark query words (and their stems) inside text, without using innerHTML.
  const fragment = document.createDocumentFragment();
  const words = (query || "").toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) || [];
  const stems = [...new Set(words.map((w) => (w.length >= 6 ? w.slice(0, w.length - 2) : w)))];
  if (!stems.length) { fragment.append(text); return fragment; }
  const escaped = stems.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${escaped.join("|")})[\\p{L}\\p{N}]*`, "giu");
  let last = 0;
  for (const match of text.matchAll(re)) {
    fragment.append(text.slice(last, match.index));
    fragment.append(h("mark", { text: match[0] }));
    last = match.index + match[0].length;
  }
  fragment.append(text.slice(last));
  return fragment;
}

function activeThesis() {
  return app.theses.find((t) => t.id === app.activeThesisId) || null;
}

// ------------------------------------------------------------------ state & sources

async function refreshState() {
  try {
    app.state = await api("/api/state");
  } catch (err) {
    console.error(err);
    return setTimeout(refreshState, 5000);
  }
  renderState();
  const running = app.state.progress.running;
  if (app.wasRunning && !running) {
    refreshSources();
    refreshTheses();
    toast("Индексация завершена");
  }
  app.wasRunning = running;
  setTimeout(refreshState, running ? 1200 : 8000);
}

function renderState() {
  const { embedder, progress, folders, tools, chunks } = app.state;
  const badge = document.getElementById("embedder-badge");
  if (embedder.semantic) {
    const mods = embedder.modalities.map((m) => ({ text: "текст", image: "изображения", audio: "звук" })[m]).join(", ");
    badge.textContent = `EmbeddingGemma 2 · ${mods}`;
    badge.title = embedder.name;
    badge.classList.remove("warn");
  } else {
    badge.textContent = "Лексический режим: семантика выключена";
    badge.title = "Установите extra ml (uv sync --extra ml), чтобы включить EmbeddingGemma 2";
    badge.classList.add("warn");
  }
  const imageButton = document.getElementById("image-btn");
  const canImage = embedder.modalities.includes("image");
  imageButton.disabled = !canImage;
  imageButton.title = canImage
    ? "Искать по изображению (можно перетащить или вставить из буфера)"
    : "Поиск по изображению требует EmbeddingGemma 2";

  const box = document.getElementById("progress");
  box.hidden = !progress.running;
  if (progress.running) {
    const pct = progress.total ? Math.round((progress.done / progress.total) * 100) : 0;
    document.getElementById("progress-fill").style.width = `${pct}%`;
    document.getElementById("progress-text").textContent =
      `Индексация ${progress.done}/${progress.total}: ${progress.current}${progress.detail ? " — " + progress.detail : ""}`;
  }

  const list = document.getElementById("folders");
  list.replaceChildren(
    ...folders.map((f) => {
      const counts = Object.entries(f.sources).map(([k, v]) => `${STATUS_LABELS[k] || k}: ${v}`).join(", ");
      return h("div", { class: `folder${f.available ? "" : " offline"}` },
        h("div", {},
          h("div", { text: f.path }),
          h("div", { class: "muted", text: f.available ? (counts || "пусто") : `Папка недоступна. Каталог сохранён. ${counts}` })),
        h("button", {
          class: "ghost small", type: "button", title: "Убрать папку из индекса", text: "✕",
          onclick: async () => {
            if (!confirm(`Убрать папку из индекса?\n${f.path}\n\nФайлы на диске не удаляются. Связи в тезисах сохранятся как снимки.`)) return;
            await api(`/api/folders/${f.id}`, { method: "DELETE" });
            refreshAll();
          },
        }));
    }),
  );

  const toolNotes = [];
  if (!tools.ffmpeg) toolNotes.push("ffmpeg не найден: кадры и звук не индексируются");
  if (!tools.tesseract) toolNotes.push("tesseract не найден: OCR выключен");
  if (!tools.asr) toolNotes.push("распознавание речи не установлено: для аудио и видео нужны субтитры .srt/.vtt рядом с файлом");
  const total = Object.values(chunks).reduce((a, b) => a + b, 0);
  const stats = total ? `Фрагментов в индексе: ${total} (${Object.entries(chunks).map(([k, v]) => `${MODALITY_LABELS[k] || k} ${v}`).join(", ")}).` : "";
  document.getElementById("tools").textContent = [stats, ...toolNotes].filter(Boolean).join(" ");
}

async function refreshSources() {
  const sources = await api("/api/sources");
  const list = document.getElementById("sources");
  if (!sources.length) {
    list.replaceChildren(h("li", { class: "muted", text: "Добавьте папку с документами, записями, видео и изображениями." }));
    return;
  }
  list.replaceChildren(
    ...sources.map((s) =>
      h("li", {},
        h("div", { class: "name" },
          h("span", { class: `status-dot ${s.status}`, title: STATUS_LABELS[s.status] || s.status }),
          h("span", { class: "path", text: s.rel_path })),
        h("div", { class: "note", text: [KIND_LABELS[s.kind], s.pages ? `${s.pages} стр.` : null, s.duration ? fmtTime(s.duration) : null, s.sidecar_path ? "субтитры" : null, `фрагментов: ${s.chunks}`, s.version > 1 ? `версия ${s.version}` : null].filter(Boolean).join(" · ") }),
        s.notes ? h("div", { class: "note", text: s.notes }) : null,
        s.error ? h("div", { class: "err", text: s.error }) : null)),
  );
}

// ------------------------------------------------------------------ search

function selectedModalities() {
  const boxes = [...document.querySelectorAll('input[name="modality"]')];
  const checked = boxes.filter((b) => b.checked).map((b) => b.value);
  return checked.length === boxes.length ? null : checked;
}

async function runSearch(query, task) {
  const q = query.trim();
  if (!q) return;
  app.lastSearch = { kind: "text", query: q, task };
  setResultsHead(`Ищу: «${q}»…`);
  try {
    const results = await api("/api/search", {
      method: "POST",
      body: { q, task, mode: document.getElementById("mode").value, modalities: selectedModalities(), limit: 30 },
    });
    renderResults(results, { query: q, headline: `Найдено ${results.length} по запросу «${q}»` });
  } catch (err) {
    setResultsHead(`Ошибка поиска: ${err.message}`);
  }
}

async function runImageSearch(file) {
  if (!app.state?.embedder.modalities.includes("image")) {
    toast("Поиск по изображению требует EmbeddingGemma 2 с vision-энкодером");
    return;
  }
  app.lastSearch = { kind: "image", file };
  const head = document.getElementById("results-head");
  const url = URL.createObjectURL(file);
  head.replaceChildren(h("img", { src: url, alt: "Изображение-запрос" }), h("span", { text: "Ищу похожие материалы…" }));
  const form = new FormData();
  form.append("file", file, file.name || "query.png");
  const mods = selectedModalities();
  const params = new URLSearchParams({ limit: "30" });
  (mods || []).forEach((m) => params.append("modalities", m));
  try {
    const results = await api(`/api/search/image?${params}`, { method: "POST", body: form });
    renderResults(results, { query: "", headline: `Похожие на изображение: ${results.length}`, image: url });
  } catch (err) {
    setResultsHead(`Ошибка поиска по изображению: ${err.message}`);
  }
}

async function runSimilar(chunkId, label) {
  app.lastSearch = { kind: "similar", chunkId, label };
  setResultsHead("Ищу похожие фрагменты…");
  const results = await api(`/api/chunks/${chunkId}/similar?limit=30`);
  renderResults(results, { query: "", headline: `Похожие на фрагмент из ${label}: ${results.length}` });
}

async function runCounterpoints(thesis) {
  app.lastSearch = { kind: "counter", thesisId: thesis.id };
  setResultsHead("Ищу материалы, которые могут изменить вывод…");
  const results = await api(`/api/theses/${thesis.id}/counterpoints?limit=15`);
  renderResults(results, {
    query: thesis.title,
    headline: `Другая точка зрения для «${thesis.title}»: ${results.length} кандидатов для проверки. Это не исчерпывающий поиск противоречий.`,
  });
}

function setResultsHead(text) {
  document.getElementById("results-head").replaceChildren(h("span", { text }));
}

function renderResults(results, { query, headline, image }) {
  app.results = results;
  const head = document.getElementById("results-head");
  head.replaceChildren(...[image ? h("img", { src: image, alt: "Изображение-запрос" }) : null, h("span", { text: headline })].filter(Boolean));
  const container = document.getElementById("results");
  if (!results.length) {
    container.replaceChildren(h("div", { class: "empty" },
      h("strong", { text: "Ничего не найдено" }),
      "Попробуйте другие слова, уберите фильтры или проверьте, что папка проиндексирована."));
    return;
  }
  const thesis = activeThesis();
  const inThesis = new Set((thesis?.evidence || []).map((e) => e.chunk_id));
  container.replaceChildren(...results.map((r) => resultCard(r, query, thesis, inThesis)));
}

function resultCard(r, query, thesis, inThesis) {
  const card = document.getElementById("tpl-result").content.firstElementChild.cloneNode(true);
  card.querySelector(".modality").textContent = MODALITY_LABELS[r.modality] || r.modality;
  const link = card.querySelector(".source-link");
  link.textContent = r.source;
  link.addEventListener("click", () => openViewer(r.chunk_id));
  card.querySelector(".locator").textContent = locatorText(r);
  if (r.source_status === "missing") card.querySelector(".card-head").append(h("span", { class: "tag-new", text: "оригинал недоступен" }));
  if (r.new_source) card.querySelector(".card-head").append(h("span", { class: "tag-new", text: "новый источник для тезиса" }));

  const body = card.querySelector(".card-body");
  if (r.has_preview && r.modality !== "text") {
    body.append(h("img", { src: `/api/chunks/${r.chunk_id}/preview`, alt: `${MODALITY_LABELS[r.modality]} из ${r.source}`, loading: "lazy" }));
    if (r.text) body.append(h("div", { class: "ocr clamp" }, "Распознанный текст: ", highlight(r.text, query)));
  } else if (r.modality === "audio") {
    body.append(h("p", { class: "muted", text: "Звуковой фрагмент (по звучанию, без текста). Откройте, чтобы прослушать." }));
  } else {
    body.append(h("p", { class: "clamp" }, highlight(r.text, query)));
  }

  const why = [];
  if (r.lexical_rank) why.push("совпадение слов");
  if (r.similarity != null) why.push(`смысловая близость ${r.similarity.toFixed(2)}`);
  card.querySelector(".why").textContent = why.join(" · ");
  card.querySelector(".why").title = "Близость не является вероятностью истинности";

  const actions = card.querySelector(".actions");
  actions.append(
    h("button", { class: "small ghost", type: "button", text: "Открыть", onclick: () => openViewer(r.chunk_id) }),
    h("button", { class: "small ghost", type: "button", text: "Похожие", onclick: () => runSimilar(r.chunk_id, r.source) }),
  );
  if (thesis) {
    if (inThesis.has(r.chunk_id)) {
      actions.append(h("span", { class: "in-thesis", text: "✓ в тезисе" }));
    } else {
      for (const role of Object.keys(ROLE_SHORT)) {
        actions.append(h("button", {
          class: "small role-btn", type: "button", dataset: { role }, text: `+ ${ROLE_SHORT[role]}`,
          title: `Добавить к тезису «${thesis.title}» как «${ROLE_LABELS[role]}»`,
          onclick: () => addEvidence(thesis.id, r.chunk_id, role),
        }));
      }
    }
  } else {
    actions.append(h("span", { class: "why", text: "Выберите тезис справа, чтобы добавлять фрагменты" }));
  }
  return card;
}

// ------------------------------------------------------------------ theses

async function refreshTheses() {
  app.theses = await api("/api/theses");
  if (app.activeThesisId && !activeThesis()) setActive(null);
  if (!app.activeThesisId && app.theses.length) setActive(app.theses[0].id, false);
  renderTheses();
}

function setActive(id, rerender = true) {
  app.activeThesisId = id;
  store("activeThesis", id);
  if (rerender) {
    renderTheses();
    if (app.results.length) renderResults(app.results, currentResultsContext());
  }
}

function currentResultsContext() {
  const headline = document.getElementById("results-head").textContent;
  const image = document.querySelector("#results-head img")?.src;
  return { query: app.lastSearch?.query || "", headline, image };
}

function renderTheses() {
  const container = document.getElementById("theses");
  if (!app.theses.length) {
    container.replaceChildren(h("div", { class: "empty" },
      h("strong", { text: "Пока нет тезисов" }),
      "Сформулируйте утверждение, для которого нужно собрать проверяемые основания."));
    return;
  }
  container.replaceChildren(...app.theses.map(thesisCard));
}

function thesisCard(t) {
  const active = t.id === app.activeThesisId;
  const title = h("p", {
    class: "thesis-title", contenteditable: "true", spellcheck: "true", text: t.title,
    "aria-label": "Формулировка тезиса",
    onkeydown: (e) => { if (e.key === "Enter") { e.preventDefault(); e.target.blur(); } },
    onblur: (e) => saveThesisTitle(t, e.target.textContent),
  });
  const meta = h("div", { class: "thesis-meta" },
    active ? h("span", { class: "active-flag", text: "● активный" }) : null,
    h("span", { text: `фрагментов: ${t.evidence.length}` }),
    t.checked ? h("span", { class: "ok", text: `✓ проверено: ${t.checked}` }) : null,
    t.needs_recheck ? h("span", { class: "warn", text: `⚠ перепроверить: ${t.needs_recheck}` }) : null,
    t.revision > 1 ? h("span", { text: `ред. ${t.revision}` }) : null);

  const card = h("article", { class: `thesis${active ? " active" : ""}` },
    h("div", { class: "thesis-head", onclick: (e) => { if (!e.target.isContentEditable) setActive(t.id); } }, title, meta),
    h("div", { class: "thesis-tools" },
      h("button", {
        class: "small", type: "button", text: "Найти основания",
        title: "Поиск по тезису. Можно выделить часть формулировки — тогда ищется только она.",
        onmousedown: (e) => e.preventDefault(),
        onclick: () => {
          const selection = window.getSelection()?.toString().trim();
          const usable = selection && title.contains(window.getSelection().anchorNode) ? selection : t.title;
          setActive(t.id);
          document.getElementById("query").value = usable;
          document.getElementById("task").value = "FactChecking";
          runSearch(usable, "FactChecking");
        },
      }),
      h("button", { class: "small", type: "button", text: "Другая точка зрения", onclick: () => { setActive(t.id); runCounterpoints(t); } }),
      h("button", {
        class: "small ghost", type: "button", text: "Удалить",
        onclick: async () => {
          if (!confirm(`Удалить тезис и его подборку (${t.evidence.length} фрагм.)?`)) return;
          await api(`/api/theses/${t.id}`, { method: "DELETE" });
          refreshTheses();
        },
      })));

  for (const role of Object.keys(ROLE_LABELS)) {
    const items = t.evidence.filter((e) => e.role === role);
    if (!items.length) continue;
    card.append(h("section", { class: "role-group", dataset: { role } },
      h("h4", { text: `${ROLE_LABELS[role]} · ${items.length}` }),
      ...items.map(evidenceItem)));
  }
  return card;
}

async function saveThesisTitle(t, text) {
  const value = text.replace(/\s+/g, " ").trim();
  if (!value || value === t.title) { renderTheses(); return; }
  let minor = false;
  if (t.checked) {
    minor = confirm("Это только исправление опечатки, смысл тезиса не изменился?\n\nOK: проверки фрагментов сохранятся.\nОтмена: проверенные фрагменты будут отмечены для повторной проверки.");
  }
  try {
    await api(`/api/theses/${t.id}`, { method: "PATCH", body: { title: value, minor } });
  } catch (err) { toast(err.message); }
  refreshTheses();
}

function evidenceItem(e) {
  const roleSelect = h("select", {
    "aria-label": "Роль фрагмента",
    onchange: (ev) => patchEvidence(e.id, { role: ev.target.value }),
  }, ...Object.entries(ROLE_LABELS).map(([value, label]) => h("option", { value, selected: value === e.role, text: label })));
  const checkbox = h("input", {
    type: "checkbox", checked: e.context_checked,
    onchange: (ev) => patchEvidence(e.id, { context_checked: ev.target.checked }),
  });
  const note = h("textarea", {
    class: "note", rows: "1", placeholder: "Заметка: зачем фрагмент, сомнения, что уточнить", text: e.note,
    onchange: (ev) => patchEvidence(e.id, { note: ev.target.value }, false),
  });
  note.value = e.note;

  return h("div", { class: "evidence" },
    h("div", { class: "src" },
      h("span", { class: "modality", text: MODALITY_LABELS[e.modality] || e.modality }),
      h("button", { class: "source-link", type: "button", text: e.source, onclick: () => openEvidence(e) }),
      h("span", { class: "locator", text: locatorText(e) })),
    e.has_preview && e.chunk_id ? h("img", { class: "thumb", src: `/api/chunks/${e.chunk_id}/preview`, alt: "", loading: "lazy" }) : null,
    e.excerpt ? h("blockquote", { text: e.excerpt }) : null,
    e.current_excerpt ? h("details", { class: "diff" },
      h("summary", { text: "Текст в новой версии источника" }),
      h("p", { text: e.current_excerpt })) : null,
    ...e.issues.map((i) => h("div", { class: "issue", text: `⚠ ${i.message}` })),
    h("div", { class: "controls" },
      roleSelect,
      h("label", { class: "check", title: "Я открыл оригинал и проверил окружающий контекст" }, checkbox,
        e.needs_recheck && e.context_checked ? "проверить снова" : "контекст проверен"),
      h("button", {
        class: "link-btn danger", type: "button", text: "убрать",
        onclick: async () => { await api(`/api/evidence/${e.id}`, { method: "DELETE" }); await refreshTheses(); rerenderResults(); },
      })),
    note);
}

async function addEvidence(thesisId, chunkId, role) {
  try {
    await api(`/api/theses/${thesisId}/evidence`, { method: "POST", body: { chunk_id: chunkId, role } });
    await refreshTheses();
    rerenderResults();
    toast(`Добавлено: ${ROLE_LABELS[role]}`);
  } catch (err) { toast(err.message); }
}

async function patchEvidence(id, body, rerender = true) {
  try {
    await api(`/api/evidence/${id}`, { method: "PATCH", body });
    if (rerender) await refreshTheses();
    else app.theses = await api("/api/theses");
  } catch (err) { toast(err.message); }
}

function rerenderResults() {
  if (app.results.length) renderResults(app.results, currentResultsContext());
}

// ------------------------------------------------------------------ viewer

const viewer = document.getElementById("viewer");
viewer.querySelector("[data-close]").addEventListener("click", () => viewer.close());
viewer.addEventListener("close", () => document.getElementById("viewer-original").replaceChildren());

function openEvidence(e) {
  if (e.chunk_id) return openViewer(e.chunk_id);
  // The fragment no longer exists in the index: show the saved snapshot.
  showViewer({
    title: e.source,
    meta: `${locatorText(e)} · снимок фрагмента, сохранённый при добавлении`,
    original: h("div", { class: "offline-note", text: e.issues.map((i) => i.message).join("; ") || "Фрагмент недоступен" }),
    context: [h("p", { class: "target", text: e.excerpt })],
  });
}

async function openViewer(chunkId) {
  let c;
  try { c = await api(`/api/chunks/${chunkId}`); } catch (err) { toast(err.message); return; }
  const fileUrl = `/api/sources/${c.source_id}/file`;
  const original = h("div", { class: "viewer-original-inner" });
  const missing = c.status === "missing";

  if (missing) {
    original.append(h("div", { class: "offline-note", text: "Оригинал недоступен: файл перемещён или диск отключён. Ниже — сохранённый в индексе текст." }));
  } else if (c.kind === "document" && c.rel_path.toLowerCase().endsWith(".pdf")) {
    original.append(h("iframe", { src: `${fileUrl}#page=${c.page || 1}`, title: `PDF: ${c.rel_path}` }));
  } else if (c.kind === "image") {
    original.append(h("img", { src: fileUrl, alt: c.rel_path }));
  } else if (c.kind === "video" || c.kind === "audio") {
    const media = h(c.kind === "video" ? "video" : "audio", { src: fileUrl, controls: true, preload: "metadata" });
    const start = c.t_start || 0;
    const end = c.t_end;
    media.addEventListener("loadedmetadata", () => { media.currentTime = start; }, { once: true });
    let stopAt = null;
    media.addEventListener("timeupdate", () => { if (stopAt != null && media.currentTime >= stopAt) { media.pause(); stopAt = null; } });
    original.append(media, h("div", { class: "play-row" },
      h("button", { class: "small primary", type: "button", text: `▶ Фрагмент ${fmtTime(start)}–${fmtTime(end)}`, onclick: () => { media.currentTime = start; stopAt = end; media.play(); } }),
      h("button", { class: "small", type: "button", text: "−10 с", onclick: () => { media.currentTime = Math.max(0, media.currentTime - 10); } }),
      h("button", { class: "small", type: "button", text: "+10 с", onclick: () => { media.currentTime += 10; } }),
      h("span", { class: "muted", text: c.modality === "video" ? "Найден по кадру" : c.modality === "audio" ? "Найден по звучанию" : "Найден по речи" })));
    if (c.has_preview && c.modality === "video") original.append(h("img", { src: `/api/chunks/${c.id}/preview`, alt: "Найденный кадр" }));
  } else if (c.kind === "transcript") {
    original.append(h("div", { class: "offline-note", text: "Транскрипт без медиафайла. Положите аудио или видео с тем же именем рядом, чтобы слушать оригинал." }));
  } else {
    original.append(h("p", { class: "muted" }, "Документ открыт в виде извлечённого текста. ", h("a", { href: fileUrl, target: "_blank", rel: "noopener", text: "Открыть исходный файл" })));
  }
  if (c.modality === "image" && c.has_preview && c.kind !== "image") {
    original.append(h("img", { src: `/api/chunks/${c.id}/preview`, alt: `Страница ${c.page}` }));
  }

  const context = c.context.length
    ? c.context.map((p) => h("p", { class: p.id === c.id ? "target" : "" },
        locatorText(p) ? h("span", { class: "loc", text: locatorText(p) }) : null, p.text))
    : [h("p", { class: "muted", text: c.text || "Для этого фрагмента нет извлечённого текста." })];

  const thesis = activeThesis();
  const actions = [];
  if (thesis && !c.linked_theses.some((l) => l.thesis_id === thesis.id)) {
    for (const role of Object.keys(ROLE_SHORT)) {
      actions.push(h("button", { class: "small role-btn", type: "button", dataset: { role }, text: `+ ${ROLE_SHORT[role]}`, onclick: async () => { await addEvidence(thesis.id, c.id, role); openViewer(c.id); } }));
    }
  }
  actions.push(h("button", { class: "small ghost", type: "button", text: "Похожие", onclick: () => { viewer.close(); runSimilar(c.id, c.rel_path); } }));

  showViewer({
    title: c.rel_path,
    meta: [KIND_LABELS[c.kind], MODALITY_LABELS[c.modality], locatorText(c), `извлечено: ${c.extractor}`, `версия файла ${c.source_version}`].filter(Boolean).join(" · "),
    original,
    context,
    links: c.linked_theses,
    actions,
  });
}

function showViewer({ title, meta, original, context, links = [], actions = [] }) {
  document.getElementById("viewer-title").textContent = title;
  document.getElementById("viewer-meta").textContent = meta;
  document.getElementById("viewer-original").replaceChildren(original);
  document.getElementById("viewer-context").replaceChildren(...context);
  document.getElementById("viewer-links").replaceChildren(...(links.length ? [
    h("h3", { text: "В тезисах" }),
    ...links.map((l) => h("p", { class: "muted", text: `${ROLE_LABELS[l.role]}: ${l.title}` })),
  ] : []));
  document.getElementById("viewer-actions").replaceChildren(...actions);
  if (!viewer.open) viewer.showModal();
  document.querySelector(".viewer-context .target")?.scrollIntoView({ block: "center" });
}

// ------------------------------------------------------------------ wiring

function refreshAll() {
  refreshSources();
  refreshTheses();
}

document.getElementById("folder-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("folder-input");
  try {
    await api("/api/folders", { method: "POST", body: { path: input.value.trim() } });
    input.value = "";
    toast("Папка добавлена, начинаю индексацию");
    app.wasRunning = true;
    refreshSources();
  } catch (err) { toast(err.message); }
});

document.getElementById("reindex").addEventListener("click", async () => {
  await api("/api/reindex", { method: "POST" });
  app.wasRunning = true;
  toast("Проверяю изменения в папках");
});

document.getElementById("search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  runSearch(document.getElementById("query").value, document.getElementById("task").value);
});
document.getElementById("query").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); e.target.form.requestSubmit(); }
});
document.querySelectorAll('input[name="modality"], #mode').forEach((el) => el.addEventListener("change", () => {
  if (app.lastSearch?.kind === "text") runSearch(app.lastSearch.query, app.lastSearch.task);
  else if (app.lastSearch?.kind === "image") runImageSearch(app.lastSearch.file);
}));

document.getElementById("image-btn").addEventListener("click", () => document.getElementById("image-input").click());
document.getElementById("image-input").addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) runImageSearch(file);
  e.target.value = "";
});

const pane = document.getElementById("search-pane");
const overlay = document.getElementById("drop-overlay");
pane.addEventListener("dragover", (e) => {
  if ([...(e.dataTransfer?.items || [])].some((i) => i.type.startsWith("image/"))) { e.preventDefault(); overlay.hidden = false; }
});
pane.addEventListener("dragleave", (e) => { if (!pane.contains(e.relatedTarget)) overlay.hidden = true; });
pane.addEventListener("drop", (e) => {
  overlay.hidden = true;
  const file = [...(e.dataTransfer?.files || [])].find((f) => f.type.startsWith("image/"));
  if (file) { e.preventDefault(); runImageSearch(file); }
});
document.addEventListener("paste", (e) => {
  if (e.target.closest("textarea, input, [contenteditable]")) return;
  const file = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
  if (file) runImageSearch(file);
});

document.getElementById("thesis-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("thesis-input");
  try {
    const thesis = await api("/api/theses", { method: "POST", body: { title: input.value } });
    input.value = "";
    app.activeThesisId = thesis.id;
    store("activeThesis", thesis.id);
    await refreshTheses();
    rerenderResults();
  } catch (err) { toast(err.message); }
});
document.getElementById("thesis-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); e.target.form.requestSubmit(); }
});

refreshState();
refreshAll();
