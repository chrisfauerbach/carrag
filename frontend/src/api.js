const BASE = '';

async function request(url, options = {}) {
  const res = await fetch(`${BASE}${url}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const message = body?.detail || `Request failed: ${res.status}`;
    throw new Error(message);
  }
  return res.json();
}

export async function uploadFile(file, tags = []) {
  const form = new FormData();
  form.append('file', file);
  if (tags.length) form.append('tags', tags.join(','));
  return request('/ingest/file', { method: 'POST', body: form });
}

export async function ingestUrl(url, tags = []) {
  return request('/ingest/url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, tags }),
  });
}

export async function getModels() {
  return request('/query/models');
}

export async function queryDocuments(question, topK = 5, history = [], model = null, chatId = null, tags = []) {
  const payload = { question, top_k: topK, history };
  if (model) payload.model = model;
  if (chatId) payload.chat_id = chatId;
  if (tags.length) payload.tags = tags;
  return request('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function listDocuments() {
  return request('/documents');
}

export async function deleteDocument(id) {
  return request(`/documents/${id}`, { method: 'DELETE' });
}

export async function queryDocumentsStream(
  question,
  { topK = 5, history = [], model = null, chatId = null, tags = [], onToken, onSources, onDone, onError, signal }
) {
  const payload = { question, top_k: topK, history };
  if (model) payload.model = model;
  if (chatId) payload.chat_id = chatId;
  if (tags.length) payload.tags = tags;

  const res = await fetch(`${BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep incomplete line in buffer

    let currentEvent = null;
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith('data: ') && currentEvent) {
        const data = JSON.parse(line.slice(6));
        if (currentEvent === 'token' && onToken) onToken(data.token);
        else if (currentEvent === 'sources' && onSources) onSources(data.sources);
        else if (currentEvent === 'done' && onDone) onDone(data);
        else if (currentEvent === 'error' && onError) onError(data.error);
        currentEvent = null;
      }
    }
  }
}

export async function updateDocumentTags(id, tags) {
  return request(`/documents/${id}/tags`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tags }),
  });
}

export async function getDocumentChunks(id) {
  return request(`/documents/${id}/chunks`);
}

export async function getDocumentSimilarity(threshold = 0) {
  return request(`/documents/similarity?threshold=${threshold}`);
}

// --- Metrics ---

export async function getMetrics(minutes = 60) {
  return request(`/metrics?minutes=${minutes}`);
}

// --- Chat Sessions ---

export async function createChat(title = null) {
  const payload = {};
  if (title) payload.title = title;
  return request('/chats', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function listChats() {
  return request('/chats');
}

export async function getChat(chatId) {
  return request(`/chats/${chatId}`);
}

export async function renameChat(chatId, title) {
  return request(`/chats/${chatId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
}

export async function deleteChat(chatId) {
  return request(`/chats/${chatId}`, { method: 'DELETE' });
}

// --- Prompts ---

export async function listPrompts() {
  return request('/prompts');
}

export async function getPrompt(key) {
  return request(`/prompts/${key}`);
}

export async function updatePrompt(key, content) {
  return request(`/prompts/${key}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  });
}

export async function resetPrompt(key) {
  return request(`/prompts/${key}/reset`, { method: 'POST' });
}
