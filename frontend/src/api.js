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

export async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  return request('/ingest/file', { method: 'POST', body: form });
}

export async function ingestUrl(url) {
  return request('/ingest/url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
}

export async function getModels() {
  return request('/query/models');
}

export async function queryDocuments(question, topK = 5, history = [], model = null) {
  const payload = { question, top_k: topK, history };
  if (model) payload.model = model;
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

export async function getDocumentChunks(id) {
  return request(`/documents/${id}/chunks`);
}

export async function getDocumentSimilarity(threshold = 0) {
  return request(`/documents/similarity?threshold=${threshold}`);
}
