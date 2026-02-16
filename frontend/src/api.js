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

export async function queryDocuments(question, topK = 5) {
  return request('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  });
}

export async function listDocuments() {
  return request('/documents');
}

export async function deleteDocument(id) {
  return request(`/documents/${id}`, { method: 'DELETE' });
}
