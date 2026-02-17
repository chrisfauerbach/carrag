import { useState, useEffect } from 'react';
import { listPrompts, updatePrompt, resetPrompt } from '../api';
import StatusMessage from './StatusMessage';
import './AdminPanel.css';

export default function AdminPanel() {
  const [prompts, setPrompts] = useState([]);
  const [drafts, setDrafts] = useState({});
  const [saving, setSaving] = useState({});
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPrompts();
  }, []);

  async function fetchPrompts() {
    setLoading(true);
    try {
      const data = await listPrompts();
      setPrompts(data.prompts);
      const initial = {};
      for (const p of data.prompts) {
        initial[p.key] = p.content;
      }
      setDrafts(initial);
    } catch (err) {
      setStatus({ type: 'error', message: `Failed to load prompts: ${err.message}` });
    }
    setLoading(false);
  }

  function handleChange(key, value) {
    setDrafts((prev) => ({ ...prev, [key]: value }));
  }

  function isDirty(key) {
    const original = prompts.find((p) => p.key === key);
    return original && drafts[key] !== original.content;
  }

  async function handleSave(key) {
    setSaving((prev) => ({ ...prev, [key]: true }));
    try {
      const result = await updatePrompt(key, drafts[key]);
      setPrompts((prev) =>
        prev.map((p) => (p.key === key ? { ...p, content: result.content, updated_at: result.updated_at } : p))
      );
      setStatus({ type: 'success', message: `Saved "${prompts.find((p) => p.key === key)?.name}"` });
    } catch (err) {
      setStatus({ type: 'error', message: `Save failed: ${err.message}` });
    }
    setSaving((prev) => ({ ...prev, [key]: false }));
  }

  async function handleReset(key) {
    const name = prompts.find((p) => p.key === key)?.name || key;
    if (!window.confirm(`Reset "${name}" to its default content?`)) return;

    setSaving((prev) => ({ ...prev, [key]: true }));
    try {
      const result = await resetPrompt(key);
      setPrompts((prev) =>
        prev.map((p) => (p.key === key ? { ...p, content: result.content, updated_at: result.updated_at } : p))
      );
      setDrafts((prev) => ({ ...prev, [key]: result.content }));
      setStatus({ type: 'success', message: `Reset "${name}" to default` });
    } catch (err) {
      setStatus({ type: 'error', message: `Reset failed: ${err.message}` });
    }
    setSaving((prev) => ({ ...prev, [key]: false }));
  }

  if (loading) {
    return <div className="admin-panel"><p className="admin-loading">Loading prompts...</p></div>;
  }

  return (
    <div className="admin-panel">
      {status && (
        <StatusMessage
          type={status.type}
          message={status.message}
          onDismiss={() => setStatus(null)}
        />
      )}

      <p className="admin-description">
        Edit the system prompts used by the RAG pipeline and auto-tagger. Changes take effect immediately.
      </p>

      {prompts.map((prompt) => (
        <div key={prompt.key} className="prompt-card">
          <div className="prompt-header">
            <h3>{prompt.name}</h3>
            <span className="prompt-key">{prompt.key}</span>
          </div>
          <p className="prompt-description">{prompt.description}</p>

          {prompt.variables.length > 0 && (
            <div className="prompt-variables">
              <span className="variables-label">Variables:</span>
              {prompt.variables.map((v) => (
                <span key={v} className="variable-chip">{`{${v}}`}</span>
              ))}
            </div>
          )}

          <textarea
            className="prompt-textarea"
            value={drafts[prompt.key] ?? prompt.content}
            onChange={(e) => handleChange(prompt.key, e.target.value)}
            rows={8}
            disabled={saving[prompt.key]}
          />

          <div className="prompt-actions">
            <button
              className="btn-save"
              onClick={() => handleSave(prompt.key)}
              disabled={saving[prompt.key] || !isDirty(prompt.key)}
            >
              {saving[prompt.key] ? 'Saving...' : 'Save'}
            </button>
            <button
              className="btn-reset"
              onClick={() => handleReset(prompt.key)}
              disabled={saving[prompt.key]}
            >
              Reset to Default
            </button>
          </div>

          {prompt.updated_at && (
            <div className="prompt-meta">
              Last updated: {new Date(prompt.updated_at).toLocaleString()}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
