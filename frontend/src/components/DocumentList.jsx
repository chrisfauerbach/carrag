import { useState, useEffect } from 'react';
import { listDocuments, deleteDocument, updateDocumentTags } from '../api';
import StatusMessage from './StatusMessage';
import DocumentDetail from './DocumentDetail';
import './DocumentList.css';

export default function DocumentList() {
  const [docs, setDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('');
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [editingTagsId, setEditingTagsId] = useState(null);
  const [editingTagsValue, setEditingTagsValue] = useState('');

  async function fetchDocs() {
    setLoading(true);
    setError('');
    try {
      const data = await listDocuments();
      setDocs(data.documents);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  }

  useEffect(() => {
    fetchDocs();
  }, []);

  async function handleDelete(id, name) {
    if (!window.confirm(`Delete "${name}" and all its chunks?`)) return;
    try {
      await deleteDocument(id);
      setDocs((prev) => prev.filter((d) => d.document_id !== id));
    } catch (err) {
      setError(err.message);
    }
  }

  function handleStartEditTags(e, doc) {
    e.stopPropagation();
    setEditingTagsId(doc.document_id);
    setEditingTagsValue((doc.tags || []).join(', '));
  }

  async function handleSaveTags(e, docId) {
    e.stopPropagation();
    const newTags = editingTagsValue.split(',').map((t) => t.trim()).filter(Boolean);
    try {
      await updateDocumentTags(docId, newTags);
      setDocs((prev) =>
        prev.map((d) => (d.document_id === docId ? { ...d, tags: newTags } : d))
      );
    } catch (err) {
      setError(err.message);
    }
    setEditingTagsId(null);
  }

  function handleCancelEditTags(e) {
    e.stopPropagation();
    setEditingTagsId(null);
  }

  function formatDate(dateStr) {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleDateString();
  }

  if (selectedDoc) {
    return (
      <DocumentDetail
        doc={selectedDoc}
        onBack={() => setSelectedDoc(null)}
      />
    );
  }

  const needle = filter.toLowerCase();
  const filteredDocs = needle
    ? docs.filter(
        (d) =>
          d.filename.toLowerCase().includes(needle) ||
          d.source_type.toLowerCase().includes(needle) ||
          (d.tags || []).some((t) => t.toLowerCase().includes(needle))
      )
    : docs;

  return (
    <div className="document-list">
      <div className="doc-list-header">
        <h2>Documents</h2>
        <button className="refresh-btn" onClick={fetchDocs} disabled={loading}>
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {docs.length > 0 && (
        <div className="doc-filter">
          <input
            className="filter-input"
            type="text"
            placeholder="Filter by name or type..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          {filter && (
            <button className="clear-filter-btn" onClick={() => setFilter('')}>
              Clear
            </button>
          )}
          <span className="filter-count">
            {filteredDocs.length} of {docs.length}
          </span>
        </div>
      )}

      {error && (
        <StatusMessage
          type="error"
          message={error}
          onDismiss={() => setError('')}
        />
      )}

      {!loading && docs.length === 0 && !error && (
        <div className="empty-state">
          No documents ingested yet. Go to the Upload tab to add some.
        </div>
      )}

      {docs.length > 0 && filteredDocs.length === 0 && (
        <div className="empty-state">
          No documents match "{filter}".
        </div>
      )}

      {filteredDocs.length > 0 && (
        <table className="doc-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Type</th>
              <th>Tags</th>
              <th>Chunks</th>
              <th>Created</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filteredDocs.map((doc) => (
              <tr
                key={doc.document_id}
                className="doc-row-clickable"
                onClick={() => setSelectedDoc(doc)}
              >
                <td>{doc.filename}</td>
                <td>{doc.source_type}</td>
                <td onClick={(e) => e.stopPropagation()}>
                  {editingTagsId === doc.document_id ? (
                    <div className="tag-edit">
                      <input
                        type="text"
                        value={editingTagsValue}
                        onChange={(e) => setEditingTagsValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSaveTags(e, doc.document_id);
                          if (e.key === 'Escape') handleCancelEditTags(e);
                        }}
                        autoFocus
                      />
                      <button onClick={(e) => handleSaveTags(e, doc.document_id)}>Save</button>
                      <button onClick={handleCancelEditTags}>Cancel</button>
                    </div>
                  ) : (
                    <div className="tag-display" onClick={(e) => handleStartEditTags(e, doc)}>
                      {(doc.tags || []).length > 0
                        ? doc.tags.map((t) => (
                            <span key={t} className="tag-badge">{t}</span>
                          ))
                        : <span className="no-tags">click to add</span>}
                    </div>
                  )}
                </td>
                <td>{doc.chunk_count}</td>
                <td>{formatDate(doc.created_at)}</td>
                <td>
                  <button
                    className="delete-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(doc.document_id, doc.filename);
                    }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
