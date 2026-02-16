import { useState, useEffect } from 'react';
import { listDocuments, deleteDocument } from '../api';
import StatusMessage from './StatusMessage';
import './DocumentList.css';

export default function DocumentList() {
  const [docs, setDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('');

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

  function formatDate(dateStr) {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleDateString();
  }

  const needle = filter.toLowerCase();
  const filteredDocs = needle
    ? docs.filter(
        (d) =>
          d.filename.toLowerCase().includes(needle) ||
          d.source_type.toLowerCase().includes(needle)
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
              <th>Chunks</th>
              <th>Created</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filteredDocs.map((doc) => (
              <tr key={doc.document_id}>
                <td>{doc.filename}</td>
                <td>{doc.source_type}</td>
                <td>{doc.chunk_count}</td>
                <td>{formatDate(doc.created_at)}</td>
                <td>
                  <button
                    className="delete-btn"
                    onClick={() => handleDelete(doc.document_id, doc.filename)}
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
