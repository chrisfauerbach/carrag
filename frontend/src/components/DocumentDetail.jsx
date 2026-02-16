import { useState, useEffect } from 'react';
import { getDocumentChunks } from '../api';
import StatusMessage from './StatusMessage';
import './DocumentDetail.css';

export default function DocumentDetail({ doc, onBack }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    async function fetchChunks() {
      setLoading(true);
      setError('');
      try {
        const result = await getDocumentChunks(doc.document_id);
        setData(result);
      } catch (err) {
        setError(err.message);
      }
      setLoading(false);
    }
    fetchChunks();
  }, [doc.document_id]);

  return (
    <div className="document-detail">
      <div className="detail-header">
        <button className="back-btn" onClick={onBack}>Back</button>
        <h2>{doc.filename}</h2>
        <div className="detail-meta">
          <span className="type-badge">{data?.source_type || doc.source_type}</span>
          {data && <span className="chunk-count">{data.chunk_count} chunks</span>}
        </div>
      </div>

      {error && (
        <StatusMessage type="error" message={error} onDismiss={() => setError('')} />
      )}

      {loading && <StatusMessage type="info" message="Loading chunks..." />}

      {data && (
        <div className="chunk-list">
          {data.chunks.map((chunk) => (
            <div className="chunk-card" key={chunk.chunk_index}>
              <div className="chunk-card-header">
                <span>Chunk {chunk.chunk_index}</span>
                <span>chars {chunk.char_start}–{chunk.char_end}</span>
              </div>
              <pre className="chunk-content">{chunk.content}</pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
