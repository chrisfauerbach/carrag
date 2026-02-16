import { useState, useRef } from 'react';
import { uploadFile, ingestUrl } from '../api';
import StatusMessage from './StatusMessage';
import './UploadPanel.css';

export default function UploadPanel() {
  const [url, setUrl] = useState('');
  const [dragover, setDragover] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  function addResult(entry) {
    setResults((prev) => [entry, ...prev]);
  }

  async function handleFiles(files) {
    setUploading(true);
    setError('');
    for (const file of files) {
      try {
        const data = await uploadFile(file);
        addResult({
          name: data.filename,
          chunks: data.chunk_count,
          ok: true,
        });
      } catch (err) {
        addResult({ name: file.name, error: err.message, ok: false });
      }
    }
    setUploading(false);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragover(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length) handleFiles(files);
  }

  function handleDragOver(e) {
    e.preventDefault();
    setDragover(true);
  }

  function handleDragLeave() {
    setDragover(false);
  }

  function handleBrowse() {
    fileInputRef.current?.click();
  }

  function handleFileInput(e) {
    const files = Array.from(e.target.files);
    if (files.length) handleFiles(files);
    e.target.value = '';
  }

  async function handleIngestUrl() {
    if (!url.trim()) return;
    setUploading(true);
    setError('');
    try {
      const data = await ingestUrl(url.trim());
      addResult({
        name: data.filename,
        chunks: data.chunk_count,
        ok: true,
      });
      setUrl('');
    } catch (err) {
      addResult({ name: url.trim(), error: err.message, ok: false });
    }
    setUploading(false);
  }

  function handleUrlKeyDown(e) {
    if (e.key === 'Enter') handleIngestUrl();
  }

  return (
    <div className="upload-panel">
      {error && (
        <StatusMessage
          type="error"
          message={error}
          onDismiss={() => setError('')}
        />
      )}

      <div
        className={`drop-zone ${dragover ? 'dragover' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleBrowse}
      >
        <p>{uploading ? 'Uploading...' : 'Drop files here'}</p>
        <button
          className="browse-btn"
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            handleBrowse();
          }}
          disabled={uploading}
        >
          Browse files
        </button>
        <div className="hint">PDF, TXT, MD</div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.md,.text,.markdown"
          multiple
          hidden
          onChange={handleFileInput}
        />
      </div>

      <div className="url-section">
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={handleUrlKeyDown}
          placeholder="https://example.com/page"
          disabled={uploading}
        />
        <button
          onClick={handleIngestUrl}
          disabled={uploading || !url.trim()}
        >
          Ingest URL
        </button>
      </div>

      {results.length > 0 && (
        <div className="results-log">
          <h3>Results</h3>
          {results.map((r, i) => (
            <div
              key={i}
              className={`result-entry ${r.ok ? 'success' : 'error'}`}
            >
              <span>{r.name}</span>
              <span className="chunk-count">
                {r.ok ? `${r.chunks} chunks` : r.error}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
