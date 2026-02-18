import { useState, useRef, useEffect, useCallback } from 'react';
import { uploadFile, ingestUrl, listJobs, cancelJob } from '../api';
import StatusMessage from './StatusMessage';
import './UploadPanel.css';

const ACTIVE_STATUSES = ['queued', 'parsing', 'tagging', 'embedding', 'indexing'];

function JobEntry({ job, onCancel }) {
  const isActive = ACTIVE_STATUSES.includes(job.status);
  const isSuccess = job.status === 'completed';
  const isError = job.status === 'failed';
  const isCancelled = job.status === 'cancelled';

  const statusClass = isSuccess ? 'success' : isError ? 'error' : isCancelled ? 'cancelled' : 'active';

  const stageLabels = {
    queued: 'Queued',
    parsing: 'Parsing content...',
    tagging: 'Generating tags...',
    embedding: 'Embedding chunks...',
    indexing: 'Indexing...',
  };

  const progressPercent = job.total_chunks > 0
    ? Math.round((job.embedded_chunks / job.total_chunks) * 100)
    : 0;

  // Truncate long filenames/URLs
  const displayName = job.filename.length > 60
    ? job.filename.slice(0, 57) + '...'
    : job.filename;

  return (
    <div className={`job-entry ${statusClass}`}>
      <div className="job-header">
        <span className="job-filename" title={job.filename}>{displayName}</span>
        <span className={`job-badge ${statusClass}`}>{job.status}</span>
      </div>

      {isActive && (
        <div className="job-progress">
          <span className="job-stage">{stageLabels[job.status] || job.status}</span>
          {job.status === 'embedding' && job.total_chunks > 0 && (
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${progressPercent}%` }}
              />
              <span className="progress-label">{job.embedded_chunks}/{job.total_chunks}</span>
            </div>
          )}
          <button
            className="cancel-btn"
            onClick={() => onCancel(job.job_id)}
            title="Cancel job"
          >
            Cancel
          </button>
        </div>
      )}

      {isSuccess && (
        <span className="job-result">
          {job.chunk_count} chunks
          {job.tags && job.tags.length > 0 && ` · ${job.tags.join(', ')}`}
        </span>
      )}

      {isError && <span className="job-error">{job.error}</span>}
    </div>
  );
}

export default function UploadPanel() {
  const [url, setUrl] = useState('');
  const [tags, setTags] = useState('');
  const [dragover, setDragover] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [jobs, setJobs] = useState([]);
  const pollRef = useRef(null);

  const hasActiveJobs = jobs.some((j) => ACTIVE_STATUSES.includes(j.status));

  const fetchJobs = useCallback(async () => {
    try {
      const data = await listJobs();
      setJobs(data.jobs);
    } catch {
      // ignore polling errors
    }
  }, []);

  // Load historical jobs on mount
  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  // Poll while active jobs exist
  useEffect(() => {
    if (hasActiveJobs) {
      pollRef.current = setInterval(fetchJobs, 2000);
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [hasActiveJobs, fetchJobs]);

  function parseTags() {
    return tags.split(',').map((t) => t.trim()).filter(Boolean);
  }

  async function handleFiles(files) {
    setUploading(true);
    setError('');
    const parsed = parseTags();
    for (const file of files) {
      try {
        await uploadFile(file, parsed);
      } catch (err) {
        setError(err.message);
      }
    }
    setUploading(false);
    await fetchJobs();
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

  const fileInputRef = useRef(null);

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
      await ingestUrl(url.trim(), parseTags());
      setUrl('');
    } catch (err) {
      setError(err.message);
    }
    setUploading(false);
    await fetchJobs();
  }

  function handleUrlKeyDown(e) {
    if (e.key === 'Enter') handleIngestUrl();
  }

  async function handleCancel(jobId) {
    try {
      await cancelJob(jobId);
      await fetchJobs();
    } catch {
      // ignore
    }
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

      <div className="tags-section">
        <input
          type="text"
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          placeholder="Tags (comma-separated, e.g. research, ml)"
          disabled={uploading}
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

      <div className="jobs-log">
        <h3>Ingestion Jobs</h3>
        {jobs.length === 0 ? (
          <p className="jobs-empty">No ingestion jobs yet. Upload a file or ingest a URL to get started.</p>
        ) : (
          jobs.map((job) => (
            <JobEntry key={job.job_id} job={job} onCancel={handleCancel} />
          ))
        )}
      </div>
    </div>
  );
}
