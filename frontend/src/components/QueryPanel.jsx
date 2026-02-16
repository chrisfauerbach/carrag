import { useState, useRef, useEffect } from 'react';
import { queryDocuments } from '../api';
import SourceCard from './SourceCard';
import StatusMessage from './StatusMessage';
import './QueryPanel.css';

export default function QueryPanel() {
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(5);
  const [showTopK, setShowTopK] = useState(false);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const timerRef = useRef(null);

  useEffect(() => {
    return () => clearInterval(timerRef.current);
  }, []);

  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    setResult(null);
    setError('');
    setElapsed(0);

    const start = Date.now();
    timerRef.current = setInterval(() => {
      setElapsed(((Date.now() - start) / 1000).toFixed(1));
    }, 100);

    try {
      const data = await queryDocuments(question.trim(), topK);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      clearInterval(timerRef.current);
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleAsk();
    }
  }

  return (
    <div className="query-panel">
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question about your documents..."
        disabled={loading}
      />

      <div className="query-controls">
        <button
          className="ask-btn"
          onClick={handleAsk}
          disabled={loading || !question.trim()}
        >
          {loading ? 'Thinking...' : 'Ask'}
        </button>

        <button
          className="topk-toggle"
          onClick={() => setShowTopK(!showTopK)}
        >
          {showTopK ? 'Hide options' : 'Options'}
        </button>

        {showTopK && (
          <div className="topk-control">
            <label>Sources:</label>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </div>
        )}
      </div>

      {loading && (
        <div className="thinking">Thinking... {elapsed}s</div>
      )}

      {error && (
        <StatusMessage
          type="error"
          message={error}
          onDismiss={() => setError('')}
        />
      )}

      {result && (
        <>
          <div className="answer-section">
            <div className="answer-box">{result.answer}</div>
            <div className="answer-meta">
              Model: {result.model} &middot;{' '}
              {(result.duration_ms / 1000).toFixed(1)}s
            </div>
          </div>

          {result.sources?.length > 0 && (
            <div className="sources-section">
              <h3>Sources ({result.sources.length})</h3>
              {result.sources.map((source, i) => (
                <SourceCard key={i} index={i} source={source} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
