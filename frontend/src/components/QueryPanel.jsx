import { useState, useRef, useEffect } from 'react';
import { queryDocuments, getModels } from '../api';
import ChatMessage from './ChatMessage';
import StatusMessage from './StatusMessage';
import './QueryPanel.css';

export default function QueryPanel() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [topK, setTopK] = useState(5);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [showTopK, setShowTopK] = useState(false);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState('');
  const timerRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    getModels()
      .then((data) => {
        setModels(data.models);
        setSelectedModel(data.default);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    return () => clearInterval(timerRef.current);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  function buildHistory() {
    return messages.map((m) => ({ role: m.role, content: m.content }));
  }

  async function handleSend() {
    const question = input.trim();
    if (!question) return;

    const userMsg = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setError('');
    setElapsed(0);

    const start = Date.now();
    timerRef.current = setInterval(() => {
      setElapsed(((Date.now() - start) / 1000).toFixed(1));
    }, 100);

    try {
      const history = buildHistory();
      const data = await queryDocuments(question, topK, history, selectedModel || null);
      const assistantMsg = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        model: data.model,
        duration_ms: data.duration_ms,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(err.message);
    } finally {
      clearInterval(timerRef.current);
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSend();
    }
  }

  function handleNewChat() {
    setMessages([]);
    setError('');
    setInput('');
  }

  const hasMessages = messages.length > 0;

  return (
    <div className="query-panel">
      <div className="chat-thread">
        {!hasMessages && !loading && (
          <div className="chat-empty">
            Ask a question about your documents to start a conversation.
          </div>
        )}

        {messages.map((msg, i) => (
          <ChatMessage key={i} message={msg} />
        ))}

        {loading && (
          <div className="thinking">Thinking... {elapsed}s</div>
        )}

        <div ref={chatEndRef} />
      </div>

      {error && (
        <StatusMessage
          type="error"
          message={error}
          onDismiss={() => setError('')}
        />
      )}

      <div className="chat-input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={hasMessages ? 'Ask a follow-up...' : 'Ask a question about your documents...'}
          disabled={loading}
          rows={2}
        />
        <div className="chat-controls">
          <button
            className="ask-btn"
            onClick={handleSend}
            disabled={loading || !input.trim()}
          >
            {loading ? 'Thinking...' : 'Send'}
          </button>

          {hasMessages && (
            <button className="new-chat-btn" onClick={handleNewChat} disabled={loading}>
              New Chat
            </button>
          )}

          <button
            className="topk-toggle"
            onClick={() => setShowTopK(!showTopK)}
          >
            {showTopK ? 'Hide options' : 'Options'}
          </button>

          {showTopK && (
            <>
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
              {models.length > 0 && (
                <div className="model-control">
                  <label>Model:</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    {models.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
