import { useState, useRef, useEffect, useCallback } from 'react';
import { queryDocuments, queryDocumentsStream, getModels, createChat, getChat } from '../api';
import ChatMessage from './ChatMessage';
import ChatSidebar from './ChatSidebar';
import StatusMessage from './StatusMessage';
import './QueryPanel.css';

export default function QueryPanel() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [topK, setTopK] = useState(10);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [filterTags, setFilterTags] = useState('');
  const [showTopK, setShowTopK] = useState(false);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState('');
  const [activeChatId, setActiveChatId] = useState(null);
  const [sidebarRefreshKey, setSidebarRefreshKey] = useState(0);
  const timerRef = useRef(null);
  const chatEndRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    getModels()
      .then((data) => {
        setModels(data.models);
        setSelectedModel(data.default);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      abortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  function buildHistory() {
    return messages
      .filter((m) => !m.streaming)
      .map((m) => ({ role: m.role, content: m.content }));
  }

  const handleSend = useCallback(async () => {
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

    const history = buildHistory();
    const abortController = new AbortController();
    abortRef.current = abortController;

    // Create a chat session if we don't have one
    let chatId = activeChatId;
    if (!chatId) {
      try {
        const chat = await createChat();
        chatId = chat.chat_id;
        setActiveChatId(chatId);
      } catch {}
    }

    // Add placeholder assistant message for streaming
    const placeholderIdx = messages.length + 1; // +1 for the user message we just added
    setMessages((prev) => [
      ...prev,
      { role: 'assistant', content: '', streaming: true },
    ]);

    const parsedTags = filterTags.split(',').map((t) => t.trim()).filter(Boolean);
    let streamFailed = false;

    try {
      await queryDocumentsStream(question, {
        topK,
        history,
        model: selectedModel || null,
        chatId,
        tags: parsedTags,
        signal: abortController.signal,
        onToken: (token) => {
          setMessages((prev) => {
            const updated = [...prev];
            const msg = updated[placeholderIdx];
            if (msg) {
              updated[placeholderIdx] = { ...msg, content: msg.content + token };
            }
            return updated;
          });
        },
        onSources: (sources) => {
          setMessages((prev) => {
            const updated = [...prev];
            const msg = updated[placeholderIdx];
            if (msg) {
              updated[placeholderIdx] = { ...msg, sources };
            }
            return updated;
          });
        },
        onDone: (data) => {
          setMessages((prev) => {
            const updated = [...prev];
            const msg = updated[placeholderIdx];
            if (msg) {
              updated[placeholderIdx] = {
                ...msg,
                streaming: false,
                model: data.model,
                duration_ms: data.duration_ms,
              };
            }
            return updated;
          });
        },
        onError: (errMsg) => {
          streamFailed = true;
          setError(errMsg);
        },
      });
    } catch {
      streamFailed = true;
    }

    // Refresh sidebar after stream completes — the server persists messages
    // after the done SSE event, so we must wait for the full response to end.
    if (!streamFailed) {
      setSidebarRefreshKey((k) => k + 1);
    }

    if (streamFailed) {
      // Remove the streaming placeholder
      setMessages((prev) => prev.filter((_, i) => i !== placeholderIdx));

      // Fall back to non-streaming
      try {
        const data = await queryDocuments(question, topK, history, selectedModel || null, chatId, parsedTags);
        const assistantMsg = {
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
          model: data.model,
          duration_ms: data.duration_ms,
        };
        setMessages((prev) => [...prev, assistantMsg]);
        setSidebarRefreshKey((k) => k + 1);
      } catch (err) {
        setError(err.message);
      }
    }

    clearInterval(timerRef.current);
    setLoading(false);
  }, [input, messages, topK, selectedModel, activeChatId, filterTags]);

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSend();
    }
  }

  function handleNewChat() {
    abortRef.current?.abort();
    setMessages([]);
    setError('');
    setInput('');
    setActiveChatId(null);
    setSidebarRefreshKey((k) => k + 1);
  }

  async function handleSelectChat(chatId) {
    if (chatId === activeChatId) return;
    abortRef.current?.abort();
    setError('');
    setInput('');
    setActiveChatId(chatId);
    try {
      const data = await getChat(chatId);
      setMessages(
        data.messages.map((m) => ({
          role: m.role,
          content: m.content,
          sources: m.sources || undefined,
          model: m.model || undefined,
          duration_ms: m.duration_ms || undefined,
        }))
      );
    } catch {
      setError('Failed to load chat');
    }
  }

  const hasMessages = messages.length > 0;
  const isStreaming = messages.some((m) => m.streaming);

  return (
    <div className="query-panel-with-sidebar">
      <ChatSidebar
        activeChatId={activeChatId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        refreshKey={sidebarRefreshKey}
      />
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

        {loading && !isStreaming && (
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
              <div className="tags-filter-control">
                <label>Tags:</label>
                <input
                  type="text"
                  value={filterTags}
                  onChange={(e) => setFilterTags(e.target.value)}
                  placeholder="e.g. research, ml"
                />
              </div>
            </>
          )}
        </div>
      </div>
      </div>
    </div>
  );
}
