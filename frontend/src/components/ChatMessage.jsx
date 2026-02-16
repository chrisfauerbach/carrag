import MarkdownContent from './MarkdownContent';
import './ChatMessage.css';

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`chat-message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-label">{isUser ? 'You' : 'Assistant'}</div>
      <div className="message-bubble">
        {isUser ? (
          <p className="user-text">{message.content}</p>
        ) : (
          <>
            <MarkdownContent content={message.content} />
            {message.streaming && <span className="streaming-cursor" />}
          </>
        )}
      </div>

      {!isUser && message.sources?.length > 0 && (
        <details className="message-sources">
          <summary>Sources ({message.sources.length})</summary>
          <ul>
            {message.sources.map((s, i) => (
              <li key={i}>
                <strong>{s.metadata?.filename || 'unknown'}</strong>
                <span className="source-score">({(s.score * 100).toFixed(0)}%)</span>
                <p className="source-snippet">{s.content.slice(0, 200)}...</p>
              </li>
            ))}
          </ul>
        </details>
      )}

      {!isUser && message.model && (
        <div className="message-meta">
          {message.model} &middot; {(message.duration_ms / 1000).toFixed(1)}s
        </div>
      )}
    </div>
  );
}
