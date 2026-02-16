import { useState, useEffect, useRef } from 'react';
import { listChats, renameChat, deleteChat } from '../api';
import './ChatSidebar.css';

function timeAgo(dateStr) {
  const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function ChatSidebar({ activeChatId, onSelectChat, onNewChat, refreshKey }) {
  const [chats, setChats] = useState([]);
  const [renamingId, setRenamingId] = useState(null);
  const [renameValue, setRenameValue] = useState('');
  const renameRef = useRef(null);

  useEffect(() => {
    listChats()
      .then((data) => setChats(data.chats))
      .catch(() => {});
  }, [refreshKey]);

  useEffect(() => {
    if (renamingId && renameRef.current) {
      renameRef.current.focus();
      renameRef.current.select();
    }
  }, [renamingId]);

  function handleRenameStart(chat) {
    setRenamingId(chat.chat_id);
    setRenameValue(chat.title);
  }

  async function handleRenameSubmit(chatId) {
    const title = renameValue.trim();
    if (title) {
      try {
        await renameChat(chatId, title);
        setChats((prev) =>
          prev.map((c) => (c.chat_id === chatId ? { ...c, title } : c))
        );
      } catch {}
    }
    setRenamingId(null);
  }

  async function handleDelete(chatId) {
    try {
      await deleteChat(chatId);
      setChats((prev) => prev.filter((c) => c.chat_id !== chatId));
      if (activeChatId === chatId) {
        onNewChat();
      }
    } catch {}
  }

  return (
    <div className="chat-sidebar">
      <div className="chat-sidebar-header">
        <button onClick={onNewChat}>New Chat</button>
      </div>
      <div className="chat-sidebar-list">
        {chats.length === 0 && (
          <div className="chat-sidebar-empty">No chats yet</div>
        )}
        {chats.map((chat) => (
          <div
            key={chat.chat_id}
            className={`chat-sidebar-item ${activeChatId === chat.chat_id ? 'active' : ''}`}
            onClick={() => onSelectChat(chat.chat_id)}
          >
            {renamingId === chat.chat_id ? (
              <input
                ref={renameRef}
                className="chat-sidebar-rename-input"
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                onBlur={() => handleRenameSubmit(chat.chat_id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleRenameSubmit(chat.chat_id);
                  if (e.key === 'Escape') setRenamingId(null);
                }}
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <>
                <div className="chat-sidebar-item-title">{chat.title}</div>
                <div className="chat-sidebar-item-meta">
                  {chat.message_count} msgs &middot; {timeAgo(chat.updated_at)}
                </div>
                <div className="chat-sidebar-item-actions">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRenameStart(chat);
                    }}
                  >
                    Rename
                  </button>
                  <button
                    className="delete-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(chat.chat_id);
                    }}
                  >
                    Delete
                  </button>
                </div>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
