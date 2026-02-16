import './StatusMessage.css';

export default function StatusMessage({ type, message, onDismiss }) {
  if (!message) return null;

  return (
    <div className={`status-message ${type}`}>
      <span>{message}</span>
      {onDismiss && (
        <button className="dismiss" onClick={onDismiss}>
          &times;
        </button>
      )}
    </div>
  );
}
