import { useState } from 'react';
import './SourceCard.css';

const TRUNCATE_LENGTH = 200;

export default function SourceCard({ index, source }) {
  const [expanded, setExpanded] = useState(false);
  const text = source.content || '';
  const needsTruncation = text.length > TRUNCATE_LENGTH;
  const displayText =
    needsTruncation && !expanded
      ? text.slice(0, TRUNCATE_LENGTH) + '...'
      : text;

  const filename = source.metadata?.filename || 'Unknown';
  const score = (source.score * 100).toFixed(1);

  return (
    <div className="source-card">
      <div className="source-card-header">
        <span>
          <span className="source-number">[{index + 1}]</span> {filename}
        </span>
        <span className="score">{score}% match</span>
      </div>
      <div className="source-card-text">{displayText}</div>
      {needsTruncation && (
        <button className="toggle" onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  );
}
