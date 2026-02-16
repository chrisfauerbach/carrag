import ReactMarkdown from 'react-markdown';
import './MarkdownContent.css';

export default function MarkdownContent({ content }) {
  return (
    <div className="markdown-content">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
}
