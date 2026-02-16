import { useState } from 'react';
import QueryPanel from './components/QueryPanel';
import UploadPanel from './components/UploadPanel';
import DocumentList from './components/DocumentList';
import SimilarityMap from './components/SimilarityMap';

const TABS = ['Query', 'Upload', 'Documents', 'Similarity'];

export default function App() {
  const [activeTab, setActiveTab] = useState('Query');

  return (
    <div className="app">
      <header className="app-header">
        <h1>Carrag</h1>
        <p>Local RAG with Elasticsearch + Ollama</p>
      </header>

      <nav className="tabs">
        {TABS.map((tab) => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </nav>

      {activeTab === 'Query' && <QueryPanel />}
      {activeTab === 'Upload' && <UploadPanel />}
      {activeTab === 'Documents' && <DocumentList />}
      {activeTab === 'Similarity' && <SimilarityMap />}
    </div>
  );
}
