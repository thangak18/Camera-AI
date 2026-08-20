import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import QueryPanel from './components/QueryPanel';
import ResultsGallery from './components/ResultsGallery';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('kis');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSearch = async (payload) => {
    setIsLoading(true);
    try {
      let endpoint = '';
      let body = {};
      
      // Update this URL whenever Kaggle restarts and gives a new Localtunnel link
      const API_BASE_URL = "https://neat-parts-fall.loca.lt";

      if (activeTab === 'kis') {
        endpoint = `${API_BASE_URL}/search/kis`;
        body = { query: payload, top_k: 20 };
      } else if (activeTab === 'qa') {
        endpoint = `${API_BASE_URL}/search/qa`;
        body = { description: payload.description || '', question: payload.question || '', top_k: 20 };
      } else if (activeTab === 'trake') {
        endpoint = `${API_BASE_URL}/search/trake`;
        body = { event_queries: payload };
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Bypass-Tunnel-Reminder': 'true' // Required by localtunnel to bypass warning screen
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) throw new Error('Search failed');
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error(error);
      alert('Search failed. Please ensure the backend is running on port 8000.');
    } finally {
      setIsLoading(false);
    }
  };

  // Reset results when tab changes
  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
    setResults([]);
  };

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} setActiveTab={handleTabChange} />
      
      <main className="main-content">
        <QueryPanel 
          activeTab={activeTab} 
          onSearch={handleSearch} 
          isLoading={isLoading} 
        />
        
        <ResultsGallery 
          results={results} 
          activeTab={activeTab} 
        />
      </main>
    </div>
  );
}

export default App;
