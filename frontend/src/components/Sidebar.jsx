import React from 'react';
import { Search, MessageSquare, FastForward } from 'lucide-react';
import './Sidebar.css';

const Sidebar = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'kis', name: 'Textual KIS', icon: <Search size={20} /> },
    { id: 'qa', name: 'Visual Q&A', icon: <MessageSquare size={20} /> },
    { id: 'trake', name: 'TRAKE (Temporal)', icon: <FastForward size={20} /> },
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>Camera AI</h2>
      </div>
      
      <nav className="sidebar-nav">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.icon}
            <span>{tab.name}</span>
          </button>
        ))}
      </nav>
    </div>
  );
};

export default Sidebar;
