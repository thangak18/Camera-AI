import React, { useState } from 'react';
import { Search, Plus, Trash2 } from 'lucide-react';
import './QueryPanel.css';

const QueryPanel = ({ activeTab, onSearch, isLoading }) => {
  const [query, setQuery] = useState('');
  const [events, setEvents] = useState(['', '']);

  const handleSearch = (e) => {
    e.preventDefault();
    if (activeTab === 'trake') {
      const validEvents = events.filter(e => e.trim());
      if (validEvents.length > 0) onSearch(validEvents);
    } else {
      if (query.trim()) onSearch(query);
    }
  };

  const updateEvent = (index, value) => {
    const newEvents = [...events];
    newEvents[index] = value;
    setEvents(newEvents);
  };

  const addEvent = () => setEvents([...events, '']);
  
  const removeEvent = (index) => {
    if (events.length > 2) {
      setEvents(events.filter((_, i) => i !== index));
    }
  };

  return (
    <div className="query-panel panel animate-fade-in">
      <div className="panel-header">
        <h3>
          {activeTab === 'kis' && "Textual Known-Item Search"}
          {activeTab === 'qa' && "Visual Question Answering"}
          {activeTab === 'trake' && "Temporal Event Search (TRAKE)"}
        </h3>
        <p className="text-secondary">
          {activeTab === 'kis' && "Find specific frames using text descriptions."}
          {activeTab === 'qa' && "Ask questions and get answers based on visual content."}
          {activeTab === 'trake' && "Search for a chronological sequence of events."}
        </p>
      </div>

      <form onSubmit={handleSearch} className="query-form">
        {activeTab === 'trake' ? (
          <div className="events-list">
            {events.map((ev, i) => (
              <div key={i} className="event-input-group">
                <span className="event-number">{i + 1}</span>
                <input
                  type="text"
                  className="input-field"
                  placeholder={`Describe event ${i + 1}...`}
                  value={ev}
                  onChange={(e) => updateEvent(i, e.target.value)}
                />
                {events.length > 2 && (
                  <button type="button" className="btn-icon danger" onClick={() => removeEvent(i)}>
                    <Trash2 size={18} />
                  </button>
                )}
              </div>
            ))}
            <button type="button" className="btn-secondary add-event-btn" onClick={addEvent}>
              <Plus size={16} /> Add Event
            </button>
          </div>
        ) : (
          <div className="single-input-group">
            <textarea
              className="input-field query-textarea"
              placeholder={activeTab === 'kis' ? "e.g., 'A person wearing a red shirt riding a bicycle'" : "e.g., 'What color is the car?'"}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={3}
            />
          </div>
        )}

        <div className="form-actions">
          <button type="submit" className="btn-primary" disabled={isLoading}>
            {isLoading ? (
              <span className="loading-spinner"></span>
            ) : (
              <Search size={18} />
            )}
            {isLoading ? 'Searching...' : 'Execute Search'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default QueryPanel;
