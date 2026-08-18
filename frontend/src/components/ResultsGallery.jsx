import React from 'react';
import './ResultsGallery.css';

const ResultsGallery = ({ results, activeTab }) => {
  if (!results || results.length === 0) {
    return (
      <div className="empty-state panel animate-fade-in">
        <p className="text-secondary">No results yet. Enter a query above to start searching.</p>
      </div>
    );
  }

  // Generate random image from unsplash based on video/frame id for aesthetics
  const getImageUrl = (id) => `https://source.unsplash.com/random/400x300?sig=${id}`;

  return (
    <div className="results-gallery">
      {results.map((result, index) => (
        <div key={`${result.video_id}-${index}`} className="result-card panel animate-fade-in" style={{animationDelay: `${index * 0.05}s`}}>
          <div className="card-image-wrapper">
            <img 
              src={getImageUrl(`${result.video_id}-${result.frame_id || result.frame_ids?.[0]}`)} 
              alt="Result" 
              className="card-image"
              loading="lazy"
            />
            {result.score !== undefined && (
              <div className="score-badge">
                {(result.score * 100).toFixed(1)}%
              </div>
            )}
            <div className="rank-badge">#{index + 1}</div>
          </div>
          
          <div className="card-content">
            <div className="meta-info">
              <span className="video-id" title="Video ID">{result.video_id}</span>
              {activeTab !== 'trake' && (
                <span className="frame-id" title="Frame ID">Frame: {result.frame_id}</span>
              )}
            </div>

            {activeTab === 'qa' && (
              <div className="qa-answer">
                <span className="label">Answer:</span>
                <p>{result.answer}</p>
              </div>
            )}

            {activeTab === 'trake' && result.frame_ids && (
              <div className="trake-frames">
                <span className="label">Sequence:</span>
                <div className="frame-pills">
                  {result.frame_ids.map((fid, i) => (
                    <span key={i} className="frame-pill">{fid}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default ResultsGallery;
