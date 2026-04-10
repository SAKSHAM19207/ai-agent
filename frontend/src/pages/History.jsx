import { useState, useEffect } from 'react';

export default function History() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const loaded = JSON.parse(localStorage.getItem('applied_jobs')) || [];
    setHistory(loaded.reverse()); // newest first
  }, []);

  return (
    <div>
      <h2>Application History</h2>
      {history.length === 0 ? (
        <p style={{ color: '#94a3b8' }}>You haven't applied to any jobs yet.</p>
      ) : (
        <div style={{ marginTop: '2rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {history.map((job, idx) => (
            <div key={idx} className="gen-item" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ margin: '0 0 0.5rem 0', color: '#e2e8f0' }}>{job.title}</h3>
                <p style={{ margin: 0, color: '#94a3b8' }}>{job.company}</p>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span className="match-badge">Applied</span>
                <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.8rem', color: '#64748b' }}>
                  {new Date(job.applied_date).toLocaleDateString()}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
