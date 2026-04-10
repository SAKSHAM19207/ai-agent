import { useState, useEffect } from 'react';

export default function Jobs() {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeJob, setActiveJob] = useState(null);
  const [coverLetter, setCoverLetter] = useState('');
  const [qa, setQa] = useState([]);
  const [loadingAction, setLoadingAction] = useState(false);

  useEffect(() => {
    fetchJobs();
  }, []);

  const fetchJobs = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/jobs');
      const data = await res.json();
      setJobs(data.jobs);
      localStorage.setItem("latest_jobs", JSON.stringify(data.jobs));
    } catch(err) {
      console.error(err);
    }
    setLoading(false);
  };

  const generateLetter = async (job) => {
    setLoadingAction(true);
    setActiveJob(job);
    setQa([]);
    setCoverLetter('');
    try {
      const res = await fetch('/api/generate_cover_letter', {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify(job)
      });
      const data = await res.json();
      setCoverLetter(data.cover_letter);
    } catch (err) {
      console.error(err);
    }
    setLoadingAction(false);
  };

  const prepInterview = async (job) => {
    setLoadingAction(true);
    setActiveJob(job);
    setCoverLetter('');
    setQa([]);
    try {
      const res = await fetch('/api/prep_interview', {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify(job)
      });
      const data = await res.json();
      setQa(data.qa);
    } catch(err) {
      console.error(err);
    }
    setLoadingAction(false);
  };

  const handleApply = (job) => {
    // Save to history
    const history = JSON.parse(localStorage.getItem('applied_jobs')) || [];
    if (!history.some(h => h.id === job.id)) {
      history.push({ ...job, applied_date: new Date().toISOString() });
      localStorage.setItem('applied_jobs', JSON.stringify(history));
    }
    // Redirect to remote job URL
    if (job.url) {
      window.open(job.url, '_blank');
    } else {
      alert("Application link not found.");
    }
  };

  return (
    <div className="two-col">
      <div className="col">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3>Matches for your profile</h3>
          <button onClick={fetchJobs} disabled={loading} className="btn-secondary">{loading ? 'Fetching...' : 'Refresh Jobs'}</button>
        </div>
        
        <div className="jobs-list" style={{ marginTop: '1rem' }}>
          {jobs.length === 0 && !loading && <span style={{color:'#64748b', fontSize:'0.9rem'}}>No jobs found.</span>}
          {jobs.map(job => (
            <div key={job.id} className="job-block" style={{ border: activeJob?.id === job.id ? '1px solid #818cf8' : '' }}>
              <div className="job-title-row">
                <h4>{job.title} <span style={{ fontWeight: 400, color: '#94a3b8' }}>@{job.company}</span></h4>
                <span className="match-badge">{job.match}% Match</span>
              </div>
              <div className="job-actions">
                <button onClick={() => generateLetter(job)} disabled={loadingAction} className="btn-action">Cover Letter</button>
                <button onClick={() => prepInterview(job)} disabled={loadingAction} className="btn-action">Prep Interview</button>
                <button onClick={() => handleApply(job)} className="btn-apply pulse">APPLY NOW</button>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="col right-col">
        <h3>Agent Workspace</h3>
        <div className="gen-box">
          {loadingAction && <div style={{textAlign:'center', color:'#94a3b8', padding: '2rem'}}>Agent is thinking...</div>}
          
          {coverLetter && !loadingAction && (
            <div className="gen-item">
              <h4 className="gen-title">Cover Letter for {activeJob?.company}</h4>
              <p style={{whiteSpace:'pre-wrap'}}>{coverLetter}</p>
            </div>
          )}
          
          {qa.length > 0 && !loadingAction && (
            <div className="gen-item">
              <h4 className="gen-title">Interview Prep for {activeJob?.company}</h4>
              {qa.map((q, idx) => (
                <div key={idx} style={{marginBottom:'1.5rem'}}>
                  <strong style={{color:'#e2e8f0'}}>Q{idx + 1}) {q.Q}</strong>
                  <p style={{color:'#94a3b8', marginTop:'0.5rem', paddingLeft:'1rem', borderLeft:'2px solid #818cf8'}}>
                    {q.A}
                  </p>
                </div>
              ))}
            </div>
          )}
          {!coverLetter && qa.length === 0 && !loadingAction && (
            <div className="empty-state">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
              <p>Select Cover Letter or Prep Interview to launch the generative agent.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
