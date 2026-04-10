import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Optimize() {
  const [resumeText, setResumeText] = useState('');
  const [optimized, setOptimized] = useState('');
  const [atsScore, setAtsScore] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const saved = localStorage.getItem("raw_resume");
    if (saved) {
      setResumeText(saved);
    } else {
      navigate('/');
    }
  }, [navigate]);

  const optimizeResume = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/optimize_resume', {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ resume_text: resumeText })
      });
      const data = await res.json();
      setOptimized(data.optimized);
      setAtsScore(data.ats_score);
      localStorage.setItem("optimized_resume", data.optimized);
      localStorage.setItem("ats_score", data.ats_score);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div className="two-col">
      <div className="col">
        <h3>Raw Resume Extracted</h3>
        <textarea 
          value={resumeText}
          onChange={(e) => setResumeText(e.target.value)}
          className="resume-box"
          style={{ height: '300px' }}
        />
        <button onClick={optimizeResume} disabled={loading} className="btn-secondary" style={{ marginTop: '1rem' }}>
          {loading ? 'Optimizing...' : 'Optimize Resume & Calculate ATS'}
        </button>
      </div>

      <div className="col right-col">
        {atsScore !== null && (
          <div className="gen-item" style={{ marginBottom: '2rem', textAlign: 'center' }}>
            <h3 style={{ margin: 0, color: '#e2e8f0' }}>ATS Score</h3>
            <div style={{ fontSize: '3rem', fontWeight: 'bold', color: atsScore > 75 ? '#4ade80' : '#facc15' }}>
              {atsScore}%
            </div>
          </div>
        )}
        
        {optimized ? (
          <div className="gen-item">
            <h4 className="gen-title">Optimized Highlights for ATS</h4>
            <p style={{whiteSpace:'pre-wrap'}}>{optimized}</p>
            <div style={{ marginTop: '2rem' }}>
              <button onClick={() => navigate('/jobs')} className="btn-primary pulse">Find Matching Jobs</button>
            </div>
          </div>
        ) : (
          <div className="empty-state" style={{ height: '100%' }}>
            <p>Click optimize to generate your ATS score and updated resume bullet points.</p>
          </div>
        )}
      </div>
    </div>
  );
}
