import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function UploadResume() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch('/api/upload_resume', {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.text) {
        localStorage.setItem("raw_resume", data.text);
        localStorage.setItem("resume_filename", file.name);
        navigate('/optimize');
      } else {
        alert("Failed to parse PDF.");
      }
    } catch (err) {
      console.error(err);
      alert("Error uploading PDF");
    }
    setLoading(false);
  };

  return (
    <div className="upload-container" style={{ textAlign: 'center', marginTop: '4rem' }}>
      <h2 style={{ fontSize: '3.5rem', fontWeight: 800, marginBottom: '1rem', background: 'linear-gradient(to right, #e0e7ff, #a5b4fc)', WebkitBackgroundClip: 'text', color: 'transparent', letterSpacing: '-0.02em' }}>
        Elevate Your Career
      </h2>
      <p style={{ color: '#94a3b8', fontSize: '1.25rem', marginBottom: '3rem', maxWidth: '500px', margin: '0 auto 3rem auto', lineHeight: 1.6 }}>
        Drop your PDF resume to automatically calculate your ATS match, generate custom cover letters, and unlock tailored prep.
      </p>
      
      <div className="upload-card">
        <div className="upload-icon-wrapper">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="upload-icon">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
        </div>
        
        <label className="file-input-label">
          {file ? <span style={{ color: '#fff' }}>{file.name}</span> : "Browse PDF Document"}
          <input type="file" accept="application/pdf" onChange={handleFileChange} className="hidden-input" />
        </label>
        
        <div style={{ marginTop: '2.5rem', width: '100%' }}>
          <button className="btn-primary pulse" onClick={handleUpload} disabled={!file || loading} style={{ width: '100%', padding: '1.1rem', fontSize: '1.1rem' }}>
            {loading ? 'Analyzing your profile...' : 'Unlock Dashboard ✨'}
          </button>
        </div>
      </div>
    </div>
  );
}
