import { useEffect } from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import UploadResume from './pages/UploadResume'
import Optimize from './pages/Optimize'
import Jobs from './pages/Jobs'
import History from './pages/History'
import './index.css'

function App() {
  const location = useLocation()

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' })
  }, [location.pathname])

  return (
    <div className="dashboard full">
      <div className="header">
        <h1>AI Job App Agent</h1>
        <nav className="nav-links" style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <Link to="/" className={location.pathname === '/' ? 'active-link' : ''}>Home</Link>
          <Link to="/optimize" className={location.pathname === '/optimize' ? 'active-link' : ''}>Optimize</Link>
          <Link to="/jobs" className={location.pathname === '/jobs' ? 'active-link' : ''}>Jobs</Link>
          <Link to="/history" className={location.pathname === '/history' ? 'active-link' : ''}>History</Link>
        </nav>
      </div>
      
      <div className="content-container" style={{ padding: '2rem' }}>
        <Routes>
          <Route path="/" element={<UploadResume />} />
          <Route path="/optimize" element={<Optimize />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </div>
    </div>
  )
}

export default App
