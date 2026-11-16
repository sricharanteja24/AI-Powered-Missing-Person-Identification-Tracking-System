import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle,
  FileText,
  LogOut,
  ShieldAlert,
  Upload,
  UserPlus,
  Video,
  XCircle,
} from 'lucide-react';

const API_BASE = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) || '';

export const getToken = () => localStorage.getItem('mp_token');
export const setToken = (token) => {
  if (token) localStorage.setItem('mp_token', token);
  else localStorage.removeItem('mp_token');
};

export async function apiRequest(path, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  if (!(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch {}
    throw new Error(detail);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/pdf')) return res.blob();
  return res.json();
}

const mediaUrl = (fullPath, folder) => {
  if (!fullPath) return null;
  const filename = String(fullPath).split(/[/\\]/).pop();
  return filename ? `/media/${folder}/${filename}` : null;
};

const formatVideoTime = (sec) => {
  const s = Number(sec || 0);
  const mins = Math.floor(s / 60);
  const rem = Math.floor(s % 60);
  const mmss = `${String(mins).padStart(2, '0')}:${String(rem).padStart(2, '0')}`;
  const whole = Math.round(s);
  const mod100 = whole % 100;
  const suffix =
    mod100 >= 11 && mod100 <= 13
      ? 'th'
      : whole % 10 === 1
      ? 'st'
      : whole % 10 === 2
      ? 'nd'
      : whole % 10 === 3
      ? 'rd'
      : 'th';
  return `${mmss} (${whole}${suffix} sec / ${s.toFixed(2)}s)`;
};

export default function App() {
  const [user, setUser] = useState(null);
  const [tab, setTab] = useState('cases');
  const [authMode, setAuthMode] = useState('login');
  const [authForm, setAuthForm] = useState({ username: '', password: '', full_name: '', email: '' });
  const [cases, setCases] = useState([]);
  const [videos, setVideos] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [auditLogs, setAuditLogs] = useState([]);
  const [error, setError] = useState('');
  const [statusMsg, setStatusMsg] = useState('');
  const [caseForm, setCaseForm] = useState({
    case_id: '',
    full_name: '',
    age: '',
    gender: '',
    last_known_location: '',
    last_seen_date: '',
    description: '',
  });
  const [refPhoto, setRefPhoto] = useState(null);
  const [cameraId, setCameraId] = useState('CAM-01');

  const loadDashboard = async () => {
    try {
      const [c, v, a, l] = await Promise.all([
        apiRequest('/api/cases'),
        apiRequest('/api/videos'),
        apiRequest('/api/alerts'),
        apiRequest('/api/alerts/audit-logs?limit=50'),
      ]);
      setCases(c.cases || []);
      setVideos(v.videos || []);
      setAlerts(a.alerts || []);
      setAuditLogs(l.audit_logs || []);
    } catch (e) {
      setError(e.message);
    }
  };

  useEffect(() => {
    if (getToken()) {
      apiRequest('/api/auth/me')
        .then((res) => {
          setUser(res.user);
          loadDashboard();
        })
        .catch(() => setToken(null));
    }
  }, []);

  useEffect(() => {
    if (!user) return;
    const hasActive = videos.some((v) => v.status === 'queued' || v.status === 'processing');
    if (!hasActive) return;
    const id = setInterval(loadDashboard, 3000);
    return () => clearInterval(id);
  }, [user, videos]);

  const handleAuth = async (e) => {
    e.preventDefault();
    setError('');
    try {
      if (authMode === 'register') {
        await apiRequest('/api/auth/register', {
          method: 'POST',
          body: JSON.stringify({ ...authForm, role: 'investigator' }),
        });
      }
      const res = await apiRequest('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ username: authForm.username, password: authForm.password }),
      });
      setToken(res.access_token);
      setUser(res.user);
      loadDashboard();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleCreateCase = async (e) => {
    e.preventDefault();
    setError('');
    setStatusMsg('');
    try {
      const created = await apiRequest('/api/cases', {
        method: 'POST',
        body: JSON.stringify({
          ...caseForm,
          age: caseForm.age ? Number(caseForm.age) : null,
        }),
      });
      if (refPhoto && created?.case?.id) {
        const fd = new FormData();
        fd.append('file', refPhoto);
        const photoRes = await apiRequest(`/api/cases/${created.case.id}/photo`, {
          method: 'POST',
          body: fd,
        });
        setStatusMsg(
          `Case ${created.case.case_id} created. Face detected: ${photoRes.face_detected ? 'Yes' : 'No'} (${photoRes.face_quality_note})`
        );
      } else {
        setStatusMsg(`Case ${created.case.case_id} created.`);
      }
      setCaseForm({
        case_id: '',
        full_name: '',
        age: '',
        gender: '',
        last_known_location: '',
        last_seen_date: '',
        description: '',
      });
      setRefPhoto(null);
      loadDashboard();
    } catch (err) {
      setError(err.message);
    }
  };

  const handlePhotoUpload = async (caseDbId, file) => {
    if (!file) return;
    setError('');
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await apiRequest(`/api/cases/${caseDbId}/photo`, { method: 'POST', body: fd });
      setStatusMsg(`Reference photo updated. Face detected: ${res.face_detected ? 'Yes' : 'No'} (${res.face_quality_note})`);
      loadDashboard();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleVideoUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError('');
    const fd = new FormData();
    fd.append('camera_id', cameraId);
    fd.append('file', file);
    try {
      await apiRequest('/api/videos/upload', { method: 'POST', body: fd });
      setStatusMsg(`Uploaded ${file.name} for processing.`);
      loadDashboard();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleVerifyAlert = async (alertId, verification_status) => {
    try {
      await apiRequest(`/api/alerts/${alertId}/verify`, {
        method: 'POST',
        body: JSON.stringify({ verification_status, verification_notes: `Reviewed by ${user.sub || user.username}` }),
      });
      loadDashboard();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDownloadReport = async (caseDbId, caseCode) => {
    try {
      const blob = await apiRequest(`/api/reports/${caseDbId}/pdf`);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `investigation_report_${caseCode}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4 bg-slate-950">
        <form onSubmit={handleAuth} className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold text-lg">
            <ShieldAlert className="w-6 h-6" />
            <span>Missing Person Investigation Portal</span>
          </div>
          {error && <div className="p-2.5 text-sm bg-red-950/60 border border-red-800 text-red-200 rounded">{error}</div>}
          <input
            className="w-full px-3 py-2 rounded bg-slate-800 border border-slate-700 text-sm"
            placeholder="Username"
            value={authForm.username}
            onChange={(e) => setAuthForm({ ...authForm, username: e.target.value })}
            required
          />
          {authMode === 'register' && (
            <>
              <input
                className="w-full px-3 py-2 rounded bg-slate-800 border border-slate-700 text-sm"
                placeholder="Full Name"
                value={authForm.full_name}
                onChange={(e) => setAuthForm({ ...authForm, full_name: e.target.value })}
                required
              />
              <input
                className="w-full px-3 py-2 rounded bg-slate-800 border border-slate-700 text-sm"
                type="email"
                placeholder="Email"
                value={authForm.email}
                onChange={(e) => setAuthForm({ ...authForm, email: e.target.value })}
                required
              />
            </>
          )}
          <input
            className="w-full px-3 py-2 rounded bg-slate-800 border border-slate-700 text-sm"
            type="password"
            placeholder="Password (min 8 chars)"
            value={authForm.password}
            onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
            required
          />
          <button className="w-full py-2 rounded bg-emerald-600 hover:bg-emerald-500 font-medium text-sm">
            {authMode === 'login' ? 'Sign In' : 'Create Investigator Account'}
          </button>
          <button
            type="button"
            onClick={() => setAuthMode(authMode === 'login' ? 'register' : 'login')}
            className="w-full text-xs text-slate-400 hover:text-slate-200"
          >
            {authMode === 'login' ? 'Need an account? Register' : 'Already registered? Sign In'}
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/70 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2 font-semibold text-emerald-400">
          <ShieldAlert className="w-5 h-5" />
          <span>AI Missing Person Tracking & Investigation</span>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-slate-400">
            {user.sub || user.username} ({user.role})
          </span>
          <button
            onClick={() => {
              setToken(null);
              setUser(null);
            }}
            className="flex items-center gap-1 text-slate-300 hover:text-white"
          >
            <LogOut className="w-4 h-4" /> Logout
          </button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-6 space-y-6">
        <div className="p-3 rounded-lg bg-amber-950/40 border border-amber-800/60 text-amber-200 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>
            Responsible AI Notice: Similarity scores are investigative leads only and do not establish identity. Every candidate match requires human verification.
          </span>
        </div>

        {error && <div className="p-3 rounded bg-red-950/60 border border-red-800 text-red-200 text-sm">{error}</div>}
        {statusMsg && <div className="p-3 rounded bg-emerald-950/60 border border-emerald-800 text-emerald-200 text-sm">{statusMsg}</div>}

        <nav className="flex gap-2 border-b border-slate-800 pb-2">
          {[
            ['cases', 'Missing Cases', cases.length],
            ['videos', 'CCTV Videos', videos.length],
            ['alerts', 'Candidate Alerts', alerts.length],
            ['audit', 'Audit Logs', auditLogs.length],
          ].map(([key, label, count]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium ${
                tab === key ? 'bg-emerald-600 text-white' : 'bg-slate-900 text-slate-300 hover:bg-slate-800'
              }`}
            >
              {label} ({count})
            </button>
          ))}
        </nav>

        {tab === 'cases' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <form onSubmit={handleCreateCase} className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3 h-fit">
              <h2 className="font-semibold flex items-center gap-2 text-sm">
                <UserPlus className="w-4 h-4 text-emerald-400" /> Register Missing Person
              </h2>
              <input className="w-full px-3 py-2 rounded bg-slate-800 border border-slate-700 text-sm" placeholder="Case ID (e.g. MP-2026-001)" value={caseForm.case_id} onChange={(e) => setCaseForm({ ...caseForm, case_id: e.target.value })} required />
              <input className="w-full px-3 py-2 rounded bg-slate-800 border border-slate-700 text-sm" placeholder="Full Name" value={caseForm.full_name} onChange={(e) => setCaseForm({ ...caseForm, full_name: e.target.value })} required />
              <div className="grid grid-cols-2 gap-2">
                <input className="px-3 py-2 rounded bg-slate-800 border border-slate-700 text-sm" type="number" placeholder="Age" value={caseForm.age} onChange={(e) => setCaseForm({ ...caseForm, age: e.target.value })} />
                <input className="px-3 py-2 rounded bg-slate-800 border border-slate-700 text-sm" placeholder="Gender" value={caseForm.gender} onChange={(e) => setCaseForm({ ...caseForm, gender: e.target.value })} />
              </div>
              <input className="w-full px-3 py-2 rounded bg-slate-800 border border-slate-700 text-sm" placeholder="Last Known Location" value={caseForm.last_known_location} onChange={(e) => setCaseForm({ ...caseForm, last_known_location: e.target.value })} required />
              <input className="w-full px-3 py-2 rounded bg-slate-800 border border-slate-700 text-sm" type="date" value={caseForm.last_seen_date} onChange={(e) => setCaseForm({ ...caseForm, last_seen_date: e.target.value })} required />
              <textarea className="w-full px-3 py-2 rounded bg-slate-800 border border-slate-700 text-sm" rows={2} placeholder="Clothing / Notes" value={caseForm.description} onChange={(e) => setCaseForm({ ...caseForm, description: e.target.value })} />

              <label className="block text-xs text-slate-300 space-y-1">
                <span>Missing Person Reference Photo</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setRefPhoto(e.target.files?.[0] || null)}
                  className="block w-full text-xs text-slate-300 file:mr-3 file:py-1.5 file:px-3 file:rounded file:border-0 file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700"
                  required
                />
              </label>

              <button className="w-full py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-sm font-medium">Create Case & Extract Features</button>
            </form>

            <div className="lg:col-span-2 space-y-3">
              {cases.map((c) => {
                const photoUrl = mediaUrl(c.photo_path, 'photos');
                return (
                  <div key={c.id} className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      {photoUrl && (
                        <img src={photoUrl} alt={c.full_name} className="w-14 h-14 object-cover rounded-lg border border-slate-700" />
                      )}
                      <div>
                        <div className="font-semibold text-sm">{c.case_id} — {c.full_name}</div>
                        <div className="text-xs text-slate-400">
                          Last seen: {c.last_known_location} ({c.last_seen_date}) • Face Emb: {c.has_face_embedding ? 'Ready' : 'Missing'} • Re-ID Emb: {c.has_reid_embedding ? 'Ready' : 'Missing'}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <label className="cursor-pointer px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs flex items-center gap-1">
                        <Upload className="w-3.5 h-3.5" /> Update Photo
                        <input type="file" accept="image/*" className="hidden" onChange={(e) => handlePhotoUpload(c.id, e.target.files?.[0])} />
                      </label>
                      <button onClick={() => handleDownloadReport(c.id, c.case_id)} className="px-3 py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-xs flex items-center gap-1">
                        <FileText className="w-3.5 h-3.5" /> PDF Report
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {tab === 'videos' && (
          <div className="space-y-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center gap-3">
              <Video className="w-5 h-5 text-emerald-400" />
              <input className="px-3 py-1.5 rounded bg-slate-800 border border-slate-700 text-sm" value={cameraId} onChange={(e) => setCameraId(e.target.value)} placeholder="Camera ID (e.g. CAM-01)" />
              <label className="cursor-pointer px-4 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-sm font-medium">
                Upload CCTV Video
                <input type="file" accept="video/*" className="hidden" onChange={handleVideoUpload} />
              </label>
            </div>
            <div className="space-y-2">
              {videos.map((v) => (
                <div key={v.id} className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
                  <div>
                    <div className="font-medium text-sm">{v.video_name} ({v.camera_id})</div>
                    <div className="text-xs text-slate-400">Status: {v.status} • Progress: {v.progress_pct}%</div>
                  </div>
                  <div className="w-40 bg-slate-800 rounded-full h-2 overflow-hidden">
                    <div className="bg-emerald-500 h-2" style={{ width: `${v.progress_pct}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === 'alerts' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {alerts.map((a) => {
              const evUrl = mediaUrl(a.evidence_image_path, 'evidence');
              return (
                <div key={a.id} className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="font-semibold text-sm">{a.case_code} — {a.missing_person_name}</div>
                      <div className="text-xs text-slate-400">
                        {a.camera_id} • Track #{a.track_id} • Frame {a.frame_number}
                      </div>
                      <div className="text-xs font-medium text-emerald-400 mt-0.5">
                        Detected at {formatVideoTime(a.timestamp_sec)}
                      </div>
                    </div>
                    <span className="px-2 py-0.5 rounded text-xs bg-slate-800">{a.verification_status}</span>
                  </div>
                  {evUrl && (
                    <img
                      src={evUrl}
                      alt={`Candidate match at ${a.timestamp_sec}s`}
                      className="w-full max-h-64 object-contain rounded-lg border border-slate-800 bg-black"
                    />
                  )}
                  <div className="text-xs text-slate-300">
                    Tier: <b>{a.match_tier}</b> | Face Sim: {a.face_similarity ?? 'N/A'} | Re-ID Sim: {a.reid_similarity ?? 'N/A'}
                  </div>
                  <div className="text-xs text-amber-300">{a.limitation_note}</div>
                  <div className="flex gap-2 pt-1">
                    <button onClick={() => handleVerifyAlert(a.id, 'verified_candidate')} className="px-3 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-xs flex items-center gap-1">
                      <CheckCircle className="w-3.5 h-3.5" /> Verify Candidate
                    </button>
                    <button onClick={() => handleVerifyAlert(a.id, 'rejected_candidate')} className="px-3 py-1 rounded bg-red-600 hover:bg-red-500 text-xs flex items-center gap-1">
                      <XCircle className="w-3.5 h-3.5" /> Reject
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {tab === 'audit' && (
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-800 text-slate-300">
                <tr>
                  <th className="p-3">Time</th>
                  <th className="p-3">User</th>
                  <th className="p-3">Action</th>
                  <th className="p-3">Resource</th>
                  <th className="p-3">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {auditLogs.map((log) => (
                  <tr key={log.id}>
                    <td className="p-3 text-slate-400">{log.created_at}</td>
                    <td className="p-3">{log.username || 'system'}</td>
                    <td className="p-3 font-medium">{log.action}</td>
                    <td className="p-3">{log.resource_type} #{log.resource_id}</td>
                    <td className="p-3 text-slate-400">{log.details}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}