# AI-Powered Missing Person Identification & Tracking System

An industry-standard, human-in-the-loop CCTV investigation platform combining **Ultralytics YOLO11s**, **ByteTrack**, **InsightFace (`buffalo_l` RetinaFace + ArcFace)**, and **OSNet-AIN (`osnet_ain_x1_0`)** behind a **FastAPI** backend and **React + Vite + Tailwind** dashboard.

> **Responsible AI Notice:** Similarity scores are uncalibrated investigative leads and do **not** establish identity. Every candidate match requires human verification, and appearance-only matches are explicitly flagged when facial evidence is unavailable.

## Quick Start (Local Development)

### 1. Backend
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`. Register the first investigator/admin account, create a missing person case, upload a reference photograph, upload a CCTV video, review candidate alerts, and export a PDF investigation report.

### 3. Run Automated Tests
```bash
cd backend
python -m unittest tests/test_e2e.py
```

### 4. Docker Deployment
```bash
docker compose up --build
```
