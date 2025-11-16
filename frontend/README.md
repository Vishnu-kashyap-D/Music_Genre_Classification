# Music Genre Classifier - Frontend

Professional React frontend for the Music Genre Classification web application.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Backend connection

The analysis view calls the Flask backend at `http://127.0.0.1:5000` by default. Make sure the backend is running before uploading audio:

```bash
python app.py
```

You can quickly verify connectivity by visiting `http://127.0.0.1:5000/health` in your browser. If the backend runs on a different host or port, configure it via an environment variable (see below).

### Environment variables

Create a `.env` file inside `frontend/` if you need to override defaults:

```
VITE_API_BASE_URL=http://localhost:8000
```

Restart the dev server after changing `.env` so Vite picks up the new value.

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` folder.

## Features

- 🎨 Dark/Light mode toggle
- 🎵 Drag & drop audio upload
- 🎤 Live audio recording with waveform visualization
- 📊 Interactive radar chart for genre similarity
- 📈 Confidence bars with color coding
- 🎧 Waveform player with timeline heatmap
- 📸 Spectrogram visualization
- 📊 SNR (Signal-to-Noise Ratio) meter
- 📱 Social sharing (WhatsApp, Instagram, Twitter)
- 📄 PDF report generation
- 📜 Analysis history (last 5 analyses)

## Tech Stack

- React 18
- Vite
- Tailwind CSS
- Framer Motion (animations)
- Chart.js / react-chartjs-2 (charts)
- WaveSurfer.js (audio visualization)
- jsPDF (PDF generation)
- html2canvas (image generation)
- Lucide React (icons)

