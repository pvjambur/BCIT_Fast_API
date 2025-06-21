project_root/
│
├── static/
│   ├── bat_species/ (pre-populated with species images)
│   │   ├── Hipposideros_lankadiva.jpg
│   │   ├── Hipposideros_speoris.jpg
│   │   └── ... (other species images)
│   │
│   ├── temp/ (for downloaded files)
│   │   ├── Sensor.txt
│   │   ├── Camera.jpg
│   │   ├── Spectogram.jpg
│   │   └── Audio.wav
│   │
│   └── styles.css
│
├── models/
│   ├── efficientnet_b0_bat.pth
│   └── classes.json
│
├── templates/
│   └── index.html (single HTML file)
│
├── app.py (FastAPI main application)
├── drive_client.py (Google Drive operations)
├── model_predictor.py (Spectogram classification)
└── requirements.txt