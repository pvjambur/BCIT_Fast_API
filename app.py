from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from pathlib import Path
import time
from datetime import datetime
import json
from drive_client import DriveClient
from models.predict import classify_image
import threading
import re
from typing import Dict, List, Optional
import os
import wave
import contextlib
import zipfile
import pandas as pd
import io
import csv
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import requests

app = FastAPI()
drive_client = DriveClient()

GROQ_API_KEY = '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'
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL = 'llama3-70b-8192'

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
app.mount("/downloads", StaticFiles(directory="downloads"), name="downloads")
templates = Jinja2Templates(directory="templates")

# Start the auto-refresh thread
drive_client.start_auto_refresh()

# Custom filters
def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    if isinstance(value, (int, float)):
        value = datetime.fromtimestamp(value)
    return value.strftime(format)

templates.env.filters['datetimeformat'] = datetimeformat

def parse_sensor_data(sensor_path: Path) -> Dict:
    """Parse sensor data from Sensor.txt file"""
    sensor_data = {
        'client_id': '1',  # Default to client 1 if not found
        'temperature_C': 0,
        'humidity_%': 0,
        'light_lux': 0,
        'pressure_hPa': 0,
        'gps_status': 'not linked',
        'extra_info': []
    }
    
    try:
        with open(sensor_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = ''.join(c for c in line if c.isprintable())
                if not line:
                    continue
                
                # Extract client ID if present
                if match := re.match(r"Client(\d+)", line):
                    sensor_data['client_id'] = match.group(1)
                
                # Match sensor data patterns
                if match := re.match(r".*Temp:\s*([\d.]+)C", line):
                    sensor_data['temperature_C'] = float(match.group(1))
                elif match := re.match(r".*Humidity:\s*([\d.]+)%", line):
                    sensor_data['humidity_%'] = float(match.group(1))
                elif match := re.match(r".*Light:\s*([\d.]+)\s*lux", line):
                    sensor_data['light_lux'] = float(match.group(1))
                elif match := re.match(r".*Pressure:\s*([\d.]+)\s*hPa", line):
                    sensor_data['pressure_hPa'] = float(match.group(1))
                elif "GPS not linked" in line:
                    sensor_data['gps_status'] = "not linked"
                elif "GPS linked" in line:
                    sensor_data['gps_status'] = "linked"
                else:
                    sensor_data['extra_info'].append(line)
                    
    except Exception as e:
        print(f"Error reading sensor data: {e}")
    
    return sensor_data

def get_audio_metadata(audio_path: Path) -> Dict:
    """Extract metadata from WAV file"""
    metadata = {
        'size': f"{audio_path.stat().st_size / 1024:.2f} KB",
        'modified': datetime.fromtimestamp(audio_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        'duration': 0,
        'sample_rate': 0,
        'channels': 0,
        'sample_width': 0
    }
    
    try:
        with contextlib.closing(wave.open(str(audio_path), 'r')) as f:
            metadata['duration'] = f.getnframes() / float(f.getframerate())
            metadata['sample_rate'] = f.getframerate()
            metadata['channels'] = f.getnchannels()
            metadata['sample_width'] = f.getsampwidth() * 8  # in bits
    except Exception as e:
        print(f"Error reading audio metadata: {e}")
    
    return metadata

def generate_pdf_report(report_data: Dict) -> bytes:
    """Generate PDF report using ReportLab"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph(f"Bat Monitoring Report - {report_data['title']}", styles['Title']))
    elements.append(Spacer(1, 0.25*inch))
    
    # Summary from Groq
    if 'summary' in report_data:
        elements.append(Paragraph("Executive Summary", styles['Heading2']))
        elements.append(Paragraph(report_data['summary'], styles['BodyText']))
        elements.append(Spacer(1, 0.25*inch))
    
    # Client Information
    if 'clients' in report_data:
        elements.append(Paragraph("Client Information", styles['Heading2']))
        client_data = [["Client ID", "Status", "Last Update", "Detections"]]
        for client in report_data['clients']:
            client_data.append([
                client['id'],
                client['status'],
                datetimeformat(client['last_update']),
                str(client['detections'])
            ])
        client_table = Table(client_data)
        client_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(client_table)
        elements.append(Spacer(1, 0.25*inch))
    
    # Species Data
    if 'species' in report_data:
        elements.append(Paragraph("Species Detected", styles['Heading2']))
        species_data = [["Species", "Count", "Confidence"]]
        for species in report_data['species']:
            species_data.append([
                species['name'],
                str(species['count']),
                f"{species['confidence']}%"
            ])
        species_table = Table(species_data)
        species_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(species_table)
        elements.append(Spacer(1, 0.25*inch))
    
    # Activity Patterns
    if 'activity' in report_data:
        elements.append(Paragraph("Activity Patterns", styles['Heading2']))
        activity_data = [["Time", "Detections"]]
        for time, count in report_data['activity'].items():
            activity_data.append([time, str(count)])
        activity_table = Table(activity_data)
        activity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(activity_table)
        elements.append(Spacer(1, 0.25*inch))
    
    # Spectogram Image
    if 'spectogram_path' in report_data and os.path.exists(report_data['spectogram_path']):
        elements.append(Paragraph("Sample Spectogram", styles['Heading2']))
        img = Image(report_data['spectogram_path'], width=4*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.25*inch))
    
    # Generate PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

def get_groq_summary(context: str) -> str:
    """Get summary from Groq API"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Provide a concise executive summary (3-5 sentences) for this bat monitoring data."
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            "temperature": 0.7
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error getting Groq summary: {e}")
        return "Could not generate summary. Please check Groq connection."

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, folder: str = None, client: str = None):
    folders = drive_client.get_folder_status()
    selected_folder = None
    data = {
        "request": request,
        "folders": folders,
        "timestamp": int(time.time()),
        "sensor_data": None,
        "camera_image": None,
        "spectogram_image": None,
        "audio_file": None,
        "species_prediction": None,
        "confidence": None,
        "species_image": None,
        "client_id": None,
        "audio_metadata": None,
        "clients": drive_client.get_clients_status(),
        "selected_client": client,
        "bat_stats": drive_client.get_bat_stats()
    }

    if folders:
        if folder:
            selected_folder = next((f for f in folders if f['name'] == folder), None)
        else:
            # If client is specified, get their latest folder
            if client:
                client_folders = [f for f in folders if f.get('client_id') == client and f['complete']]
                if client_folders:
                    selected_folder = sorted(client_folders, key=lambda x: x['name'])[-1]
            else:
                # Otherwise get the latest complete folder
                complete_folders = [f for f in folders if f['complete']]
                if complete_folders:
                    selected_folder = sorted(complete_folders, key=lambda x: x['name'])[-1]

        if selected_folder:
            folder_path = Path(selected_folder['path'])
            data['selected_folder'] = selected_folder['name']
            data['client_id'] = selected_folder.get('client_id', '1')

            # Load sensor data
            if selected_folder['sensor']:
                sensor_data = parse_sensor_data(folder_path / "Sensor.txt")
                data['sensor_data'] = sensor_data
                data['client_id'] = sensor_data['client_id']

            # Process spectogram if exists
            if selected_folder['spectogram']:
                spectogram_path = folder_path / "Spectogram.jpg"
                try:
                    species, confidence = classify_image(str(spectogram_path))
                    data['species_prediction'] = species
                    data['confidence'] = confidence
                    species_image_path = Path(f"static/bat_species/{species.replace(' ', '_')}.jpg")
                    if species_image_path.exists():
                        data['species_image'] = f"static/bat_species/{species.replace(' ', '_')}.jpg"
                    else:
                        data['species_image'] = "static/bat_species/Unknown.jpg"
                    data['spectogram_image'] = f"static/temp/{selected_folder['name']}/Spectogram.jpg"
                except Exception as e:
                    print(f"Error processing spectogram: {e}")
                    data['species_prediction'] = "Unknown"
                    data['confidence'] = 0.0
                    data['species_image'] = "static/bat_species/Unknown.jpg"

            # Add other files to data if they exist
            if selected_folder['camera']:
                data['camera_image'] = f"static/temp/{selected_folder['name']}/Camera.jpg"
            if selected_folder['audio']:
                data['audio_file'] = f"static/temp/{selected_folder['name']}/Audio.wav"
                # Get audio metadata
                audio_path = folder_path / "Audio.wav"
                if audio_path.exists():
                    data['audio_metadata'] = get_audio_metadata(audio_path)

    return templates.TemplateResponse("index.html", data)

@app.get("/api/folders")
async def get_folders():
    try:
        folders = drive_client.get_folder_status()
        return JSONResponse({
            "folders": sorted(folders, key=lambda x: x['name']),
            "timestamp": int(time.time()),
            "clients": drive_client.get_clients_status(),
            "bat_stats": drive_client.get_bat_stats()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/refresh")
async def force_refresh():
    try:
        drive_client.process_drive()
        return JSONResponse({
            "status": "success",
            "message": "Drive refresh initiated",
            "timestamp": int(time.time())
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/client/{client_id}")
async def get_client_data(client_id: str):
    try:
        client_data = drive_client.get_client_data(client_id)
        return JSONResponse({
            "status": "success",
            "client_id": client_id,
            "data": client_data,
            "timestamp": int(time.time())
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bat_stats")
async def get_bat_statistics():
    try:
        stats = drive_client.get_bat_stats()
        return JSONResponse({
            "status": "success",
            "stats": stats,
            "timestamp": int(time.time())
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AskResponse(BaseModel):
    answer: str
    question: str

class AskRequest(BaseModel):
    question: str
    context: str = None

class UploadResponse(BaseModel):
    message: str
    filename: str

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        messages = [{
            "role": "system",
            "content": "You are a bat call monitoring assistant. Analyze the provided bat call data and answer questions professionally."
        }]
        
        if request.context:
            messages.append({
                "role": "user",
                "content": f"Context: {request.context}\n\nQuestion: {request.question}"
            })
        else:
            messages.append({
                "role": "user",
                "content": request.question
            })
        
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.7
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        answer = result['choices'][0]['message']['content']
        
        return AskResponse(
            answer=answer,
            question=request.question
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/generate_report")
async def generate_report(
    report_type: str,
    client_id: Optional[str] = None,
    species: Optional[str] = None,
    date_range: Optional[str] = None
):
    try:
        # Get data for report
        stats = drive_client.get_bat_stats()
        clients = drive_client.get_clients_status()
        folders = drive_client.get_folder_status()
        
        # Filter data based on parameters
        if client_id:
            folders = [f for f in folders if f.get('client_id') == client_id]
            clients = [c for c in clients if c['id'] == client_id]
        
        # Prepare report data
        report_data = {
            "title": f"Bat Monitoring Report - {report_type.capitalize()}",
            "clients": clients,
            "species": [{"name": k, "count": v, "confidence": 0} for k, v in stats.get('species_count', {}).items()],
            "activity": stats.get('activity_patterns', {}),
            "timestamp": int(time.time())
        }
        
        # Add spectogram if available
        if folders and folders[0]['spectogram']:
            report_data['spectogram_path'] = str(Path(folders[0]['path']) / "Spectogram.jpg")
        
        # Get summary from Groq
        context_str = json.dumps({
            "report_type": report_type,
            "client_id": client_id,
            "species": species,
            "stats": stats,
            "clients": clients
        }, indent=2)
        
        report_data['summary'] = get_groq_summary(context_str)
        
        # Generate PDF
        pdf_bytes = generate_pdf_report(report_data)
        
        # Return as downloadable file
        headers = {
            "Content-Disposition": f"attachment; filename=bat_report_{report_type}_{int(time.time())}.pdf"
        }
        return StreamingResponse(io.BytesIO(pdf_bytes), headers=headers, media_type="application/pdf")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download_data")
async def download_data(
    client_id: Optional[str] = None,
    species: Optional[str] = None,
    date_range: Optional[str] = None,
    include_sensor: bool = False,
    include_audio: bool = False,
    include_spectogram: bool = False,
    include_camera: bool = False,
    include_metadata: bool = False,
    format: str = "zip"  # or "csv"
):
    try:
        folders = drive_client.get_folder_status()
        
        # Filter folders based on parameters
        if client_id:
            folders = [f for f in folders if f.get('client_id') == client_id]
        if date_range:
            # Implement date filtering logic here
            pass
        
        if format == "zip":
            # Create ZIP file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                for folder in folders:
                    folder_path = Path(folder['path'])
                    
                    # Add selected files to ZIP
                    if include_sensor and folder['sensor']:
                        zip_file.write(folder_path / "Sensor.txt", f"{folder['name']}/Sensor.txt")
                    
                    if include_audio and folder['audio']:
                        zip_file.write(folder_path / "Audio.wav", f"{folder['name']}/Audio.wav")
                    
                    if include_spectogram and folder['spectogram']:
                        zip_file.write(folder_path / "Spectogram.jpg", f"{folder['name']}/Spectogram.jpg")
                    
                    if include_camera and folder['camera']:
                        zip_file.write(folder_path / "Camera.jpg", f"{folder['name']}/Camera.jpg")
                    
                    if include_metadata:
                        # Create metadata file
                        metadata = {
                            "folder": folder['name'],
                            "client_id": folder.get('client_id', '1'),
                            "timestamp": folder.get('timestamp', 0),
                            "species": folder.get('species', 'Unknown'),
                            "confidence": folder.get('confidence', 0)
                        }
                        zip_file.writestr(f"{folder['name']}/metadata.json", json.dumps(metadata, indent=2))
            
            zip_buffer.seek(0)
            headers = {
                "Content-Disposition": f"attachment; filename=bat_data_{int(time.time())}.zip"
            }
            return StreamingResponse(zip_buffer, headers=headers, media_type="application/zip")
        
        elif format == "csv":
            # Create CSV data
            csv_data = io.StringIO()
            writer = csv.writer(csv_data)
            
            # Write header
            writer.writerow([
                "Folder", "Client ID", "Timestamp", "Species", 
                "Confidence", "Sensor Data", "Audio", "Spectogram", "Camera"
            ])
            
            # Write rows
            for folder in folders:
                writer.writerow([
                    folder['name'],
                    folder.get('client_id', '1'),
                    folder.get('timestamp', 0),
                    folder.get('species', 'Unknown'),
                    folder.get('confidence', 0),
                    "Yes" if folder['sensor'] else "No",
                    "Yes" if folder['audio'] else "No",
                    "Yes" if folder['spectogram'] else "No",
                    "Yes" if folder['camera'] else "No"
                ])
            
            csv_buffer = io.BytesIO(csv_data.getvalue().encode())
            headers = {
                "Content-Disposition": f"attachment; filename=bat_data_{int(time.time())}.csv"
            }
            return StreamingResponse(csv_buffer, headers=headers, media_type="text/csv")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid format specified")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/groq/chat")
async def groq_chat(question: str = Form(...), history: str = Form("[]")):
    try:
        history_data = json.loads(history)
        
        messages = [
            {
                "role": "system",
                "content": "You are a bat call monitoring assistant. Analyze the provided bat call data and answer questions professionally."
            }
        ]
        
        # Add history if available
        for item in history_data:
            messages.append({
                "role": "user" if item["sender"] == "user" else "assistant",
                "content": item["message"]
            })
        
        # Add current question
        messages.append({
            "role": "user",
            "content": question
        })
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.7
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        answer = result['choices'][0]['message']['content']
        
        return JSONResponse({
            "answer": answer,
            "success": True
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".ppt"}

# Define the directory for uploaded documents
DOC_DIR = Path("documents")

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, XLSX, XLS, PPTX, and PPT files are allowed")
    
    contents = await file.read()
    if len(contents) > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 200MB)")
    
    file_path = DOC_DIR / file.filename
    if file_path.exists():
        raise HTTPException(status_code=409, detail="File already exists")
    
    try:
        DOC_DIR.mkdir(exist_ok=True, parents=True)
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        return UploadResponse(
            message="File uploaded successfully",
            filename=file.filename
        )
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/api/generate/report")
async def generate_report(
    report_type: str = Form(...),
    client_id: str = Form(None),
    start_date: str = Form(None),
    end_date: str = Form(None)
):
    try:
        # Get data based on parameters
        # This would query your database in a real application
        report_data = get_report_data(report_type, client_id, start_date, end_date)
        
        # Generate PDF
        pdf_path = create_pdf_report(report_data, report_type)
        
        return FileResponse(
            pdf_path,
            media_type='application/pdf',
            filename=f"bat_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_pdf_report(data, report_type):
    styles = getSampleStyleSheet()
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    filename = f"bat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(report_dir, filename)
    
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    elements = []
    
    # Title
    elements.append(Paragraph(f"Bat Call Monitoring Report - {report_type.capitalize()}", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Add content based on report type
    if report_type == "daily":
        # Add daily report content
        pass
    elif report_type == "species":
        # Add species report content
        pass
    
    # Build PDF
    doc.build(elements)
    return filepath

@app.get("/api/download/recording/{recording_id}")
async def download_recording(recording_id: str):
    try:
        # In a real app, this would get the recording from your storage
        recording_dir = f"recordings/{recording_id}"
        
        if not os.path.exists(recording_dir):
            raise HTTPException(status_code=404, detail="Recording not found")
            
        # Create zip file in memory
        zip_filename = f"bat_recording_{recording_id}.zip"
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            for root, dirs, files in os.walk(recording_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_file.write(file_path, os.path.basename(file_path))
                    
            # Add metadata
            metadata = {
                "recording_id": recording_id,
                "timestamp": datetime.now().isoformat(),
                "species": "Pipistrellus pipistrellus",  # Would come from your data
                "confidence": 0.92  # Would come from your data
            }
            
            zip_file.writestr("metadata.json", json.dumps(metadata))
            
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment;filename={zip_filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/client/{client_id}")
async def download_client_data(client_id: str, items: str = "all"):
    try:
        # Determine which items to include
        include_items = items.split(',') if items != "all" else [
            "sensor", "camera", "spectrogram", "audio", "metadata"
        ]
        
        # Create zip file
        zip_filename = f"client_{client_id}_data.zip"
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            # Add selected files
            if "sensor" in include_items:
                # Add sensor data
                pass
                
            if "camera" in include_items:
                # Add camera images
                pass
                
            # Add other requested items...
            
            # Add metadata
            metadata = {
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "items_included": include_items
            }
            zip_file.writestr("metadata.json", json.dumps(metadata))
            
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment;filename={zip_filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/excel")
async def export_excel(
    client_id: str = None,
    start_date: str = None,
    end_date: str = None
):
    try:
        # Get data (would query your database in real app)
        data = get_export_data(client_id, start_date, end_date)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create Excel file in memory
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Bat Calls')
            
        excel_buffer.seek(0)
        
        return StreamingResponse(
            excel_buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment;filename=bat_calls_export.xlsx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def get_report_data(report_type, client_id, start_date, end_date):
    # This would query your database in a real application
    return {
        "type": report_type,
        "client": client_id,
        "period": f"{start_date} to {end_date}",
        "data": []  # Actual data would go here
    }

def get_export_data(client_id, start_date, end_date):
    # This would query your database in a real application
    return [
        {"timestamp": "2023-06-15 21:42:15", "species": "Pipistrellus pipistrellus", "confidence": 0.92},
        # More data...
    ]

@app.on_event("shutdown")
def shutdown_event():
    drive_client.stop_auto_refresh()