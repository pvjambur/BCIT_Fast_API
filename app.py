from fastapi import FastAPI, Request, HTTPException
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
import io
import csv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import ollama

app = FastAPI()
drive_client = DriveClient()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
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
    
    # Summary from Ollama
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

def get_ollama_summary(context: str) -> str:
    """Get summary from Ollama"""
    try:
        response = ollama.generate(
            model='llama3',
            prompt=f"Provide a concise executive summary (3-5 sentences) for this bat monitoring data: {context}"
        )
        return response['response']
    except Exception as e:
        print(f"Error getting Ollama summary: {e}")
        return "Could not generate summary. Please check Ollama connection."

def get_ollama_response(question: str, context: str) -> str:
    """Get response from Ollama with context"""
    try:
        response = ollama.generate(
            model='llama3',
            prompt=f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        )
        return response['response']
    except Exception as e:
        print(f"Error getting Ollama response: {e}")
        return "I couldn't process your request. Please try again later."

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

@app.get("/api/ask_ai")
async def ask_ai(question: str, client_id: Optional[str] = None, species: Optional[str] = None):
    try:
        # Get context data
        stats = drive_client.get_bat_stats()
        clients = drive_client.get_clients_status()
        
        # Prepare context string
        context = {
            "total_detections": stats.get('total_detections', 0),
            "species": stats.get('species_count', {}),
            "clients": clients,
            "question": question
        }
        
        if client_id:
            context['current_client'] = client_id
        if species:
            context['current_species'] = species
        
        context_str = json.dumps(context, indent=2)
        
        # Get response from Ollama
        response = get_ollama_response(question, context_str)
        
        return JSONResponse({
            "status": "success",
            "response": response,
            "timestamp": int(time.time())
        })
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
        
        # Get summary from Ollama
        context_str = json.dumps({
            "report_type": report_type,
            "client_id": client_id,
            "species": species,
            "stats": stats,
            "clients": clients
        }, indent=2)
        
        report_data['summary'] = get_ollama_summary(context_str)
        
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

@app.on_event("shutdown")
def shutdown_event():
    drive_client.stop_auto_refresh()