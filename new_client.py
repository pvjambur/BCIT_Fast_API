import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from googleapiclient.errors import HttpError
import logging
from pathlib import Path
import time
import threading
import re
from typing import Dict, List
from datetime import datetime
import json
import wave
import contextlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriveClient:
    def __init__(self):
        self.drive = self._initialize_drive()
        self.root_folder_id = 'root'
        self.local_temp = "static/temp"
        self.running = False
        self.refresh_interval = 30  # seconds
        Path(self.local_temp).mkdir(parents=True, exist_ok=True)
        self.clients_data = {}
        self.bat_stats = {
            'total_detections': 0,
            'species_count': {},
            'client_stats': {},
            'hourly_activity': {},
            'last_updated': 0
        }
        
        # Client locations (latitude, longitude)
        self.client_locations = {
            '1': (12.922971, 77.500997),
            '2': (12.922881, 77.500950)
        }

    def _initialize_drive(self):
        try:
            gauth = GoogleAuth()
            # Try to load saved credentials
            gauth.LoadCredentialsFile("credentials.json")
            if gauth.credentials is None:
                # Authenticate if they're not there
                gauth.LocalWebserverAuth()
            elif gauth.access_token_expired:
                # Refresh them if expired
                gauth.Refresh()
            else:
                # Initialize the saved creds
                gauth.Authorize()
            # Save the current credentials to a file
            gauth.SaveCredentialsFile("credentials.json")
            return GoogleDrive(gauth)
        except Exception as e:
            logger.error(f"Error initializing drive: {e}")
            # Fall back to settings.yaml if credentials.json doesn't exist
            if not Path("credentials.json").exists():
                logger.info("Trying to authenticate using settings.yaml")
                gauth = GoogleAuth(settings_file='settings.yaml')
                gauth.LocalWebserverAuth()
                gauth.SaveCredentialsFile("credentials.json")
                return GoogleDrive(gauth)
            raise

    def start_auto_refresh(self):
        self.running = True
        def refresh_loop():
            while self.running:
                try:
                    self.process_drive()
                    self.update_bat_stats()
                except Exception as e:
                    logger.error(f"Error during auto-refresh: {str(e)}")
                time.sleep(self.refresh_interval)
        
        thread = threading.Thread(target=refresh_loop, daemon=True)
        thread.start()
        logger.info("Started auto-refresh of Google Drive folders")

    def stop_auto_refresh(self):
        self.running = False
        logger.info("Stopped auto-refresh of Google Drive folders")

    def process_drive(self):
        logger.info("Checking Google Drive for new BatCalls folders...")
        try:
            query = "title contains 'BatCalls' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            folders = self.drive.ListFile({'q': query}).GetList()
            
            if not folders:
                logger.info("No BatCalls folders found in Google Drive")
                return
            
            for folder in folders:
                self.process_folder(folder)
                
        except Exception as e:
            logger.error(f"Error processing drive: {str(e)}")

    def process_folder(self, folder):
        folder_name = folder['title']
        safe_folder_name = folder_name.replace(' ', '_')
        local_folder = Path(self.local_temp) / safe_folder_name
        local_folder.mkdir(exist_ok=True)
        
        logger.info(f"Processing folder: {folder_name}")
        
        query = f"'{folder['id']}' in parents and trashed=false"
        files = self.drive.ListFile({'q': query}).GetList()
        
        for file in files:
            try:
                if 'sensor' in file['title'].lower() and file['title'].endswith('.txt'):
                    self._download_file(file, local_folder / "Sensor.txt")
                    self._update_client_data(local_folder / "Sensor.txt")
                elif 'camera' in file['title'].lower() and file['title'].endswith('.jpg'):
                    self._download_file(file, local_folder / "Camera.jpg")
                elif 'spectogram' in file['title'].lower() and file['title'].endswith('.jpg'):
                    self._download_file(file, local_folder / "Spectogram.jpg")
                elif 'audio' in file['title'].lower() and file['title'].endswith('.wav'):
                    self._download_file(file, local_folder / "Audio.wav")
            except Exception as e:
                logger.error(f"Error processing file {file['title']}: {str(e)}")

    def _update_client_data(self, sensor_path: Path):
        """Update client data from sensor file"""
        try:
            with open(sensor_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if match := re.search(r"Client(\d+)", content):
                    client_id = match.group(1)
                    if client_id not in self.clients_data:
                        self.clients_data[client_id] = {
                            'last_update': time.time(),
                            'folders': []
                        }
                    self.clients_data[client_id]['last_update'] = time.time()
                    self.clients_data[client_id]['folders'].append(sensor_path.parent.name)
        except Exception as e:
            logger.error(f"Error updating client data: {e}")

    def _download_file(self, file, destination):
        try:
            if not destination.exists():
                file.GetContentFile(str(destination))
                logger.info(f"Downloaded: {file['title']} to {destination}")
            else:
                # Check if remote file is newer
                remote_mtime = file['modifiedDate']
                local_mtime = datetime.fromtimestamp(destination.stat().st_mtime).isoformat()
                if remote_mtime > local_mtime:
                    file.GetContentFile(str(destination))
                    logger.info(f"Updated: {file['title']} to {destination}")
                else:
                    logger.debug(f"File already up-to-date: {destination}")
        except HttpError as e:
            if e.resp.status == 403:
                try:
                    file.GetContentFile(str(destination), acknowledgeAbuse=True)
                    logger.info(f"Downloaded with acknowledgeAbuse: {file['title']}")
                except Exception as e:
                    logger.error(f"Failed to download with acknowledgeAbuse: {str(e)}")
            else:
                logger.error(f"HTTP error downloading {file['title']}: {str(e)}")
        except Exception as e:
            logger.error(f"Error downloading {file['title']}: {str(e)}")

    def get_folder_status(self):
        folders = []
        for folder in Path(self.local_temp).iterdir():
            if folder.is_dir() and "BatCalls" in folder.name:
                status = {
                    'name': folder.name,
                    'path': str(folder),
                    'sensor': (folder / "Sensor.txt").exists(),
                    'camera': (folder / "Camera.jpg").exists(),
                    'spectogram': (folder / "Spectogram.jpg").exists(),
                    'audio': (folder / "Audio.wav").exists()
                }
                status['complete'] = all([status['sensor'], status['camera'], 
                                        status['spectogram'], status['audio']])
                
                # Get client ID from sensor file if exists
                if status['sensor']:
                    try:
                        with open(folder / "Sensor.txt", 'r', encoding='utf-8') as f:
                            content = f.read()
                            if match := re.search(r"Client(\d+)", content):
                                status['client_id'] = match.group(1)
                    except Exception as e:
                        logger.error(f"Error reading client ID from {folder}: {e}")
                        status['client_id'] = '1'
                else:
                    status['client_id'] = '1'
                
                folders.append(status)
        return folders

    def get_clients_status(self) -> Dict:
        """Get status of all clients"""
        clients = {}
        for folder in Path(self.local_temp).iterdir():
            if folder.is_dir() and "BatCalls" in folder.name:
                sensor_file = folder / "Sensor.txt"
                if sensor_file.exists():
                    try:
                        with open(sensor_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if match := re.search(r"Client(\d+)", content):
                                client_id = match.group(1)
                                if client_id not in clients:
                                    clients[client_id] = {
                                        'folders': [],
                                        'last_update': os.path.getmtime(sensor_file),
                                        'location': self.client_locations.get(client_id, (0, 0))
                                    }
                                clients[client_id]['folders'].append(folder.name)
                    except Exception as e:
                        logger.error(f"Error reading client data from {sensor_file}: {e}")
        
        # Add default client if none found
        if not clients:
            clients['1'] = {
                'folders': [],
                'last_update': time.time(),
                'location': self.client_locations.get('1', (0, 0))
            }
        
        return clients

    def get_client_data(self, client_id: str) -> Dict:
        """Get detailed data for a specific client"""
        client_folders = []
        for folder in Path(self.local_temp).iterdir():
            if folder.is_dir() and "BatCalls" in folder.name:
                sensor_file = folder / "Sensor.txt"
                if sensor_file.exists():
                    try:
                        with open(sensor_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if f"Client{client_id}" in content:
                                client_folders.append({
                                    'name': folder.name,
                                    'sensor_data': self._parse_sensor_content(content),
                                    'has_spectogram': (folder / "Spectogram.jpg").exists(),
                                    'has_audio': (folder / "Audio.wav").exists(),
                                    'timestamp': os.path.getmtime(sensor_file)
                                })
                    except Exception as e:
                        logger.error(f"Error reading client data from {sensor_file}: {e}")
        
        return {
            'client_id': client_id,
            'folders': client_folders,
            'last_update': time.time(),
            'location': self.client_locations.get(client_id, (0, 0))
        }

    def _parse_sensor_content(self, content: str) -> Dict:
        """Parse sensor data from file content"""
        data = {
            'temperature_C': 0,
            'humidity_%': 0,
            'light_lux': 0,
            'pressure_hPa': 0,
            'gps_status': 'not linked'
        }
        
        if match := re.search(r"Temp:\s*([\d.]+)C", content):
            data['temperature_C'] = float(match.group(1))
        if match := re.search(r"Humidity:\s*([\d.]+)%", content):
            data['humidity_%'] = float(match.group(1))
        if match := re.search(r"Light:\s*([\d.]+)\s*lux", content):
            data['light_lux'] = float(match.group(1))
        if match := re.search(r"Pressure:\s*([\d.]+)\s*hPa", content):
            data['pressure_hPa'] = float(match.group(1))
        if "GPS linked" in content:
            data['gps_status'] = 'linked'
            
        return data

    def update_bat_stats(self):
        """Update bat detection statistics"""
        stats = {
            'total_detections': 0,
            'species_count': {},
            'client_stats': {},
            'hourly_activity': {str(h): 0 for h in range(24)},
            'last_updated': time.time()
        }
        
        # Process all folders to gather statistics
        for folder in Path(self.local_temp).iterdir():
            if folder.is_dir() and "BatCalls" in folder.name:
                spectogram = folder / "Spectogram.jpg"
                sensor = folder / "Sensor.txt"
                
                if spectogram.exists() and sensor.exists():
                    try:
                        # Get client ID
                        client_id = '1'
                        with open(sensor, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if match := re.search(r"Client(\d+)", content):
                                client_id = match.group(1)
                        
                        # Update client stats
                        if client_id not in stats['client_stats']:
                            stats['client_stats'][client_id] = 0
                        stats['client_stats'][client_id] += 1
                        
                        # Get timestamp from folder name or file mtime
                        try:
                            timestamp = os.path.getmtime(sensor)
                            hour = datetime.fromtimestamp(timestamp).hour
                            stats['hourly_activity'][str(hour)] += 1
                        except:
                            pass
                        
                        # Count as detection
                        stats['total_detections'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing stats for {folder}: {e}")
        
        self.bat_stats = stats
        self._save_stats_to_file()
    
    def _save_stats_to_file(self):
        """Save statistics to a JSON file for AI to access"""
        try:
            with open('static/bat_stats.json', 'w') as f:
                json.dump(self.bat_stats, f)
        except Exception as e:
            logger.error(f"Error saving bat stats: {e}")

    def get_bat_stats(self):
        """Get bat detection statistics"""
        # Try to load from file if not in memory
        if not self.bat_stats or time.time() - self.bat_stats.get('last_updated', 0) > 3600:
            try:
                with open('static/bat_stats.json', 'r') as f:
                    self.bat_stats = json.load(f)
            except:
                self.update_bat_stats()
        
        return self.bat_stats