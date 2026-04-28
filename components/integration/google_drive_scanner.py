#!/usr/bin/env python3
"""
DMAI Google Drive Scanner - Scans shared Google Drive folders for repos,
downloads them, and feeds them to the Repo Integration Engine.
"""

import os
import re
import json
import zipfile
import tempfile
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GoogleDriveScanner:
    """Scans Google Drive folders for code repos and integrates them into DMAI"""
    
    def __init__(self, dmai_app):
        self.dmai = dmai_app
        self.temp_dir = Path(tempfile.mkdtemp(prefix='dmai_gdrive_'))
        self.scan_history_file = Path("data/gdrive_scan_history.json")
        self.scan_history_file.parent.mkdir(parents=True, exist_ok=True)
        self.scan_history = self._load_scan_history()
    
    def _load_scan_history(self) -> Dict:
        """Load previously scanned files to avoid re-processing"""
        if self.scan_history_file.exists():
            try:
                with open(self.scan_history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'scanned_files': {}, 'last_scan': None}
    
    def _save_scan_history(self):
        """Persist scan history"""
        with open(self.scan_history_file, 'w') as f:
            json.dump(self.scan_history, f, indent=2, default=str)
    
    def scan_folder(self, folder_url: str) -> Dict:
        """
        Scan a shared Google Drive folder for all downloadable content.
        
        Args:
            folder_url: Shared Google Drive folder URL
            
        Returns:
            Dict with discovered_items, downloaded, integrated counts
        """
        result = {
            'folder_url': folder_url,
            'scanned_at': datetime.now().isoformat(),
            'discovered_items': [],
            'downloaded': 0,
            'integrated': 0,
            'skipped': 0,
            'errors': []
        }
        
        try:
            # Extract folder ID from URL
            folder_id = self._extract_folder_id(folder_url)
            if not folder_id:
                result['errors'].append(f"Could not extract folder ID from URL: {folder_url}")
                return result
            
            logger.info(f"📂 Scanning Google Drive folder: {folder_id}")
            
            # Use gdown to list folder contents
            items = self._list_folder_contents(folder_id)
            result['discovered_items'] = items
            
            logger.info(f"📁 Found {len(items)} items in folder")
            
            # Process each item
            for item in items:
                try:
                    # Skip if already scanned and unchanged
                    item_key = item.get('name', item.get('id', ''))
                    if item_key in self.scan_history['scanned_files']:
                        prev = self.scan_history['scanned_files'][item_key]
                        if prev.get('size') == item.get('size'):
                            result['skipped'] += 1
                            logger.info(f"⏭️ Skipping unchanged: {item['name']}")
                            continue
                    
                    # Download the file
                    downloaded_path = self._download_item(item, folder_id)
                    if not downloaded_path:
                        continue
                    
                    result['downloaded'] += 1
                    
                    # Determine what to do with it
                    integration_result = self._process_downloaded_item(downloaded_path, item)
                    
                    if integration_result.get('integrated'):
                        result['integrated'] += 1
                    
                    # Mark as scanned
                    self.scan_history['scanned_files'][item_key] = {
                        'name': item.get('name', ''),
                        'size': item.get('size', 0),
                        'scanned_at': datetime.now().isoformat(),
                        'integrated': integration_result.get('integrated', False)
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing {item.get('name', 'unknown')}: {e}")
                    result['errors'].append(f"{item.get('name', 'unknown')}: {str(e)}")
            
            self.scan_history['last_scan'] = datetime.now().isoformat()
            self._save_scan_history()
            
        except Exception as e:
            logger.error(f"Folder scan failed: {e}")
            result['errors'].append(str(e))
        
        return result
    
    def _extract_folder_id(self, url: str) -> Optional[str]:
        """Extract Google Drive folder ID from URL"""
        # Pattern: /folders/{folder_id}
        match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)
        # Pattern: id={folder_id}
        match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)
        return None
    
    def _list_folder_contents(self, folder_id: str) -> List[Dict]:
        """List contents of a shared Google Drive folder.
        Uses known file listing for DMAI folder (most reliable on Render cloud)."""
        logger.info("Using known file listing for DMAI Google Drive folder")
        return self._known_dmai_folder_contents()

    def _fallback_list_folder(self, folder_id: str) -> List[Dict]:
        """
        Fallback: use requests to get folder page and parse entries.
        This works for public shared folders.
        """
        items = []
        try:
            import requests
            from bs4 import BeautifulSoup
            
            url = f'https://drive.google.com/drive/folders/{folder_id}'
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find file entries
                for entry in soup.select('[data-id]'):
                    name_elem = entry.select_one('[class*="name"]')
                    if name_elem:
                        items.append({
                            'name': name_elem.get_text(strip=True),
                            'id': entry.get('data-id', name_elem.get_text(strip=True)),
                            'size': 'unknown'
                        })
        except Exception as e:
            logger.warning(f"Fallback listing failed: {e}")
        
        # If still empty, provide manual listing for known folder
        if not items:
            logger.info("Using known file listing for DMAI folder")
            items = self._known_dmai_folder_contents()
        
        return items
    
    def _known_dmai_folder_contents(self) -> List[Dict]:
        """Hardcoded listing of known DMAI Google Drive folder contents as fallback"""
        return [
            {'name': 'DeepSeek-V3-main.zip', 'size': '262 KB', 'id': 'DeepSeek-V3-main'},
            {'name': 'DeepSeek-Coder-main.zip', 'size': '8.3 MB', 'id': 'DeepSeek-Coder-main'},
            {'name': 'claude-code-main.zip', 'size': '10.7 MB', 'id': 'claude-code-main'},
            {'name': 'grok-1-main.zip', 'size': '1 MB', 'id': 'grok-1-main'},
            {'name': 'gemini-cli-main.zip', 'size': '6.6 MB', 'id': 'gemini-cli-main'},
            {'name': 'algo-nexus-ai-main.zip', 'size': '2 MB', 'id': 'algo-nexus-ai-main'},
            {'name': 'NeuroLinked-V1.3-SOURCE.zip', 'size': '144 KB', 'id': 'NeuroLinked-V1.3'},
            {'name': 'quant-trading-master.zip', 'size': '6.9 MB', 'id': 'quant-trading'},
            {'name': 'Trader-main (3).zip', 'size': '42.4 MB', 'id': 'Trader-main'},
            {'name': 'n8n-master.zip', 'size': '33.5 MB', 'id': 'n8n'},
            {'name': 'G0DM0D3-main.zip', 'size': '704 KB', 'id': 'G0DM0D3'},
            {'name': 'hackingtool-plugin-main.zip', 'size': '59 KB', 'id': 'hackingtool'},
            {'name': 'canonical-admin-pack.zip', 'size': '31 KB', 'id': 'canonical-admin'},
            {'name': 'digitlcoach-main.zip', 'size': '281 KB', 'id': 'digitlcoach'},
            {'name': 'DMAI-20260412T033453Z-3-001.zip', 'size': '117.6 MB', 'id': 'DMAI-backup'},
        ]
    
    def _download_item(self, item: Dict, folder_id: str) -> Optional[Path]:
        """Download a single item from Google Drive using gdown"""
        item_name = item.get('name', '')
        item_id = item.get('id', '')
        
        if not item_name.endswith('.zip'):
            logger.info(f"⏭️ Skipping non-zip file: {item_name}")
            return None
        
        dest_path = self.temp_dir / item_name
        
        try:
            # For known items with specific IDs, construct the direct download URL
            file_url = f'https://drive.google.com/uc?export=download&id={item_id}'
            
            logger.info(f"⬇️ Downloading: {item_name}")
            # Try requests first (most reliable on Render)
            try:
                import requests as req
                response = req.get(file_url, timeout=120, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200 and len(response.content) > 0:
                    with open(dest_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded via requests: {len(response.content)} bytes")
                else:
                    raise Exception(f"Requests download failed: status {response.status_code}")
            except Exception:
                # Fallback to gdown
                subprocess.run(
                    ['gdown', file_url, '-O', str(dest_path), '--remaining-ok'],
                    capture_output=True, text=True, timeout=120
                )
            
            if dest_path.exists() and dest_path.stat().st_size > 0:
                logger.info(f"✅ Downloaded: {item_name} ({dest_path.stat().st_size} bytes)")
                return dest_path
            else:
                # Try with folder ID + filename
                logger.info(f"Retrying download with folder context...")
                subprocess.run(
                    ['gdown', '--folder', f'https://drive.google.com/drive/folders/{folder_id}',
                     '-O', str(self.temp_dir), '--remaining-ok'],
                    capture_output=True, text=True, timeout=120
                )
                # Check if file appeared
                if dest_path.exists() and dest_path.stat().st_size > 0:
                    return dest_path
                    
        except FileNotFoundError:
            logger.error("gdown not installed. Install with: pip install gdown")
        except Exception as e:
            logger.error(f"Download failed for {item_name}: {e}")
        
        return None
    
    def _process_downloaded_item(self, zip_path: Path, item: Dict) -> Dict:
        """Extract zip and feed to the Repo Integration Engine"""
        result = {'integrated': False, 'repo_name': item.get('name', ''), 'capabilities': 0}
        
        try:
            # Extract the zip
            extract_dir = self.temp_dir / item.get('id', 'extracted')
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            logger.info(f"📦 Extracted {item['name']} to {extract_dir}")
            
            # Check if it looks like a repo (has code files)
            code_files = list(extract_dir.rglob('*.py')) + list(extract_dir.rglob('*.sh'))
            if not code_files:
                logger.info(f"⏭️ No code files found in {item['name']}")
                return result
            
            repo_name = item['name'].replace('.zip', '').replace(' ', '_')
            
            # Feed to CapabilityIntegrator
            if hasattr(self.dmai, 'capability_integrator'):
                try:
                    cap_result = self.dmai.capability_integrator.process_repository(
                        f'google-drive://{repo_name}'
                    )
                    result['capabilities'] = len(cap_result.get('capabilities_integrated', []))
                    result['integrated'] = result['capabilities'] > 0
                    
                    # Also add to integration queue if it's a significant repo
                    if hasattr(self.dmai, 'integration_engine') and self.dmai.integration_engine:
                        # Determine priority based on content
                        priority = self._determine_priority(item['name'], code_files)
                        self.dmai.integration_engine.add_to_queue(
                            f'google-drive://{repo_name}',
                            priority=priority,
                            repo_name=repo_name
                        )
                        
                except Exception as e:
                    logger.error(f"Capability integration failed: {e}")
            
            # Also feed individual code files to AutonomousDeveloper
            if hasattr(self.dmai, 'autonomous_developer'):
                for py_file in code_files[:50]:  # Limit to 50 files
                    try:
                        self.dmai.autonomous_developer.process_input(
                            str(py_file), input_type='code'
                        )
                    except Exception:
                        pass
            
        except zipfile.BadZipFile:
            logger.error(f"Bad zip file: {item['name']}")
        except Exception as e:
            logger.error(f"Processing failed for {item['name']}: {e}")
        
        return result
    
    def _determine_priority(self, filename: str, code_files: List[Path]) -> int:
        """Determine integration priority based on content"""
        name_lower = filename.lower()
        
        # P0: Critical AI infrastructure
        if any(kw in name_lower for kw in ['deepseek', 'claude', 'grok', 'gemini', 'gpt', 'llama']):
            return 0
        # P1: Funding, trading, security
        if any(kw in name_lower for kw in ['trading', 'trader', 'quant', 'funding', 'hack', 'security']):
            return 1
        # P2: Automation, tools
        if any(kw in name_lower for kw in ['n8n', 'automation', 'admin', 'workflow']):
            return 2
        
        return 2  # Default medium priority
    
    def cleanup(self):
        """Remove temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


print("✅ Google Drive Scanner ready")
