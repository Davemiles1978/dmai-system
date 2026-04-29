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
            
            # Download all zip files at once (more reliable)
            zip_items = [i for i in items if i.get('name', '').endswith('.zip')]
            downloaded_files = self._download_all_zips(folder_id, zip_items) if zip_items else {}
            result['downloaded'] = len(downloaded_files)
            
            # Process each downloaded file
            for item in items:
                try:
                    item_name = item.get('name', '')
                    if not item_name.endswith('.zip'):
                        continue
                    
                    # Skip if already scanned
                    item_key = item_name
                    if item_key in self.scan_history['scanned_files']:
                        prev = self.scan_history['scanned_files'][item_key]
                        if prev.get('size') == item.get('size'):
                            result['skipped'] += 1
                            continue
                    
                    downloaded_path = downloaded_files.get(item_name)
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
        """Hardcoded listing of DMAI Google Drive files with real file IDs"""
        return [
            # Individual files with real Google Drive file IDs
            {'name': 'automaton-main.zip', 'id': '16FR59wf_CuTCUl_0quNDs7PPAKbQ6p6V', 'size': 'unknown'},
            {'name': 'sky-reels-v2.zip', 'id': '1_lyFN70phgPGQEv83k9mxUcuUGBWNi4l', 'size': 'unknown'},
            {'name': 'ui-ux-pro-max-skill.zip', 'id': '1kyaJJvvwicQowdWilsDnW5U9BrRTY66p', 'size': 'unknown'},
            {'name': 'deepseek-v3.zip', 'id': '13at-8ujhxZJE6HTqXPSIWZUqSmrMBoza', 'size': 'unknown'},
            {'name': 'claude-system-specs.zip', 'id': '156F1OT2CN_TnX8eh9GQCjr3ih5dXX7BS', 'size': 'unknown'},
            {'name': 'dmai-knowledge-pack.zip', 'id': '1jmQtdJ8qYP1SXWY_osmVFcjlxi1Uj76s', 'size': 'unknown'},
            {'name': 'ai-models-pack.zip', 'id': '1vC4Yhyd3qpP6ENWk7sSe6hpvOJhB7IA4', 'size': 'unknown'},
            {'name': 'dev-tools-pack.zip', 'id': '1PeUd59LIXh6heRdDBbSTWEu2SPTWbWVa', 'size': 'unknown'},
            {'name': 'security-tools.zip', 'id': '1_eNIoorfSuFcoRuA0jgSSs_zHXJ95NOF', 'size': 'unknown'},
            {'name': 'data-analysis.zip', 'id': '1T3ei4V9zVfX-V2oBOECtHApN_xH_zlKK', 'size': 'unknown'},
            {'name': 'content-creation.zip', 'id': '1AxhQ-4P0KqsROIX7JJI8NpAqSvOuiWbw', 'size': 'unknown'},
            {'name': 'business-tools.zip', 'id': '1cfW32RO3L5VVkwozC15Cwhwa9cIxr3JB', 'size': 'unknown'},
            {'name': 'learning-resources.zip', 'id': '1kVN2zF1dueTnv2T3PJUIgIfAr_AWIe8w', 'size': 'unknown'},
            {'name': 'api-integrations.zip', 'id': '1JC80qRd3zJ2WFMSjHM4SccG1futPtdDk', 'size': 'unknown'},
            {'name': 'misc-scripts.zip', 'id': '1CoPnhgsPh6Zp9sqU_8vo6KhFTjw-Ox47', 'size': 'unknown'},
            # Folder-based items (download as whole folders)
            {'name': 'grok-main', 'id': '1US8Uc8zvuL6Lo9l7Z04RyHishp6veaye', 'size': 'folder', 'is_folder': True},
            {'name': 'algo-nexus-ai-main', 'id': '1ZRPxtRtkP62iykIKeqPI_5Dzc7F1QcIS', 'size': 'folder', 'is_folder': True},
            {'name': 'HeyGen-assets', 'id': '1bTglbds35t5ZeBNDftSgBOLT9iozyb9K', 'size': 'folder', 'is_folder': True},
        ]
    
    def _download_all_zips(self, folder_id: str, known_items: List[Dict]) -> Dict[str, Path]:
        """Download all zip files from Google Drive.
        Handles Google's confirmation page for large files."""
        downloaded = {}
        
        import requests as req
        
        for item in known_items:
            item_name = item.get('name', '')
            item_id = item.get('id', '')
            
            if not item_name.endswith('.zip') and not item.get('is_folder'):
                continue
            
            try:
                logger.info(f"⬇️ Downloading: {item_name}")
                
                # Session with cookies to handle Google's confirmation flow
                session = req.Session()
                session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
                
                if item.get('is_folder'):
                    file_url = f'https://drive.google.com/drive/folders/{item_id}'
                else:
                    file_url = f'https://drive.google.com/uc?export=download&id={item_id}'
                
                # First request - may get confirmation page for large files
                response = session.get(file_url, timeout=30, allow_redirects=True)
                
                # Check if we hit the virus scan confirmation page
                if 'confirm=' in response.url or 'confirm=' in response.text[:2000]:
                    # Extract confirmation token
                    import re
                    confirm_match = re.search(r'confirm=([0-9A-Za-z_\-]+)', response.text)
                    if confirm_match:
                        confirm_token = confirm_match.group(1)
                        logger.info(f"Got confirmation token, retrying...")
                        file_url += f'&confirm={confirm_token}'
                        response = session.get(file_url, timeout=120)
                
                # Check if we got actual content
                content_type = response.headers.get('Content-Type', '')
                if response.status_code == 200 and len(response.content) > 500:
                    # Verify it's not HTML
                    if 'text/html' not in content_type or len(response.content) > 10000:
                        dest_path = self.temp_dir / item_name
                        with open(dest_path, 'wb') as f:
                            f.write(response.content)
                        downloaded[item_name] = dest_path
                        logger.info(f"✅ Downloaded: {item_name} ({len(response.content)} bytes)")
                    else:
                        logger.warning(f"Got HTML instead of file for {item_name}")
                else:
                    logger.warning(f"Download failed for {item_name}: status {response.status_code}, size {len(response.content)}")
                    
            except Exception as e:
                logger.error(f"Download error for {item_name}: {e}")
        
        # Also try gdown as fallback if nothing downloaded
        if not downloaded:
            try:
                logger.info("Trying gdown as fallback...")
                subprocess.run(
                    ['python3', '-m', 'pip', 'install', 'gdown'],
                    capture_output=True, timeout=30
                )
                result = subprocess.run(
                    ['gdown', '--folder', f'https://drive.google.com/drive/folders/{folder_id}',
                     '-O', str(self.temp_dir), '--remaining-ok'],
                    capture_output=True, text=True, timeout=300
                )
                # Check what showed up
                for item in known_items:
                    item_name = item.get('name', '')
                    dest_path = self.temp_dir / item_name
                    if dest_path.exists() and dest_path.stat().st_size > 0:
                        downloaded[item_name] = dest_path
                        logger.info(f"✅ gdown downloaded: {item_name}")
            except Exception as e:
                logger.warning(f"gdown fallback also failed: {e}")
        
        return downloaded

    def _download_item(self, item: Dict, folder_id: str) -> Optional[Path]:
        """Download a single item from Google Drive"""
        item_name = item.get('name', '')
        item_id = item.get('id', '')
        
        is_folder = item.get('is_folder', False)
        if not item_name.endswith('.zip') and not is_folder:
            logger.info(f"⏭️ Skipping: {item_name}")
            return None
        
        dest_path = self.temp_dir / item_name
        
        try:
            if item.get('is_folder'):
                file_url = f'https://drive.google.com/drive/folders/{item_id}'
            else:
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
