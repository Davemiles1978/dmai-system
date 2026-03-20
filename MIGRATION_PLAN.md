# DMAI Services Migration to dmai-production

## Services to Migrate from Other Workspace

### 1. agi-evolution-system (WORKING)
- Type: Background Worker
- Build Command: pip install -r requirements.txt
- Start Command: python cloud_launcher.py
- Env Vars: (check in Render dashboard)
- Disk: /var/data (persistent)

### 2. DMAI-Evolution (Failed deploy - we'll fix fresh)
- Type: Background Worker
- Build Command: pip install -r requirements.txt
- Start Command: python evolution/cloud_evolution.py

### 3. dmai-cloud-evolution (WORKING)
- Type: Web Service
- Build Command: pip install -r requirements.txt
- Start Command: python cloud_web_ui.py

### 4. dmai-cloud-ui (Failed deploy)
- Type: Web Service
- Build Command: npm install (if Node) or pip install (if Python)

### 5. dmai-harvester-db (Already in dmai-production!)
- Already there - good!

## Migration Steps
1. Copy env vars from each service
2. Recreate in dmai-production with same settings
3. Test each service
4. Once verified, delete from old workspace
