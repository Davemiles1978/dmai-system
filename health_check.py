#!/usr/bin/env python3
"""
Health check endpoint for Render.com
"""
import json
import sys
from pathlib import Path
from datetime import datetime

def health_status():
    """Return health status for Render"""
    try:
        # Import orchestrator
        sys.path.append('.')
        from agi_orchestrator import AGIOrchestrator
        
        # Check if running on Render
        data_path = Path('/var/data') if Path('/var/data').exists() else Path('./data')
        status_file = data_path / 'deploy_info.json'
        
        # Get orchestrator status
        orchestrator = AGIOrchestrator()
        status = orchestrator.get_status()
        
        # Add Render-specific info
        if status_file.exists():
            with open(status_file, 'r') as f:
                deploy_info = json.load(f)
        else:
            deploy_info = {'platform': 'unknown'}
        
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'generation': status['state']['generation'],
            'health_status': status['state']['health_status'],
            'active_capabilities': status['active_capabilities'],
            'platform': deploy_info.get('platform', 'local'),
            'uptime': str(datetime.now() - datetime.fromisoformat(deploy_info.get('deployed_at', datetime.now().isoformat()))) if 'deployed_at' in deploy_info else 'unknown'
        }
        
        # Exit codes for Render health checks
        if status['state']['health_status'] == 'running':
            print(json.dumps(health, indent=2))
            return 0
        else:
            print(json.dumps({**health, 'status': 'degraded'}, indent=2))
            return 1
            
    except Exception as e:
        print(json.dumps({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, indent=2))
        return 1

if __name__ == "__main__":
    sys.exit(health_status())
