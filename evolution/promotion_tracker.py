#!/usr/bin/env python3
"""
Promotion Tracker for DMAI Evolution System
Tracks successful evolutions and promotes versions to primary status
3+ successful merges → promoted to primary
"""

import os
import json
import time
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
log_dir = Path.home() / "Library/Logs/dmai"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - promotion - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'promotion_tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('promotion_tracker')

class PromotionTracker:
    """Tracks successful evolutions and promotes versions to primary status"""
    
    def __init__(self, data_dir: str = "data/evolution"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.evolution_successes: Dict[str, List[Dict]] = {}
        self.primary_versions: List[str] = []
        self.promotion_history: List[Dict] = []
        
        # Load existing data
        self.load_state()
        
        logger.info(f"📊 Promotion Tracker initialized with {len(self.primary_versions)} primary versions")
    
    def load_state(self):
        """Load saved promotion state"""
        state_file = self.data_dir / "promotion_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.evolution_successes = state.get('successes', {})
                    self.primary_versions = state.get('primaries', [])
                    self.promotion_history = state.get('history', [])
                logger.info(f"Loaded promotion state: {len(self.primary_versions)} primaries")
            except Exception as e:
                logger.error(f"Failed to load promotion state: {e}")
    
    def save_state(self):
        """Save current promotion state"""
        state_file = self.data_dir / "promotion_state.json"
        try:
            state = {
                'successes': self.evolution_successes,
                'primaries': self.primary_versions,
                'history': self.promotion_history[-100:],  # Keep last 100
                'timestamp': datetime.datetime.now().isoformat()
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save promotion state: {e}")
    
    def track_success(self, version_id: str, success_metrics: Dict[str, Any]):
        """Track a successful evolution"""
        if version_id not in self.evolution_successes:
            self.evolution_successes[version_id] = []
        
        success_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": success_metrics
        }
        
        self.evolution_successes[version_id].append(success_record)
        
        logger.info(f"✅ Tracked success for version {version_id} (total: {len(self.evolution_successes[version_id])})")
        
        # Check for promotion
        if len(self.evolution_successes[version_id]) >= 3:
            self.promote_to_primary(version_id)
        
        # Save state after tracking
        self.save_state()
    
    def promote_to_primary(self, version_id: str):
        """Promote a version to primary status"""
        if version_id in self.primary_versions:
            logger.info(f"Version {version_id} is already a primary")
            return False
        
        # Add to primary versions
        self.primary_versions.append(version_id)
        
        # Create promotion record
        promotion = {
            'version_id': version_id,
            'promoted_at': datetime.datetime.now().isoformat(),
            'success_count': len(self.evolution_successes.get(version_id, [])),
            'metrics': self.evolution_successes.get(version_id, [])[-3:],  # Last 3 successes
            'generation': len(self.primary_versions)
        }
        
        self.promotion_history.append(promotion)
        
        # Update primary config
        self.update_primary_config(version_id)
        
        logger.info(f"🎉 Version {version_id} PROMOTED to primary! (Generation {len(self.primary_versions)})")
        
        # Try to announce via voice if available
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from core_connector import voice_say
            voice_say(f"Version {version_id} promoted to primary after 3 successful evolutions")
        except:
            pass
        
        self.save_state()
        return True
    
    def update_primary_config(self, version_id: str):
        """Update configuration to use this as primary version"""
        config_file = self.data_dir / "primary_config.json"
        
        # Load existing config or create new
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except:
                config = {}
        else:
            config = {}
        
        # Update with new primary
        config['primary_version'] = version_id
        config['primary_since'] = datetime.datetime.now().isoformat()
        config['promotion_history'] = self.promotion_history
        config['total_primaries'] = len(self.primary_versions)
        
        # Save config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"⚙️ Updated primary config to version {version_id}")
    
    def get_version_status(self, version_id: str) -> Dict[str, Any]:
        """Get status for a specific version"""
        successes = self.evolution_successes.get(version_id, [])
        
        return {
            'version_id': version_id,
            'success_count': len(successes),
            'is_primary': version_id in self.primary_versions,
            'last_success': successes[-1] if successes else None,
            'success_history': successes[-5:] if successes else []  # Last 5 successes
        }
    
    def get_primary_versions(self) -> List[Dict[str, Any]]:
        """Get list of all primary versions with details"""
        primaries = []
        
        for version_id in self.primary_versions:
            status = self.get_version_status(version_id)
            
            # Find promotion record
            promotion = next(
                (p for p in self.promotion_history if p['version_id'] == version_id),
                None
            )
            
            primaries.append({
                **status,
                'promoted_at': promotion['promoted_at'] if promotion else None,
                'generation': self.primary_versions.index(version_id) + 1
            })
        
        return primaries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get promotion statistics"""
        return {
            'total_versions_tracked': len(self.evolution_successes),
            'total_primaries': len(self.primary_versions),
            'total_promotions': len(self.promotion_history),
            'recent_promotions': self.promotion_history[-5:] if self.promotion_history else [],
            'primary_versions': self.get_primary_versions(),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def reset_version(self, version_id: str):
        """Reset success count for a version (if it fails later)"""
        if version_id in self.evolution_successes:
            del self.evolution_successes[version_id]
            
            # If it was primary, demote it
            if version_id in self.primary_versions:
                self.primary_versions.remove(version_id)
                logger.warning(f"⚠️ Version {version_id} demoted from primary due to reset")
            
            self.save_state()
            logger.info(f"🔄 Reset success tracking for version {version_id}")
            return True
        return False


if __name__ == "__main__":
    # Test the promotion tracker
    tracker = PromotionTracker()
    
    print("📊 Promotion Tracker Test")
    print("=" * 50)
    
    # Test tracking successes
    test_versions = ['v1.0.1', 'v1.0.2', 'v1.1.0']
    
    for version in test_versions:
        for i in range(4):  # Track 4 successes for some versions
            tracker.track_success(version, {
                'test_score': 0.7 + (i * 0.05),
                'test_cycle': i + 1
            })
            print(f"✅ Tracked success {i+1} for {version}")
    
    # Get stats
    stats = tracker.get_stats()
    print("\n📈 Promotion Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Check specific version
    print("\n🔍 Version Status for v1.0.1:")
    print(json.dumps(tracker.get_version_status('v1.0.1'), indent=2))
    
    print("\n" + "=" * 50)
    print("✅ Promotion Tracker test complete")
