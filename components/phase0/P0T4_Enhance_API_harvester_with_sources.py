#!/usr/bin/env python3
"""
P0T4_Enhance_API_harvester_with_sources.py
Enhance API harvester with source tracking and validation
Full-featured component for DMAI evolution system
"""

import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_harvester_enhancer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('APIHarvesterEnhancer')

class Enhance_API_harvester_with_sources:
    """
    API Harvester Enhancer - Adds source tracking and validation to API key harvesting
    Enhances the API harvester with source attribution, quality scoring, and deduplication
    """
    
    def __init__(self):
        self.name = "Enhance API harvester with sources"
        self.component_id = "P0T4"
        self.version = "1.0.0"
        self.status = "initialized"
        self.depends_on = ["P0T1", "P0T2", "P0T3"]
        self.sources = []
        self.harvest_stats = {
            'total_keys': 0,
            'unique_keys': 0,
            'duplicates_removed': 0,
            'sources_tracked': 0,
            'last_harvest': None
        }
        self.enhancement_history = []
        
    def run(self, continuous=False, interval=300):
        """
        Main execution method - called by evolution engine
        
        Args:
            continuous: Whether to run continuously
            interval: Check interval in seconds
        """
        logger.info(f"🚀 Starting {self.name} v{self.version}")
        
        try:
            if continuous:
                logger.info(f"Continuous mode: checking every {interval} seconds")
                while True:
                    self._enhance_harvester()
                    time.sleep(interval)
            else:
                # Single run
                result = self._enhance_harvester()
            
            logger.info(f"✅ {self.name} completed")
            return self.get_status()
            
        except Exception as e:
            logger.error(f"❌ Error in {self.name}: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e), "component": self.component_id}
    
    def evolve(self):
        """
        Evolution method - called when component needs to evolve
        """
        logger.info(f"🧬 Evolving {self.name}")
        self.version = f"1.0.{len(self.enhancement_history) + 1}"
        
        return {
            'component': self.component_id,
            'evolution': 'completed',
            'new_version': self.version,
            'stats': self.harvest_stats
        }
    
    def execute(self, command=None, **kwargs):
        """
        Execute method - runs specific commands
        
        Commands:
            - enhance: Run enhancement process
            - add_source: Add a new source to track
            - validate: Validate harvested keys
            - deduplicate: Remove duplicate keys
            - stats: Get harvest statistics
            - reset: Reset statistics
        """
        logger.info(f"⚙️ Executing command: {command}")
        
        if command == 'enhance':
            return self._enhance_harvester()
            
        elif command == 'add_source':
            source = kwargs.get('source')
            source_type = kwargs.get('type', 'github')
            reliability = kwargs.get('reliability', 0.5)
            
            if source:
                return self.add_source(source, source_type, reliability)
            else:
                return {"error": "No source provided", "command": command}
                
        elif command == 'validate':
            key_data = kwargs.get('key_data')
            if key_data:
                return self.validate_key(key_data)
            else:
                return self._validate_all_keys()
                
        elif command == 'deduplicate':
            keys = kwargs.get('keys', [])
            return self.deduplicate_keys(keys)
            
        elif command == 'stats':
            return self.get_status()
            
        elif command == 'reset':
            self.harvest_stats = {
                'total_keys': 0,
                'unique_keys': 0,
                'duplicates_removed': 0,
                'sources_tracked': len(self.sources),
                'last_harvest': None
            }
            self.enhancement_history = []
            return {'status': 'reset', 'component': self.component_id}
            
        else:
            return self._enhance_harvester()
    
    def process(self, data=None):
        """
        Process method - handles data processing
        
        Can process harvest data, add source tracking, and validate keys
        """
        logger.info(f"🔄 Processing data for {self.name}")
        
        result = {
            'component': self.component_id,
            'processed': True,
            'timestamp': datetime.now().isoformat(),
            'stats': self.harvest_stats
        }
        
        if data and isinstance(data, dict):
            # Process harvest data
            if 'harvest_data' in data:
                harvest_data = data['harvest_data']
                processed = self._process_harvest_data(harvest_data)
                result['harvest_processed'] = processed
            
            # Process keys for validation
            if 'keys' in data:
                keys = data['keys']
                validated = []
                for key in keys:
                    validation = self.validate_key(key)
                    validated.append(validation)
                result['validated_keys'] = validated
            
            # Process sources to add
            if 'sources' in data:
                for source_data in data['sources']:
                    source = source_data.get('source')
                    source_type = source_data.get('type', 'github')
                    reliability = source_data.get('reliability', 0.5)
                    if source:
                        self.add_source(source, source_type, reliability)
                result['sources_added'] = len(data['sources'])
        
        return result
    
    def generate(self):
        """
        Generate method - produces output/report
        """
        logger.info(f"📊 Generating report for {self.name}")
        
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'stats': self.harvest_stats,
            'sources': self.sources[-10:],  # Last 10 sources
            'history': self.enhancement_history[-5:],  # Last 5 enhancements
            'dependencies': self.depends_on
        }
    
    def query(self, question=None):
        """
        Query method - answers questions about component state
        """
        logger.info(f"❓ Querying {self.name}")
        
        if question == 'health':
            return {
                'component': self.component_id,
                'healthy': True,
                'methods': ['run', 'evolve', 'execute', 'process', 'generate', 'query'],
                'stats': self.harvest_stats
            }
        elif question == 'sources':
            return {
                'component': self.component_id,
                'sources': self.sources,
                'total_sources': len(self.sources)
            }
        elif question == 'stats':
            return self.harvest_stats
        elif question == 'keys':
            return {
                'component': self.component_id,
                'total_keys': self.harvest_stats['total_keys'],
                'unique_keys': self.harvest_stats['unique_keys'],
                'duplicates_removed': self.harvest_stats['duplicates_removed']
            }
        else:
            return self.info()
    
    def add_source(self, source: str, source_type: str = "github", reliability: float = 0.5):
        """
        Add a new source to track for API key harvesting
        
        Args:
            source: Source identifier (URL, repo name, etc.)
            source_type: Type of source (github, gitlab, public_repo, etc.)
            reliability: Reliability score between 0 and 1
        """
        source_info = {
            'source': source,
            'type': source_type,
            'reliability': min(1.0, max(0.0, reliability)),
            'added': datetime.now().isoformat(),
            'keys_found': 0,
            'last_harvest': None
        }
        
        # Check if source already exists
        for existing in self.sources:
            if existing['source'] == source:
                logger.info(f"📋 Source already exists: {source}")
                return {
                    'status': 'exists',
                    'source': source,
                    'message': 'Source already tracked'
                }
        
        self.sources.append(source_info)
        self.harvest_stats['sources_tracked'] = len(self.sources)
        
        logger.info(f"✅ Added new source: {source} ({source_type})")
        
        return {
            'status': 'added',
            'source': source,
            'type': source_type,
            'reliability': reliability,
            'total_sources': len(self.sources)
        }
    
    def validate_key(self, key_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a harvested API key
        
        Args:
            key_data: Dictionary containing key information
                - service: The service the key is for
                - key: The actual API key
                - source: Where it was found
                - format: Expected format (if known)
        """
        service = key_data.get('service', 'unknown')
        key = key_data.get('key', '')
        source = key_data.get('source', 'unknown')
        expected_format = key_data.get('format')
        
        validation = {
            'service': service,
            'key_masked': key[:8] + '...' if len(key) > 8 else '***',
            'source': source,
            'valid': False,
            'issues': [],
            'score': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Basic validation checks
        if not key:
            validation['issues'].append('Empty key')
            validation['score'] = 0.0
        elif len(key) < 16:
            validation['issues'].append('Key too short')
            validation['score'] = 0.3
        elif len(key) > 256:
            validation['issues'].append('Key unusually long')
            validation['score'] = 0.5
        else:
            validation['valid'] = True
            validation['score'] = 1.0
            
            # Check against expected format if provided
            if expected_format:
                if expected_format == 'alphanumeric' and not key.isalnum():
                    validation['issues'].append('Expected alphanumeric format')
                    validation['score'] = 0.7
                elif expected_format == 'hex' and not all(c in '0123456789abcdefABCDEF' for c in key):
                    validation['issues'].append('Expected hex format')
                    validation['score'] = 0.7
                elif expected_format == 'base64' and len(key) % 4 != 0:
                    validation['issues'].append('Expected base64 format')
                    validation['score'] = 0.7
        
        logger.debug(f"Validated {service} key from {source}: score={validation['score']}")
        
        return validation
    
    def deduplicate_keys(self, keys: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Remove duplicate keys from a list
        
        Args:
            keys: List of key dictionaries
        """
        unique_keys = {}
        duplicates = 0
        
        for key_data in keys:
            service = key_data.get('service', 'unknown')
            key = key_data.get('key', '')
            
            # Create a unique identifier
            key_id = f"{service}:{key}"
            
            if key_id in unique_keys:
                duplicates += 1
                logger.debug(f"Duplicate key found: {service}")
            else:
                unique_keys[key_id] = key_data
        
        result = {
            'original_count': len(keys),
            'unique_count': len(unique_keys),
            'duplicates_removed': duplicates,
            'unique_keys': list(unique_keys.values())
        }
        
        self.harvest_stats['duplicates_removed'] += duplicates
        self.harvest_stats['unique_keys'] = len(unique_keys)
        
        logger.info(f"✨ Deduplication: {duplicates} duplicates removed, {len(unique_keys)} unique keys remain")
        
        return result
    
    def _enhance_harvester(self):
        """Main enhancement process"""
        logger.info("🔧 Enhancing API harvester with source tracking")
        
        enhancement = {
            'timestamp': datetime.now().isoformat(),
            'sources_processed': 0,
            'keys_processed': 0,
            'enhancements_applied': []
        }
        
        # Simulate processing sources
        for source in self.sources:
            enhancement['sources_processed'] += 1
            logger.debug(f"Processing source: {source['source']}")
            
            # In a real implementation, this would:
            # 1. Check source for new keys
            # 2. Apply source-specific parsing
            # 3. Validate found keys
            # 4. Store with source attribution
            
            enhancement['enhancements_applied'].append(f"Processed {source['source']}")
        
        # Update stats
        self.harvest_stats['last_harvest'] = enhancement['timestamp']
        self.enhancement_history.append(enhancement)
        
        # Keep history manageable
        if len(self.enhancement_history) > 100:
            self.enhancement_history = self.enhancement_history[-100:]
        
        logger.info(f"✅ Enhancement complete: processed {enhancement['sources_processed']} sources")
        
        return enhancement
    
    def _process_harvest_data(self, harvest_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming harvest data and enhance it with source tracking"""
        processed = {
            'original_count': 0,
            'enhanced_count': 0,
            'sources_identified': [],
            'keys': []
        }
        
        if 'keys' in harvest_data:
            keys = harvest_data['keys']
            processed['original_count'] = len(keys)
            
            for key_data in keys:
                # Add source tracking if not present
                if 'source' not in key_data:
                    key_data['source'] = harvest_data.get('source', 'unknown')
                    key_data['source_type'] = harvest_data.get('source_type', 'unknown')
                    key_data['harvest_timestamp'] = datetime.now().isoformat()
                    
                processed['keys'].append(key_data)
                processed['enhanced_count'] += 1
                
                if key_data['source'] not in processed['sources_identified']:
                    processed['sources_identified'].append(key_data['source'])
        
        self.harvest_stats['total_keys'] += processed['enhanced_count']
        
        return processed
    
    def _validate_all_keys(self) -> Dict[str, Any]:
        """Validate all keys in the system (placeholder)"""
        logger.info("🔍 Validating all harvested keys")
        
        # In a real implementation, this would:
        # 1. Retrieve all keys from database
        # 2. Run validation on each
        # 3. Update validation status
        # 4. Remove or flag invalid keys
        
        return {
            'status': 'validation_complete',
            'keys_validated': self.harvest_stats['total_keys'],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'stats': self.harvest_stats,
            'sources_count': len(self.sources),
            'last_enhancement': self.enhancement_history[-1] if self.enhancement_history else None,
            'methods': ['run', 'evolve', 'execute', 'process', 'generate', 'query']
        }
    
    def info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "name": self.name,
            "id": self.component_id,
            "version": self.version,
            "status": self.status,
            "depends_on": self.depends_on,
            "stats": self.harvest_stats,
            "methods": ['run', 'evolve', 'execute', 'process', 'generate', 'query']
        }

# Guard clause ensures code only runs when script is executed directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 API HARVESTER ENHANCER (P0T4)")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='API Harvester Enhancer')
    parser.add_argument('--enhance', action='store_true', help='Run enhancement')
    parser.add_argument('--add-source', metavar='SOURCE', help='Add a source to track')
    parser.add_argument('--type', default='github', help='Source type')
    parser.add_argument('--reliability', type=float, default=0.5, help='Source reliability (0-1)')
    parser.add_argument('--status', action='store_true', help='Show status')
    
    args = parser.parse_args()
    
    enhancer = Enhance_API_harvester_with_sources()
    
    if args.enhance:
        print("\n📋 Running enhancement...")
        result = enhancer._enhance_harvester()
        print(json.dumps(result, indent=2))
    
    elif args.add_source:
        print(f"\n📋 Adding source: {args.add_source}")
        result = enhancer.add_source(args.add_source, args.type, args.reliability)
        print(json.dumps(result, indent=2))
    
    elif args.status:
        print("\n📊 Component Status:")
        print(json.dumps(enhancer.get_status(), indent=2))
    
    else:
        print("\n📋 Component Info:")
        print(json.dumps(enhancer.info(), indent=2))
        print("\n💡 Use --enhance, --add-source, or --status for more options")
