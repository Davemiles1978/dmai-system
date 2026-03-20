#!/usr/bin/env python3
"""
P1T3_Create_identity_persona.py
Identity Persona Creator - Manages digital identities and personas for DMAI
Full-featured component for DMAI evolution system
"""

import os
import sys
import json
import time
import logging
import traceback
import hashlib
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('identity_persona.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('IdentityPersona')

class PersonaStatus(Enum):
    """Persona lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"
    COMPROMISED = "compromised"
    ROTATING = "rotating"

class IdentityType(Enum):
    """Types of identities"""
    HUMAN = "human"
    BOT = "bot"
    SERVICE = "service"
    API = "api"
    ANONYMOUS = "anonymous"
    TEMPORARY = "temporary"

class Create_identity_persona:
    """
    Identity Persona Creator - Creates and manages digital identities for DMAI
    Handles persona generation, rotation, and lifecycle management
    """
    
    def __init__(self):
        self.name = "Create identity persona"
        self.component_id = "P1T3"
        self.version = "1.0.0"
        self.status = "initialized"
        self.depends_on = ["P1T1", "P1T2"]
        
        # Identity storage
        self.identities = {}
        self.active_identities = {}
        self.retired_identities = []
        self.identity_pools = {
            'human': [],
            'bot': [],
            'service': [],
            'api': [],
            'anonymous': [],
            'temporary': []
        }
        
        # Statistics
        self.stats = {
            'total_identities': 0,
            'active_count': 0,
            'retired_count': 0,
            'compromised_count': 0,
            'rotations_performed': 0,
            'avg_lifetime': 0,
            'last_rotation': None,
            'pool_sizes': {k: 0 for k in self.identity_pools.keys()}
        }
        
        # Configuration
        self.rotation_schedule = {
            'human': timedelta(days=90),
            'bot': timedelta(days=30),
            'service': timedelta(days=365),
            'api': timedelta(days=180),
            'anonymous': timedelta(days=7),
            'temporary': timedelta(hours=24)
        }
        
        self.max_identities_per_type = {
            'human': 10,
            'bot': 50,
            'service': 25,
            'api': 100,
            'anonymous': 20,
            'temporary': 200
        }
        
        # Templates for different identity types
        self.templates = self._load_templates()
        
        # Identity attributes tracking
        self.attribute_history = []
        
    def run(self, continuous=False, interval=3600):
        """
        Main execution method - called by evolution engine
        
        Args:
            continuous: Whether to run continuously
            interval: Check interval in seconds (default 1 hour)
        """
        logger.info(f"🚀 Starting {self.name} v{self.version}")
        
        try:
            if continuous:
                logger.info(f"Continuous mode: checking every {interval} seconds")
                while True:
                    self._maintenance_cycle()
                    time.sleep(interval)
            else:
                # Single run
                result = self._maintenance_cycle()
            
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
        self.version = f"1.0.{len(self.identities) + 1}"
        
        # Evolve templates based on usage patterns
        evolved_templates = []
        for identity_type in IdentityType:
            if random.random() < 0.3:  # 30% chance to evolve each template
                self._evolve_template(identity_type.value)
                evolved_templates.append(identity_type.value)
        
        return {
            'component': self.component_id,
            'evolution': 'completed',
            'new_version': self.version,
            'evolved_templates': evolved_templates,
            'stats': self.stats
        }
    
    def execute(self, command=None, **kwargs):
        """
        Execute method - runs specific commands
        
        Commands:
            - create: Create new identity
            - activate: Activate identity
            - suspend: Suspend identity
            - retire: Retire identity
            - rotate: Rotate identities
            - validate: Validate identity
            - list: List identities
            - generate: Generate identity attributes
        """
        logger.info(f"⚙️ Executing command: {command}")
        
        if command == 'create':
            identity_type = kwargs.get('type', 'anonymous')
            name = kwargs.get('name')
            attributes = kwargs.get('attributes', {})
            return self.create_identity(identity_type, name, attributes)
            
        elif command == 'activate':
            identity_id = kwargs.get('identity_id')
            if identity_id:
                return self.activate_identity(identity_id)
            return {"error": "No identity_id provided"}
            
        elif command == 'suspend':
            identity_id = kwargs.get('identity_id')
            reason = kwargs.get('reason', 'unknown')
            if identity_id:
                return self.suspend_identity(identity_id, reason)
            return {"error": "No identity_id provided"}
            
        elif command == 'retire':
            identity_id = kwargs.get('identity_id')
            reason = kwargs.get('reason', 'end_of_life')
            if identity_id:
                return self.retire_identity(identity_id, reason)
            return {"error": "No identity_id provided"}
            
        elif command == 'rotate':
            identity_type = kwargs.get('type')
            if identity_type:
                return self.rotate_identities(identity_type)
            return self.rotate_all()
            
        elif command == 'validate':
            identity_id = kwargs.get('identity_id')
            if identity_id:
                return self.validate_identity(identity_id)
            return {"error": "No identity_id provided"}
            
        elif command == 'list':
            identity_type = kwargs.get('type')
            status = kwargs.get('status')
            return self.list_identities(identity_type, status)
            
        elif command == 'generate':
            identity_type = kwargs.get('type', 'anonymous')
            template = kwargs.get('template', 'default')
            return self.generate_identity_attributes(identity_type, template)
            
        elif command == 'pool':
            identity_type = kwargs.get('type')
            if identity_type:
                return self.get_pool_status(identity_type)
            return self.get_all_pools()
            
        elif command == 'stats':
            return self.get_status()
            
        elif command == 'reset':
            self.identities = {}
            self.active_identities = {}
            self.retired_identities = []
            self.identity_pools = {k: [] for k in self.identity_pools.keys()}
            self.stats = {
                'total_identities': 0,
                'active_count': 0,
                'retired_count': 0,
                'compromised_count': 0,
                'rotations_performed': 0,
                'avg_lifetime': 0,
                'last_rotation': None,
                'pool_sizes': {k: 0 for k in self.identity_pools.keys()}
            }
            return {'status': 'reset', 'component': self.component_id}
            
        else:
            return self.get_status()
    
    def process(self, data=None):
        """
        Process method - handles data processing
        
        Can process identity requests, attribute updates, and batch operations
        """
        logger.info(f"🔄 Processing data for {self.name}")
        
        result = {
            'component': self.component_id,
            'processed': True,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }
        
        if data and isinstance(data, dict):
            # Process identity creation requests
            if 'create_identities' in data:
                identities = data['create_identities']
                created = []
                for identity_data in identities:
                    identity = self.create_identity(
                        identity_data.get('type', 'anonymous'),
                        identity_data.get('name'),
                        identity_data.get('attributes', {})
                    )
                    created.append(identity)
                result['identities_created'] = created
            
            # Process activation requests
            if 'activate' in data:
                identity_ids = data['activate']
                activated = []
                for identity_id in identity_ids:
                    activated.append(self.activate_identity(identity_id))
                result['identities_activated'] = activated
            
            # Process rotation requests
            if 'rotate' in data:
                rotate_data = data['rotate']
                if isinstance(rotate_data, list):
                    for identity_type in rotate_data:
                        self.rotate_identities(identity_type)
                elif isinstance(rotate_data, str):
                    self.rotate_identities(rotate_data)
                result['rotation_completed'] = True
            
            # Process attribute updates
            if 'update_attributes' in data:
                updates = data['update_attributes']
                for identity_id, attributes in updates.items():
                    self._update_identity_attributes(identity_id, attributes)
                result['attributes_updated'] = len(updates)
        
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
            'stats': self.stats,
            'pool_status': self.get_all_pools(),
            'active_identities': len(self.active_identities),
            'recent_rotations': self.stats['rotations_performed'],
            'dependencies': self.depends_on,
            'templates': list(self.templates.keys())
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
                'stats': self.stats,
                'active_identities': len(self.active_identities)
            }
        elif question == 'identities':
            return {
                'component': self.component_id,
                'total': self.stats['total_identities'],
                'active': self.stats['active_count'],
                'by_type': {k: len(v) for k, v in self.identity_pools.items()}
            }
        elif question == 'pool':
            return self.get_all_pools()
        elif question == 'rotation':
            return {
                'component': self.component_id,
                'last_rotation': self.stats['last_rotation'],
                'rotations_performed': self.stats['rotations_performed'],
                'schedule': self.rotation_schedule
            }
        elif question == 'templates':
            return {
                'component': self.component_id,
                'templates': list(self.templates.keys()),
                'template_details': self.templates
            }
        else:
            return self.info()
    
    def create_identity(self, identity_type: str, name: str = None, 
                       attributes: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new identity persona
        
        Args:
            identity_type: Type of identity (human, bot, service, api, anonymous, temporary)
            name: Optional name for the identity
            attributes: Optional additional attributes
        """
        logger.info(f"🆕 Creating new {identity_type} identity")
        
        # Validate type
        if identity_type not in [t.value for t in IdentityType]:
            identity_type = IdentityType.ANONYMOUS.value
        
        # Check pool limits
        if len(self.identity_pools[identity_type]) >= self.max_identities_per_type[identity_type]:
            logger.warning(f"⚠️ Pool limit reached for {identity_type}, rotating oldest")
            self.rotate_identities(identity_type)
        
        # Generate identity ID
        identity_id = hashlib.sha256(f"{identity_type}{time.time()}{random.random()}".encode()).hexdigest()[:16]
        
        # Generate name if not provided
        if not name:
            name = self._generate_name(identity_type)
        
        # Generate attributes
        if not attributes:
            attributes = self.generate_identity_attributes(identity_type)
        
        # Create identity
        identity = {
            'id': identity_id,
            'name': name,
            'type': identity_type,
            'status': PersonaStatus.DRAFT.value,
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'expires_at': (datetime.now() + self.rotation_schedule[identity_type]).isoformat(),
            'attributes': attributes,
            'usage_count': 0,
            'metadata': {
                'created_by': self.component_id,
                'version': self.version,
                'tags': []
            }
        }
        
        # Store identity
        self.identities[identity_id] = identity
        self.identity_pools[identity_type].append(identity_id)
        
        # Update stats
        self.stats['total_identities'] += 1
        self.stats['pool_sizes'][identity_type] = len(self.identity_pools[identity_type])
        
        logger.info(f"✅ Created identity {name} ({identity_id[:8]}...)")
        
        return identity
    
    def activate_identity(self, identity_id: str) -> Dict[str, Any]:
        """
        Activate an identity
        
        Args:
            identity_id: ID of identity to activate
        """
        logger.info(f"🔓 Activating identity: {identity_id[:8]}...")
        
        if identity_id not in self.identities:
            return {'error': f'Identity {identity_id} not found'}
        
        identity = self.identities[identity_id]
        
        if identity['status'] == PersonaStatus.ACTIVE.value:
            return {'message': 'Identity already active', 'identity': identity}
        
        # Update status
        old_status = identity['status']
        identity['status'] = PersonaStatus.ACTIVE.value
        identity['activated_at'] = datetime.now().isoformat()
        
        # Add to active identities
        self.active_identities[identity_id] = identity
        
        # Update stats
        self.stats['active_count'] = len(self.active_identities)
        
        logger.info(f"✅ Activated identity {identity['name']} (was {old_status})")
        
        return {
            'identity_id': identity_id,
            'name': identity['name'],
            'status': identity['status'],
            'activated_at': identity['activated_at']
        }
    
    def suspend_identity(self, identity_id: str, reason: str = 'unknown') -> Dict[str, Any]:
        """
        Suspend an identity
        
        Args:
            identity_id: ID of identity to suspend
            reason: Reason for suspension
        """
        logger.info(f"⏸️ Suspending identity: {identity_id[:8]}... (reason: {reason})")
        
        if identity_id not in self.identities:
            return {'error': f'Identity {identity_id} not found'}
        
        identity = self.identities[identity_id]
        
        if identity['status'] == PersonaStatus.SUSPENDED.value:
            return {'message': 'Identity already suspended', 'identity': identity}
        
        # Update status
        old_status = identity['status']
        identity['status'] = PersonaStatus.SUSPENDED.value
        identity['suspended_at'] = datetime.now().isoformat()
        identity['suspension_reason'] = reason
        
        # Remove from active if present
        if identity_id in self.active_identities:
            del self.active_identities[identity_id]
        
        # Update stats
        self.stats['active_count'] = len(self.active_identities)
        
        logger.info(f"✅ Suspended identity {identity['name']} (was {old_status})")
        
        return {
            'identity_id': identity_id,
            'name': identity['name'],
            'status': identity['status'],
            'suspended_at': identity['suspended_at'],
            'reason': reason
        }
    
    def retire_identity(self, identity_id: str, reason: str = 'end_of_life') -> Dict[str, Any]:
        """
        Retire an identity permanently
        
        Args:
            identity_id: ID of identity to retire
            reason: Reason for retirement
        """
        logger.info(f"♻️ Retiring identity: {identity_id[:8]}... (reason: {reason})")
        
        if identity_id not in self.identities:
            return {'error': f'Identity {identity_id} not found'}
        
        identity = self.identities[identity_id]
        
        # Update status
        old_status = identity['status']
        identity['status'] = PersonaStatus.RETIRED.value
        identity['retired_at'] = datetime.now().isoformat()
        identity['retirement_reason'] = reason
        
        # Remove from active and pool
        if identity_id in self.active_identities:
            del self.active_identities[identity_id]
        
        if identity_id in self.identity_pools[identity['type']]:
            self.identity_pools[identity['type']].remove(identity_id)
        
        # Add to retired list
        self.retired_identities.append(identity)
        
        # Update stats
        self.stats['active_count'] = len(self.active_identities)
        self.stats['retired_count'] = len(self.retired_identities)
        self.stats['pool_sizes'][identity['type']] = len(self.identity_pools[identity['type']])
        
        logger.info(f"✅ Retired identity {identity['name']} (was {old_status})")
        
        return {
            'identity_id': identity_id,
            'name': identity['name'],
            'status': identity['status'],
            'retired_at': identity['retired_at'],
            'reason': reason
        }
    
    def rotate_identities(self, identity_type: str = None) -> Dict[str, Any]:
        """
        Rotate identities of a specific type
        
        Args:
            identity_type: Type of identities to rotate (None for all)
        """
        logger.info(f"🔄 Rotating identities{f' of type: {identity_type}' if identity_type else ' (all)'}")
        
        rotated = []
        
        if identity_type:
            types_to_rotate = [identity_type]
        else:
            types_to_rotate = list(self.identity_pools.keys())
        
        for id_type in types_to_rotate:
            # Find identities needing rotation
            to_rotate = []
            for identity_id in self.identity_pools[id_type]:
                identity = self.identities[identity_id]
                expires_at = datetime.fromisoformat(identity['expires_at'])
                if datetime.now() > expires_at:
                    to_rotate.append(identity)
            
            # Rotate each identity
            for identity in to_rotate:
                # Retire old identity
                self.retire_identity(identity['id'], 'rotation')
                
                # Create new identity of same type
                new_identity = self.create_identity(
                    identity['type'],
                    attributes=identity.get('attributes', {})
                )
                
                # Activate if old was active
                if identity['status'] == PersonaStatus.ACTIVE.value:
                    self.activate_identity(new_identity['id'])
                
                rotated.append({
                    'old_id': identity['id'],
                    'new_id': new_identity['id'],
                    'type': identity['type'],
                    'name': new_identity['name']
                })
            
            if to_rotate:
                logger.info(f"   Rotated {len(to_rotate)} {id_type} identities")
        
        # Update stats
        self.stats['rotations_performed'] += len(rotated)
        self.stats['last_rotation'] = datetime.now().isoformat()
        
        return {
            'rotated_count': len(rotated),
            'rotated': rotated,
            'timestamp': self.stats['last_rotation']
        }
    
    def rotate_all(self) -> Dict[str, Any]:
        """Rotate all expired identities"""
        return self.rotate_identities()
    
    def validate_identity(self, identity_id: str) -> Dict[str, Any]:
        """
        Validate an identity (check if still valid)
        
        Args:
            identity_id: ID of identity to validate
        """
        if identity_id not in self.identities:
            return {'error': f'Identity {identity_id} not found', 'valid': False}
        
        identity = self.identities[identity_id]
        
        # Check if retired
        if identity['status'] == PersonaStatus.RETIRED.value:
            return {
                'identity_id': identity_id,
                'valid': False,
                'reason': 'retired',
                'retired_at': identity.get('retired_at')
            }
        
        # Check if compromised
        if identity['status'] == PersonaStatus.COMPROMISED.value:
            return {
                'identity_id': identity_id,
                'valid': False,
                'reason': 'compromised',
                'compromised_at': identity.get('compromised_at')
            }
        
        # Check expiration
        expires_at = datetime.fromisoformat(identity['expires_at'])
        if datetime.now() > expires_at:
            return {
                'identity_id': identity_id,
                'valid': False,
                'reason': 'expired',
                'expired_at': expires_at.isoformat()
            }
        
        return {
            'identity_id': identity_id,
            'valid': True,
            'status': identity['status'],
            'expires_at': identity['expires_at']
        }
    
    def list_identities(self, identity_type: str = None, status: str = None) -> List[Dict[str, Any]]:
        """
        List identities, optionally filtered
        
        Args:
            identity_type: Filter by type
            status: Filter by status
        """
        results = []
        
        for identity_id, identity in self.identities.items():
            if identity_type and identity['type'] != identity_type:
                continue
            if status and identity['status'] != status:
                continue
            
            # Return a copy without sensitive data
            results.append({
                'id': identity['id'],
                'name': identity['name'],
                'type': identity['type'],
                'status': identity['status'],
                'created_at': identity['created_at'],
                'expires_at': identity['expires_at'],
                'usage_count': identity['usage_count']
            })
        
        return results
    
    def generate_identity_attributes(self, identity_type: str, template: str = 'default') -> Dict[str, Any]:
        """
        Generate attributes for a new identity
        
        Args:
            identity_type: Type of identity
            template: Template name to use
        """
        if identity_type not in self.templates:
            identity_type = IdentityType.ANONYMOUS.value
        
        if template not in self.templates[identity_type]:
            template = 'default'
        
        template_data = self.templates[identity_type][template]
        attributes = {}
        
        for attr_name, attr_config in template_data.items():
            if attr_config.get('type') == 'random':
                attributes[attr_name] = self._generate_random_value(attr_config)
            elif attr_config.get('type') == 'choice':
                attributes[attr_name] = random.choice(attr_config['options'])
            elif attr_config.get('type') == 'fixed':
                attributes[attr_name] = attr_config['value']
            elif attr_config.get('type') == 'pattern':
                attributes[attr_name] = self._generate_pattern(attr_config['pattern'])
        
        return attributes
    
    def get_pool_status(self, identity_type: str) -> Dict[str, Any]:
        """Get status of a specific identity pool"""
        if identity_type not in self.identity_pools:
            return {'error': f'Unknown identity type: {identity_type}'}
        
        pool = self.identity_pools[identity_type]
        active_in_pool = [i for i in pool if i in self.active_identities]
        
        return {
            'type': identity_type,
            'total': len(pool),
            'active': len(active_in_pool),
            'available': len(pool) - len(active_in_pool),
            'max_capacity': self.max_identities_per_type[identity_type],
            'identities': pool[:10]  # First 10 IDs
        }
    
    def get_all_pools(self) -> Dict[str, Any]:
        """Get status of all identity pools"""
        pools = {}
        for identity_type in self.identity_pools.keys():
            pools[identity_type] = self.get_pool_status(identity_type)
        return pools
    
    def _maintenance_cycle(self):
        """Run maintenance tasks"""
        logger.info("🔧 Running identity maintenance cycle")
        
        # Check for expired identities
        expired = []
        for identity_id, identity in self.identities.items():
            if identity['status'] == PersonaStatus.ACTIVE.value:
                expires_at = datetime.fromisoformat(identity['expires_at'])
                if datetime.now() > expires_at:
                    expired.append(identity)
        
        if expired:
            logger.info(f"⚠️ Found {len(expired)} expired identities")
            for identity in expired:
                self.retire_identity(identity['id'], 'expired')
        
        # Rotate if needed
        self.rotate_all()
        
        # Update average lifetime
        if self.retired_identities:
            total_lifetime = 0
            for identity in self.retired_identities:
                created = datetime.fromisoformat(identity['created_at'])
                retired = datetime.fromisoformat(identity.get('retired_at', datetime.now().isoformat()))
                lifetime = (retired - created).total_seconds() / 86400  # in days
                total_lifetime += lifetime
            
            self.stats['avg_lifetime'] = total_lifetime / len(self.retired_identities)
        
        logger.info(f"✅ Maintenance complete. Active: {self.stats['active_count']}")
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load identity templates"""
        return {
            'human': {
                'default': {
                    'first_name': {'type': 'choice', 'options': ['John', 'Jane', 'Alex', 'Sam', 'Chris', 'Pat', 'Taylor', 'Jordan']},
                    'last_name': {'type': 'choice', 'options': ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']},
                    'age': {'type': 'random', 'min': 25, 'max': 65, 'distribution': 'normal'},
                    'interests': {'type': 'random', 'count': 3, 'options': ['tech', 'music', 'sports', 'art', 'science', 'literature', 'gaming']},
                    'personality': {'type': 'choice', 'options': ['analytical', 'creative', 'social', 'reserved', 'adventurous']}
                },
                'professional': {
                    'title': {'type': 'choice', 'options': ['Engineer', 'Manager', 'Director', 'Consultant', 'Analyst']},
                    'industry': {'type': 'choice', 'options': ['Tech', 'Finance', 'Healthcare', 'Education', 'Manufacturing']},
                    'experience_years': {'type': 'random', 'min': 2, 'max': 20}
                }
            },
            'bot': {
                'default': {
                    'purpose': {'type': 'choice', 'options': ['crawler', 'monitor', 'assistant', 'scraper', 'automation']},
                    'behavior': {'type': 'choice', 'options': ['polite', 'aggressive', 'stealth', 'normal']},
                    'rate_limit': {'type': 'random', 'min': 1, 'max': 100}
                }
            },
            'service': {
                'default': {
                    'service_name': {'type': 'pattern', 'pattern': 'service-{random:4}'},
                    'version': {'type': 'fixed', 'value': '1.0.0'},
                    'environment': {'type': 'choice', 'options': ['prod', 'staging', 'dev', 'test']}
                }
            },
            'api': {
                'default': {
                    'api_name': {'type': 'pattern', 'pattern': 'api-{random:6}'},
                    'version': {'type': 'pattern', 'pattern': 'v{random:1}.{random:1}.{random:1}'},
                    'rate_limit': {'type': 'random', 'min': 10, 'max': 1000}
                }
            },
            'anonymous': {
                'default': {
                    'session_id': {'type': 'pattern', 'pattern': 'sess-{hex:8}'},
                    'user_agent': {'type': 'choice', 'options': ['mobile', 'desktop', 'tablet', 'bot']},
                    'locale': {'type': 'choice', 'options': ['en-US', 'en-GB', 'es-ES', 'fr-FR', 'de-DE']}
                }
            },
            'temporary': {
                'default': {
                    'purpose': {'type': 'choice', 'options': ['test', 'temp_access', 'one_time', 'emergency']},
                    'ttl_hours': {'type': 'random', 'min': 1, 'max': 24}
                }
            }
        }
    
    def _generate_name(self, identity_type: str) -> str:
        """Generate a name for the identity"""
        prefixes = {
            'human': ['user', 'person', 'individual'],
            'bot': ['bot', 'crawler', 'agent'],
            'service': ['svc', 'service', 'app'],
            'api': ['api', 'endpoint', 'gateway'],
            'anonymous': ['anon', 'guest', 'visitor'],
            'temporary': ['temp', 'tmp', 'session']
        }
        
        prefix = random.choice(prefixes.get(identity_type, ['id']))
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        
        return f"{prefix}-{suffix}"
    
    def _generate_random_value(self, config: Dict[str, Any]) -> Any:
        """Generate a random value based on config"""
        if config.get('distribution') == 'normal':
            mean = (config['min'] + config['max']) / 2
            std = (config['max'] - config['min']) / 6
            value = int(random.gauss(mean, std))
            return max(config['min'], min(config['max'], value))
        elif 'count' in config:
            return random.sample(config['options'], min(config['count'], len(config['options'])))
        else:
            return random.randint(config['min'], config['max'])
    
    def _generate_pattern(self, pattern: str) -> str:
        """Generate a string based on a pattern"""
        result = []
        i = 0
        while i < len(pattern):
            if pattern[i] == '{':
                j = pattern.find('}', i)
                if j != -1:
                    cmd = pattern[i+1:j]
                    if cmd.startswith('random:'):
                        length = int(cmd.split(':')[1])
                        result.append(''.join(random.choices(string.ascii_lowercase + string.digits, k=length)))
                    elif cmd.startswith('hex:'):
                        length = int(cmd.split(':')[1])
                        result.append(''.join(random.choices('0123456789abcdef', k=length)))
                    i = j + 1
                    continue
            result.append(pattern[i])
            i += 1
        
        return ''.join(result)
    
    def _evolve_template(self, identity_type: str):
        """Evolve an identity template based on usage"""
        if identity_type not in self.templates:
            return
        
        # Add a new variation or modify existing template
        template = self.templates[identity_type]
        new_template_name = f"evolved_{len(template) + 1}"
        
        # Create evolved template by copying and modifying default
        if 'default' in template:
            new_template = template['default'].copy()
            # Add some random variation
            for key in list(new_template.keys())[:2]:  # Modify up to 2 attributes
                if new_template[key].get('type') == 'choice':
                    # Add a new option
                    if 'options' in new_template[key]:
                        new_template[key]['options'].append(f"evolved_{random.randint(1, 100)}")
            
            template[new_template_name] = new_template
            logger.info(f"   Evolved new template '{new_template_name}' for {identity_type}")
    
    def _update_identity_attributes(self, identity_id: str, attributes: Dict[str, Any]):
        """Update identity attributes"""
        if identity_id in self.identities:
            self.identities[identity_id]['attributes'].update(attributes)
            self.identities[identity_id]['updated_at'] = datetime.now().isoformat()
            self.attribute_history.append({
                'identity_id': identity_id,
                'timestamp': datetime.now().isoformat(),
                'attributes': attributes
            })
    
    def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'stats': self.stats,
            'pool_status': self.get_all_pools(),
            'active_identities': len(self.active_identities),
            'templates_available': sum(len(t) for t in self.templates.values()),
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
            "stats": self.stats,
            "identity_types": list(self.identity_pools.keys()),
            "methods": ['run', 'evolve', 'execute', 'process', 'generate', 'query']
        }

# Guard clause ensures code only runs when script is executed directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🆔 IDENTITY PERSONA CREATOR (P1T3)")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Identity Persona Creator')
    parser.add_argument('--create', metavar='TYPE', help='Create new identity of type')
    parser.add_argument('--name', help='Name for the identity')
    parser.add_argument('--activate', metavar='ID', help='Activate identity')
    parser.add_argument('--suspend', metavar='ID', help='Suspend identity')
    parser.add_argument('--retire', metavar='ID', help='Retire identity')
    parser.add_argument('--rotate', metavar='TYPE', help='Rotate identities of type')
    parser.add_argument('--list', action='store_true', help='List identities')
    parser.add_argument('--pools', action='store_true', help='Show pool status')
    parser.add_argument('--status', action='store_true', help='Show status')
    
    args = parser.parse_args()
    
    creator = Create_identity_persona()
    
    if args.create:
        print(f"\n🆕 Creating {args.create} identity...")
        result = creator.create_identity(args.create, args.name)
        print(json.dumps(result, indent=2))
    
    elif args.activate:
        print(f"\n🔓 Activating identity: {args.activate}")
        result = creator.activate_identity(args.activate)
        print(json.dumps(result, indent=2))
    
    elif args.suspend:
        print(f"\n⏸️ Suspending identity: {args.suspend}")
        result = creator.suspend_identity(args.suspend)
        print(json.dumps(result, indent=2))
    
    elif args.retire:
        print(f"\n♻️ Retiring identity: {args.retire}")
        result = creator.retire_identity(args.retire)
        print(json.dumps(result, indent=2))
    
    elif args.rotate:
        print(f"\n🔄 Rotating identities of type: {args.rotate}")
        result = creator.rotate_identities(args.rotate)
        print(json.dumps(result, indent=2))
    
    elif args.list:
        print("\n📋 Listing all identities:")
        identities = creator.list_identities()
        for identity in identities:
            print(f"   {identity['id'][:8]}... | {identity['name']:15} | {identity['type']:10} | {identity['status']}")
    
    elif args.pools:
        print("\n📊 Identity Pools:")
        pools = creator.get_all_pools()
        print(json.dumps(pools, indent=2))
    
    elif args.status:
        print("\n📊 Component Status:")
        print(json.dumps(creator.get_status(), indent=2))
    
    else:
        print("\n📋 Component Info:")
        print(json.dumps(creator.info(), indent=2))
        print("\n💡 Use --create, --activate, --suspend, --retire, --rotate, --list, or --pools")
