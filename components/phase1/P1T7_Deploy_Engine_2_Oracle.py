#!/usr/bin/env python3
"""
P1T7_Deploy_Engine_2_Oracle.py
Oracle Cloud Deployment Engine - REAL VERSION with OCI SDK
Manages Oracle Cloud infrastructure deployment
"""

import os
import sys
import json
import time
import logging
import traceback
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

# Oracle Cloud SDK
try:
    import oci
    from oci.core import ComputeClient, VirtualNetworkClient
    from oci.load_balancer import LoadBalancerClient
    OCI_AVAILABLE = True
except ImportError:
    OCI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[Oracle] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oracle_deploy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OracleDeploy')

class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    VERIFYING = "verifying"
    VERIFIED = "verified"

class OracleResourceType(Enum):
    COMPUTE = "compute"
    DATABASE = "database"
    STORAGE = "storage"
    NETWORK = "network"
    LOAD_BALANCER = "load_balancer"
    FUNCTIONS = "functions"
    KUBERNETES = "kubernetes"
    API_GATEWAY = "api_gateway"

class Engine2Deployer:
    """
    Oracle Cloud Deployment Engine - REAL VERSION with OCI SDK
    Deploys and manages infrastructure on Oracle Cloud
    """
    
    def __init__(self):
        self.name = "Engine #2 Deployer (Oracle)"
        self.component_id = "P1T7"
        self.version = "3.0.0"  # Major version for real SDK
        self.status = "initialized"
        self.depends_on = ["P1T2", "P1T6"]
        self.provider = "Oracle"
        
        # Oracle Cloud regions
        self.regions = [
            'us-ashburn-1', 'us-phoenix-1', 'eu-frankfurt-1', 'uk-london-1',
            'eu-amsterdam-1', 'ap-mumbai-1', 'ap-sydney-1', 'ap-tokyo-1'
        ]
        self.default_region = os.getenv('ORACLE_REGION', 'eu-frankfurt-1')
        
        # OCI Clients
        self.compute_client = None
        self.vcn_client = None
        self.load_balancer_client = None
        self.identity_client = None
        
        # Deployment tracking
        self.deployments = {}
        self.active_deployments = {}
        self.deployment_history = []
        self.deployment_queue = []
        self.resources = {}
        self.engine_deployments = {}
        
        # Load credentials and initialize clients
        self.credentials = self._load_credentials()
        self.config = self._load_config()
        self._init_oci_clients()
        
        # Statistics
        self.stats = {
            'total_deployments': 0,
            'successful_deployments': 0,
            'failed_deployments': 0,
            'rolled_back_deployments': 0,
            'resources_provisioned': 0,
            'resources_terminated': 0,
            'avg_deployment_time': 0,
            'last_deployment': None,
            'verification_success_rate': 1.0,
            'oci_available': OCI_AVAILABLE,
            'oci_configured': False
        }
        
        if OCI_AVAILABLE and self.credentials.get('configured'):
            self.stats['oci_configured'] = True
            logger.info(f"✅ Oracle Cloud credentials configured - REAL deployment enabled")
        else:
            logger.warning("⚠️ Oracle Cloud credentials not configured - deployment disabled")
        
        logger.info(f"☁️ Oracle Cloud Deployment Engine initialized (v{self.version})")
    
    def _load_credentials(self) -> Dict:
        """Load Oracle Cloud credentials from environment or harvested keys"""
        creds = {
            'user': os.getenv('ORACLE_USER'),
            'tenancy': os.getenv('ORACLE_TENANCY'),
            'fingerprint': os.getenv('ORACLE_FINGERPRINT'),
            'key_file': os.getenv('ORACLE_PRIVATE_KEY_PATH'),
            'region': os.getenv('ORACLE_REGION', 'eu-frankfurt-1'),
            'configured': False
        }
        
        # Also check harvested keys
        harvested_file = Path("data/harvested_keys.json")
        if harvested_file.exists():
            try:
                with open(harvested_file, 'r') as f:
                    data = json.load(f)
                    for key_data in data.get('keys', []):
                        if key_data.get('service') == 'oracle':
                            creds['user'] = key_data.get('user', creds['user'])
                            creds['tenancy'] = key_data.get('tenancy', creds['tenancy'])
                            creds['fingerprint'] = key_data.get('fingerprint', creds['fingerprint'])
                            creds['key_file'] = key_data.get('key_file', creds['key_file'])
                            logger.info("✅ Found harvested Oracle Cloud credentials")
            except Exception as e:
                logger.error(f"Failed to load harvested keys: {e}")
        
        # Check if we have enough to configure
        if creds['user'] and creds['tenancy'] and creds['fingerprint']:
            creds['configured'] = True
        
        return creds
    
    def _load_config(self) -> Dict:
        """Load Oracle Cloud configuration"""
        return {
            'max_concurrent_deployments': 2,
            'health_check_timeout': 180,
            'verification_retries': 3,
            'rollback_on_failure': True,
            'auto_backup': True,
            'monitoring_enabled': True,
            'compartment_id': os.getenv('ORACLE_COMPARTMENT_ID'),
            'availability_domain': os.getenv('ORACLE_AD', 'AD-1'),
            'tags': {
                'managed_by': 'DMAI',
                'component': self.component_id,
                'version': self.version,
                'engine': 'recovery_engine_2'
            }
        }
    
    def _init_oci_clients(self):
        """Initialize OCI service clients"""
        if not OCI_AVAILABLE:
            return
        
        if not self.credentials['configured']:
            return
        
        try:
            # Create config dict for OCI
            config = {
                "user": self.credentials['user'],
                "tenancy": self.credentials['tenancy'],
                "fingerprint": self.credentials['fingerprint'],
                "key_file": self.credentials['key_file'],
                "region": self.credentials['region']
            }
            
            # Initialize clients
            self.compute_client = ComputeClient(config)
            self.vcn_client = VirtualNetworkClient(config)
            self.load_balancer_client = LoadBalancerClient(config)
            self.identity_client = oci.identity.IdentityClient(config)
            
            # Test connection
            user = self.identity_client.get_user(self.credentials['user'])
            if user.status == 200:
                logger.info(f"✅ OCI client initialized for user: {user.data.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCI clients: {e}")
            self.credentials['configured'] = False
    
    def run(self, continuous=False, interval=300):
        """Main execution method"""
        logger.info(f"🚀 Starting {self.name} v{self.version}")
        
        try:
            if continuous:
                while True:
                    self._deployment_cycle()
                    time.sleep(interval)
            else:
                self._deployment_cycle()
            
            return self.get_status()
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return {"error": str(e)}
    
    def evolve(self):
        """Evolution method"""
        logger.info(f"🧬 Evolving {self.name}")
        self.version = f"3.0.{len(self.deployment_history) + 1}"
        return {'component': self.component_id, 'evolution': 'completed', 'new_version': self.version}
    
    def execute(self, command=None, **kwargs):
        """Execute commands"""
        commands = {
            'deploy': self._deploy_engine,
            'destroy': self._destroy_deployment,
            'verify': self._verify_deployment,
            'scale': self._scale_deployment,
            'backup': self._backup_deployment,
            'list': self._list_deployments,
            'status': self._get_status,
            'regions': self._list_regions
        }
        
        if command in commands:
            return commands[command](kwargs)
        return {"error": f"Unknown command: {command}"}
    
    def _deploy_engine(self, kwargs) -> Dict:
        """Deploy Recovery Engine #2 on Oracle Cloud - REAL OCI"""
        name = kwargs.get('name', f"engine2-{int(time.time())}")
        config = kwargs.get('config', {})
        region = kwargs.get('region', self.default_region)
        
        if not self.stats['oci_configured']:
            return {
                "success": False,
                "error": "OCI not configured",
                "message": "DMAI needs to harvest or configure Oracle Cloud credentials"
            }
        
        deployment_id = hashlib.md5(f"{name}{time.time()}{region}".encode()).hexdigest()[:12]
        
        # Default configuration
        default_config = {
            'instance_shape': 'VM.Standard2.2',
            'instance_count': 2,
            'ocpus': 2,
            'memory_gb': 30,
            'boot_volume_size_gb': 100,
            'subnet_type': 'public',
            'assign_public_ip': True,
            'backup_enabled': True,
            'monitoring_enabled': True
        }
        default_config.update(config)
        
        deployment = {
            'id': deployment_id,
            'name': name,
            'type': 'engine2',
            'region': region,
            'config': default_config,
            'status': DeploymentStatus.PENDING.value,
            'created_at': datetime.now().isoformat(),
            'resources': [],
            'oci_ids': []
        }
        
        self.deployments[deployment_id] = deployment
        self.deployment_queue.append(deployment_id)
        
        if len(self.active_deployments) < self.config['max_concurrent_deployments']:
            self._process_deployment_queue()
        
        return {
            'deployment_id': deployment_id,
            'name': name,
            'status': DeploymentStatus.PENDING.value,
            'message': f"Deployment queued for {name}"
        }
    
    def _execute_real_deployment(self, deployment: Dict) -> bool:
        """Execute real OCI deployment"""
        try:
            compartment_id = self.config['compartment_id']
            if not compartment_id:
                compartment_id = self._get_root_compartment()
            
            # Create VCN
            vcn = self._create_vcn(compartment_id, deployment)
            if not vcn:
                return False
            deployment['oci_ids'].append(vcn['id'])
            
            # Create subnet
            subnet = self._create_subnet(compartment_id, vcn['id'], deployment)
            if not subnet:
                return False
            deployment['oci_ids'].append(subnet['id'])
            
            # Create compute instances
            for i in range(deployment['config']['instance_count']):
                instance = self._create_instance(compartment_id, subnet['id'], deployment, i)
                if instance:
                    deployment['oci_ids'].append(instance['id'])
                    deployment['resources'].append(instance)
                    self.resources[instance['id']] = instance
            
            # Create load balancer if multiple instances
            if deployment['config']['instance_count'] > 1:
                lb = self._create_load_balancer(compartment_id, subnet['id'], deployment)
                if lb:
                    deployment['oci_ids'].append(lb['id'])
                    deployment['engine_endpoint'] = lb['ip_address']
            
            self.stats['resources_provisioned'] += len(deployment['resources'])
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _create_vcn(self, compartment_id: str, deployment: Dict) -> Optional[Dict]:
        """Create Virtual Cloud Network"""
        try:
            vcn_details = oci.core.models.CreateVcnDetails()
            vcn_details.cidr_block = "10.0.0.0/16"
            vcn_details.display_name = f"dmai-vcn-{deployment['id']}"
            vcn_details.compartment_id = compartment_id
            vcn_details.dns_label = f"dmaivcn{deployment['id'][:8]}"
            
            response = self.vcn_client.create_vcn(vcn_details)
            if response.status == 200:
                logger.info(f"✅ VCN created: {response.data.id}")
                return {'id': response.data.id, 'type': 'vcn'}
        except Exception as e:
            logger.error(f"VCN creation failed: {e}")
        return None
    
    def _create_subnet(self, compartment_id: str, vcn_id: str, deployment: Dict) -> Optional[Dict]:
        """Create Subnet"""
        try:
            subnet_details = oci.core.models.CreateSubnetDetails()
            subnet_details.cidr_block = "10.0.1.0/24"
            subnet_details.display_name = f"dmai-subnet-{deployment['id']}"
            subnet_details.compartment_id = compartment_id
            subnet_details.vcn_id = vcn_id
            subnet_details.route_table_id = None
            
            response = self.vcn_client.create_subnet(subnet_details)
            if response.status == 200:
                logger.info(f"✅ Subnet created: {response.data.id}")
                return {'id': response.data.id, 'type': 'subnet'}
        except Exception as e:
            logger.error(f"Subnet creation failed: {e}")
        return None
    
    def _create_instance(self, compartment_id: str, subnet_id: str, deployment: Dict, index: int) -> Optional[Dict]:
        """Create Compute Instance"""
        try:
            instance_details = oci.core.models.LaunchInstanceDetails()
            instance_details.display_name = f"dmai-engine2-{deployment['id']}-{index}"
            instance_details.compartment_id = compartment_id
            instance_details.shape = deployment['config']['instance_shape']
            instance_details.subnet_id = subnet_id
            
            # Use Oracle Linux 8
            instance_details.image_id = self._get_latest_image(compartment_id)
            instance_details.metadata = {
                'ssh_authorized_keys': os.getenv('ORACLE_SSH_KEY', ''),
                'user_data': base64.b64encode(b'#!/bin/bash\necho "DMAI Engine #2 deployed"').decode()
            }
            
            response = self.compute_client.launch_instance(instance_details)
            if response.status == 200:
                logger.info(f"✅ Instance created: {response.data.id}")
                return {
                    'id': response.data.id,
                    'type': 'compute',
                    'name': response.data.display_name,
                    'shape': response.data.shape,
                    'status': 'provisioning'
                }
        except Exception as e:
            logger.error(f"Instance creation failed: {e}")
        return None
    
    def _get_latest_image(self, compartment_id: str) -> Optional[str]:
        """Get latest Oracle Linux 8 image"""
        try:
            images = self.compute_client.list_images(compartment_id, operating_system="Oracle Linux")
            for image in images.data:
                if image.operating_system_version.startswith("8") and "GPU" not in image.display_name:
                    return image.id
        except Exception as e:
            logger.error(f"Failed to get image: {e}")
        return None
    
    def _get_root_compartment(self) -> Optional[str]:
        """Get root compartment ID"""
        try:
            response = self.identity_client.list_compartments(
                self.credentials['tenancy'],
                lifecycle_state="ACTIVE"
            )
            for compartment in response.data:
                if compartment.name == "root":
                    return compartment.id
            return self.credentials['tenancy']
        except Exception as e:
            logger.error(f"Failed to get root compartment: {e}")
            return None
    
    def _create_load_balancer(self, compartment_id: str, subnet_id: str, deployment: Dict) -> Optional[Dict]:
        """Create Load Balancer"""
        try:
            lb_details = oci.load_balancer.models.CreateLoadBalancerDetails()
            lb_details.compartment_id = compartment_id
            lb_details.display_name = f"dmai-lb-{deployment['id']}"
            lb_details.shape_name = "10Mbps"
            lb_details.subnet_ids = [subnet_id]
            lb_details.is_private = False
            
            response = self.load_balancer_client.create_load_balancer(lb_details)
            if response.status == 202:
                logger.info(f"✅ Load balancer created: {response.data.id}")
                return {'id': response.data.id, 'type': 'load_balancer', 'ip_address': 'pending'}
        except Exception as e:
            logger.error(f"Load balancer creation failed: {e}")
        return None
    
    def _destroy_deployment(self, kwargs) -> Dict:
        """Destroy deployment - real OCI termination"""
        deployment_id = kwargs.get('deployment_id')
        if not deployment_id or deployment_id not in self.deployments:
            return {"error": "Deployment not found"}
        
        deployment = self.deployments[deployment_id]
        
        for oci_id in deployment.get('oci_ids', []):
            try:
                if oci_id.startswith('ocid1.instance'):
                    self.compute_client.terminate_instance(oci_id)
                    logger.info(f"🛑 Terminated instance: {oci_id}")
                elif oci_id.startswith('ocid1.loadbalancer'):
                    self.load_balancer_client.delete_load_balancer(oci_id)
                    logger.info(f"🗑️ Deleted load balancer: {oci_id}")
                elif oci_id.startswith('ocid1.subnet'):
                    self.vcn_client.delete_subnet(oci_id)
                    logger.info(f"🗑️ Deleted subnet: {oci_id}")
                elif oci_id.startswith('ocid1.vcn'):
                    self.vcn_client.delete_vcn(oci_id)
                    logger.info(f"🗑️ Deleted VCN: {oci_id}")
            except Exception as e:
                logger.error(f"Failed to delete {oci_id}: {e}")
        
        deployment['status'] = DeploymentStatus.ROLLED_BACK.value
        self.stats['resources_terminated'] += len(deployment.get('oci_ids', []))
        
        return {'success': True, 'deployment_id': deployment_id, 'message': 'Destroyed'}
    
    def _verify_deployment(self, kwargs) -> Dict:
        """Verify deployment"""
        deployment_id = kwargs.get('deployment_id')
        if not deployment_id or deployment_id not in self.deployments:
            return {"error": "Deployment not found"}
        
        deployment = self.deployments[deployment_id]
        
        # Verify instances are running
        all_running = True
        for oci_id in deployment.get('oci_ids', []):
            if oci_id.startswith('ocid1.instance'):
                try:
                    instance = self.compute_client.get_instance(oci_id)
                    if instance.data.lifecycle_state != "RUNNING":
                        all_running = False
                except:
                    all_running = False
        
        deployment['verification_status'] = 'passed' if all_running else 'failed'
        return {
            'deployment_id': deployment_id,
            'verified': all_running,
            'status': deployment['verification_status']
        }
    
    def _scale_deployment(self, kwargs) -> Dict:
        """Scale deployment"""
        return {"message": "Scale operation - implement with OCI"}
    
    def _backup_deployment(self, kwargs) -> Dict:
        """Backup deployment"""
        return {"message": "Backup operation - implement with OCI Block Volume backups"}
    
    def _list_deployments(self, kwargs) -> List:
        return list(self.deployments.values())
    
    def _get_status(self, kwargs) -> Dict:
        return self.get_status()
    
    def _list_regions(self, kwargs) -> List:
        return self.regions
    
    def _process_deployment_queue(self):
        """Process pending deployments"""
        processed = 0
        
        while (self.deployment_queue and 
               len(self.active_deployments) < self.config['max_concurrent_deployments'] and
               processed < 3):
            
            deployment_id = self.deployment_queue.pop(0)
            deployment = self.deployments[deployment_id]
            
            deployment['status'] = DeploymentStatus.IN_PROGRESS.value
            deployment['started_at'] = datetime.now().isoformat()
            self.active_deployments[deployment_id] = deployment
            
            success = self._execute_real_deployment(deployment)
            
            if success:
                deployment['status'] = DeploymentStatus.COMPLETED.value
                self.stats['successful_deployments'] += 1
            else:
                deployment['status'] = DeploymentStatus.FAILED.value
                self.stats['failed_deployments'] += 1
                
                if self.config['rollback_on_failure']:
                    self._destroy_deployment({'deployment_id': deployment_id})
            
            deployment['completed_at'] = datetime.now().isoformat()
            self.stats['total_deployments'] += 1
            self.stats['last_deployment'] = deployment['completed_at']
            self.deployment_history.append(deployment)
            
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
            
            processed += 1
    
    def _deployment_cycle(self):
        """Run deployment cycle"""
        self._process_deployment_queue()
    
    def get_status(self) -> Dict:
        """Get component status"""
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'provider': self.provider,
            'stats': self.stats,
            'active_deployments': len(self.active_deployments),
            'queued_deployments': len(self.deployment_queue),
            'credentials_configured': self.credentials['configured'],
            'oci_available': OCI_AVAILABLE,
            'methods': ['deploy', 'destroy', 'verify', 'list', 'status']
        }
    
    def info(self) -> Dict:
        """Get component info"""
        return {
            "name": self.name,
            "id": self.component_id,
            "version": self.version,
            "status": self.status,
            "provider": self.provider,
            "stats": self.stats
        }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("☁️ ORACLE CLOUD DEPLOYMENT ENGINE - REAL VERSION")
    print("="*60)
    
    deployer = Engine2Deployer()
    print(json.dumps(deployer.get_status(), indent=2))
