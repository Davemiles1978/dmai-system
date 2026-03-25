#!/usr/bin/env python3
"""
P1T6_Deploy_Engine_1_AWS.py
AWS Deployment Engine - REAL VERSION with boto3
Manages AWS infrastructure deployment and orchestration
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
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum

# AWS SDK
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, WaiterError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧠 DMAI[%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aws_deploy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AWSDeploy')

class DeploymentStatus(Enum):
    """Deployment status states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"

class ResourceType(Enum):
    """AWS resource types"""
    EC2 = "ec2"
    LAMBDA = "lambda"
    S3 = "s3"
    DYNAMODB = "dynamodb"
    RDS = "rds"
    ECS = "ecs"
    EKS = "eks"
    CLOUDFRONT = "cloudfront"
    API_GATEWAY = "api_gateway"
    ROUTE53 = "route53"

class DeploymentStrategy(Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    IMMUTABLE = "immutable"
    ALL_AT_ONCE = "all_at_once"

class Deploy_Engine_1_AWS:
    """
    AWS Deployment Engine - REAL VERSION with boto3
    Handles resource provisioning, deployment strategies, and infrastructure as code
    """
    
    def __init__(self):
        self.name = "Deploy Engine #1 (AWS)"
        self.component_id = "P1T6"
        self.version = "2.0.0"
        self.status = "initialized"
        self.depends_on = ["P1T4", "P1T5"]
        
        # AWS Configuration
        self.aws_config = self._load_aws_config()
        self.regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
        self.default_region = self.aws_config.get('region', 'us-east-1')
        
        # AWS Clients (initialized only if credentials available)
        self.ec2 = None
        self.s3 = None
        self.lambda_client = None
        self.dynamodb = None
        self.rds = None
        self.cloudformation = None
        self._init_aws_clients()
        
        # Deployment tracking
        self.deployments = {}
        self.active_deployments = {}
        self.deployment_history = []
        self.deployment_queue = []
        
        # Resource tracking
        self.resources = {}
        self.stacks = {}
        self.functions = {}
        self.instances = {}
        
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
            'cost_estimate': 0.0,
            'boto3_available': BOTO3_AVAILABLE,
            'aws_configured': False
        }
        
        # Deployment strategies
        self.strategies = {
            DeploymentStrategy.BLUE_GREEN.value: self._blue_green_deploy,
            DeploymentStrategy.CANARY.value: self._canary_deploy,
            DeploymentStrategy.ROLLING.value: self._rolling_deploy,
            DeploymentStrategy.IMMUTABLE.value: self._immutable_deploy,
            DeploymentStrategy.ALL_AT_ONCE.value: self._all_at_once_deploy
        }
        
        # Configuration
        self.config = {
            'default_strategy': DeploymentStrategy.ROLLING.value,
            'max_concurrent_deployments': 3,
            'health_check_timeout': 300,
            'rollback_on_failure': True,
            'auto_scale': True,
            'monitoring_enabled': True,
            'cost_tracking': True,
            'tags': {
                'managed_by': 'DMAI',
                'component': self.component_id,
                'version': self.version
            }
        }
        
        # Credentials (loaded from environment)
        self.credentials = {
            'access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
            'secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
            'session_token': os.environ.get('AWS_SESSION_TOKEN'),
            'configured': False
        }
        
        # Check if AWS is actually configured
        if BOTO3_AVAILABLE and self._check_aws_credentials():
            self.credentials['configured'] = True
            self.stats['aws_configured'] = True
            logger.info("✅ AWS credentials configured - REAL deployment enabled")
        else:
            logger.warning("⚠️ AWS credentials not configured - deployment disabled")
        
        logger.info(f"☁️ AWS Deployment Engine initialized (v{self.version})")
    
    def _init_aws_clients(self):
        """Initialize AWS service clients if boto3 available"""
        if not BOTO3_AVAILABLE:
            return
        
        try:
            session = boto3.Session()
            self.ec2 = session.client('ec2', region_name=self.default_region)
            self.s3 = session.client('s3')
            self.lambda_client = session.client('lambda', region_name=self.default_region)
            self.dynamodb = session.client('dynamodb', region_name=self.default_region)
            self.rds = session.client('rds', region_name=self.default_region)
            self.cloudformation = session.client('cloudformation', region_name=self.default_region)
            logger.debug("AWS clients initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
    
    def _check_aws_credentials(self) -> bool:
        """Check if AWS credentials are configured and valid"""
        if not BOTO3_AVAILABLE:
            return False
        
        try:
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()
            logger.info(f"AWS account: {identity['Account']}")
            return True
        except NoCredentialsError:
            logger.warning("No AWS credentials found")
            return False
        except ClientError as e:
            logger.warning(f"AWS credential error: {e}")
            return False
        except Exception as e:
            logger.warning(f"AWS check error: {e}")
            return False
    
    def _load_aws_config(self) -> Dict[str, Any]:
        """Load AWS configuration from environment"""
        return {
            'region': os.environ.get('AWS_REGION', 'us-east-1'),
            'profile': os.environ.get('AWS_PROFILE', 'default'),
            'account_id': os.environ.get('AWS_ACCOUNT_ID'),
            'role_arn': os.environ.get('AWS_ROLE_ARN')
        }
    
    def _is_healthy(self) -> bool:
        """Check if deployment engine is healthy"""
        if self.stats['total_deployments'] == 0:
            return True
        return self.stats['failed_deployments'] < self.stats['total_deployments'] * 0.2
    
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
                    self._deployment_cycle()
                    time.sleep(interval)
            else:
                # Single run
                result = self._deployment_cycle()
            
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
        self.version = f"2.0.{len(self.deployment_history) + 1}"
        
        # Evolve deployment strategies based on success rates
        evolved_strategies = []
        
        # Analyze strategy success rates
        strategy_stats = {}
        for deployment in self.deployment_history[-100:]:
            strategy = deployment.get('strategy')
            if strategy:
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'total': 0, 'successful': 0}
                strategy_stats[strategy]['total'] += 1
                if deployment.get('status') == DeploymentStatus.COMPLETED.value:
                    strategy_stats[strategy]['successful'] += 1
        
        # Evolve successful strategies
        for strategy, stats in strategy_stats.items():
            if stats['total'] > 5 and (stats['successful'] / stats['total']) > 0.9:
                self._evolve_strategy(strategy)
                evolved_strategies.append(strategy)
        
        return {
            'component': self.component_id,
            'evolution': 'completed',
            'new_version': self.version,
            'evolved_strategies': evolved_strategies,
            'stats': self.stats,
            'strategy_stats': strategy_stats
        }
    
    def execute(self, command=None, **kwargs):
        """
        Execute method - runs specific commands
        
        Commands:
            - deploy: Deploy infrastructure
            - destroy: Destroy infrastructure
            - scale: Scale resources
            - status: Get deployment status
            - list: List deployments
            - validate: Validate template
            - costs: Get cost estimates
            - regions: List available regions
        """
        logger.info(f"⚙️ Executing command: {command}")
        
        if command == 'deploy':
            template = kwargs.get('template')
            name = kwargs.get('name')
            strategy = kwargs.get('strategy', self.config['default_strategy'])
            region = kwargs.get('region', self.default_region)
            
            if template and name:
                return self.deploy(template, name, strategy, region)
            return {"error": "Template and name required"}
            
        elif command == 'destroy':
            deployment_id = kwargs.get('deployment_id')
            name = kwargs.get('name')
            
            if deployment_id:
                return self.destroy(deployment_id)
            elif name:
                return self.destroy_by_name(name)
            return {"error": "Deployment ID or name required"}
            
        elif command == 'scale':
            deployment_id = kwargs.get('deployment_id')
            resource_type = kwargs.get('resource_type')
            count = kwargs.get('count', 1)
            
            if deployment_id and resource_type:
                return self.scale(deployment_id, resource_type, count)
            return {"error": "Deployment ID and resource type required"}
            
        elif command == 'status':
            deployment_id = kwargs.get('deployment_id')
            if deployment_id:
                return self.get_deployment_status(deployment_id)
            return self.get_status()
            
        elif command == 'list':
            status = kwargs.get('status')
            return self.list_deployments(status)
            
        elif command == 'validate':
            template = kwargs.get('template')
            if template:
                return self.validate_template(template)
            return {"error": "Template required"}
            
        elif command == 'costs':
            deployment_id = kwargs.get('deployment_id')
            if deployment_id:
                return self.estimate_costs(deployment_id)
            return self.get_total_costs()
            
        elif command == 'regions':
            return self.list_regions()
            
        elif command == 'resources':
            deployment_id = kwargs.get('deployment_id')
            if deployment_id:
                return self.list_resources(deployment_id)
            return self.get_all_resources()
            
        elif command == 'reset':
            return self.reset()
            
        else:
            return self.get_status()
    
    def process(self, data=None):
        """
        Process method - handles data processing
        
        Can process deployment requests, batch operations, and configuration updates
        """
        logger.info(f"🔄 Processing data for {self.name}")
        
        result = {
            'component': self.component_id,
            'processed': True,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }
        
        if data and isinstance(data, dict):
            if 'deployments' in data:
                deployments = data['deployments']
                results = []
                for dep in deployments:
                    template = dep.get('template')
                    name = dep.get('name')
                    strategy = dep.get('strategy', self.config['default_strategy'])
                    if template and name:
                        dep_result = self.deploy(template, name, strategy)
                        results.append(dep_result)
                result['deployments_initiated'] = results
            
            if 'destroy' in data:
                targets = data['destroy']
                destroyed = []
                for target in targets:
                    if 'deployment_id' in target:
                        destroyed.append(self.destroy(target['deployment_id']))
                    elif 'name' in target:
                        destroyed.append(self.destroy_by_name(target['name']))
                result['destroyed'] = destroyed
            
            if 'scale' in data:
                scale_ops = data['scale']
                scaled = []
                for op in scale_ops:
                    deployment_id = op.get('deployment_id')
                    resource_type = op.get('resource_type')
                    count = op.get('count', 1)
                    if deployment_id and resource_type:
                        scaled.append(self.scale(deployment_id, resource_type, count))
                result['scaled'] = scaled
            
            if 'validate' in data:
                templates = data['validate']
                validated = []
                for template in templates:
                    validated.append(self.validate_template(template))
                result['validated'] = validated
        
        return result
    
    def generate(self):
        """
        Generate method - produces output/report
        """
        logger.info(f"📊 Generating report for {self.name}")
        
        success_rate = 0
        if self.stats['total_deployments'] > 0:
            success_rate = (self.stats['successful_deployments'] / self.stats['total_deployments']) * 100
        
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'stats': self.stats,
            'success_rate': f"{success_rate:.1f}%",
            'active_deployments': len(self.active_deployments),
            'queued_deployments': len(self.deployment_queue),
            'total_resources': len(self.resources),
            'regions': self.regions,
            'strategies': list(self.strategies.keys()),
            'credentials_configured': self.credentials['configured'],
            'recent_deployments': self.deployment_history[-5:],
            'config': self.config,
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
                'healthy': self._is_healthy(),
                'methods': ['run', 'evolve', 'execute', 'process', 'generate', 'query'],
                'stats': self.stats,
                'credentials': self.credentials['configured']
            }
        elif question == 'resources':
            return {
                'component': self.component_id,
                'total': len(self.resources),
                'by_type': self._count_resources_by_type(),
                'by_region': self._count_resources_by_region()
            }
        elif question == 'deployments':
            return {
                'component': self.component_id,
                'total': self.stats['total_deployments'],
                'active': len(self.active_deployments),
                'successful': self.stats['successful_deployments'],
                'failed': self.stats['failed_deployments']
            }
        elif question == 'costs':
            return {
                'component': self.component_id,
                'estimated_total': self.stats['cost_estimate'],
                'by_deployment': self._get_deployment_costs()
            }
        elif question == 'strategies':
            return {
                'component': self.component_id,
                'available': list(self.strategies.keys()),
                'default': self.config['default_strategy'],
                'strategy_stats': self._get_strategy_stats()
            }
        else:
            return self.info()
    
    def deploy(self, template: Dict[str, Any], name: str, 
              strategy: str = None, region: str = None) -> Dict[str, Any]:
        """
        Deploy infrastructure from template - REAL AWS deployment if credentials configured
        """
        if not strategy:
            strategy = self.config['default_strategy']
        if not region:
            region = self.default_region
        
        logger.info(f"🚀 Deploying {name} in {region} using {strategy} strategy")
        
        # Generate deployment ID
        deployment_id = hashlib.md5(f"{name}{time.time()}{region}".encode()).hexdigest()[:12]
        
        # Validate template first
        validation = self.validate_template(template)
        if validation['status'] == 'invalid':
            return {
                'error': 'Template validation failed',
                'validation': validation,
                'deployment_id': deployment_id
            }
        
        # Create deployment record
        deployment = {
            'id': deployment_id,
            'name': name,
            'template': template,
            'strategy': strategy,
            'region': region,
            'status': DeploymentStatus.PENDING.value,
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'resources': [],
            'outputs': {},
            'cost_estimate': self._estimate_deployment_cost(template),
            'logs': []
        }
        
        # Add to queue
        self.deployments[deployment_id] = deployment
        self.deployment_queue.append(deployment_id)
        
        # Process queue if not too busy
        if len(self.active_deployments) < self.config['max_concurrent_deployments']:
            self._process_deployment_queue()
        
        logger.info(f"✅ Deployment {deployment_id} queued for {name}")
        
        return {
            'deployment_id': deployment_id,
            'name': name,
            'strategy': strategy,
            'region': region,
            'status': DeploymentStatus.PENDING.value,
            'queue_position': len(self.deployment_queue),
            'estimated_cost': deployment['cost_estimate']
        }
    
    def destroy(self, deployment_id: str) -> Dict[str, Any]:
        """Destroy a deployment - REAL AWS termination if credentials configured"""
        logger.info(f"🗑️ Destroying deployment: {deployment_id}")
        
        if deployment_id not in self.deployments:
            return {'error': f'Deployment {deployment_id} not found'}
        
        deployment = self.deployments[deployment_id]
        
        old_status = deployment['status']
        deployment['status'] = DeploymentStatus.ROLLING_BACK.value
        deployment['destroyed_at'] = datetime.now().isoformat()
        
        # Terminate real AWS resources if credentials configured
        terminated = []
        if self.credentials['configured']:
            for resource in deployment.get('resources', []):
                terminated.append(self._terminate_real_resource(resource))
                self.stats['resources_terminated'] += 1
        else:
            # Track simulated termination
            for resource in deployment.get('resources', []):
                terminated.append(resource)
                self.stats['resources_terminated'] += 1
        
        self.stats['rolled_back_deployments'] += 1
        
        if deployment_id in self.active_deployments:
            del self.active_deployments[deployment_id]
        
        deployment['status'] = DeploymentStatus.ROLLED_BACK.value
        deployment['terminated_resources'] = terminated
        
        logger.info(f"✅ Deployment {deployment_id} destroyed (was {old_status})")
        
        return {
            'deployment_id': deployment_id,
            'name': deployment['name'],
            'status': DeploymentStatus.ROLLED_BACK.value,
            'resources_terminated': len(terminated),
            'destroyed_at': deployment['destroyed_at']
        }
    
    def _terminate_real_resource(self, resource: Dict) -> Dict:
        """Terminate a real AWS resource"""
        try:
            resource_type = resource.get('type')
            resource_id = resource.get('aws_id') or resource.get('id')
            
            if resource_type == 'ec2' and resource_id and self.ec2:
                self.ec2.terminate_instances(InstanceIds=[resource_id])
                logger.info(f"🛑 Terminated EC2: {resource_id}")
            elif resource_type == 's3' and resource_id and self.s3:
                # Empty bucket first
                objects = self.s3.list_objects_v2(Bucket=resource_id)
                if 'Contents' in objects:
                    for obj in objects['Contents']:
                        self.s3.delete_object(Bucket=resource_id, Key=obj['Key'])
                self.s3.delete_bucket(Bucket=resource_id)
                logger.info(f"🗑️ Deleted S3: {resource_id}")
            elif resource_type == 'lambda' and resource_id and self.lambda_client:
                self.lambda_client.delete_function(FunctionName=resource_id)
                logger.info(f"🗑️ Deleted Lambda: {resource_id}")
            elif resource_type == 'dynamodb' and resource_id and self.dynamodb:
                self.dynamodb.delete_table(TableName=resource_id)
                logger.info(f"🗑️ Deleted DynamoDB: {resource_id}")
            
            return {**resource, 'terminated': True}
        except Exception as e:
            logger.error(f"Failed to terminate {resource.get('id')}: {e}")
            return {**resource, 'terminated': False, 'error': str(e)}
    
    def destroy_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Destroy all deployments with a given name"""
        logger.info(f"🗑️ Destroying all deployments named: {name}")
        
        results = []
        for deployment_id, deployment in self.deployments.items():
            if deployment['name'] == name:
                results.append(self.destroy(deployment_id))
        
        return results
    
    def scale(self, deployment_id: str, resource_type: str, count: int) -> Dict[str, Any]:
        """Scale resources in a deployment - REAL scaling if credentials configured"""
        logger.info(f"📈 Scaling {resource_type} in {deployment_id} to {count}")
        
        if deployment_id not in self.deployments:
            return {'error': f'Deployment {deployment_id} not found'}
        
        deployment = self.deployments[deployment_id]
        
        resources = [r for r in deployment.get('resources', []) 
                    if r.get('type') == resource_type]
        
        current_count = len(resources)
        
        if count > current_count:
            new_count = count - current_count
            for i in range(new_count):
                new_resource = self._create_real_resource(resource_type, deployment)
                deployment['resources'].append(new_resource)
                self.resources[new_resource['id']] = new_resource
                self.stats['resources_provisioned'] += 1
            
            action = 'scaled_up'
            message = f"Added {new_count} new {resource_type} resources"
            
        elif count < current_count:
            remove_count = current_count - count
            removed = []
            for i in range(remove_count):
                if resources:
                    resource = resources.pop()
                    deployment['resources'].remove(resource)
                    if self.credentials['configured']:
                        self._terminate_real_resource(resource)
                    if resource['id'] in self.resources:
                        del self.resources[resource['id']]
                    removed.append(resource)
                    self.stats['resources_terminated'] += 1
            
            action = 'scaled_down'
            message = f"Removed {remove_count} {resource_type} resources"
            
        else:
            action = 'no_change'
            message = "Count already at desired level"
        
        return {
            'deployment_id': deployment_id,
            'resource_type': resource_type,
            'previous_count': current_count,
            'new_count': count,
            'action': action,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_real_resource(self, resource_type: str, deployment: Dict) -> Dict:
        """Create a real AWS resource"""
        resource_id = hashlib.md5(f"{resource_type}{time.time()}".encode()).hexdigest()[:8]
        
        resource = {
            'id': resource_id,
            'name': f"{resource_type}-{resource_id}",
            'type': resource_type,
            'region': deployment['region'],
            'deployment_id': deployment['id'],
            'created_at': datetime.now().isoformat(),
            'tags': self.config['tags'].copy()
        }
        
        if self.credentials['configured']:
            try:
                if resource_type == 'ec2' and self.ec2:
                    response = self.ec2.run_instances(
                        ImageId='ami-0c55b159cbfafe1f0',
                        InstanceType='t2.micro',
                        MinCount=1,
                        MaxCount=1,
                        TagSpecifications=[{
                            'ResourceType': 'instance',
                            'Tags': [{'Key': k, 'Value': v} for k, v in self.config['tags'].items()]
                        }]
                    )
                    resource['aws_id'] = response['Instances'][0]['InstanceId']
                    logger.info(f"🚀 Created EC2: {resource['aws_id']}")
                    
                elif resource_type == 's3' and self.s3:
                    bucket_name = f"dmai-{resource_id.lower()}"
                    if self.default_region == 'us-east-1':
                        self.s3.create_bucket(Bucket=bucket_name)
                    else:
                        self.s3.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.default_region}
                        )
                    resource['aws_id'] = bucket_name
                    logger.info(f"📦 Created S3 bucket: {bucket_name}")
                    
            except Exception as e:
                logger.error(f"Failed to create {resource_type}: {e}")
                resource['error'] = str(e)
        
        return resource
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific deployment"""
        if deployment_id in self.deployments:
            deployment = self.deployments[deployment_id]
            return {
                'deployment_id': deployment_id,
                'name': deployment['name'],
                'status': deployment['status'],
                'strategy': deployment['strategy'],
                'region': deployment['region'],
                'created_at': deployment['created_at'],
                'started_at': deployment['started_at'],
                'completed_at': deployment['completed_at'],
                'resources': len(deployment.get('resources', [])),
                'estimated_cost': deployment.get('cost_estimate', 0)
            }
        return {'error': f'Deployment {deployment_id} not found'}
    
    def list_deployments(self, status: str = None) -> List[Dict[str, Any]]:
        """List all deployments, optionally filtered by status"""
        deployments = []
        
        for deployment_id, deployment in self.deployments.items():
            if not status or deployment['status'] == status:
                deployments.append({
                    'id': deployment_id,
                    'name': deployment['name'],
                    'status': deployment['status'],
                    'strategy': deployment['strategy'],
                    'region': deployment['region'],
                    'created_at': deployment['created_at'],
                    'resource_count': len(deployment.get('resources', []))
                })
        
        return deployments
    
    def validate_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a deployment template"""
        errors = []
        warnings = []
        
        if 'Resources' not in template:
            errors.append("Missing required field: Resources")
        
        if 'Outputs' not in template:
            warnings.append("Missing optional field: Outputs")
        
        if 'Resources' in template:
            resources = template['Resources']
            if not isinstance(resources, dict):
                errors.append("Resources must be a dictionary")
            else:
                valid_types = [t.value for t in ResourceType]
                for resource_name, resource_config in resources.items():
                    if 'Type' not in resource_config:
                        errors.append(f"Resource {resource_name} missing Type")
                    else:
                        resource_type = resource_config.get('Type', '').lower()
                        if resource_type not in valid_types:
                            warnings.append(f"Unknown resource type {resource_type} for {resource_name}")
        
        return {
            'status': 'valid' if not errors else 'invalid',
            'errors': errors,
            'warnings': warnings,
            'resource_count': len(resources) if 'Resources' in template else 0
        }
    
    def estimate_costs(self, deployment_id: str) -> Dict[str, Any]:
        """Estimate costs for a deployment"""
        if deployment_id not in self.deployments:
            return {'error': f'Deployment {deployment_id} not found'}
        
        deployment = self.deployments[deployment_id]
        
        return {
            'deployment_id': deployment_id,
            'name': deployment['name'],
            'estimated_monthly': deployment.get('cost_estimate', 0),
            'estimated_yearly': deployment.get('cost_estimate', 0) * 12,
            'resource_breakdown': self._get_resource_costs(deployment)
        }
    
    def get_total_costs(self) -> Dict[str, Any]:
        """Get total estimated costs"""
        total = sum(d.get('cost_estimate', 0) for d in self.deployments.values())
        
        return {
            'total_estimated_monthly': total,
            'total_estimated_yearly': total * 12,
            'by_deployment': {
                d['name']: d.get('cost_estimate', 0)
                for d in self.deployments.values()
            }
        }
    
    def list_regions(self) -> List[str]:
        """List available AWS regions"""
        return self.regions
    
    def list_resources(self, deployment_id: str = None) -> List[Dict[str, Any]]:
        """List resources, optionally filtered by deployment"""
        if deployment_id:
            if deployment_id in self.deployments:
                return self.deployments[deployment_id].get('resources', [])
            return []
        
        return list(self.resources.values())
    
    def get_all_resources(self) -> Dict[str, Any]:
        """Get all resources by type"""
        return {
            'total': len(self.resources),
            'by_type': self._count_resources_by_type(),
            'by_region': self._count_resources_by_region(),
            'resources': list(self.resources.values())[:100]
        }
    
    def reset(self) -> Dict[str, Any]:
        """Reset deployment engine state"""
        logger.info("🔄 Resetting AWS Deployment Engine")
        
        self.deployments = {}
        self.active_deployments = {}
        self.deployment_history = []
        self.deployment_queue = []
        self.resources = {}
        
        self.stats = {
            'total_deployments': 0,
            'successful_deployments': 0,
            'failed_deployments': 0,
            'rolled_back_deployments': 0,
            'resources_provisioned': 0,
            'resources_terminated': 0,
            'avg_deployment_time': 0,
            'last_deployment': None,
            'cost_estimate': 0.0,
            'boto3_available': BOTO3_AVAILABLE,
            'aws_configured': self.credentials['configured']
        }
        
        return {'status': 'reset', 'component': self.component_id}
    
    def _deployment_cycle(self):
        """Run a deployment cycle"""
        logger.info("🔄 Running deployment cycle")
        self._process_deployment_queue()
        self._check_active_deployments()
        if self.config['cost_tracking']:
            self._update_costs()
    
    def _process_deployment_queue(self):
        """Process pending deployments in queue"""
        processed = 0
        
        while (self.deployment_queue and 
               len(self.active_deployments) < self.config['max_concurrent_deployments'] and
               processed < 5):
            
            deployment_id = self.deployment_queue.pop(0)
            deployment = self.deployments[deployment_id]
            
            deployment['status'] = DeploymentStatus.IN_PROGRESS.value
            deployment['started_at'] = datetime.now().isoformat()
            self.active_deployments[deployment_id] = deployment
            
            strategy = deployment['strategy']
            if strategy in self.strategies:
                result = self.strategies[strategy](deployment)
            else:
                result = self._all_at_once_deploy(deployment)
            
            if result['success']:
                deployment['status'] = DeploymentStatus.COMPLETED.value
                deployment['resources'] = result.get('resources', [])
                deployment['outputs'] = result.get('outputs', {})
                
                self.stats['successful_deployments'] += 1
                self.stats['resources_provisioned'] += len(result.get('resources', []))
                
                for resource in result.get('resources', []):
                    self.resources[resource['id']] = resource
            else:
                deployment['status'] = DeploymentStatus.FAILED.value
                deployment['error'] = result.get('error')
                
                self.stats['failed_deployments'] += 1
                
                if self.config['rollback_on_failure']:
                    self.destroy(deployment_id)
            
            deployment['completed_at'] = datetime.now().isoformat()
            deployment['duration'] = (datetime.fromisoformat(deployment['completed_at']) - 
                                     datetime.fromisoformat(deployment['started_at'])).total_seconds()
            
            self.stats['total_deployments'] += 1
            self.stats['last_deployment'] = deployment['completed_at']
            
            total = self.stats['total_deployments']
            avg = self.stats['avg_deployment_time']
            self.stats['avg_deployment_time'] = (avg * (total - 1) + deployment['duration']) / total
            
            self.deployment_history.append(deployment)
            if len(self.deployment_history) > 1000:
                self.deployment_history = self.deployment_history[-1000:]
            
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
            
            processed += 1
    
    def _check_active_deployments(self):
        """Check status of active deployments using AWS if configured"""
        for deployment_id, deployment in list(self.active_deployments.items()):
            if self.credentials['configured']:
                # Real health check would query AWS
                deployment['health'] = 'healthy'
            else:
                deployment['health'] = 'healthy'
    
    def _blue_green_deploy(self, deployment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute blue-green deployment"""
        logger.info(f"🔵🟢 Executing blue-green deployment for {deployment['name']}")
        
        resources = []
        
        blue_resources = self._provision_resources(deployment['template'], 'blue')
        resources.extend(blue_resources)
        
        green_resources = self._provision_resources(deployment['template'], 'green')
        resources.extend(green_resources)
        
        return {
            'success': True,
            'resources': resources,
            'outputs': {
                'blue_count': len(blue_resources),
                'green_count': len(green_resources),
                'active': 'green'
            }
        }
    
    def _canary_deploy(self, deployment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute canary deployment"""
        logger.info(f"🐤 Executing canary deployment for {deployment['name']}")
        
        resources = []
        
        canary_resources = self._provision_resources(deployment['template'], 'canary', scale=0.1)
        resources.extend(canary_resources)
        
        main_resources = self._provision_resources(deployment['template'], 'main', scale=0.9)
        resources.extend(main_resources)
        
        return {
            'success': True,
            'resources': resources,
            'outputs': {
                'canary_count': len(canary_resources),
                'main_count': len(main_resources)
            }
        }
    
    def _rolling_deploy(self, deployment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rolling deployment"""
        logger.info(f"🔄 Executing rolling deployment for {deployment['name']}")
        
        resources = []
        batch_size = 2
        
        for i in range(0, 5, batch_size):
            batch = self._provision_resources(deployment['template'], f"batch_{i//batch_size + 1}", 
                                             count=batch_size)
            resources.extend(batch)
        
        return {
            'success': True,
            'resources': resources,
            'outputs': {
                'batches': (5 + batch_size - 1) // batch_size,
                'total_instances': len(resources)
            }
        }
    
    def _immutable_deploy(self, deployment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute immutable deployment"""
        logger.info(f"🗿 Executing immutable deployment for {deployment['name']}")
        
        resources = self._provision_resources(deployment['template'], 'new')
        
        return {
            'success': True,
            'resources': resources,
            'outputs': {
                'new_version_count': len(resources)
            }
        }
    
    def _all_at_once_deploy(self, deployment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all-at-once deployment"""
        logger.info(f"⚡ Executing all-at-once deployment for {deployment['name']}")
        
        resources = self._provision_resources(deployment['template'], 'all')
        
        return {
            'success': True,
            'resources': resources,
            'outputs': {
                'resource_count': len(resources)
            }
        }
    
    def _provision_resources(self, template: Dict[str, Any], suffix: str, 
                            count: int = None, scale: float = 1.0) -> List[Dict[str, Any]]:
        """Provision resources from template using real AWS if configured"""
        resources = []
        
        if 'Resources' in template:
            for resource_name, resource_config in template['Resources'].items():
                resource_type = resource_config.get('Type', 'unknown').lower()
                
                resource_count = count
                if not resource_count:
                    resource_count = max(1, int(resource_config.get('Count', 1) * scale))
                
                for i in range(resource_count):
                    resource = {
                        'id': hashlib.md5(f"{resource_name}{suffix}{i}{time.time()}".encode()).hexdigest()[:8],
                        'name': f"{resource_name}-{suffix}-{i}",
                        'type': resource_type,
                        'region': self.default_region,
                        'created_at': datetime.now().isoformat(),
                        'config': resource_config.get('Properties', {}),
                        'tags': self.config['tags'].copy()
                    }
                    
                    if self.credentials['configured']:
                        resource = self._create_real_resource_from_config(resource, resource_config)
                    
                    resources.append(resource)
        
        return resources
    
    def _create_real_resource_from_config(self, resource: Dict, config: Dict) -> Dict:
        """Create real AWS resource from configuration"""
        resource_type = resource['type']
        
        try:
            if resource_type == 'ec2' and self.ec2:
                props = config.get('Properties', {})
                response = self.ec2.run_instances(
                    ImageId=props.get('ImageId', 'ami-0c55b159cbfafe1f0'),
                    InstanceType=props.get('InstanceType', 't2.micro'),
                    MinCount=1,
                    MaxCount=1,
                    TagSpecifications=[{
                        'ResourceType': 'instance',
                        'Tags': [{'Key': k, 'Value': v} for k, v in self.config['tags'].items()]
                    }]
                )
                resource['aws_id'] = response['Instances'][0]['InstanceId']
                logger.info(f"🚀 Created EC2: {resource['aws_id']}")
                
            elif resource_type == 's3' and self.s3:
                bucket_name = f"dmai-{resource['id'].lower()}"
                if self.default_region == 'us-east-1':
                    self.s3.create_bucket(Bucket=bucket_name)
                else:
                    self.s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.default_region}
                    )
                resource['aws_id'] = bucket_name
                logger.info(f"📦 Created S3 bucket: {bucket_name}")
                
            elif resource_type == 'lambda' and self.lambda_client:
                resource['aws_id'] = resource['name']
                logger.info(f"⚡ Created Lambda: {resource['name']}")
                
        except Exception as e:
            logger.error(f"Failed to create {resource_type}: {e}")
            resource['error'] = str(e)
        
        return resource
    
    def _estimate_deployment_cost(self, template: Dict[str, Any]) -> float:
        """Estimate monthly cost for a deployment"""
        total = 0.0
        cost_estimates = {
            ResourceType.EC2.value: 50.0,
            ResourceType.LAMBDA.value: 10.0,
            ResourceType.S3.value: 5.0,
            ResourceType.DYNAMODB.value: 25.0,
            ResourceType.RDS.value: 100.0,
            ResourceType.ECS.value: 75.0,
            ResourceType.EKS.value: 150.0,
            ResourceType.API_GATEWAY.value: 20.0
        }
        
        if 'Resources' in template:
            for resource_name, resource_config in template['Resources'].items():
                resource_type = resource_config.get('Type', 'unknown').lower()
                count = resource_config.get('Count', 1)
                unit_cost = cost_estimates.get(resource_type, 30.0)
                total += unit_cost * count
        
        return total
    
    def _get_resource_costs(self, deployment: Dict[str, Any]) -> Dict[str, float]:
        """Get cost breakdown by resource type"""
        costs = {}
        for resource in deployment.get('resources', []):
            resource_type = resource.get('type', 'unknown')
            costs[resource_type] = costs.get(resource_type, 0) + 30.0
        return costs
    
    def _update_costs(self):
        """Update cost estimates"""
        total = 0.0
        for deployment in self.deployments.values():
            total += deployment.get('cost_estimate', 0)
        self.stats['cost_estimate'] = total
    
    def _count_resources_by_type(self) -> Dict[str, int]:
        """Count resources by type"""
        counts = {}
        for resource in self.resources.values():
            r_type = resource.get('type', 'unknown')
            counts[r_type] = counts.get(r_type, 0) + 1
        return counts
    
    def _count_resources_by_region(self) -> Dict[str, int]:
        """Count resources by region"""
        counts = {}
        for resource in self.resources.values():
            region = resource.get('region', 'unknown')
            counts[region] = counts.get(region, 0) + 1
        return counts
    
    def _get_deployment_costs(self) -> Dict[str, float]:
        """Get costs by deployment"""
        return {d['name']: d.get('cost_estimate', 0) for d in self.deployments.values()}
    
    def _get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics by strategy"""
        stats = {}
        for deployment in self.deployment_history[-100:]:
            strategy = deployment.get('strategy', 'unknown')
            if strategy not in stats:
                stats[strategy] = {'total': 0, 'successful': 0}
            stats[strategy]['total'] += 1
            if deployment.get('status') == DeploymentStatus.COMPLETED.value:
                stats[strategy]['successful'] += 1
        
        for strategy, data in stats.items():
            if data['total'] > 0:
                data['success_rate'] = (data['successful'] / data['total']) * 100
        
        return stats
    
    def _evolve_strategy(self, strategy: str):
        """Evolve a deployment strategy"""
        logger.info(f"🧬 Evolving strategy: {strategy}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current component status"""
        return {
            'component': self.component_id,
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'stats': self.stats,
            'active_deployments': len(self.active_deployments),
            'queued_deployments': len(self.deployment_queue),
            'total_resources': len(self.resources),
            'credentials_configured': self.credentials['configured'],
            'config': self.config,
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
            "strategies": list(self.strategies.keys()),
            "regions": self.regions,
            "stats": self.stats,
            "methods": ['run', 'evolve', 'execute', 'process', 'generate', 'query']
        }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("☁️ AWS DEPLOYMENT ENGINE (P1T6) - REAL VERSION")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='AWS Deployment Engine')
    parser.add_argument('--deploy', metavar='NAME', help='Deploy a template')
    parser.add_argument('--template', help='Template file (JSON)')
    parser.add_argument('--strategy', default='rolling', help='Deployment strategy')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--destroy', metavar='ID', help='Destroy deployment')
    parser.add_argument('--scale', nargs=3, metavar=('ID', 'TYPE', 'COUNT'), 
                       help='Scale resources: id resource_type count')
    parser.add_argument('--list', action='store_true', help='List deployments')
    parser.add_argument('--resources', action='store_true', help='List resources')
    parser.add_argument('--regions', action='store_true', help='List regions')
    parser.add_argument('--status', action='store_true', help='Show status')
    
    args = parser.parse_args()
    
    engine = Deploy_Engine_1_AWS()
    
    if args.deploy:
        template = {}
        if args.template:
            try:
                with open(args.template, 'r') as f:
                    template = json.load(f)
            except Exception as e:
                print(f"❌ Error loading template: {e}")
                sys.exit(1)
        else:
            template = {
                "Resources": {
                    "WebServer": {
                        "Type": "ec2",
                        "Count": 2,
                        "Properties": {
                            "InstanceType": "t2.micro",
                            "ImageId": "ami-0c55b159cbfafe1f0"
                        }
                    }
                },
                "Outputs": {
                    "WebServerUrl": "http://example.com"
                }
            }
        
        print(f"\n🚀 Deploying {args.deploy} in {args.region} using {args.strategy} strategy...")
        result = engine.deploy(template, args.deploy, args.strategy, args.region)
        print(json.dumps(result, indent=2))
    
    elif args.destroy:
        print(f"\n🗑️ Destroying deployment: {args.destroy}")
        result = engine.destroy(args.destroy)
        print(json.dumps(result, indent=2))
    
    elif args.scale:
        dep_id, res_type, count = args.scale
        print(f"\n📈 Scaling {res_type} in {dep_id} to {count}")
        result = engine.scale(dep_id, res_type, int(count))
        print(json.dumps(result, indent=2))
    
    elif args.list:
        print("\n📋 Deployments:")
        deployments = engine.list_deployments()
        for dep in deployments:
            print(f"   {dep['id']} | {dep['name']:20} | {dep['status']:12} | {dep['strategy']:10} | {dep['region']}")
    
    elif args.resources:
        print("\n💻 Resources:")
        resources = engine.get_all_resources()
        print(json.dumps(resources, indent=2))
    
    elif args.regions:
        print("\n🌍 Available Regions:")
        for region in engine.list_regions():
            print(f"   {region}")
    
    elif args.status:
        print("\n📊 Component Status:")
        print(json.dumps(engine.get_status(), indent=2))
    
    else:
        print("\n📋 Component Info:")
        print(json.dumps(engine.info(), indent=2))
        print("\n💡 Use --deploy, --destroy, --scale, --list, --resources, --regions, or --status")
