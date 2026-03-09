

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

    def process_provider_update(self, evaluator_name, new_version):
        """
        Process a provider update by merging with internal evolution
        """
        print(f"\n🔄 Processing provider update for {evaluator_name} v{new_version}")
        
        from evolution.version_merger import version_merger
        
        # Get internal version path
        internal_path = f"/Users/davidmiles/Desktop/dmai-system/evolution/evaluators/{evaluator_name}_evaluator.py"
        
        # Create provider info
        provider_info = {
            'version': new_version,
            'source': self.provider_sources.get(evaluator_name, {}).get('url', 'unknown'),
            'detected_at': datetime.now().isoformat()
        }
        
        # Merge versions
        merged_path = version_merger.merge_versions(
            evaluator_name, 
            internal_path, 
            provider_info
        )
        
        if merged_path:
            print(f"   ✅ Successfully merged into super evolved version")
            
            # Check if this evaluator is ready for promotion
            if version_merger.should_replace_internal(evaluator_name, merged_path):
                print(f"   ⚠️  {evaluator_name} has undergone sufficient evolution")
                print(f"      Consider promoting merged version to primary")
            
            return merged_path
        
        return None
    
    def check_and_process_all_updates(self):
        """Check for updates and process them through merger"""
        updates = self.check_all_providers()
        
        processed = []
        for update in updates:
            merged = self.process_provider_update(
                update['provider'],
                update['to']
            )
            if merged:
                processed.append({
                    'provider': update['provider'],
                    'from': update['from'],
                    'to': update['to'],
                    'merged_path': merged
                })
        
        return processed
