#!/usr/bin/env python3
"""
Run the best evolved version of any AI model
"""

import os
import sys
import json
import importlib
from pathlib import Path

class BestVersionRunner:
    def __init__(self):
        self.best_path = Path.cwd() / "checkpoints" / "best_versions"
        self.repos_path = Path.cwd() / "repos"
    
    def list_best_versions(self):
        """List all available best versions"""
        if not self.best_path.exists():
            print("❌ No best versions found yet")
            return []
        
        versions = []
        for repo_dir in self.best_path.iterdir():
            if repo_dir.is_dir():
                py_files = list(repo_dir.glob("*.py"))
                if py_files:
                    versions.append({
                        "repo": repo_dir.name,
                        "files": [f.name for f in py_files],
                        "path": repo_dir
                    })
        
        return versions
    
    def run_model(self, repo_name, file_name):
        """Run a specific best version"""
        model_path = self.best_path / repo_name / file_name
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return
        
        print(f"\n🚀 RUNNING BEST VERSION: {repo_name}/{file_name}")
        print("=" * 60)
        
        # Import and run the module
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("best_model", model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for common entry points
            if hasattr(module, 'main'):
                module.main()
            elif hasattr(module, 'run'):
                module.run()
            elif hasattr(module, 'generate'):
                # Interactive mode
                print("\n📝 Interactive mode - type 'quit' to exit")
                while True:
                    prompt = input("\nYou: ")
                    if prompt.lower() == 'quit':
                        break
                    response = module.generate(prompt)
                    print(f"AI: {response}")
            else:
                print("✅ Module loaded successfully")
                print(f"📦 Available: {dir(module)}")
                
        except Exception as e:
            print(f"❌ Error running model: {e}")
    
    def interactive_menu(self):
        """Show interactive menu to choose best version"""
        versions = self.list_best_versions()
        
        if not versions:
            print("❌ No best versions found. Run evolution first!")
            return
        
        print("\n🏆 BEST EVOLVED VERSIONS AVAILABLE:")
        print("=" * 60)
        
        for i, v in enumerate(versions, 1):
            print(f"{i}. {v['repo']}")
            for f in v['files']:
                print(f"   📄 {f}")
        
        print("\n0. Exit")
        
        try:
            choice = int(input("\nSelect version to run: "))
            if choice == 0:
                return
            if 1 <= choice <= len(versions):
                selected = versions[choice-1]
                if len(selected['files']) == 1:
                    self.run_model(selected['repo'], selected['files'][0])
                else:
                    print("\nMultiple files found. Which one?")
                    for j, f in enumerate(selected['files'], 1):
                        print(f"{j}. {f}")
                    file_choice = int(input("Choose: "))
                    if 1 <= file_choice <= len(selected['files']):
                        self.run_model(selected['repo'], selected['files'][file_choice-1])
        except:
            print("Invalid choice")

if __name__ == "__main__":
    runner = BestVersionRunner()
    
    if len(sys.argv) > 2:
        runner.run_model(sys.argv[1], sys.argv[2])
    else:
        runner.interactive_menu()
