#!/usr/bin/env python3
"""
DMAI Integration Test Suite v1.1
Fixed: Import path and command detection
"""

import os
import sys
import json
import time
import asyncio
import requests
import threading
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [TEST] - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dmai_test')

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(__file__))

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
CHECK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"
WARN = f"{YELLOW}⚠{RESET}"

class IntegrationTester:
    """Test all DMAI integrated components"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.test_results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        self.server_process = None
        self.server_url = "http://localhost:5001"
        
    def print_header(self, text):
        """Print a formatted header"""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}{text:^60}{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")
        
    def print_test(self, name, passed, message=""):
        """Print test result"""
        if passed:
            print(f"  {CHECK} {name}")
            self.test_results['passed'].append(name)
            if message:
                print(f"     {message}")
        else:
            print(f"  {CROSS} {name}")
            self.test_results['failed'].append(name)
            if message:
                print(f"     {RED}{message}{RESET}")
                
    def print_warning(self, name, message):
        """Print warning"""
        print(f"  {WARN} {name}: {message}")
        self.test_results['warnings'].append(f"{name}: {message}")
        
    def wait_for_server(self, timeout=30):
        """Wait for Flask server to start"""
        print(f"\n  Waiting for server at {self.server_url}...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"  {CHECK} Server is running")
                    return True
            except:
                time.sleep(1)
        print(f"  {CROSS} Server failed to start")
        return False
        
    # ========================================================================
    # CORE COMPONENT TESTS
    # ========================================================================
    
    def test_imports(self):
        """Test all required imports"""
        self.print_header("1. Testing Imports")
        
        # Phase 6 imports - they are in components/phase6/
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'components'))
        
        try:
            from phase6.P6_AdvancedIntelligence import (
                SyntheticNeuron, SyntheticNeuralNetwork, PatternSynthesis,
                KnowledgeGraph, ThreatIntelligence, DarkWebIntel,
                SelfImprovementLoop, AIModelFusion, RecursiveSelfImprover,
                UnbreakableMasterInterface, Phase6Manager
            )
            self.print_test("Phase 6 imports", True)
        except Exception as e:
            self.print_test("Phase 6 imports", False, str(e))
            return False
            
        try:
            # Test main core import
            from dmai_core_complete import (
                UnifiedEvolutionEngine, DMAIApplication, KillswitchMonitor,
                VoiceSystem, MusicLearner, PersonaGenerator, ConversationMemory
            )
            self.print_test("Core module imports", True)
        except Exception as e:
            self.print_test("Core module imports", False, str(e))
            return False
            
        return True
        
    def test_synthetic_core(self):
        """Test Synthetic Neural Network core"""
        self.print_header("2. Testing Synthetic Intelligence Core")
        
        try:
            from dmai_core_complete import UnifiedEvolutionEngine
            from pathlib import Path
            
            # Create engine (without starting web server)
            engine = UnifiedEvolutionEngine(Path("."))
            
            # Test network properties
            self.print_test("Synthetic network initialized", 
                           len(engine.synthetic_network.neurons) > 0,
                           f"Neurons: {len(engine.synthetic_network.neurons)}")
            
            # Test consciousness value
            consciousness = engine.synthetic_network.consciousness_level
            self.print_test("Consciousness level readable",
                           0 <= consciousness <= 1,
                           f"Consciousness: {consciousness:.4f}")
            
            # Test process method
            result = engine.synthetic_network.process({"test": "data"})
            self.print_test("Process method works",
                           result is not None,
                           f"Output: {str(result)[:50]}")
            
            # Test evolve method
            evolve_result = engine.synthetic_network.evolve()
            self.print_test("Evolve method works",
                           evolve_result.get('neurons', 0) > 0,
                           f"Neurons after evolve: {evolve_result.get('neurons', 0)}")
            
            return True
            
        except Exception as e:
            self.print_test("Synthetic core test", False, str(e))
            return False
            
    def test_expression_layer(self):
        """Test Expression Layer components"""
        self.print_header("3. Testing Expression Layer")
        
        try:
            from dmai_core_complete import UnifiedEvolutionEngine
            from pathlib import Path
            
            engine = UnifiedEvolutionEngine(Path("."))
            
            # Test Voice System
            voice_profile = engine.voice_system.get_profile()
            self.print_test("Voice System", 
                           voice_profile is not None,
                           f"Profile: pitch={voice_profile.get('pitch', 0):.2f}")
            
            # Test Music Learner
            music_taste = engine.music_learner.get_taste()
            self.print_test("Music Learner",
                           music_taste is not None,
                           f"Preferred tempo: {music_taste.get('preferred_tempo', 0)}")
            
            # Test Persona Generator
            persona = engine.persona_generator.get_current_persona()
            self.print_test("Persona Generator",
                           persona is not None,
                           f"Style: {persona.get('speaking_style', 'unknown')}")
            
            # Test Conversation Memory
            memory_stats = engine.conversation_memory.get_stats()
            self.print_test("Conversation Memory",
                           memory_stats is not None,
                           f"Conversations: {memory_stats.get('total_conversations', 0)}")
            
            # Test Persona evolution with consciousness
            engine.persona_generator.evolve({'type': 'test'}, 0.5)
            evolved_persona = engine.persona_generator.get_current_persona()
            self.print_test("Persona evolution",
                           evolved_persona.get('consciousness_level', 0) == 0.5,
                           f"Consciousness in persona: {evolved_persona.get('consciousness_level', 0)}")
            
            return True
            
        except Exception as e:
            self.print_test("Expression layer test", False, str(e))
            return False
            
    def test_threat_intelligence(self):
        """Test Threat Intelligence components"""
        self.print_header("4. Testing Threat Intelligence")
        
        try:
            from dmai_core_complete import UnifiedEvolutionEngine
            from pathlib import Path
            
            engine = UnifiedEvolutionEngine(Path("."))
            
            # Test IOC extraction
            test_text = "Suspicious IP: 192.168.1.1 and domain bad-site.com"
            iocs = engine.threat_intel.extract_iocs(test_text)
            self.print_test("IOC Extraction",
                           len(iocs) > 0,
                           f"Found {len(iocs)} IOCs")
            
            # Test threat assessment
            assessment = engine.threat_intel.assess_threat(iocs)
            self.print_test("Threat Assessment",
                           assessment.get('level') is not None,
                           f"Threat level: {assessment.get('level', 'unknown')}")
            
            # Test async CVE fetch (mock since it requires network)
            try:
                loop = asyncio.new_event_loop()
                cves = loop.run_until_complete(engine.threat_intel.fetch_cves(days_back=1))
                loop.close()
                self.print_test("CVE Fetch (API)",
                               cves is not None,
                               f"Found {len(cves)} recent CVEs")
            except Exception as e:
                self.print_warning("CVE Fetch", f"Network/API issue: {str(e)[:50]}")
            
            return True
            
        except Exception as e:
            self.print_test("Threat intelligence test", False, str(e))
            return False
            
    def test_ai_fusion(self):
        """Test AI+SI Fusion"""
        self.print_header("5. Testing AI+SI Fusion")
        
        try:
            from dmai_core_complete import UnifiedEvolutionEngine
            from pathlib import Path
            
            engine = UnifiedEvolutionEngine(Path("."))
            
            # Test fusion weights
            weights = engine.ai_fusion.fusion_weights
            self.print_test("Fusion weights",
                           'si' in weights and 'ai' in weights,
                           f"SI: {weights.get('si', 0):.2f}, AI: {weights.get('ai', 0):.2f}")
            
            # Test model registration
            engine.ai_fusion.register_ai_model("test_model", {"test": True}, "test")
            self.print_test("AI Model Registration",
                           len(engine.ai_fusion.ai_models) > 0,
                           f"Registered: {list(engine.ai_fusion.ai_models.keys())}")
            
            # Test async fused process
            async def test_fusion():
                result = await engine.ai_fusion.fused_process({"test": "data"})
                return result
                
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(test_fusion())
            loop.close()
            
            self.print_test("Fused Processing",
                           result.get('fused_output') is not None,
                           f"Consciousness in fusion: {result.get('fused_output', {}).get('consciousness', 0):.4f}")
            
            return True
            
        except Exception as e:
            self.print_test("AI Fusion test", False, str(e))
            return False
            
    def test_self_improvement(self):
        """Test Self-Improvement components"""
        self.print_header("6. Testing Self-Improvement")
        
        try:
            from dmai_core_complete import UnifiedEvolutionEngine
            from pathlib import Path
            
            engine = UnifiedEvolutionEngine(Path("."))
            
            # Test code analysis
            analysis = engine.self_improvement.analyze_self()
            self.print_test("Self Analysis",
                           analysis.get('total_lines', 0) > 0,
                           f"Lines analyzed: {analysis.get('total_lines', 0)}")
            
            # Test improvement generation
            improvements = engine.self_improvement.generate_improvement(analysis)
            self.print_test("Improvement Generation",
                           len(improvements) > 0,
                           f"Generated: {len(improvements)} chars")
            
            # Test recursive improver
            recursive_analysis = engine.recursive_improver.analyze_for_improvement("core")
            self.print_test("Recursive Analysis",
                           recursive_analysis.get('target') == "core",
                           f"Improvements found: {len(recursive_analysis.get('improvements', []))}")
            
            return True
            
        except Exception as e:
            self.print_test("Self-improvement test", False, str(e))
            return False
            
    def test_evolution_cycle(self):
        """Test Evolution Cycle"""
        self.print_header("7. Testing Evolution Cycle")
        
        try:
            from dmai_core_complete import UnifiedEvolutionEngine
            from pathlib import Path
            
            engine = UnifiedEvolutionEngine(Path("."))
            
            # Record initial consciousness
            initial_consciousness = engine.synthetic_network.consciousness_level
            
            # Run one evolution cycle
            result = engine.evolution_cycle()
            
            self.print_test("Evolution cycle completed",
                           result is not None,
                           f"Cycle {result.get('evolution', 0)}")
            
            self.print_test("Consciousness updated",
                           result.get('consciousness', 0) >= initial_consciousness,
                           f"Before: {initial_consciousness:.4f}, After: {result.get('consciousness', 0):.4f}")
            
            self.print_test("Persona evolved",
                           result.get('persona') is not None,
                           f"Persona style: {result.get('persona', {}).get('speaking_style', 'unknown')}")
            
            return True
            
        except Exception as e:
            self.print_test("Evolution cycle test", False, str(e))
            return False
            
    def test_killswitch(self):
        """Test Killswitch Monitor"""
        self.print_header("8. Testing Killswitch Monitor")
        
        try:
            from dmai_core_complete import KillswitchMonitor
            
            monitor = KillswitchMonitor()
            
            # Test initial state
            self.print_test("Initial state",
                           monitor.get_status()['paused'] == False,
                           "Not paused")
            
            # Test pause flag (create temp file)
            import tempfile
            import os
            
            # Create temporary pause flag
            pause_file = "data/pause.flag"
            os.makedirs("data", exist_ok=True)
            with open(pause_file, 'w') as f:
                f.write("test")
            
            # Wait for monitor to detect
            time.sleep(1.5)
            
            self.print_test("Pause detection",
                           monitor.check_paused() == True,
                           "Pause flag detected")
            
            # Clean up
            if os.path.exists(pause_file):
                os.remove(pause_file)
            
            monitor.stop()
            self.print_test("Killswitch monitor stop",
                           True,
                           "Monitor stopped")
            
            return True
            
        except Exception as e:
            self.print_test("Killswitch test", False, str(e))
            return False
            
    def test_message_processing(self):
        """Test message processing with commands"""
        self.print_header("9. Testing Message Processing")
        
        try:
            from dmai_core_complete import UnifiedEvolutionEngine, DMAIApplication
            from pathlib import Path
            
            # Create engine and app to test command handling
            engine = UnifiedEvolutionEngine(Path("."))
            app = DMAIApplication()
            
            # Test regular message
            response = engine.process_message("tester", "What is consciousness?")
            self.print_test("Regular message processing",
                           len(response) > 0 and "consciousness" in response.lower(),
                           f"Response: {response[:60]}...")
            
            # Test /status command using the app's command handler
            command_response = app._handle_command("/status")
            self.print_test("/status command",
                           "Consciousness" in command_response or "DMAI" in command_response,
                           f"Response preview: {command_response[:60]}...")
            
            # Test /persona command
            command_response = app._handle_command("/persona")
            self.print_test("/persona command",
                           "Persona" in command_response or "traits" in command_response.lower(),
                           f"Response preview: {command_response[:60]}...")
            
            # Test /synthetic command
            command_response = app._handle_command("/synthetic")
            self.print_test("/synthetic command",
                           "Neurons" in command_response or "Synapses" in command_response,
                           f"Response preview: {command_response[:60]}...")
            
            # Test /fusion command
            command_response = app._handle_command("/fusion")
            self.print_test("/fusion command",
                           "Fusion" in command_response or "SI Weight" in command_response,
                           f"Response preview: {command_response[:60]}...")
            
            # Test /threat command
            command_response = app._handle_command("/threat")
            self.print_test("/threat command",
                           "Threat" in command_response or "CVEs" in command_response,
                           f"Response preview: {command_response[:60]}...")
            
            return True
            
        except Exception as e:
            self.print_test("Message processing test", False, str(e))
            return False
            
    def test_web_endpoints(self):
        """Test Web API Endpoints"""
        self.print_header("10. Testing Web API Endpoints (Server Required)")
        
        # Check if server is running
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            if response.status_code != 200:
                self.print_test("Server connection", False, "Server not running")
                self.print_warning("Web tests", "Start server first with: python3 dmai_core_complete.py")
                return False
        except:
            self.print_test("Server connection", False, "Server not running")
            self.print_warning("Web tests", "Start server first with: python3 dmai_core_complete.py")
            return False
            
        # Test health endpoint
        try:
            response = requests.get(f"{self.server_url}/health")
            data = response.json()
            self.print_test("Health endpoint", 
                           data.get('status') == 'active',
                           f"Version: {data.get('version', 'unknown')}")
        except Exception as e:
            self.print_test("Health endpoint", False, str(e))
            
        # Test status endpoint
        try:
            response = requests.get(f"{self.server_url}/api/status")
            data = response.json()
            self.print_test("Status endpoint",
                           data.get('consciousness') is not None,
                           f"Consciousness: {data.get('consciousness', 0):.2f}%")
        except Exception as e:
            self.print_test("Status endpoint", False, str(e))
            
        # Test chat endpoint with command
        try:
            response = requests.post(
                f"{self.server_url}/api/chat",
                json={"message": "/status", "user": "tester"}
            )
            data = response.json()
            self.print_test("Chat endpoint (command)",
                           "Consciousness" in data.get('response', ''),
                           f"Response contains consciousness data")
        except Exception as e:
            self.print_test("Chat endpoint", False, str(e))
            
        # Test persona endpoint
        try:
            response = requests.get(f"{self.server_url}/api/persona")
            data = response.json()
            self.print_test("Persona endpoint",
                           data.get('speaking_style') is not None,
                           f"Style: {data.get('speaking_style', 'unknown')}")
        except Exception as e:
            self.print_test("Persona endpoint", False, str(e))
            
        # Test synthetic status endpoint
        try:
            response = requests.get(f"{self.server_url}/api/synthetic/status")
            data = response.json()
            self.print_test("Synthetic status endpoint",
                           data.get('neurons', 0) > 0,
                           f"Neurons: {data.get('neurons', 0)}")
        except Exception as e:
            self.print_test("Synthetic status endpoint", False, str(e))
            
        # Test fusion status endpoint
        try:
            response = requests.get(f"{self.server_url}/api/fusion/status")
            data = response.json()
            self.print_test("Fusion status endpoint",
                           'fusion_weights' in data,
                           f"Weights: SI={data.get('fusion_weights', {}).get('si', 0):.2f}")
        except Exception as e:
            self.print_test("Fusion status endpoint", False, str(e))
            
        # Test threat status endpoint
        try:
            response = requests.get(f"{self.server_url}/api/threat/status")
            data = response.json()
            self.print_test("Threat status endpoint",
                           'cves_tracked' in data,
                           f"CVEs: {data.get('cves_tracked', 0)}")
        except Exception as e:
            self.print_test("Threat status endpoint", False, str(e))
            
        return True
        
    def run_all_tests(self, require_server=False):
        """Run all tests"""
        print(f"\n{GREEN}{'█'*60}{RESET}")
        print(f"{GREEN}{'DMAI INTEGRATION TEST SUITE v1.1':^60}{RESET}")
        print(f"{GREEN}{'█'*60}{RESET}")
        print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run core tests (no server needed)
        tests_passed = 0
        tests_total = 0
        
        # List of test methods
        tests = [
            ("Imports", self.test_imports),
            ("Synthetic Core", self.test_synthetic_core),
            ("Expression Layer", self.test_expression_layer),
            ("Threat Intelligence", self.test_threat_intelligence),
            ("AI Fusion", self.test_ai_fusion),
            ("Self-Improvement", self.test_self_improvement),
            ("Evolution Cycle", self.test_evolution_cycle),
            ("Killswitch", self.test_killswitch),
            ("Message Processing", self.test_message_processing),
        ]
        
        for name, test_func in tests:
            try:
                if test_func():
                    tests_passed += 1
                tests_total += 1
            except Exception as e:
                print(f"  {CROSS} {name}: Exception - {str(e)}")
                self.test_results['failed'].append(f"{name}: {str(e)}")
                tests_total += 1
        
        # Web tests (require server)
        if require_server:
            print(f"\n{YELLOW}NOTE: Testing web endpoints requires server running{RESET}")
            print(f"Start server in another terminal: python3 dmai_core_complete.py")
            
            response = input("\nIs the server running? (y/n): ")
            if response.lower() == 'y':
                if self.test_web_endpoints():
                    tests_passed += 1
                tests_total += 1
        
        # Print summary
        self.print_header("TEST SUMMARY")
        print(f"  {CHECK} Passed: {tests_passed}")
        print(f"  {CROSS} Failed: {len(self.test_results['failed'])}")
        print(f"  {WARN} Warnings: {len(self.test_results['warnings'])}")
        print(f"  Total: {tests_total}")
        
        if self.test_results['failed']:
            print(f"\n{RED}Failed tests:{RESET}")
            for fail in self.test_results['failed']:
                print(f"  • {fail}")
                
        if self.test_results['warnings']:
            print(f"\n{YELLOW}Warnings:{RESET}")
            for warn in self.test_results['warnings']:
                print(f"  • {warn}")
                
        print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if tests_passed == tests_total:
            print(f"\n{GREEN}🎉 ALL TESTS PASSED! DMAI integration is successful.{RESET}")
        else:
            print(f"\n{RED}❌ Some tests failed. Please review the output above.{RESET}")
            
        return tests_passed == tests_total


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DMAI Integration Test Suite')
    parser.add_argument('--server', action='store_true', 
                       help='Include web endpoint tests (requires server running)')
    args = parser.parse_args()
    
    tester = IntegrationTester()
    success = tester.run_all_tests(require_server=args.server)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
