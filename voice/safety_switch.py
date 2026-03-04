import os
import json
import threading
import time
from datetime import datetime

class DMASafetySwitch:
    def __init__(self):
        self.switch_file = "dmai_safety.json"
        self.master_only = ["david"]
        self.emergency_stop = False
        self.paused = False
        self.load_state()
    
    def load_state(self):
        if os.path.exists(self.switch_file):
            with open(self.switch_file, 'r') as f:
                state = json.load(f)
                self.emergency_stop = state.get('emergency_stop', False)
                self.paused = state.get('paused', False)
    
    def save_state(self):
        with open(self.switch_file, 'w') as f:
            json.dump({
                'emergency_stop': self.emergency_stop,
                'paused': self.paused,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def pause(self, user_voice_id):
        if user_voice_id in self.master_only:
            self.paused = True
            self.save_state()
            return True, "DMAI paused. Say 'resume' to continue."
        return False, "Only master can pause DMAI."
    
    def resume(self, user_voice_id):
        if user_voice_id in self.master_only:
            self.paused = False
            self.save_state()
            return True, "DMAI resumed."
        return False, "Only master can resume DMAI."
    
    def kill(self, user_voice_id, passphrase=None):
        if user_voice_id in self.master_only:
            if passphrase == "DMAI terminate authority override":
                self.emergency_stop = True
                self.save_state()
                return True, "DMAI emergency stop activated. System will shutdown."
        return False, "Kill requires master authentication and override passphrase."
    
    def check_paused(self):
        return self.paused or self.emergency_stop

safety = DMASafetySwitch()
