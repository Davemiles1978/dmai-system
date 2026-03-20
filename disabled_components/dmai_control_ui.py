#!/usr/bin/env python3
"""DMAI Control UI - Type commands + Multi-biometric auth"""
import sys
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import json
from datetime import datetime
import subprocess

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent)))))

from voice.safety_switch import safety
from voice.auth.voice_auth import VoiceAuth
from language_learning.processor.language_learner import LanguageLearner

class DMAIControlUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🧬 DMAI Master Control")
        self.window.geometry("900x700")
        self.window.configure(bg='#1a1a2f')
        
        # Authentication state
        self.authenticated = False
        self.auth_method = None
        self.master_id = "david"
        self.backup_codes = ["DMAI-OVERRIDE-2026", "MASTER-KEY-789"]
        
        # Initialize components
        self.safety = safety
        self.voice_auth = VoiceAuth()
        self.learner = LanguageLearner()
        
        self.setup_ui()
        self.check_voice_status()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.window, text="🧬 DMAI MASTER CONTROL", 
                        font=("Arial", 20, "bold"), 
                        bg='#1a1a2f', fg='#6e8efb')
        title.pack(pady=10)
        
        # Status Frame
        status_frame = tk.Frame(self.window, bg='#16213e', relief=tk.RAISED, bd=2)
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, text="⚠️ NOT AUTHENTICATED", 
                                     font=("Arial", 14, "bold"),
                                     bg='#16213e', fg='#ff4444')
        self.status_label.pack(pady=5)
        
        # Auth Frame
        auth_frame = tk.Frame(self.window, bg='#1a1a2f')
        auth_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(auth_frame, text="Authentication Method:", 
                bg='#1a1a2f', fg='#e0e0e0').pack()
        
        self.auth_var = tk.StringVar(value="voice")
        tk.Radiobutton(auth_frame, text="🎤 Voice Biometric", variable=self.auth_var, 
                      value="voice", bg='#1a1a2f', fg='#e0e0e0', 
                      selectcolor='#1a1a2f', command=self.toggle_auth).pack()
        tk.Radiobutton(auth_frame, text="🔐 Backup Code", variable=self.auth_var, 
                      value="backup", bg='#1a1a2f', fg='#e0e0e0',
                      selectcolor='#1a1a2f', command=self.toggle_auth).pack()
        tk.Radiobutton(auth_frame, text="👤 Master Password", variable=self.auth_var,
                      value="password", bg='#1a1a2f', fg='#e0e0e0',
                      selectcolor='#1a1a2f', command=self.toggle_auth).pack()
        
        # Auth Input Frame
        self.auth_input_frame = tk.Frame(self.window, bg='#1a1a2f')
        self.auth_input_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.auth_entry = tk.Entry(self.auth_input_frame, width=50, show="*",
                                   bg='#0a0a1a', fg='#e0e0e0', insertbackground='#6e8efb')
        self.auth_entry.pack(side=tk.LEFT, padx=5)
        
        self.auth_button = tk.Button(self.auth_input_frame, text="Authenticate",
                                     command=self.authenticate,
                                     bg='#6e8efb', fg='white', font=("Arial", 10, "bold"))
        self.auth_button.pack(side=tk.LEFT, padx=5)
        
        # Status Display
        self.auth_status = tk.Label(self.window, text="", bg='#1a1a2f', fg='#e0e0e0')
        self.auth_status.pack(pady=5)
        
        # Command Input Frame
        cmd_frame = tk.Frame(self.window, bg='#1a1a2f')
        cmd_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(cmd_frame, text="Enter Command:", bg='#1a1a2f', fg='#e0e0e0').pack(anchor=tk.W)
        
        self.cmd_entry = tk.Entry(cmd_frame, width=70, state='disabled',
                                  bg='#0a0a1a', fg='#e0e0e0', insertbackground='#6e8efb')
        self.cmd_entry.pack(fill=tk.X, pady=5)
        self.cmd_entry.bind('<Return>', lambda e: self.send_command())
        
        self.send_button = tk.Button(cmd_frame, text="Send Command", command=self.send_command,
                                     state='disabled', bg='#6e8efb', fg='white',
                                     font=("Arial", 10, "bold"))
        self.send_button.pack(pady=5)
        
        # Output Display
        output_frame = tk.Frame(self.window, bg='#16213e')
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(output_frame, text="System Output:", bg='#16213e', fg='#e0e0e0').pack(anchor=tk.W)
        
        self.output_area = scrolledtext.ScrolledText(output_frame, height=15,
                                                     bg='#0a0a1a', fg='#a777e3',
                                                     font=("Courier", 10))
        self.output_area.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Safety Controls Frame
        safety_frame = tk.Frame(self.window, bg='#1a1a2f')
        safety_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(safety_frame, text="Emergency Controls:", bg='#1a1a2f', fg='#e0e0e0').pack()
        
        btn_frame = tk.Frame(safety_frame, bg='#1a1a2f')
        btn_frame.pack()
        
        tk.Button(btn_frame, text="⏸️ Pause", command=self.pause_system,
                 bg='#ffaa00', fg='black', width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="▶️ Resume", command=self.resume_system,
                 bg='#00aa00', fg='white', width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🛑 Kill", command=self.kill_system,
                 bg='#ff4444', fg='white', width=10).pack(side=tk.LEFT, padx=5)
        
        # Info Bar
        info_frame = tk.Frame(self.window, bg='#16213e')
        info_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.vocab_label = tk.Label(info_frame, text="Vocabulary: Loading...",
                                    bg='#16213e', fg='#e0e0e0')
        self.vocab_label.pack(side=tk.LEFT, padx=10)
        
        self.gen_label = tk.Label(info_frame, text="Generation: ?",
                                  bg='#16213e', fg='#e0e0e0')
        self.gen_label.pack(side=tk.RIGHT, padx=10)
        
        self.update_info()
        
    def toggle_auth(self):
        if self.auth_var.get() == "voice":
            self.auth_entry.config(show="*")
            self.auth_entry.delete(0, tk.END)
        elif self.auth_var.get() == "backup":
            self.auth_entry.config(show="")
            self.auth_entry.delete(0, tk.END)
        else:  # password
            self.auth_entry.config(show="*")
            self.auth_entry.delete(0, tk.END)
            
    def check_voice_status(self):
        try:
            if 'master' in self.voice_auth.voiceprints:
                self.log_output("✅ Voice biometrics enrolled and ready")
            else:
                self.log_output("⚠️ Voice not enrolled - use backup methods")
        except:
            self.log_output("⚠️ Voice auth unavailable")
    
    def authenticate(self):
        method = self.auth_var.get()
        value = self.auth_entry.get()
        
        if method == "voice":
            self.log_output("🎤 Voice auth selected - say something after clicking OK")
            # Trigger voice capture in separate thread
            threading.Thread(target=self.voice_auth_thread).start()
            
        elif method == "backup":
            if value in self.backup_codes:
                self.authenticated = True
                self.auth_method = "backup"
                self.status_label.config(text="✅ AUTHENTICATED (Backup Code)", fg='#00ff00')
                self.enable_controls()
                self.log_output("✅ Authenticated via backup code")
            else:
                self.log_output("❌ Invalid backup code")
                
        elif method == "password":
            # Simple password - in production, use proper hash
            if value == "Master2026!":
                self.authenticated = True
                self.auth_method = "password"
                self.status_label.config(text="✅ AUTHENTICATED (Password)", fg='#00ff00')
                self.enable_controls()
                self.log_output("✅ Authenticated via password")
            else:
                self.log_output("❌ Invalid password")
    
    def voice_auth_thread(self):
        import sounddevice as sd
        import numpy as np
        
        self.log_output("🎤 Recording 3 seconds of voice...")
        recording = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        
        is_match, confidence = self.voice_auth.verify(recording.flatten(), 16000)
        
        if is_match:
            self.authenticated = True
            self.auth_method = "voice"
            self.status_label.config(text=f"✅ AUTHENTICATED (Voice - {confidence:.0%})", fg='#00ff00')
            self.enable_controls()
            self.log_output(f"✅ Voice authentication successful ({confidence:.0%} confidence)")
        else:
            self.log_output(f"❌ Voice authentication failed ({confidence:.0%} confidence)")
    
    def enable_controls(self):
        self.cmd_entry.config(state='normal')
        self.send_button.config(state='normal')
        self.auth_entry.config(state='disabled')
        self.auth_button.config(state='disabled')
    
    def send_command(self):
        if not self.authenticated:
            self.log_output("❌ Please authenticate first")
            return
        
        cmd = self.cmd_entry.get().strip()
        if not cmd:
            return
        
        self.log_output(f">>> {cmd}")
        self.cmd_entry.delete(0, tk.END)
        
        # Process command
        if cmd.lower() in ["pause", "pause system"]:
            self.pause_system()
        elif cmd.lower() in ["resume", "resume system"]:
            self.resume_system()
        elif cmd.lower() in ["kill", "kill system", "emergency stop"]:
            self.kill_system()
        elif cmd.lower().startswith("vocab") or "words" in cmd.lower():
            self.show_vocabulary()
        elif cmd.lower() == "status":
            self.show_status()
        else:
            self.log_output(f"🤖 DMAI: Command received - '{cmd}'")
    
    def pause_system(self):
        if not self.authenticated:
            self.log_output("❌ Authenticate first")
            return
        success, msg = self.safety.pause(self.master_id)
        self.log_output(f"⏸️ {msg}")
        if success:
            self.log_output("⚠️ System PAUSED - Say 'resume' to continue")
    
    def resume_system(self):
        if not self.authenticated:
            self.log_output("❌ Authenticate first")
            return
        success, msg = self.safety.resume(self.master_id)
        self.log_output(f"▶️ {msg}")
    
    def kill_system(self):
        if not self.authenticated:
            self.log_output("❌ Authenticate first")
            return
        # Require override for kill
        if messagebox.askyesno("Emergency Kill", "Are you ABSOLUTELY sure? This will shutdown DMAI."):
            success, msg = self.safety.kill(self.master_id, "DMAI terminate authority override")
            self.log_output(f"🛑 {msg}")
            if success:
                self.log_output("💀 DMAI terminated")
                self.authenticated = False
                self.status_label.config(text="⚠️ SYSTEM TERMINATED", fg='#ff4444')
    
    def show_vocabulary(self):
        stats = self.learner.get_stats()
        self.log_output(f"📚 DMAI knows {stats['vocabulary_size']} words")
    
    def show_status(self):
        self.log_output(f"🔐 Auth: {self.auth_method if self.authenticated else 'None'}")
        self.log_output(f"⏸️ Paused: {self.safety.check_paused()}")
        stats = self.learner.get_stats()
        self.log_output(f"📚 Vocabulary: {stats['vocabulary_size']}")
    
    def update_info(self):
        try:
            stats = self.learner.get_stats()
            self.vocab_label.config(text=f"Vocabulary: {stats['vocabulary_size']} words")
            
            # Try to get generation
            try:
                with open('ai_core/evolution/current_generation.txt', 'r') as f:
                    gen = f.read().strip()
                self.gen_label.config(text=f"Generation: {gen}")
            except:
                pass
                
        except:
            pass
        
        # Update every 30 seconds
        self.window.after(30000, self.update_info)
    
    def log_output(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_area.insert(tk.END, f"[{timestamp}] {text}\n")
        self.output_area.see(tk.END)
    
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    ui = DMAIControlUI()
    ui.run()
