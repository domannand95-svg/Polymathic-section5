# Full integrated PyQt5 ritual orchestration application

import sys
import asyncio
import threading
import json
import random
import math
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QTextEdit, QSpinBox, QFormLayout,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen
import numpy as np
import pygame

pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Helper functions to load/save JSON
def load_json_file(filepath):
    if Path(filepath).is_file():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_json_file(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# Ritual Grammar expansion and archetypes helpers
def archetype_by_symbol(symbol, archetypes):
    for arch in archetypes:
        if arch['symbol'] == symbol:
            return arch
    return {"symbol": symbol, "polarity": "None", "element": "None", "gender": "neutral"}

def weighted_random_choice(expansions):
    total = sum(exp.get('weight', 1) for exp in expansions)
    r = random.uniform(0, total)
    upto = 0
    for exp in expansions:
        w = exp.get('weight', 1)
        if upto + w >= r:
            return exp
        upto += w
    return expansions[0] if expansions else None

def expand_symbol(symbol, archetypes, grammar_rules, max_depth=4, depth=0):
    if depth >= max_depth or symbol not in grammar_rules:
        return [archetype_by_symbol(symbol, archetypes)]
    rule = grammar_rules[symbol]
    expansions = rule.get("expansions", [])
    if rule.get("optional", False) and random.random() < 0.1:
        return []
    expansion = weighted_random_choice(expansions)
    if expansion is None:
        return [archetype_by_symbol(symbol, archetypes)]
    result = []
    for sym in expansion.get("sequence", [symbol]):
        result.extend(expand_symbol(sym, archetypes, grammar_rules, max_depth, depth + 1))
    return result

# Dummy Ritual Transformer Interface
class RitualTransformerInterface:
    def prepare_and_infer(self, symbolic_sequence):
        tokens = [token["symbol"] for token in symbolic_sequence]
        return tokens + ["ת", "ש"]

# Virtual Device Simulator
class RichVirtualDeviceController(QObject):
    state_changed = pyqtSignal(dict)
    log_generated = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.connected = True
        self.command_queue = asyncio.Queue()
        self.device_state = {"coil_power": 0, "last_command": None, "error_mode": False, "sensor_value": 0.0, "status": "Idle"}
        self._running = False
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self.run_loop, daemon=True).start()
        
    def run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self.device_main_loop())
        
    async def device_main_loop(self):
        self._running = True
        while self._running:
            try:
                cmd = await asyncio.wait_for(self.command_queue.get(), timeout=0.1)
                await self.handle_command(cmd)
            except asyncio.TimeoutError:
                self.simulate_state()
            except Exception as e:
                self.log_generated.emit(f"Device error: {e}")
                
    async def handle_command(self, cmd):
        self.device_state["last_command"] = cmd
        self.device_state["status"] = "Processing"
        self.log_generated.emit(f"Processing command: {cmd}")
        await asyncio.sleep(0.3)
        if cmd.startswith("SET POWER"):
            try:
                val = int(cmd.split()[-1])
                if 0 <= val <= 100:
                    self.device_state["coil_power"] = val
                    self.device_state["status"] = "Power Set"
                    self.log_generated.emit(f"Power set to {val}")
                else:
                    self.device_state["status"] = "Error: Power out of range"
                    self.device_state["error_mode"] = True
                    self.log_generated.emit("Power out of range")
            except:
                self.device_state["status"] = "Error: Invalid power value"
                self.device_state["error_mode"] = True
                self.log_generated.emit("Invalid power value")
        elif cmd.startswith("TRIGGER"):
            token = cmd.split()[-1]
            self.device_state["status"] = f"Triggered token {token}"
            self.log_generated.emit(f"Triggered token {token}")
            await asyncio.sleep(0.5)
        else:
            self.device_state["status"] = "Unknown Command"
            self.log_generated.emit(f"Unknown command: {cmd}")
        self.state_changed.emit(self.device_state.copy())
        
    def simulate_state(self):
        t = datetime.now().timestamp()
        self.device_state["sensor_value"] = 50 + 10 * math.sin(t / 5)
        if self.device_state.get("error_mode") and random.random() < 0.01:
            self.device_state["error_mode"] = False
            self.device_state["status"] = "Error Cleared"
            self.log_generated.emit("Error cleared")
        self.state_changed.emit(self.device_state.copy())
        
    def send_command(self, command):
        if self.connected:
            self._loop.call_soon_threadsafe(self.command_queue.put_nowait, command)
        else:
            self.log_generated.emit("Device not connected")
            
    def disconnect(self):
        self._running = False
        self.connected = False
        self.log_generated.emit("Device disconnected")

# Audio Synthesizer
class RitualAudioSynth:
    def __init__(self):
        self.freq_map = {"Yang": 523.25, "Yin": 392.00, "Balanced": 440.00}
        self.channel = pygame.mixer.Channel(0)
        
    def play_tone_for_token(self, token, polarity):
        freq = self.freq_map.get(polarity, 440.0)
        duration = 400
        sound = self._create_tone(freq, duration)
        self.channel.play(sound)
        
    def _create_tone(self, freq, duration_ms):
        sample_rate = 22050
        n_samples = int(sample_rate * duration_ms / 1000)
        buf = np.zeros((n_samples, 2), dtype=np.int16)
        max_ampl = 2**15 - 1
        for s in range(n_samples):
            val = int(max_ampl * 0.4 * math.sin(2 * math.pi * freq * s / sample_rate))
            buf[s][0] = val
            buf[s][1] = val
        return pygame.sndarray.make_sound(buf)

# Fractal Visual Widget
class FractalVisualWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.phase = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(40)
        
    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        c = self.rect().center()
        max_r = min(self.width(), self.height()) // 2 - 30
        self.phase += 0.12
        
        for i in range(6):
            r = max_r * (i + 1) / 6
            a = self.phase + i * 1.2
            col = QColor(
                max(min(int(150 + 100*math.sin(a)),255),0),
                max(min(int(100 + 100*math.sin(a + 2)),255),0),
                max(min(int(150 + 100*math.sin(a + 4)),255),0),
                180
            )
            pen = QPen(col, 3)
            painter.setPen(pen)
            points = []
            for j in range(100):
                angle = j * (2*math.pi/100) * (1 + i*0.3) + a
                x = c.x() + r * math.cos(angle) * (1 + 0.1*math.sin(j))
                y = c.y() + r * math.sin(angle) * (1 + 0.1*math.cos(j))
                points.append(QPointF(x,y))
            for p1, p2 in zip(points, points[1:]):
                painter.drawLine(p1, p2)

# Orchestrator
class RitualOrchestrator:
    def __init__(self, archetypes, grammar_rules, transformer, device, logger, event_bus):
        self.archetypes = archetypes
        self.grammar_rules = grammar_rules
        self.transformer = transformer
        self.device = device
        self.logger = logger
        self.event_bus = event_bus
        self.max_depth = 4
        self._running = False
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        
    async def _wait_if_paused(self):
        await self._pause_event.wait()
        
    async def _expand_and_predict(self):
        expanded = expand_symbol("S", self.archetypes, self.grammar_rules, self.max_depth)
        predicted = self.transformer.prepare_and_infer(expanded)
        return expanded, predicted
    
    async def _trigger_ritual_actions(self, tokens):
        for token in tokens:
            await self._wait_if_paused()
            if not self._running:
                break
            arch = next((a for a in self.archetypes if a['symbol'] == token), {"polarity": "None"})
            self.logger.log("ritual_token", {"token": token, "attributes": arch})
            await self.event_bus.put({"token": token, "attributes": arch})
            self.device.send_command(f"TRIGGER {token}")
            
    async def run(self):
        self._running = True
        self._paused = False
        self._pause_event.set()
        self.logger.log("ritual_start", {})
        try:
            expanded, predicted = await self._expand_and_predict()
            self.on_update_expansion(expanded)
            self.on_update_predictions(predicted)
            await self._trigger_ritual_actions(predicted)
        finally:
            self._running = False
            self._paused = False
            self._pause_event.set()
            self.logger.log("ritual_end", {})
    
    def start(self):
        if self._running:
            return
        self._running = True
        self._paused = False
        self._pause_event.set()
        asyncio.run(self.run())
        
    def pause(self):
        if not self._running:
            return
        self._paused = True
        self._pause_event.clear()
        
    def resume(self):
        if not self._running:
            return
        self._paused = False
        self._pause_event.set()
        
    def stop(self):
        self._running = False
        self._paused = False
        self._pause_event.set()
        
    def on_update_expansion(self, seq):
        pass
    
    def on_update_predictions(self, tokens):
        pass

# Event Logger
class EventLogger:
    def __init__(self):
        self.events = []
    def log(self, event_type, details):
        ts = datetime.now().isoformat()
        self.events.append({"timestamp": ts, "type": event_type, "details": details})
    def export_log(self, file):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(self.events, f, indent=2)

# Parameter Profile Manager Widget
class ProfileManagerWidget(QWidget):
    profile_changed = pyqtSignal(dict)
    
    def __init__(self, profile):
        super().__init__()
        self.profile = profile.copy()
        self.init_ui()
        
    def init_ui(self):
        layout = QFormLayout()
        self.recursion = QSpinBox()
        self.recursion.setRange(1, 10)
        self.recursion.setValue(self.profile.get("recursion_depth", 4))
        self.recursion.valueChanged.connect(self.update_profile)
        layout.addRow("Recursion Depth", self.recursion)
        
        self.speed = QSpinBox()
        self.speed.setRange(100, 3000)
        self.speed.setSingleStep(50)
        self.speed.setValue(self.profile.get("ritual_speed", 500))
        self.speed.valueChanged.connect(self.update_profile)
        layout.addRow("Ritual Speed (ms)", self.speed)
        
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Profile")
        self.load_btn = QPushButton("Load Profile")
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.load_btn)
        layout.addRow(btn_layout)
        
        self.save_btn.clicked.connect(self.save_profile)
        self.load_btn.clicked.connect(self.load_profile)
        self.setLayout(layout)
        
    def update_profile(self):
        self.profile['recursion_depth'] = self.recursion.value()
        self.profile['ritual_speed'] = self.speed.value()
        self.profile_changed.emit(self.profile)
        
    def save_profile(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Profile", "", "JSON Files (*.json)")
        if path:
            save_json_file(path, self.profile)
            QMessageBox.information(self, "Info", f"Profile saved to {path}")
            
    def load_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Profile", "", "JSON Files (*.json)")
        if path:
            p = load_json_file(path)
            if not isinstance(p, dict):
                QMessageBox.warning(self, "Error", "Invalid profile file!")
                return
            self.profile = p
            self.recursion.setValue(self.profile.get('recursion_depth', 4))
            self.speed.setValue(self.profile.get('ritual_speed', 500))
            self.profile_changed.emit(self.profile)
            QMessageBox.information(self, "Info", f"Profile loaded from {path}")

# Main GUI — combining everything
class RitualApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Ritual Orchestration")
        self.resize(1200, 900)
        
        self.archetypes = load_json_file('archetypes.json')
        self.grammar_rules = load_json_file('grammar_rules.json')
        
        self.logger = EventLogger()
        self.device = RichVirtualDeviceController()
        self.audio = RitualAudioSynth()
        self.transformer = RitualTransformerInterface()
        self.event_bus = asyncio.Queue()
        
        self.orch = RitualOrchestrator(self.archetypes, self.grammar_rules, self.transformer, self.device, self.logger, self.event_bus)
        self.orch.on_update_expansion = self.update_expansion
        self.orch.on_update_predictions = self.update_predictions
        self.device.state_changed.connect(self.device_status_update)
        self.device.log_generated.connect(self.append_device_log)
        
        self.profile_manager = ProfileManagerWidget({'recursion_depth': 4, 'ritual_speed': 500})
        self.profile_manager.profile_changed.connect(self.apply_profile)
        
        self.init_ui()
        
        self._async_loop = asyncio.new_event_loop()
        threading.Thread(target=self._async_loop.run_forever, daemon=True).start()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        
        self.start_btn.clicked.connect(self.start_ritual)
        self.stop_btn.clicked.connect(self.stop_ritual)
        
        layout.addLayout(controls)
        
        self.device_status = QLabel("Device Status: Idle")
        layout.addWidget(self.device_status)
        
        self.device_log = QTextEdit()
        self.device_log.setReadOnly(True)
        layout.addWidget(self.device_log)
        
        self.fractal_visual = FractalVisualWidget()
        self.fractal_visual.setMinimumHeight(200)
        layout.addWidget(self.fractal_visual)
        
        self.token_list = QListWidget()
        layout.addWidget(QLabel("Ritual Tokens"))
        layout.addWidget(self.token_list)
        
        layout.addWidget(self.profile_manager)
        
        self.setLayout(layout)
        
    def start_ritual(self):
        self.append_device_log("Starting ritual...")
        asyncio.run_coroutine_threadsafe(self.orch.run(), self._async_loop)
        
    def stop_ritual(self):
        self.orch.stop()
        self.append_device_log("Ritual stopped.")
    
    def update_expansion(self, seq):
        self.token_list.clear()
        for t in seq:
            it = QListWidgetItem(t.get("symbol", "?"))
            col = {"Yang": "#E63946", "Yin": "#1D3557", "Balanced": "#2A9D8F"}.get(t.get("polarity", "None"), "#6c757d")
            it.setForeground(QColor(col))
            self.token_list.addItem(it)
            
    def update_predictions(self, tokens):
        for t in tokens:
            pol = "Balanced"
            for a in self.archetypes:
                if a["symbol"] == t:
                    pol = a.get("polarity", pol)
                    break
            self.audio.play_tone_for_token(t, pol)
            
    def device_status_update(self, state):
        stat = state.get("status", "Unknown")
        power = state.get("coil_power", 0)
        sensor = state.get("sensor_value", 0)
        self.device_status.setText(f"Device: {stat} | Power: {power} | Sensor: {sensor:.2f}")
        
    def append_device_log(self, msg):
        self.device_log.append(msg)
        
    def apply_profile(self, profile):
        self.orch.max_depth = profile.get("recursion_depth", self.orch.max_depth)
        self.append_device_log(f"Applied profile with recursion {self.orch.max_depth}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RitualApp()
    gui.show()
    sys.exit(app.exec_())
    import sys
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QListWidget, QFileDialog, QApplication
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Utility functions for token frequency and timing analysis

def compute_token_frequency(events):
    freq = {}
    for e in events:
        if e['type'] == 'ritual_token':
            token = e['details'].get('token')
            freq[token] = freq.get(token, 0) + 1
    return freq

def compute_inter_token_intervals(events):
    timestamps = [datetime.fromisoformat(e['timestamp']) for e in events if e['type'] == 'ritual_token']
    intervals = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i-1]).total_seconds()
        intervals.append(delta)
    return intervals

class MatplotlibPlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_bar(self, data, title="Bar Chart", xlabel="", ylabel=""):
        self.ax.clear()
        keys = list(data.keys())
        values = list(data.values())
        self.ax.bar(keys, values, color='teal')
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.canvas.draw()

    def plot_hist(self, data, bins=10, title="Histogram", xlabel="", ylabel=""):
        self.ax.clear()
        self.ax.hist(data, bins=bins, color='coral', edgecolor='black')
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.canvas.draw()

class RitualReplayController:
    def __init__(self, events, update_expansion_cb, update_prediction_cb, log_cb):
        self.events = [e for e in events if e['type'] == 'ritual_token']
        self.update_expansion = update_expansion_cb
        self.update_prediction = update_prediction_cb
        self.log = log_cb
        self.index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_event)
        self.running = False

    def start(self):
        if not self.running and self.events:
            self.index = 0
            self.running = True
            self.log("Replay started")
            self.timer.start(500)

    def pause(self):
        if self.running:
            self.timer.stop()
            self.running = False
            self.log("Replay paused")

    def resume(self):
        if not self.running and self.index < len(self.events):
            self.timer.start(500)
            self.running = True
            self.log("Replay resumed")

    def stop(self):
        self.timer.stop()
        self.running = False
        self.index = 0
        self.log("Replay stopped")

    def next_event(self):
        if self.index >= len(self.events):
            self.stop()
            self.log("Replay ended")
            return
        event = self.events[self.index]
        token = event['details'].get('token')
        self.update_expansion([{"symbol": token, "polarity": "Balanced"}])
        self.update_prediction([token])
        self.log(f"Replay token: {token} ({self.index + 1}/{len(self.events)})")
        self.index += 1

class SessionAnalyticsReplayWidget(QWidget):
    def __init__(self, update_expansion_cb, update_prediction_cb):
        super().__init__()
        self.setWindowTitle("Session Analytics & Replay")
        self.resize(800, 600)

        self.events = []
        self.replay_controller = None
        self.update_expansion = update_expansion_cb
        self.update_prediction = update_prediction_cb

        layout = QVBoxLayout()

        # Buttons for loading and replay control
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Session Log")
        self.start_btn = QPushButton("Start Replay")
        self.pause_btn = QPushButton("Pause Replay")
        self.resume_btn = QPushButton("Resume Replay")
        self.stop_btn = QPushButton("Stop Replay")
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.resume_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        # Analytics plots
        self.freq_plot = MatplotlibPlotWidget()
        self.freq_label = QLabel("Token Frequency")
        layout.addWidget(self.freq_label)
        layout.addWidget(self.freq_plot)

        self.interval_plot = MatplotlibPlotWidget()
        self.interval_label = QLabel("Inter-Token Intervals (seconds)")
        layout.addWidget(self.interval_label)
        layout.addWidget(self.interval_plot)

        # Replay log and status
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(QLabel("Replay Log"))
        layout.addWidget(self.log_text)

        self.setLayout(layout)

        # Connect buttons
        self.load_btn.clicked.connect(self.load_session_log)
        self.start_btn.clicked.connect(self.start_replay)
        self.pause_btn.clicked.connect(self.pause_replay)
        self.resume_btn.clicked.connect(self.resume_replay)
        self.stop_btn.clicked.connect(self.stop_replay)

    def load_session_log(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Session Log", "", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.events = json.load(f)
        except Exception as e:
            self.log_text.append(f"Failed to load log: {e}")
            return

        self.log_text.append(f"Loaded session log from {file_path}")

        # Compute and plot token frequency
        freq = compute_token_frequency(self.events)
        self.freq_plot.plot_bar(freq, xlabel="Tokens", ylabel="Count")

        # Compute and plot inter-token timing intervals
        intervals = compute_inter_token_intervals(self.events)
        if intervals:
            self.interval_plot.plot_hist(intervals, bins=20, xlabel="Seconds", ylabel="Frequency")
        else:
            self.interval_plot.ax.clear()
            self.interval_plot.canvas.draw()

        # Enable replay buttons
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        # Setup replay controller with callbacks
        self.replay_controller = RitualReplayController(self.events, self.update_expansion, self.update_prediction, self.append_log)

    def start_replay(self):
        if self.replay_controller:
            self.replay_controller.start()
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    def pause_replay(self):
        if self.replay_controller:
            self.replay_controller.pause()
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)

    def resume_replay(self):
        if self.replay_controller:
            self.replay_controller.resume()
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)

    def stop_replay(self):
        if self.replay_controller:
            self.replay_controller.stop()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def append_log(self, text):
        self.log_text.append(text)


if __name__ == "__main__":
    # For standalone test
    app = QApplication(sys.argv)
    w = SessionAnalyticsReplayWidget(lambda e: print("Expansion:", e), lambda p: print("Prediction:", p))
    w.show()
    sys.exit(app.exec_())