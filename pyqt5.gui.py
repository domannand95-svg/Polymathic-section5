import sys
import asyncio
import threading
import json
import random
import time
from datetime import datetime
import pygame
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidgetItem, QLabel, QTextEdit, QSplitter, QListWidget,
    QSlider, QFormLayout, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QBrush, QPainter, QPen

# Initialize pygame mixer for audio
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# --- Data Loading ---
def load_archetypes(filepath='archetypes.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_grammar_rules(filepath='grammar_rules.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get('rules', {})

# --- Fractal Grammar Expansion (unchanged) ---
def archetype_by_symbol(symbol, archetypes):
    for arch in archetypes:
        if arch['symbol'] == symbol:
            return arch
    return None

def weighted_random_choice(expansions):
    total_weight = sum(exp.get('weight',1.0) for exp in expansions)
    r = random.uniform(0, total_weight)
    upto = 0
    for exp in expansions:
        w = exp.get('weight',1.0)
        if upto + w >= r:
            return exp
        upto += w
    return expansions[0] if expansions else None

def expand_symbol(symbol, archetypes, grammar_rules, max_depth=4, current_depth=0):
    if current_depth >= max_depth or symbol not in grammar_rules:
        arch = archetype_by_symbol(symbol, archetypes)
        return [{
            "symbol": arch.get("symbol", symbol) if arch else symbol,
            "polarity": arch.get("polarity", "Balanced") if arch else "Balanced",
            "element": arch.get("element", "None") if arch else "None",
            "gender": arch.get("gender", "neutral") if arch else "neutral",
        }]
    rule = grammar_rules[symbol]
    expansions = rule.get("expansions", [])
    if rule.get("optional", False) and random.random() < 0.1:
        return []
    expansion = weighted_random_choice(expansions)
    if expansion is None:
        arch = archetype_by_symbol(symbol, archetypes)
        return [{
            "symbol": arch.get("symbol", symbol) if arch else symbol,
            "polarity": arch.get("polarity", "Balanced") if arch else "Balanced",
            "element": arch.get("element", "None") if arch else "None",
            "gender": arch.get("gender", "neutral") if arch else "neutral",
        }]
    sequence = expansion.get("sequence", [symbol])
    result = []
    for sym in sequence:
        result.extend(expand_symbol(sym, archetypes, grammar_rules, max_depth, current_depth + 1))
    return result

# --- Dummy Transformer ---
class RitualTransformerInterface:
    def prepare_and_infer(self, symbolic_sequence):
        return [token['symbol'] for token in symbolic_sequence] + ["ת", "ש"]

# --- Audio (pygame) Synthesizer ---
POLARITY_NOTE_MAP = {
    "Yang": 72,      # C5
    "Yin": 60,       # C4
    "Balanced": 67,  # G4
    "None": 64       # E4
}

async def ritual_action_play_note(symbol, archetype):
    polarity = archetype.get('polarity', 'None')
    note = POLARITY_NOTE_MAP.get(polarity, 64)
    print(f"Playing note {note} for {symbol} ({polarity})")
    # Placeholder: Sleep simulates note duration
    await asyncio.sleep(0.5)

async def ritual_action_default(symbol):
    print(f"Ritual action: {symbol} - Default")
    await asyncio.sleep(0.3)

# --- Device Abstraction (stub) ---
class DeviceController:
    def __init__(self):
        self.connected = False  # Stub: no actual device

    async def send_command(self, command):
        print(f"Device command: {command}")
        await asyncio.sleep(0)  # Simulate async

    async def trigger_coil(self, intensity, duration):
        await self.send_command(f"coil {intensity}")
        await asyncio.sleep(duration)
        await self.send_command("coil off")

# --- Event Logger ---
class EventLogger:
    def __init__(self):
        self.events = []

    def log(self, event_type, details):
        timestamp = datetime.now().isoformat()
        entry = {"timestamp": timestamp, "type": event_type, "details": details}
        self.events.append(entry)
        print(f"Log Event: {entry}")

# --- Async Event Bus ---
class AsyncEventBus:
    def __init__(self):
        self.subscribers = []

    def subscribe(self, coro):
        self.subscribers.append(coro)

    async def publish(self, event):
        await asyncio.gather(*(c(event) for c in self.subscribers))

# --- Ritual Orchestrator ---
class RitualOrchestrator:
    def __init__(self, archetypes, grammar_rules, transformer, device_ctrl, logger, event_bus):
        self.archetypes = archetypes
        self.grammar_rules = grammar_rules
        self.transformer = transformer
        self.device_ctrl = device_ctrl
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
        expanded_seq = expand_symbol("S", self.archetypes, self.grammar_rules, self.max_depth)
        predicted_tokens = self.transformer.prepare_and_infer(expanded_seq)
        return expanded_seq, predicted_tokens

    async def _trigger_ritual_actions(self, tokens):
        for token in tokens:
            await self._wait_if_paused()
            if not self._running:
                break
            arch = self.archetypes.get(token, {"polarity":"None"})
            self.logger.log("ritual_token", {"token": token, "attributes": arch})
            await self.event_bus.publish({"token": token, "attributes": arch})
            await ritual_action_play_note(token, arch)
            await self.device_ctrl.send_command(f"Trigger {token}")

    async def run(self):
        self._running = True
        self._paused = False
        self._pause_event.set()
        self.logger.log("ritual_start", {})
        try:
            expanded_seq, predicted_tokens = await self._expand_and_predict()
            self.on_update_expansion(expanded_seq)
            self.on_update_predictions(predicted_tokens)
            await self._trigger_ritual_actions(predicted_tokens)
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

    # Hooks for GUI
    def on_update_expansion(self, expanded_seq):
        pass

    def on_update_predictions(self, predicted_tokens):
        pass

# --- Glyph Rendering with Animated Color Shifts ---
class AnimatedGlyphWidget(QListWidget):
    def __init__(self, archetypes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.archetypes = {a['symbol']: a for a in archetypes}
        self.symbol_font = QFont("Segoe UI Symbol", 28)
        self._colors = {
            "Yang": QColor("#E63946"),
            "Yin": QColor("#1D3557"),
            "Balanced": QColor("#2A9D8F"),
            "None": QColor("#6c757d"),
        }
        self.base_colors = self._colors.copy()
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._animate_colors)
        self.animation_timer.start(100)  # 10fps animation
        self._phase = 0

    def add_glyphs(self, symbols):
        self.clear()
        for sym in symbols:
            item = QListWidgetItem(sym)
            item.setFont(self.symbol_font)
            arch = self.archetypes.get(sym)
            if arch:
                polarity = arch.get('polarity', 'None')
                item.setForeground(QBrush(self._colors.get(polarity, QColor("#000000"))))
            self.addItem(item)
        self._base_symbols = symbols

    def _animate_colors(self):
        # Simple pulsating color effect on glyphs by shifting RGB channels
        new_colors = {}
        self._phase += 1
        import math
        for polarity, color in self.base_colors.items():
            hue_shift = (math.sin(self._phase / 10) + 1) / 2  # 0..1 oscillation
            r = min(max(int(color.red() * hue_shift), 0), 255)
            g = min(max(int(color.green() * hue_shift), 0), 255)
            b = min(max(int(color.blue() * hue_shift), 0), 255)
            new_colors[polarity] = QColor(r, g, b)
        self._colors = new_colors
        # Update item colors
        for idx in range(self.count()):
            item = self.item(idx)
            sym = item.text()
            arch = self.archetypes.get(sym)
            if arch:
                polarity = arch.get('polarity', 'None')
                item.setForeground(QBrush(self._colors.get(polarity, QColor("#000000"))))

# --- GUI ---
class Communicator(QObject):
    update_expansion = pyqtSignal(list)
    update_predictions = pyqtSignal(list)

class RitualGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Ritual Orchestration System")
        self.resize(1000, 700)

        self.comm = Communicator()
        self.comm.update_expansion.connect(self.update_expansion_view)
        self.comm.update_predictions.connect(self.update_predictions_view)

        self.archetypes = load_archetypes()
        self.grammar_rules = load_grammar_rules()
        self.device_controller = DeviceController()
        self.event_logger = EventLogger()
        self.event_bus = AsyncEventBus()
        self.transformer = RitualTransformerInterface()
        self.orch = RitualOrchestrator(
            self.archetypes,
            self.grammar_rules,
            self.transformer,
            self.device_controller,
            self.event_logger,
            self.event_bus
        )
        self.orch.on_update_expansion = lambda seq: self.comm.update_expansion.emit(seq)
        self.orch.on_update_predictions = lambda tokens: self.comm.update_predictions.emit(tokens)

        self._async_loop = None

        # Subscribe widgets to ritual token events for multimedia effects
        self.event_bus.subscribe(self.on_ritual_token_event)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Ritual")
        self.pause_btn = QPushButton("Pause Ritual")
        self.resume_btn = QPushButton("Resume Ritual")
        self.stop_btn = QPushButton("Stop Ritual")

        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.resume_btn)
        button_layout.addWidget(self.stop_btn)

        # Expanded ritual glyphs with animated visuals
        self.expansion_list = AnimatedGlyphWidget(self.archetypes)
        self.expansion_label = QLabel("Expanded Ritual Symbols")
        self.expansion_list.currentTextChanged.connect(self.show_archetype_details)

        # Predicted tokens list
        self.predicted_list = QListWidget()
        self.predicted_label = QLabel("Predicted Ritual Tokens")

        # Archetype details text
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_label = QLabel("Archetype Attributes")

        # Ritual parameter controls (recursion depth and pacing)
        param_layout = QFormLayout()
        self.recursion_depth_spin = QSpinBox()
        self.recursion_depth_spin.setMinimum(1)
        self.recursion_depth_spin.setMaximum(8)
        self.recursion_depth_spin.setValue(4)
        self.recursion_depth_spin.valueChanged.connect(self.update_recursion_depth)

        self.ritual_speed_spin = QSpinBox()
        self.ritual_speed_spin.setMinimum(100)
        self.ritual_speed_spin.setMaximum(2000)
        self.ritual_speed_spin.setValue(500)
        self.ritual_speed_spin.setSingleStep(100)
        self.ritual_speed_spin.valueChanged.connect(self.update_ritual_speed)

        param_layout.addRow("Recursion Depth:", self.recursion_depth_spin)
        param_layout.addRow("Ritual Speed (ms):", self.ritual_speed_spin)
        param_widget = QWidget()
        param_widget.setLayout(param_layout)

        # Left panel: expansion glyphs + parameters
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.expansion_label)
        left_layout.addWidget(self.expansion_list)
        left_layout.addWidget(param_widget)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # Right panel: predicted tokens + archetype details
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.predicted_label)
        right_layout.addWidget(self.predicted_list)
        right_layout.addWidget(self.details_label)
        right_layout.addWidget(self.details_text)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])

        main_layout.addLayout(button_layout)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Buttons connected
        self.start_btn.clicked.connect(self.start_ritual)
        self.pause_btn.clicked.connect(self.pause_ritual)
        self.resume_btn.clicked.connect(self.resume_ritual)
        self.stop_btn.clicked.connect(self.stop_ritual)

    def update_recursion_depth(self, val):
        self.orch.max_depth = val
        self.event_logger.log("param_change", {"recursion_depth": val})

    def update_ritual_speed(self, val):
        # Placeholder for speed usage (can use in async delays in ritual actions)
        self.event_logger.log("param_change", {"ritual_speed_ms": val})

    # Async event subscriber example
    async def on_ritual_token_event(self, event):
        # React to ritual token events for multimedia sync
        token = event.get("token")
        attrs = event.get("attributes", {})
        print(f"Multimedia visual/audio sync event received for token {token} attrs {attrs}")

    def start_ritual(self):
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        if not self._async_loop:
            self._async_loop = asyncio.new_event_loop()
            threading.Thread(target=self._async_loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self.orch.run(), self._async_loop)

    def pause_ritual(self):
        self.orch.pause()
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(True)

    def resume_ritual(self):
        self.orch.resume()
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)

    def stop_ritual(self):
        self.orch.stop()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def update_expansion_view(self, expanded_seq):
        symbols = [token['symbol'] for token in expanded_seq]
        self.expansion_list.add_glyphs(symbols)

    def update_predictions_view(self, tokens):
        self.predicted_list.clear()
        for t in tokens:
            self.predicted_list.addItem(t)

    def show_archetype_details(self, symbol):
        if not symbol:
            self.details_text.clear()
            return
        arch = next((a for a in self.archetypes if a['symbol'] == symbol), None)
        if arch:
            details = (
                f"Symbol: {arch['symbol']}
"
                f"Polarity: {arch['polarity']}
"
                f"Element: {arch['element']}
"
                f"Gender: {arch['gender']}"
            )
            self.details_text.setText(details)
        else:
            self.details_text.setText("No archetype data available.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RitualGUI()
    gui.show()
    sys.exit(app.exec_())
    import sys
import asyncio
import threading
import random
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit,
    QListWidget, QListWidgetItem, QApplication, QTabWidget
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QColor

# Simulated OSC/MIDI Network Device Client
class SimulatedDeviceClient(QObject):
    log_generated = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, device_name):
        super().__init__()
        self.device_name = device_name
        self.connected = False
        self._running = False
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self._simulate_status_update)

    def connect(self):
        self.connected = True
        self._running = True
        self.simulation_timer.start(1000)
        self.status_changed.emit(f"{self.device_name} connected")

    def disconnect(self):
        self.connected = False
        self._running = False
        self.simulation_timer.stop()
        self.status_changed.emit(f"{self.device_name} disconnected")

    def send_message(self, msg):
        if not self.connected:
            self.log_generated.emit(f"{self.device_name} not connected, cannot send message")
            return
        self.log_generated.emit(f"Sent to {self.device_name}: {msg}")

    def _simulate_status_update(self):
        # Periodic status updates or feedback simulation
        if self.connected:
            sensor_val = random.randint(0,100)
            self.status_changed.emit(f"{self.device_name} sensor value {sensor_val}")
            self.log_generated.emit(f"{self.device_name} sensor value updated: {sensor_val}")

# Network Multimedia Control Widget
class NetworkMultimediaWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network Multimedia Device Integration")
        layout = QVBoxLayout()

        self.devices = {}
        self.device_list = QListWidget()
        layout.addWidget(QLabel("Network Devices"))
        layout.addWidget(self.device_list)

        btn_layout = QHBoxLayout()
        self.add_device_btn = QPushButton("Add Device")
        self.remove_device_btn = QPushButton("Remove Selected")
        btn_layout.addWidget(self.add_device_btn)
        btn_layout.addWidget(self.remove_device_btn)
        layout.addLayout(btn_layout)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        layout.addWidget(QLabel("Device Logs"))
        layout.addWidget(self.status_text)

        self.setLayout(layout)

        self.add_device_btn.clicked.connect(self.add_device)
        self.remove_device_btn.clicked.connect(self.remove_selected_device)

    def add_device(self):
        name = f"Device-{len(self.devices)+1}"
        client = SimulatedDeviceClient(name)
        client.log_generated.connect(self.append_log)
        client.status_changed.connect(self.update_device_status)
        self.devices[name] = client
        self.device_list.addItem(name)
        client.connect()

    def remove_selected_device(self):
        current = self.device_list.currentItem()
        if current:
            name = current.text()
            client = self.devices.get(name)
            if client:
                client.disconnect()
                del self.devices[name]
            self.device_list.takeItem(self.device_list.row(current))

    def append_log(self, text):
        self.status_text.append(text)

    def update_device_status(self, status):
        self.status_text.append(f"Status: {status}")

    def send_message_to_all(self, msg):
        for client in self.devices.values():
            client.send_message(msg)

# Main GUI with Network Multimedia Tab Integration
class IntegratedRitualSystemWithNetwork(QWidget):
    def __init__(self, ritual_app_instance):
        super().__init__()
        self.setWindowTitle("Ritual Orchestration System with Network Multimedia")
        self.resize(1300, 900)

        self.ritual_app = ritual_app_instance

        self.tabs = QTabWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)

        self.tabs.addTab(self.ritual_app, "Ritual Orchestration")
        self.network_widget = NetworkMultimediaWidget()
        self.tabs.addTab(self.network_widget, "Network Multimedia")

        self.setLayout(main_layout)

        # Example: connect ritual events to network message sends
        # Here you should connect your ritual event bus or callbacks to send device messages
        # For demo, we simulate sending a message on ritual start:
        self.ritual_app.start_btn.clicked.connect(self.send_start_message)

    def send_start_message(self):
        self.network_widget.send_message_to_all("Ritual started")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Import or define your existing ritual app class here; for example purpose, placeholder:
    class DummyRitualApp(QWidget):
        def __init__(self):
            super().__init__()
            layout=QVBoxLayout()
            self.start_btn = QPushButton("Start Ritual")
            layout.addWidget(self.start_btn)
            self.setLayout(layout)
    ritual_app = DummyRitualApp()

    main_win = IntegratedRitualSystemWithNetwork(ritual_app)
    main_win.show()
    sys.exit(app.exec_())
    