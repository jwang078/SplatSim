"""Tkinter-based GUI components for SplatSim.

This module provides modular, thread-safe GUI components that can be used
to build control panels for SplatSim simulations.
"""

import threading
import tkinter as tk
from tkinter import ttk
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageTk


# =============================================================================
# Configuration Types
# =============================================================================

@dataclass
class GuiStyle:
    """Style configuration for the GUI."""
    font_family: str = "Helvetica"
    font_size: int = 11
    font_size_header: int = 13
    font_size_button: int = 11
    padding: str = "15"


@dataclass
class IntParam:
    """Configuration for an integer parameter field."""
    key: str
    label: str
    min_val: int = 0
    max_val: int = 100
    default: int = 0


@dataclass
class FloatParam:
    """Configuration for a float parameter with slider."""
    key: str
    label: str
    min_val: float = 0.0
    max_val: float = 1.0
    default: float = 0.0
    slider_length: int = 120


@dataclass
class BoolParam:
    """Configuration for a boolean checkbox parameter."""
    key: str
    label: str
    default: bool = False


@dataclass
class StrParam:
    """Configuration for a string text entry parameter."""
    key: str
    label: str
    default: str = ""
    width: int = 20


@dataclass
class EnumParam:
    """Configuration for an enum dropdown parameter."""
    key: str
    label: str
    enum_class: type  # The enum class to use
    default: Any = None  # Default enum member (uses first if None)


@dataclass
class ButtonConfig:
    """Configuration for a button."""
    text: str
    callback_key: str
    style: str = "TButton"


@dataclass
class ButtonGroupConfig:
    """Configuration for a group of buttons."""
    buttons: List[ButtonConfig] = field(default_factory=list)


@dataclass
class SectionConfig:
    """Configuration for a GUI section."""
    header: Optional[str] = None
    int_params: List[IntParam] = field(default_factory=list)
    float_params: List[FloatParam] = field(default_factory=list)
    bool_params: List[BoolParam] = field(default_factory=list)
    button_groups: List[ButtonGroupConfig] = field(default_factory=list)


# =============================================================================
# Base GUI Class
# =============================================================================

class ThreadedTkinterGui(ABC):
    """Base class for threaded Tkinter GUIs.

    Provides thread-safe GUI management with automatic cleanup when the
    main thread exits.
    """

    def __init__(self, title: str = "GUI", style: Optional[GuiStyle] = None):
        """Initialize the GUI.

        Args:
            title: Window title
            style: Style configuration (uses defaults if not provided)
        """
        self._title = title
        self._style = style or GuiStyle()

        # Thread-safe storage
        self._values: Dict[str, tk.Variable] = {}
        self._button_flags: Dict[str, bool] = {}
        self._enum_classes: Dict[str, type] = {}  # Maps param key -> enum class
        self._lock = threading.Lock()

        # GUI state
        self._root: Optional[tk.Tk] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_requested = False
        self._main_thread = threading.main_thread()

    def start(self):
        """Start the GUI in a separate thread."""
        if self._running:
            return
        self._running = True
        self._shutdown_requested = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the GUI and close the window (thread-safe).

        Signals the GUI thread to shut down via _shutdown_requested flag,
        which is polled by _check_shutdown every 100ms. Waits for the GUI
        thread to finish to ensure Tk resources are cleaned up from the
        correct thread (avoiding Tcl_AsyncDelete crashes).
        """
        self._shutdown_requested = True
        self._running = False

        # Wait for the GUI thread to finish — _check_shutdown (running on the
        # Tk event loop) will see _shutdown_requested and call _destroy_window
        # from the correct thread.  Use a generous timeout so the Tk thread
        # has time to process the shutdown.
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        # If the thread is still alive after the timeout, the Tk mainloop is
        # stuck.  Null out _root so that __del__ / GC won't touch a live
        # interpreter from the wrong thread.
        if self._thread is not None and self._thread.is_alive():
            self._root = None

    def _destroy_window(self):
        """Destroy window - must be called from GUI thread.

        Explicitly unregisters tk.Variable objects from the Tcl interpreter
        before destroying the window, preventing 'main thread is not in main
        loop' errors when Python's GC later calls Variable.__del__.
        """
        if self._root:
            try:
                # Unregister each tk.Variable from the Tcl interpreter while
                # we're still in the GUI thread, then neuter the _tk reference
                # so that Variable.__del__ becomes a no-op.
                with self._lock:
                    for var in self._values.values():
                        try:
                            self._root.tk.globalunsetvar(var._name)
                        except tk.TclError:
                            pass
                        # Prevent __del__ from touching the (soon-dead) Tcl interp
                        var._tk = None
                    self._values.clear()
                    self._enum_classes.clear()
                self._root.quit()
                self._root.destroy()
            except tk.TclError:
                pass
        self._root = None

    def _run(self):
        """Main GUI thread loop."""
        self._root = tk.Tk()
        self._root.title(self._title)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._configure_styles()
        self._build_ui()
        self._check_shutdown()

        try:
            self._root.mainloop()
        except Exception:
            pass
        finally:
            # Ensure cleanup runs from this thread after mainloop exits,
            # regardless of how it exited.  This prevents Tcl_AsyncDelete
            # from aborting when the Tcl interpreter is torn down by the
            # wrong thread during process exit.
            self._destroy_window()

    def _configure_styles(self):
        """Configure ttk styles."""
        s = self._style
        default_font = (s.font_family, s.font_size)
        header_font = (s.font_family, s.font_size_header, "bold")
        button_font = (s.font_family, s.font_size_button)

        style = ttk.Style()
        style.configure("TLabel", font=default_font)
        style.configure("TButton", font=button_font, padding=5)
        style.configure("TCheckbutton", font=default_font)
        style.configure("TEntry", font=default_font)
        style.configure("Header.TLabel", font=header_font)
        style.configure("Mode.TButton", font=(s.font_family, s.font_size_button, "bold"), padding=8)

    @abstractmethod
    def _build_ui(self):
        """Build the UI components. Override in subclasses."""
        pass

    def _check_shutdown(self):
        """Periodically check if shutdown was requested or main thread died."""
        if not self._main_thread.is_alive():
            self._destroy_window()
            return

        if self._shutdown_requested:
            self._destroy_window()
        elif self._root:
            self._root.after(100, self._check_shutdown)

    def _on_close(self):
        """Handle window close event."""
        self._running = False
        self._destroy_window()

    def _register_button(self, key: str):
        """Register a button callback key."""
        with self._lock:
            self._button_flags[key] = False

    def _on_button_pressed(self, key: str):
        """Handle button press."""
        with self._lock:
            self._button_flags[key] = True

    def check_button(self, key: str) -> bool:
        """Check and clear a button press flag.

        Args:
            key: The button callback key

        Returns:
            True if button was pressed since last check
        """
        with self._lock:
            pressed = self._button_flags.get(key, False)
            self._button_flags[key] = False
        return pressed

    def check_buttons(self, keys: List[str]) -> Dict[str, bool]:
        """Check and clear multiple button press flags.

        Args:
            keys: List of button callback keys

        Returns:
            Dict mapping key -> pressed state
        """
        with self._lock:
            result = {key: self._button_flags.get(key, False) for key in keys}
            for key in keys:
                self._button_flags[key] = False
        return result

    def get_value(self, key: str) -> Any:
        """Get a single value from the GUI.

        Args:
            key: The parameter key

        Returns:
            The current value
        """
        if key not in self._values:
            return None
        try:
            return self._values[key].get()
        except tk.TclError:
            return None

    def get_enum_value(self, key: str) -> Any:
        """Get an enum value from the GUI.

        Args:
            key: The parameter key for an enum dropdown

        Returns:
            The enum member corresponding to the current selection, or None
        """
        if key not in self._values or key not in self._enum_classes:
            return None
        try:
            str_value = self._values[key].get()
            enum_class = self._enum_classes[key]
            # Find the enum member with matching value
            for member in enum_class:
                if member.value == str_value:
                    return member
            return None
        except tk.TclError:
            return None

    def get_values(self) -> Dict[str, Any]:
        """Get all current values from GUI.

        Returns:
            Dictionary of parameter key -> value
        """
        result = {}
        for key, var in self._values.items():
            try:
                result[key] = var.get()
            except tk.TclError:
                pass
        return result

    def save_to_config(self, config: Dict[str, Any]):
        """Save current GUI values to a config dict.

        Searches both top-level keys and nested sub-dicts for matching keys.

        Args:
            config: Config dict to update with current values
        """
        values = self.get_values()
        for key, value in values.items():
            # Search top-level keys first, then nested sub-dicts
            if key in config:
                config_type = type(config[key])
                if config_type == bool:
                    config[key] = bool(value)
                else:
                    config[key] = config_type(value)
            else:
                for _, sub_dict in config.items():
                    if isinstance(sub_dict, dict) and key in sub_dict:
                        config_type = type(sub_dict[key])
                        if config_type == bool:
                            sub_dict[key] = bool(value)
                        else:
                            sub_dict[key] = config_type(value)
                        break


# =============================================================================
# GUI Builder Helpers
# =============================================================================

class GuiBuilder:
    """Helper class for building GUI components."""

    def __init__(self, parent: tk.Widget, gui: ThreadedTkinterGui, style: GuiStyle):
        """Initialize the builder.

        Args:
            parent: Parent frame to add widgets to
            gui: The GUI instance (for registering values/buttons)
            style: Style configuration
        """
        self._parent = parent
        self._gui = gui
        self._style = style
        self._row = 0

    @property
    def current_row(self) -> int:
        """Get the current row index."""
        return self._row

    def add_header(self, text: str) -> int:
        """Add a section header.

        Args:
            text: Header text

        Returns:
            Row index
        """
        ttk.Label(self._parent, text=text, style="Header.TLabel").grid(
            row=self._row, column=0, columnspan=2, sticky="w", pady=(5, 8)
        )
        self._row += 1
        return self._row - 1

    def add_separator(self) -> int:
        """Add a horizontal separator.

        Returns:
            Row index
        """
        ttk.Separator(self._parent, orient="horizontal").grid(
            row=self._row, column=0, columnspan=2, sticky="ew", pady=10
        )
        self._row += 1
        return self._row - 1

    def add_int_param(self, param: IntParam, initial_value: Optional[int] = None) -> int:
        """Add an integer parameter entry field.

        Args:
            param: Parameter configuration
            initial_value: Initial value (overrides param.default)

        Returns:
            Row index
        """
        value = initial_value if initial_value is not None else param.default

        ttk.Label(self._parent, text=param.label).grid(
            row=self._row, column=0, sticky="w", pady=3
        )

        var = tk.IntVar(value=int(value))
        self._gui._values[param.key] = var

        font = (self._style.font_family, self._style.font_size)
        entry = ttk.Entry(self._parent, textvariable=var, width=10, font=font)
        entry.grid(row=self._row, column=1, sticky="e", pady=3)

        self._row += 1
        return self._row - 1

    def add_float_param(self, param: FloatParam, initial_value: Optional[float] = None) -> int:
        """Add a float parameter with slider and entry.

        Args:
            param: Parameter configuration
            initial_value: Initial value (overrides param.default)

        Returns:
            Row index
        """
        value = initial_value if initial_value is not None else param.default

        ttk.Label(self._parent, text=param.label).grid(
            row=self._row, column=0, sticky="w", pady=3
        )

        var = tk.DoubleVar(value=float(value))
        self._gui._values[param.key] = var

        slider_frame = ttk.Frame(self._parent)
        slider_frame.grid(row=self._row, column=1, sticky="e", pady=3)

        slider = ttk.Scale(
            slider_frame, from_=param.min_val, to=param.max_val,
            variable=var, orient="horizontal", length=param.slider_length
        )
        slider.pack(side="left")

        font = (self._style.font_family, self._style.font_size)
        entry = ttk.Entry(slider_frame, textvariable=var, width=7, font=font)
        entry.pack(side="left", padx=(5, 0))

        self._row += 1
        return self._row - 1

    def add_bool_param(self, param: BoolParam, initial_value: Optional[bool] = None) -> int:
        """Add a boolean checkbox parameter.

        Args:
            param: Parameter configuration
            initial_value: Initial value (overrides param.default)

        Returns:
            Row index
        """
        value = initial_value if initial_value is not None else param.default

        ttk.Label(self._parent, text=param.label).grid(
            row=self._row, column=0, sticky="w", pady=3
        )

        var = tk.BooleanVar(value=bool(value))
        self._gui._values[param.key] = var

        check = ttk.Checkbutton(self._parent, variable=var)
        check.grid(row=self._row, column=1, sticky="e", pady=3)

        self._row += 1
        return self._row - 1

    def add_str_param(self, param: 'StrParam', initial_value: Optional[str] = None) -> int:
        """Add a string text entry parameter.

        Args:
            param: Parameter configuration
            initial_value: Initial value (overrides param.default)

        Returns:
            Row index
        """
        value = initial_value if initial_value is not None else param.default

        ttk.Label(self._parent, text=param.label).grid(
            row=self._row, column=0, sticky="w", pady=3
        )

        var = tk.StringVar(value=str(value))
        self._gui._values[param.key] = var

        font = (self._style.font_family, self._style.font_size)
        entry = ttk.Entry(self._parent, textvariable=var, width=param.width, font=font)
        entry.grid(row=self._row, column=1, sticky="e", pady=3)

        self._row += 1
        return self._row - 1

    def add_enum_param(self, param: 'EnumParam', initial_value: Any = None) -> int:
        """Add an enum dropdown parameter.

        Args:
            param: Parameter configuration with enum_class
            initial_value: Initial enum member (overrides param.default)

        Returns:
            Row index
        """
        # Determine the initial value
        if initial_value is not None:
            value = initial_value
        elif param.default is not None:
            value = param.default
        else:
            # Use first enum member as default
            value = list(param.enum_class)[0]

        ttk.Label(self._parent, text=param.label).grid(
            row=self._row, column=0, sticky="w", pady=3
        )

        # Create a StringVar that stores the enum's value (not name)
        # Handle both enum members and string values
        if isinstance(value, str):
            var = tk.StringVar(value=value)
        else:
            var = tk.StringVar(value=value.value)
        self._gui._values[param.key] = var

        # Store enum class for later lookup
        self._gui._enum_classes[param.key] = param.enum_class

        # Create dropdown with enum values
        values = [member.value for member in param.enum_class]
        combo = ttk.Combobox(self._parent, textvariable=var, values=values, state="readonly", width=18)
        combo.grid(row=self._row, column=1, sticky="e", pady=3)

        self._row += 1
        return self._row - 1

    def add_button_row(self, buttons: List[ButtonConfig], colspan: int = 2) -> int:
        """Add a row of buttons.

        Args:
            buttons: List of button configurations
            colspan: Column span for the button frame

        Returns:
            Row index
        """
        btn_frame = ttk.Frame(self._parent)
        btn_frame.grid(row=self._row, column=0, columnspan=colspan, pady=8)

        for btn_config in buttons:
            self._gui._register_button(btn_config.callback_key)
            btn = ttk.Button(
                btn_frame, text=btn_config.text,
                command=lambda k=btn_config.callback_key: self._gui._on_button_pressed(k),
                style=btn_config.style
            )
            btn.pack(side="left", padx=5)

        self._row += 1
        return self._row - 1


# =============================================================================
# Mode Panels
# =============================================================================

class ModePanel:
    """Base class for a mode's settings panel in the GUI.

    Subclass to define a new mode. Each panel gets its own LabelFrame that is
    shown when the mode is active and hidden otherwise.

    To add a new mode:
        1. Subclass ModePanel
        2. Set name, mode_values, and button_key as class attributes
        3. Override build() to add widgets (or leave empty for a placeholder)
        4. Pass an instance to SplatSimGui via the panels list
    """

    name: str           # Display name (used for the button and LabelFrame title)
    mode_values: set    # SERVE_MODES .value strings this panel owns
    button_key: str     # Unique key for the mode-switch button
    default_mode: str   # Mode string to transition to when the button is pressed

    def __init__(self):
        self.frame: Optional[ttk.LabelFrame] = None

    def build(self, parent: tk.Widget, gui: 'ThreadedTkinterGui',
              style: GuiStyle, config: Dict[str, Any]) -> None:
        """Build mode-specific widgets inside parent. Override in subclasses."""
        pass

    def process_buttons(self, gui: 'SplatSimGui') -> None:
        """Handle panel-specific button presses. Override in subclasses.

        Called once per frame when this panel's mode is active.
        """
        pass


class InteractiveModePanel(ModePanel):
    """Interactive mode — no configurable settings yet."""

    name = "Interactive Mode"
    mode_values = {"interactive"}
    button_key = "interactive_mode"
    default_mode = "interactive"


class TrajectoryGenModePanel(ModePanel):
    """Trajectory generation mode with all its parameter controls."""

    name = "Trajectory Gen Mode"
    mode_values = {"generate_trajectories", "generate_trajectories_idle"}
    button_key = "traj_gen_mode"
    default_mode = "generate_trajectories_idle"

    BTN_START = "start_traj"
    BTN_STOP = "stop_traj"

    def build(self, parent: tk.Widget, gui: 'ThreadedTkinterGui',
              style: GuiStyle, config: Dict[str, Any]) -> None:
        builder = GuiBuilder(parent, gui, style)
        traj_config = config.get("trajectory_generator", {})

        builder.add_str_param(
            StrParam("experiment_name", "Experiment Name", "", width=25),
            traj_config.get("experiment_name", "")
        )

        int_params = [
            IntParam("num_base_trajectories", "Num Trajectories", 1, 1000),
            IntParam("obstacles_per_base_trajectory", "Obstacles/Traj", 0, 10),
            IntParam("paths_per_obstacle", "Paths/Obstacle", 1, 5),
            IntParam("min_obstacles", "Min Obstacles", 0, 5),
            IntParam("max_obstacles", "Max Obstacles", 1, 10),
            IntParam("num_path_candidates", "Path Candidates", 1, 20),
        ]
        for param in int_params:
            builder.add_int_param(param, traj_config.get(param.key))

        float_params = [
            FloatParam("k_exp", "k_exp", 0.1, 20.0),
            FloatParam("k_sig", "k_sig", 0.1, 30.0),
            FloatParam("threshold", "threshold", 0.0, 1.0),
        ]
        for param in float_params:
            builder.add_float_param(param, traj_config.get(param.key))

        builder.add_bool_param(
            BoolParam("disable_camera_scoring_for_rrt", "Disable Cam Score"),
            traj_config.get("disable_camera_scoring_for_rrt", False)
        )

        builder.add_button_row([
            ButtonConfig("Start Traj Gen", self.BTN_START),
            ButtonConfig("Stop", self.BTN_STOP),
        ])

    def process_buttons(self, gui: 'SplatSimGui') -> None:
        start = gui.check_button(self.BTN_START)
        stop = gui.check_button(self.BTN_STOP)

        if start and gui.mode == "generate_trajectories_idle":
            gui.save_to_config(gui._config)
            gui.set_mode("generate_trajectories")
            print(f"[GUI] Started trajectory generation with config: {gui._config}")

        if stop:
            if gui.mode == "generate_trajectories":
                gui.set_mode("generate_trajectories_idle")
            elif gui.mode == "generate_trajectories_idle":
                gui.set_mode("interactive")


# =============================================================================
# SplatSim GUI Implementation
# =============================================================================

class SplatSimGui(ThreadedTkinterGui):
    """Tkinter-based GUI for SplatSim controls.

    Modes are defined as ModePanel subclasses and passed via the panels list.
    Each panel gets a mode-switch button and a LabelFrame that is shown/hidden
    automatically when the mode changes.
    """

    # Key for debug mode dropdown
    DEBUG_MODE_KEY = "debug_mode"

    def __init__(
        self,
        config: Dict[str, Any],
        initial_mode: str = "interactive",
        debug_mode_enum: Optional[type] = None,
        initial_debug_mode: Any = None,
        panels: Optional[List[ModePanel]] = None,
    ):
        """Initialize the GUI.

        Args:
            config: Dictionary of config values passed to each panel's build()
            initial_mode: Initial mode string (e.g., "interactive")
            debug_mode_enum: The DEBUG_MODES enum class (optional)
            initial_debug_mode: Initial debug mode enum member (optional)
            panels: List of ModePanel instances. Defaults to
                     [InteractiveModePanel(), TrajectoryGenModePanel()].
        """
        super().__init__(title="SplatSim Controls")
        self._config = config
        self._initial_mode = initial_mode
        self._mode_var: Optional[tk.StringVar] = None
        self._debug_mode_enum = debug_mode_enum
        self._initial_debug_mode = initial_debug_mode
        self._panels = panels if panels is not None else [
            InteractiveModePanel(),
            TrajectoryGenModePanel(),
        ]

        # Mode state (thread-safe — written by GUI thread, read by main thread)
        self._current_mode = initial_mode
        self._mode_lock = threading.Lock()

        # Build lookup: button_key -> panel
        self._panel_by_button: Dict[str, ModePanel] = {
            p.button_key: p for p in self._panels
        }

        # Camera image display state
        self._camera_frame: Optional[ttk.LabelFrame] = None
        self._camera_labels: Dict[str, tk.Label] = {}
        self._camera_name_labels: Dict[str, ttk.Label] = {}
        self._photo_refs: Dict[str, ImageTk.PhotoImage] = {}
        self._pending_images: Optional[Dict[str, np.ndarray]] = None
        self._image_lock = threading.Lock()

    def _build_ui(self):
        """Build the SplatSim UI."""
        main_frame = ttk.Frame(self._root, padding=self._style.padding)
        main_frame.grid(row=0, column=0, sticky="nsew")

        builder = GuiBuilder(main_frame, self, self._style)

        # Camera observations display area (at the top)
        builder.add_header("Camera Observations")
        self._camera_frame = ttk.LabelFrame(main_frame, text="", padding=5)
        self._camera_frame.grid(
            row=builder.current_row, column=0, columnspan=2, sticky="nsew", pady=5
        )
        builder._row += 1

        # Start polling for image updates
        self._poll_camera_images()

        builder.add_separator()

        # Current mode status display
        self._mode_var = tk.StringVar(value=f"Mode: {self._initial_mode}")
        self._values["_mode_var"] = self._mode_var
        mode_label = ttk.Label(main_frame, textvariable=self._mode_var, style="Header.TLabel")
        mode_label.grid(row=builder.current_row, column=0, columnspan=2, sticky="w", pady=(0, 10))
        builder._row += 1

        # Mode selection buttons (one per panel)
        builder.add_button_row([
            ButtonConfig(p.name, p.button_key, "Mode.TButton") for p in self._panels
        ])

        # Debug mode dropdown (if enum provided)
        if self._debug_mode_enum is not None:
            builder.add_separator()
            builder.add_header("Debug Settings")
            builder.add_enum_param(
                EnumParam(self.DEBUG_MODE_KEY, "Debug Mode", self._debug_mode_enum),
                self._initial_debug_mode
            )

        # Build each panel's settings frame
        for panel in self._panels:
            panel.frame = ttk.LabelFrame(main_frame, text=panel.name, padding=10)
            panel.frame.grid(
                row=builder.current_row, column=0, columnspan=2,
                sticky="nsew", pady=(5, 5)
            )
            # Ensure the titled border is visible even if build() adds no widgets
            panel.frame.configure(height=40)
            panel.frame.grid_propagate(False)
            builder._row += 1
            panel.build(panel.frame, self, self._style, self._config)
            # Re-enable propagation if build() added widgets, so the frame
            # grows to fit them. If still empty, the minimum height holds.
            if panel.frame.winfo_children():
                panel.frame.grid_propagate(True)

        # Show/hide panels based on initial mode
        self._update_panel_visibility(self._initial_mode)

    # ------------------------------------------------------------------
    # Mode state & transitions
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Current mode string (thread-safe read)."""
        with self._mode_lock:
            return self._current_mode

    def set_mode(self, mode: str):
        """Set the current mode, update the display, and show/hide panels.

        Thread-safe — can be called from any thread.

        Args:
            mode: The mode string (e.g., "interactive", "generate_trajectories")
        """
        with self._mode_lock:
            old = self._current_mode
            self._current_mode = mode
        if old != mode:
            print(f"[GUI] Mode: {old} -> {mode}")
        if self._mode_var is not None:
            try:
                self._mode_var.set(f"Mode: {mode}")
            except tk.TclError:
                pass
        self._update_panel_visibility(mode)

    def process_mode_transitions(self) -> None:
        """Process mode button presses and panel-specific button logic.

        Call this once per frame from the main loop. It handles:
        - Mode-switch buttons (one per panel)
        - Panel-specific buttons (e.g. Start/Stop for TrajectoryGenModePanel)
        """
        # Check if any mode-switch button was pressed
        for panel in self._panels:
            if not self.check_button(panel.button_key):
                continue
            # Ignore if already in this mode group
            if self.mode in panel.mode_values:
                break
            # Transition to the panel's default mode (first in mode_values)
            target = panel.default_mode
            self.set_mode(target)
            break

        # Let each panel handle its own buttons
        for panel in self._panels:
            if self.mode in panel.mode_values:
                panel.process_buttons(self)

    def _update_panel_visibility(self, mode: str) -> None:
        """Show the panel that owns this mode, hide all others."""
        for panel in self._panels:
            if panel.frame is None:
                continue
            try:
                if mode in panel.mode_values:
                    panel.frame.grid()
                else:
                    panel.frame.grid_remove()
            except tk.TclError:
                pass

    def get_debug_mode(self) -> Any:
        """Get the current debug mode selection.

        Returns:
            The DEBUG_MODES enum member, or None if not available.
        """
        return self.get_enum_value(self.DEBUG_MODE_KEY)

    # ------------------------------------------------------------------
    # Camera image display
    # ------------------------------------------------------------------

    def update_camera_images(self, frames: Dict[str, np.ndarray]) -> None:
        """Update camera observation images (thread-safe).

        Can be called from any thread. Images will be displayed on the next
        Tkinter poll cycle.

        Args:
            frames: Dict mapping camera name to RGB numpy array (H, W, 3), uint8.
        """
        with self._image_lock:
            self._pending_images = dict(frames)

    def _poll_camera_images(self) -> None:
        """Poll for pending image updates from other threads."""
        if not self._root or self._shutdown_requested:
            return

        with self._image_lock:
            pending = self._pending_images
            self._pending_images = None

        if pending is not None:
            self._render_camera_images(pending)

        self._root.after(33, self._poll_camera_images)  # ~30 fps

    def _render_camera_images(self, frames: Dict[str, np.ndarray]) -> None:
        """Render camera images into the GUI. Must be called from Tk thread."""
        if self._camera_frame is None:
            return

        # Determine grid layout: arrange in rows of up to 2 columns
        camera_names = sorted(frames.keys())
        num_cameras = len(camera_names)
        if num_cameras == 0:
            return
        cols = min(num_cameras, 2)
        max_thumb_width = 224

        # Remove labels for cameras that are no longer present
        stale = set(self._camera_labels.keys()) - set(camera_names)
        for name in stale:
            self._camera_labels[name].destroy()
            del self._camera_labels[name]
            self._camera_name_labels[name].destroy()
            del self._camera_name_labels[name]
            self._photo_refs.pop(name, None)

        for idx, name in enumerate(camera_names):
            row = (idx // cols) * 2  # *2 because label + image per camera
            col = idx % cols

            img_array = frames[name]
            h, w = img_array.shape[:2]
            scale = min(max_thumb_width / w, max_thumb_width / h)
            new_w, new_h = int(w * scale), int(h * scale)

            pil_img = Image.fromarray(img_array).resize(
                (new_w, new_h), Image.BILINEAR
            )
            photo = ImageTk.PhotoImage(pil_img)
            self._photo_refs[name] = photo

            # Create or update the name label
            if name not in self._camera_name_labels:
                lbl = ttk.Label(self._camera_frame, text=name, style="TLabel")
                lbl.grid(row=row, column=col, padx=5, pady=(5, 0), sticky="n")
                self._camera_name_labels[name] = lbl

            # Create or update the image label
            if name not in self._camera_labels:
                img_lbl = tk.Label(self._camera_frame, bg="black")
                img_lbl.grid(row=row + 1, column=col, padx=5, pady=(0, 5))
                self._camera_labels[name] = img_lbl

            self._camera_labels[name].configure(image=photo)
