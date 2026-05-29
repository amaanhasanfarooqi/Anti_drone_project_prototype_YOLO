"""
=============================================================
Anti-Drone System — Tracking Controller
=============================================================
Encapsulates the visual servoing logic:
  - Positional error calculation (pixel-space)
  - Proportional control for servo movement
  - Target lock detection
  - Fire cooldown management

Used by detect.py and the Jupyter notebook.
=============================================================
"""

import time
import math


class TrackingController:
    """
    Coordinate-based visual servo controller.

    Computes servo commands from the 2D pixel error between
    the detected drone centre and the frame centre.

    Parameters
    ----------
    frame_cx : int
        X coordinate of the frame centre (pixels)
    frame_cy : int
        Y coordinate of the frame centre (pixels)
    move_threshold : int
        Minimum pixel error to trigger a servo move command
    lock_threshold : int
        Pixel error below which TARGET LOCK is declared
    fire_cooldown : float
        Minimum seconds between consecutive FIRE commands
    """

    def __init__(
        self,
        frame_cx       = 320,
        frame_cy       = 240,
        move_threshold = 30,
        lock_threshold = 20,
        fire_cooldown  = 3.0,
    ):
        self.frame_cx       = frame_cx
        self.frame_cy       = frame_cy
        self.move_threshold = move_threshold
        self.lock_threshold = lock_threshold
        self.fire_cooldown  = fire_cooldown

        self._last_fire_time = 0.0

    # ──────────────────────────────────────────────────────
    # CORE: COMPUTE COMMANDS
    # ──────────────────────────────────────────────────────

    def compute(self, drone_cx: int, drone_cy: int) -> dict:
        """
        Given detected drone centre (drone_cx, drone_cy),
        return a dict of servo commands and status.

        Returns
        -------
        {
            "commands": list[str],   # e.g. ["TRACK", "RIGHT", "UP"]
            "error_x":  int,         # signed pixel error in X
            "error_y":  int,         # signed pixel error in Y
            "distance": float,       # Euclidean pixel distance to centre
            "locked":   bool,        # True if within lock threshold
            "fire":     bool,        # True if FIRE command issued
        }
        """
        error_x  = drone_cx - self.frame_cx
        error_y  = drone_cy - self.frame_cy
        distance = math.sqrt(error_x ** 2 + error_y ** 2)

        commands = ["TRACK"]

        # ── Pan (horizontal) control ───────────────────────
        if error_x > self.move_threshold:
            commands.append("RIGHT")
        elif error_x < -self.move_threshold:
            commands.append("LEFT")

        # ── Tilt (vertical) control ────────────────────────
        if error_y > self.move_threshold:
            commands.append("DOWN")
        elif error_y < -self.move_threshold:
            commands.append("UP")

        # ── Target lock + fire cooldown ────────────────────
        locked = (abs(error_x) < self.lock_threshold and
                  abs(error_y) < self.lock_threshold)
        fire   = False

        if locked:
            now = time.time()
            if now - self._last_fire_time > self.fire_cooldown:
                commands.append("FIRE")
                self._last_fire_time = now
                fire = True

        return {
            "commands": commands,
            "error_x":  error_x,
            "error_y":  error_y,
            "distance": distance,
            "locked":   locked,
            "fire":     fire,
        }

    # ──────────────────────────────────────────────────────
    # SCAN (no target)
    # ──────────────────────────────────────────────────────

    def scan(self) -> list:
        """Return scan command when no drone is detected."""
        return ["SCAN"]

    # ──────────────────────────────────────────────────────
    # UTILITY: PROPORTIONAL SERVO ANGLE (for smoother control)
    # ──────────────────────────────────────────────────────

    def proportional_step(
        self,
        error_x: int,
        error_y: int,
        gain: float = 0.05,
        max_step: int = 5,
    ) -> tuple:
        """
        Compute proportional servo step sizes.

        Instead of fixed +2° / -2° steps (as used in the basic
        Arduino code), this provides a smooth proportional response
        where larger errors produce larger steps.

        Parameters
        ----------
        error_x  : pixel error in X
        error_y  : pixel error in Y
        gain     : proportional gain (tune to taste)
        max_step : maximum step size in degrees

        Returns
        -------
        (pan_step, tilt_step) — signed degrees to add to current servo angle
        """
        pan_step  = int(max(-max_step, min(max_step, gain * error_x)))
        tilt_step = int(max(-max_step, min(max_step, gain * error_y)))
        return pan_step, tilt_step

    # ──────────────────────────────────────────────────────
    # STATUS STRING (for HUD / logging)
    # ──────────────────────────────────────────────────────

    def status_string(self, result: dict) -> str:
        locked = "🔒 LOCKED" if result["locked"] else "📡 TRACKING"
        fire   = " 🔥 FIRE!" if result["fire"]   else ""
        return (
            f"{locked}{fire} | "
            f"Err X: {result['error_x']:+4d}px  "
            f"Y: {result['error_y']:+4d}px  "
            f"Dist: {result['distance']:5.1f}px"
        )
