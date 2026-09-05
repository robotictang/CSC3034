"""A Newton demo: six-legged robot patrol in an oil-palm forest.

Run from the CSC3034 folder:
    source .venv-newton/bin/activate
    python newton_examples/oil_palm_hexapod.py --device cuda:0 --viewer gl

This is a lightweight, kinematic robotics visualisation. Use WASD to drive
the six-legged robot. When it approaches an oil palm, a detection beacon,
signal line, and live status panel identify the target.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples


PALMS = [
    (-5.5, -3.0), (-2.0, -3.2), (1.8, -3.1), (5.2, -2.8),
    (-4.3, 1.2), (0.0, 0.0), (4.1, 1.5), (-2.0, 3.6), (2.3, 3.8),
]


def segment_transform(start: tuple[float, float, float], end: tuple[float, float, float]):
    """Make a transform for a box whose long local axis is the line start-end."""
    dx, dy, dz = (end[i] - start[i] for i in range(3))
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    flat_length = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(dz, flat_length)
    half_yaw, half_pitch = yaw / 2.0, pitch / 2.0
    sy, cy = math.sin(half_yaw), math.cos(half_yaw)
    sp, cp = math.sin(half_pitch), math.cos(half_pitch)
    rotation = wp.quat(sy * sp, -cy * sp, sy * cp, cy * cp)
    midpoint = wp.vec3(
        (start[0] + end[0]) / 2.0,
        (start[1] + end[1]) / 2.0,
        (start[2] + end[2]) / 2.0,
    )
    return wp.transform(p=midpoint, q=rotation), length


class OilPalmHexapodDemo:
    """A hexapod scout with animated legs and proximity palm detection."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.robot_x = 0.0
        self.robot_y = -5.5
        self.robot_heading = math.pi / 2.0
        self.walk_time = 0.0
        self.detector_range = 3.4
        self.detected_palm: int | None = None
        self.detected_distance = float("inf")
        self.last_signal = "OFFLINE"
        self.leg_layout: list[tuple[float, float, float]] = []

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_ground_plane()

        for palm_index, (x, y) in enumerate(PALMS, start=1):
            self._add_palm(builder, palm_index, x, y)

        # The blue chassis includes a raised sensor mast and a bright green
        # camera indicator at the front.
        self.robot_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.64), q=wp.quat_identity()),
            label="hexapod_chassis",
        )
        builder.add_shape_box(self.robot_body, hx=0.72, hy=0.46, hz=0.18, color=wp.vec3(0.07, 0.19, 0.42))
        builder.add_shape_box(
            self.robot_body,
            xform=wp.transform(p=wp.vec3(0.42, 0.0, 0.20), q=wp.quat_identity()),
            hx=0.28,
            hy=0.30,
            hz=0.13,
            color=wp.vec3(0.16, 0.43, 0.78),
        )
        builder.add_shape_cylinder(
            self.robot_body,
            xform=wp.transform(p=wp.vec3(0.20, 0.0, 0.39), q=wp.quat_identity()),
            radius=0.065,
            half_height=0.18,
            color=wp.vec3(0.18, 0.18, 0.20),
        )
        builder.add_shape_sphere(
            self.robot_body,
            xform=wp.transform(p=wp.vec3(0.31, 0.0, 0.55), q=wp.quat_identity()),
            radius=0.10,
            color=wp.vec3(0.20, 1.0, 0.30),
        )
        self._add_legs(builder)
        self.target_beacon = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -50.0), q=wp.quat_identity()),
            label="palm_detection_beacon",
        )
        builder.add_shape_sphere(
            self.target_beacon,
            radius=0.16,
            color=wp.vec3(0.10, 1.0, 0.20),
        )

        self.model = builder.finalize()
        self.state = self.model.state()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(10.0, -16.0, 12.0), pitch=-24.0, yaw=-140.0)

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.gui, position="side")
            # Newton reserves WASD for camera motion by default. Disable that
            # keyboard path so the keys control this robot; mouse controls
            # still orbit, pan, and zoom the camera.
            self.viewer.gui.update_camera_from_keys = lambda _dt, _keys: None
            self.viewer.vsync = True

        self._update_scene()
        print("[INFO] Palm scout ready. Click the 3D view, then use WASD to drive.")

    @staticmethod
    def _add_palm(builder, palm_index: int, x: float, y: float) -> None:
        """Create a layered trunk, a broad crown, and fruit bunches."""
        lean_x = 0.10 * math.sin(palm_index * 1.7)
        lean_y = 0.10 * math.cos(palm_index * 1.3)
        for section in range(3):
            z = 0.42 + section * 0.78
            fraction = (section + 0.5) / 3.0
            builder.add_shape_cylinder(
                body=-1,
                xform=wp.transform(
                    p=wp.vec3(x + lean_x * fraction, y + lean_y * fraction, z),
                    q=wp.quat_identity(),
                ),
                radius=0.18 - section * 0.015,
                half_height=0.43,
                color=wp.vec3(0.30 + section * 0.03, 0.13 + section * 0.015, 0.035),
            )

        crown_x, crown_y, crown_z = x + lean_x, y + lean_y, 2.48
        for frond in range(8):
            angle = frond * math.tau / 8.0 + 0.18 * (palm_index % 2)
            start = (crown_x, crown_y, crown_z)
            end = (
                crown_x + 1.75 * math.cos(angle),
                crown_y + 1.75 * math.sin(angle),
                crown_z - 0.38 - 0.12 * (frond % 2),
            )
            xform, length = segment_transform(start, end)
            builder.add_shape_box(
                body=-1,
                xform=xform,
                hx=length / 2.0,
                hy=0.10,
                hz=0.035,
                color=wp.vec3(0.035, 0.36 + 0.045 * (palm_index % 3), 0.075),
            )
            # Feather-like leaflets make the crown look recognisably like an
            # oil palm rather than a group of plain green bars.
            for fraction in (0.30, 0.48, 0.66, 0.84):
                base_x = crown_x + 1.75 * fraction * math.cos(angle)
                base_y = crown_y + 1.75 * fraction * math.sin(angle)
                base_z = crown_z + (end[2] - crown_z) * fraction
                leaflet_length = 0.46 * (1.0 - 0.38 * fraction)
                for side in (-1.0, 1.0):
                    leaflet_end = (
                        base_x + 0.32 * math.cos(angle) - side * leaflet_length * math.sin(angle),
                        base_y + 0.32 * math.sin(angle) + side * leaflet_length * math.cos(angle),
                        base_z - 0.08,
                    )
                    leaflet_xform, leaflet_size = segment_transform((base_x, base_y, base_z), leaflet_end)
                    builder.add_shape_box(
                        body=-1,
                        xform=leaflet_xform,
                        hx=leaflet_size / 2.0,
                        hy=0.035,
                        hz=0.018,
                        color=wp.vec3(0.025, 0.27 + 0.04 * (palm_index % 3), 0.045),
                    )

        # Orange fruit bunches make the target trees recognisable at a glance.
        for offset_x, offset_y in ((0.22, 0.0), (-0.13, 0.17), (-0.10, -0.18)):
            builder.add_shape_sphere(
                body=-1,
                xform=wp.transform(
                    p=wp.vec3(crown_x + offset_x, crown_y + offset_y, 2.23),
                    q=wp.quat_identity(),
                ),
                radius=0.10,
                color=wp.vec3(0.95, 0.30, 0.03),
            )

    def _add_legs(self, builder) -> None:
        """Build six upper/lower leg pairs, each animated separately."""
        for leg_index, hip_x in enumerate((-0.48, 0.0, 0.48)):
            for side in (-1.0, 1.0):
                phase = 0.0 if (leg_index + int(side > 0)) % 2 == 0 else math.pi
                upper = builder.add_body(
                    xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.45), q=wp.quat_identity()),
                    label=f"leg_{leg_index}_{'left' if side < 0 else 'right'}_upper",
                )
                lower = builder.add_body(
                    xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.23), q=wp.quat_identity()),
                    label=f"leg_{leg_index}_{'left' if side < 0 else 'right'}_lower",
                )
                leg_colour = wp.vec3(0.92, 0.27, 0.08) if phase == 0.0 else wp.vec3(0.96, 0.58, 0.08)
                builder.add_shape_box(upper, hx=0.42, hy=0.065, hz=0.065, color=leg_colour)
                builder.add_shape_box(lower, hx=0.42, hy=0.055, hz=0.055, color=wp.vec3(0.18, 0.20, 0.24))
                self.leg_layout.append((hip_x, side, phase))

    @staticmethod
    def _to_world(robot_x: float, robot_y: float, heading: float, local: tuple[float, float, float]):
        cos_heading, sin_heading = math.cos(heading), math.sin(heading)
        local_x, local_y, local_z = local
        return (
            robot_x + cos_heading * local_x - sin_heading * local_y,
            robot_y + sin_heading * local_x + cos_heading * local_y,
            local_z,
        )

    def _key_down(self, key: str) -> bool:
        return bool(getattr(self.viewer, "is_key_down", lambda _key: False)(key))

    def _drive_robot(self) -> bool:
        """Read WASD input and prevent the robot from walking through trunks."""
        forward = float(self._key_down("w")) - float(self._key_down("s"))
        turn = float(self._key_down("d")) - float(self._key_down("a"))
        self.robot_heading += turn * math.radians(115.0) * self.frame_dt
        if forward == 0.0:
            return False

        speed = 2.0
        next_x = self.robot_x + forward * speed * math.cos(self.robot_heading) * self.frame_dt
        next_y = self.robot_y + forward * speed * math.sin(self.robot_heading) * self.frame_dt
        if any(math.hypot(next_x - palm_x, next_y - palm_y) < 0.68 for palm_x, palm_y in PALMS):
            return False
        self.robot_x, self.robot_y = next_x, next_y
        self.walk_time += self.frame_dt
        return True

    def _signal(self) -> tuple[str, float]:
        if self.detected_palm is None:
            return "NO SIGNAL", 0.0
        strength = max(0.0, 1.0 - self.detected_distance / self.detector_range)
        if self.detected_distance <= 1.15:
            return "TARGET LOCK", strength
        if self.detected_distance <= 2.20:
            return "STRONG SIGNAL", strength
        return "WEAK SIGNAL", strength

    def _update_scene(self) -> None:
        robot_x, robot_y, heading = self.robot_x, self.robot_y, self.robot_heading
        poses = [
            wp.transform(
                p=wp.vec3(robot_x, robot_y, 0.64),
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), heading),
            )
        ]

        # Alternating tripods lift and swing forward. The animation is
        # kinematic, but every leg has an independently moving upper and lower
        # segment, producing a clear six-legged walk.
        gait_phase = self.walk_time * 5.8
        for hip_x, side, phase_offset in self.leg_layout:
            swing = math.sin(gait_phase + phase_offset)
            lift = max(0.0, swing) * 0.16
            stride = 0.20 * swing
            hip = self._to_world(robot_x, robot_y, heading, (hip_x, side * 0.40, 0.59))
            knee = self._to_world(robot_x, robot_y, heading, (hip_x + stride, side * 0.82, 0.34 + lift))
            foot = self._to_world(robot_x, robot_y, heading, (hip_x - stride, side * 1.13, 0.08))
            upper_xform, _ = segment_transform(hip, knee)
            lower_xform, _ = segment_transform(knee, foot)
            poses.extend((upper_xform, lower_xform))

        distances = [math.hypot(robot_x - palm_x, robot_y - palm_y) for palm_x, palm_y in PALMS]
        nearest_index = int(np.argmin(distances))
        self.detected_distance = distances[nearest_index]
        self.detected_palm = nearest_index + 1 if self.detected_distance <= self.detector_range else None
        if self.detected_palm is None:
            poses.append(wp.transform(p=wp.vec3(0.0, 0.0, -50.0), q=wp.quat_identity()))
        else:
            palm_x, palm_y = PALMS[nearest_index]
            poses.append(wp.transform(p=wp.vec3(palm_x, palm_y, 3.25), q=wp.quat_identity()))

        pose_array = wp.array(poses, dtype=wp.transform, device=self.state.body_q.device)
        self.state.body_q.assign(pose_array)

        signal, _ = self._signal()
        if signal != self.last_signal:
            print(f"[DETECTOR] {signal}: nearest oil palm is {self.detected_distance:.2f} m away.")
            self.last_signal = signal

    def step(self) -> None:
        self.sim_time += self.frame_dt
        self._drive_robot()
        self._update_scene()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        signal, strength = self._signal()
        self.viewer.log_scalar("Oil palm signal strength", strength)
        if self.detected_palm is None:
            self.viewer.log_lines("oil_palm_signal", None, None, None)
        else:
            palm_x, palm_y = PALMS[self.detected_palm - 1]
            starts = wp.array([wp.vec3(self.robot_x, self.robot_y, 1.15)], dtype=wp.vec3, device=self.state.body_q.device)
            ends = wp.array([wp.vec3(palm_x, palm_y, 2.80)], dtype=wp.vec3, device=self.state.body_q.device)
            colour = wp.vec3(1.0 - strength, strength, 0.08)
            colours = wp.array([colour], dtype=wp.vec3, device=self.state.body_q.device)
            self.viewer.log_lines("oil_palm_signal", starts, ends, colours)
        self.viewer.end_frame()

    def gui(self, imgui) -> None:
        imgui.separator()
        signal, strength = self._signal()
        imgui.text("Oil Palm Scout: click 3D view, then drive")
        imgui.text("W/S: forward/backward    A/D: turn")
        imgui.text("Mouse: orbit, pan and zoom")
        imgui.text(f"Scan radius: {self.detector_range:.2f} m")
        imgui.text(f"Detector: {signal}")
        imgui.text(f"Nearest palm: {self.detected_distance:.2f} m")
        imgui.text(f"Signal strength: {strength * 100.0:.0f}%")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(OilPalmHexapodDemo(viewer, args), args)
