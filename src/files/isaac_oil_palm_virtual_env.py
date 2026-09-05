"""Create a realistic-looking oil-palm research plot in NVIDIA Isaac Sim.

This is a self-contained Isaac Sim virtual environment.  It uses the PBR
textures shipped with the local Isaac Sim installation, so it does not need an
internet connection or an Omniverse/Nucleus account.

Run from the ``CSC3034`` folder:

    ~/isaacsim/python.sh src/files/isaac_oil_palm_virtual_env.py

Useful options:

    # Keep the viewport open for five minutes and export the USD stage.
    ~/isaacsim/python.sh src/files/isaac_oil_palm_virtual_env.py --seconds 300

    # Build and validate the scene without opening a viewport.
    ~/isaacsim/python.sh src/files/isaac_oil_palm_virtual_env.py --headless --seconds 1

The generated USD file can later be opened directly in Isaac Sim.  This scene
is deliberately separate from ``isaac_durian_plantation.py`` so that the
original coursework example remains unchanged.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path


PARSER = argparse.ArgumentParser(description="Create an RTX-ready oil-palm virtual environment in Isaac Sim.")
PARSER.add_argument("--seconds", type=float, default=120.0, help="How long the Isaac Sim viewport stays open.")
PARSER.add_argument("--headless", action="store_true", help="Build without opening the Isaac Sim viewport.")
PARSER.add_argument(
    "--output",
    type=Path,
    default=Path("isaac_outputs/oil_palm_virtual_env.usda"),
    help="USD file written after the scene is created.",
)
ARGS, _ = PARSER.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:  # Isaac Sim 4.x compatibility
    try:
        from omni.isaac.kit import SimulationApp
    except ImportError as error:
        raise SystemExit(
            "Isaac Sim was not found. Run this file with ~/isaacsim/python.sh, not normal python3."
        ) from error


# The application must start before importing Omniverse/Isaac extension APIs.
SIMULATION_APP = SimulationApp({"headless": ARGS.headless})

import carb
import omni.appwindow
import omni.usd
from isaacsim.core.experimental.materials import OmniPbrMaterial
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

if not ARGS.headless:
    import omni.ui as ui


RNG = random.Random(3034)


def vec(value: tuple[float, float, float]) -> Gf.Vec3d:
    """Return a double precision vector used by USD transforms."""
    return Gf.Vec3d(*value)


def unit(value: Gf.Vec3d) -> Gf.Vec3d:
    """Return a safe unit vector, including for a near-zero input."""
    return value.GetNormalized() if value.GetLength() > 1e-6 else Gf.Vec3d(0.0, 0.0, 1.0)


def orientation(source_axis: tuple[float, float, float], target: Gf.Vec3d) -> Gf.Quatf:
    """Create a quaternion that points a primitive axis along ``target``."""
    quaternion = Gf.Rotation(vec(source_axis), unit(target)).GetQuat()
    return Gf.Quatf(quaternion.GetReal(), Gf.Vec3f(*quaternion.GetImaginary()))


def terrain_height(x: float, y: float) -> float:
    """A gentle, repeatable undulation in metres for a less artificial field."""
    return 0.11 * math.sin(x * 0.23) * math.cos(y * 0.19) + 0.035 * math.sin((x + y) * 0.71)


def find_asset_texture(*relative_paths: str) -> str | None:
    """Find the first local Isaac Sim texture from a list of possible paths."""
    # ``get_assets_root_path`` may prefer the public online asset server.  The
    # course workstation already has a local asset bundle, which is faster and
    # continues to work without a network connection.  ``${kit}`` resolves to
    # e.g. ``.../isaacsim/kit``, so its parent is the Isaac Sim installation.
    kit_path = Path(carb.tokens.get_tokens_interface().resolve("${kit}"))
    local_roots = (kit_path.parent / "data" / "assets" / "Assets",)
    for assets_root in local_roots:
        for relative_path in relative_paths:
            candidate = assets_root / relative_path
            if candidate.is_file():
                return str(candidate)

    # Keep the official helper call as a diagnostic/future fallback.  Remote
    # URLs are intentionally not returned here because this function promises
    # a locally installed texture and Path cannot validate a URL as a file.
    get_assets_root_path()
    return None


def build_scene(output: Path, seconds: float) -> None:
    """Populate the active stage, export it, then let the user inspect it."""
    stage = omni.usd.get_context().get_stage()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Scope.Define(stage, "/World/Plantation")
    UsdGeom.Scope.Define(stage, "/World/Vegetation")
    UsdGeom.Scope.Define(stage, "/World/Props")

    # These textures ship in Isaac Sim 6.  Materials still use their supplied
    # colour if a future Isaac release moves a texture to a different folder.
    texture_base = "Isaac/6.0/Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/common_assets/shared_textures"
    grass_diffuse = find_asset_texture(
        f"{texture_base}/CreepingBentgrass_Regular_COLOR_High.png",
        f"{texture_base}/grass_a_basecolor.png",
    )
    grass_normal = find_asset_texture(
        f"{texture_base}/CreepingBentgrass_Regular_NRM_High.png",
        f"{texture_base}/grass_a_normal.png",
    )
    grass_roughness = find_asset_texture(f"{texture_base}/CreepingBentgrass_Regular_ROUGHNESS_High.png")
    bark_diffuse = find_asset_texture(
        f"{texture_base}/washingtonia_robusta_cut_bark_01_a.png",
        f"{texture_base}/tree_bark_03_diff_2k.png",
    )
    bark_normal = find_asset_texture(
        f"{texture_base}/washingtonia_robusta_cut_bark_01_n.png",
        f"{texture_base}/tree_bark_03_nor_2k.png",
    )
    palm_normal = find_asset_texture(f"{texture_base}/royalpalmleaves_v3_mat_normal.png")

    def pbr_material(
        name: str,
        colour: tuple[float, float, float],
        *,
        diffuse_texture: str | None = None,
        normal_texture: str | None = None,
        roughness_texture: str | None = None,
        roughness: float = 0.7,
        texture_scale: tuple[float, float] = (1.0, 1.0),
    ) -> UsdShade.Material:
        """Create a real-time RTX OmniPBR material and return its USD prim."""
        path = f"/World/Looks/{name}"
        material_api = OmniPbrMaterial(path)
        material_api.set_input_values("diffuse_color_constant", list(colour))
        material_api.set_input_values("reflection_roughness_constant", roughness)
        material_api.set_input_values("project_uvw", True)
        material_api.set_input_values("texture_scale", list(texture_scale))
        if diffuse_texture:
            material_api.set_input_values("diffuse_texture", diffuse_texture)
        if normal_texture:
            material_api.set_input_values("normalmap_texture", normal_texture)
            material_api.set_input_values("bump_factor", 0.35)
        if roughness_texture:
            material_api.set_input_values("reflectionroughness_texture", roughness_texture)
        return UsdShade.Material.Get(stage, path)

    grass = pbr_material(
        "FieldGrass", (0.28, 0.43, 0.17), diffuse_texture=grass_diffuse,
        normal_texture=grass_normal, roughness_texture=grass_roughness,
        roughness=0.92, texture_scale=(8.0, 8.0),
    )
    palm_bark = pbr_material(
        "PalmBark", (0.25, 0.16, 0.08), diffuse_texture=bark_diffuse,
        normal_texture=bark_normal, roughness=0.84, texture_scale=(2.0, 6.0),
    )
    palm_leaf_dark = pbr_material(
        "PalmLeafDark", (0.035, 0.20, 0.055), normal_texture=palm_normal, roughness=0.63,
    )
    palm_leaf_light = pbr_material(
        "PalmLeafLight", (0.075, 0.34, 0.09), normal_texture=palm_normal, roughness=0.60,
    )
    leaf_stem = pbr_material("LeafStem", (0.22, 0.31, 0.055), roughness=0.72)
    fruit_ripe = pbr_material("RipeFruit", (0.66, 0.075, 0.018), roughness=0.48)
    fruit_warm = pbr_material("WarmFruit", (0.95, 0.28, 0.018), roughness=0.45)
    earth = pbr_material("AccessTrack", (0.34, 0.19, 0.075), roughness=0.96, texture_scale=(5.0, 12.0))
    stone = pbr_material("RiverStone", (0.22, 0.24, 0.20), roughness=0.86)
    painted_metal = pbr_material("SurveyMarker", (0.98, 0.73, 0.05), roughness=0.33)
    robot_paint = pbr_material("ScoutRobotPaint", (0.08, 0.19, 0.26), roughness=0.30)
    robot_joint = pbr_material("ScoutRobotJoints", (0.09, 0.10, 0.11), roughness=0.42)
    sensor_lens = pbr_material("SensorLens", (0.06, 0.68, 0.88), roughness=0.18)

    def bind(prim, material: UsdShade.Material) -> None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    def transformed(prim, position: tuple[float, float, float], scale=None, orient=None) -> None:
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(vec(position))
        if orient is not None:
            xform.AddOrientOp().Set(orient)
        if scale is not None:
            xform.AddScaleOp().Set(Gf.Vec3f(*scale))

    def cube(
        path: str,
        position: tuple[float, float, float],
        dimensions: tuple[float, float, float],
        material: UsdShade.Material,
        orient: Gf.Quatf | None = None,
    ):
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1.0)
        transformed(shape.GetPrim(), position, dimensions, orient)
        bind(shape.GetPrim(), material)
        return shape

    def sphere(
        path: str,
        position: tuple[float, float, float],
        radius: float,
        material: UsdShade.Material,
        scale: tuple[float, float, float] | None = None,
    ):
        shape = UsdGeom.Sphere.Define(stage, path)
        shape.CreateRadiusAttr(radius)
        transformed(shape.GetPrim(), position, scale)
        bind(shape.GetPrim(), material)
        return shape

    def cylinder_between(
        path: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        radius: float,
        material: UsdShade.Material,
    ):
        start_vec, end_vec = vec(start), vec(end)
        direction = end_vec - start_vec
        length = direction.GetLength()
        shape = UsdGeom.Cylinder.Define(stage, path)
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(length)
        midpoint = (start_vec + end_vec) * 0.5
        transformed(
            shape.GetPrim(), tuple(midpoint), None,
            orientation((0.0, 0.0, 1.0), direction),
        )
        bind(shape.GetPrim(), material)
        return shape

    def blade_between(
        path: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        width: float,
        material: UsdShade.Material,
    ):
        """Create a thin, broad leaflet; its long local axis is X."""
        start_vec, end_vec = vec(start), vec(end)
        direction = end_vec - start_vec
        length = direction.GetLength()
        midpoint = (start_vec + end_vec) * 0.5
        return cube(
            path,
            tuple(midpoint),
            (length, width, 0.022),
            material,
            orientation((1.0, 0.0, 0.0), direction),
        )

    # A genuine triangle mesh makes the plantation surface undulate under the
    # trees.  It can also participate in later PhysX robot simulations.
    terrain = UsdGeom.Mesh.Define(stage, "/World/Plantation/Terrain")
    x_count, y_count = 49, 41
    x_min, x_max, y_min, y_max = -16.0, 16.0, -14.0, 14.0
    points = []
    for y_index in range(y_count):
        y = y_min + (y_max - y_min) * y_index / (y_count - 1)
        for x_index in range(x_count):
            x = x_min + (x_max - x_min) * x_index / (x_count - 1)
            points.append(Gf.Vec3f(x, y, terrain_height(x, y)))
    counts, indices = [], []
    for y_index in range(y_count - 1):
        for x_index in range(x_count - 1):
            a = y_index * x_count + x_index
            counts.append(4)
            indices.extend((a, a + 1, a + x_count + 1, a + x_count))
    terrain.CreatePointsAttr(points)
    terrain.CreateFaceVertexCountsAttr(counts)
    terrain.CreateFaceVertexIndicesAttr(indices)
    terrain.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    terrain.CreateDoubleSidedAttr(True)
    bind(terrain.GetPrim(), grass)
    UsdPhysics.CollisionAPI.Apply(terrain.GetPrim())

    # A shallow access lane prevents the plantation from looking like a
    # perfectly uniform grid and gives a future robot a clear route.
    for index, x in enumerate((-2.6, 7.4)):
        cube(
            f"/World/Plantation/AccessTrack_{index}",
            (x, 0.0, terrain_height(x, 0.0) + 0.018),
            (1.6, 27.5, 0.055), earth,
        )

    palm_records: list[dict[str, float | str]] = []

    def oil_palm(name: str, x: float, y: float, height: float, phase: float) -> None:
        """Build one layered oil palm: tapered trunk, fronds, and fruit bunches."""
        base_z = terrain_height(x, y)
        root = UsdGeom.Xform.Define(stage, f"/World/Vegetation/{name}")
        root.GetPrim().CreateAttribute("plant:type", Sdf.ValueTypeNames.String).Set("Oil palm (Elaeis guineensis)")
        root.GetPrim().CreateAttribute("plant:height_m", Sdf.ValueTypeNames.Float).Set(height)
        palm_records.append({"name": name, "x": x, "y": y, "z": base_z, "height": height})

        # Slightly offset, tapered trunk sections imply the natural texture and
        # gentle curve of a mature oil-palm trunk.
        segments = 8
        trunk_points = []
        for index in range(segments + 1):
            fraction = index / segments
            trunk_points.append((
                x + 0.10 * math.sin(fraction * math.pi + phase),
                y + 0.08 * math.cos(fraction * math.pi * 1.4 + phase),
                base_z + fraction * height,
            ))
        for index, (start, end) in enumerate(zip(trunk_points[:-1], trunk_points[1:])):
            radius = 0.43 - 0.17 * (index / segments)
            section = cylinder_between(
                f"/World/Vegetation/{name}/TrunkSection_{index}", start, end, radius, palm_bark,
            )
            UsdPhysics.CollisionAPI.Apply(section.GetPrim())

        crown = vec(trunk_points[-1])
        sphere(f"/World/Vegetation/{name}/Crown", tuple(crown), 0.46, leaf_stem, (1.0, 1.0, 0.86))

        # Each frond bends downward in three pieces.  Short leaflets on both
        # sides create a recognisable feather-leaf silhouette instead of a
        # generic green ball.
        for frond_index in range(12):
            angle = phase + frond_index * math.tau / 12.0
            radial = Gf.Vec3d(math.cos(angle), math.sin(angle), 0.0)
            side = Gf.Vec3d(-math.sin(angle), math.cos(angle), 0.0)
            spread = 3.7 + 0.55 * math.sin(frond_index * 1.7 + phase)
            points_for_frond = [
                crown + Gf.Vec3d(0.0, 0.0, 0.05),
                crown + radial * (spread * 0.27) + Gf.Vec3d(0.0, 0.0, 0.18),
                crown + radial * (spread * 0.65) + Gf.Vec3d(0.0, 0.0, -0.16),
                crown + radial * spread + Gf.Vec3d(0.0, 0.0, -1.12),
            ]
            for segment_index, (start, end) in enumerate(zip(points_for_frond[:-1], points_for_frond[1:])):
                cylinder_between(
                    f"/World/Vegetation/{name}/Frond_{frond_index}/Rachis_{segment_index}",
                    tuple(start), tuple(end), 0.055 - segment_index * 0.009, leaf_stem,
                )
            for leaflet_index, fraction in enumerate((0.26, 0.43, 0.60, 0.76, 0.90)):
                # Interpolate along the last two segments, then fan leaflets
                # outwards, shorter towards the drooping end of the frond.
                curve_position = crown + radial * (spread * fraction) + Gf.Vec3d(
                    0.0, 0.0, 0.17 - max(0.0, fraction - 0.35) * 2.0,
                )
                blade_length = 1.12 - 0.42 * fraction
                for side_sign, side_name in ((-1.0, "Left"), (1.0, "Right")):
                    blade_direction = unit(
                        side * side_sign * 0.92 + radial * 0.28 + Gf.Vec3d(0.0, 0.0, -0.16)
                    )
                    blade_end = curve_position + blade_direction * blade_length
                    blade_between(
                        f"/World/Vegetation/{name}/Frond_{frond_index}/Leaflet_{leaflet_index}_{side_name}",
                        tuple(curve_position), tuple(blade_end),
                        0.19 - leaflet_index * 0.018,
                        palm_leaf_light if (frond_index + leaflet_index) % 3 else palm_leaf_dark,
                    )

        # Bunches of orange-red fruit are distinctive and are deliberately
        # placed under the crown where a camera or robot would inspect them.
        for bunch_index in range(5):
            angle = phase + bunch_index * math.tau / 5.0 + 0.24
            centre = crown + Gf.Vec3d(math.cos(angle) * 0.72, math.sin(angle) * 0.72, -0.62)
            for fruit_index in range(13):
                fruit_angle = fruit_index * math.tau / 13.0
                ring = 0.18 + 0.08 * ((fruit_index + bunch_index) % 3)
                fruit_position = centre + Gf.Vec3d(
                    math.cos(fruit_angle) * ring,
                    math.sin(fruit_angle) * ring,
                    ((fruit_index % 4) - 1.5) * 0.095,
                )
                sphere(
                    f"/World/Vegetation/{name}/FruitBunch_{bunch_index}/Fruit_{fruit_index}",
                    tuple(fruit_position), 0.09 + (fruit_index % 2) * 0.012,
                    fruit_ripe if fruit_index % 3 else fruit_warm,
                    (0.86, 0.86, 1.25),
                )

    # Oil-palm fields are planted in staggered rows.  Small height/rotation
    # variations stop the setting from appearing like cloned game objects.
    tree_locations = []
    for row_index, x in enumerate((-11.4, -6.0, -0.6, 4.8, 10.2)):
        for column_index, y in enumerate((-9.0, -3.0, 3.0, 9.0)):
            tree_locations.append((x + (column_index % 2) * 0.5, y, row_index, column_index))
    for index, (x, y, row_index, column_index) in enumerate(tree_locations):
        oil_palm(
            f"OilPalm_{index:02d}", x, y,
            height=7.2 + RNG.uniform(-0.55, 0.8),
            phase=0.21 * row_index + 0.47 * column_index + RNG.uniform(-0.12, 0.12),
        )

    # Low ferns and scattered rocks break up the otherwise clean terrain.
    def ground_fern(index: int, x: float, y: float) -> None:
        origin = Gf.Vec3d(x, y, terrain_height(x, y) + 0.05)
        for blade_index in range(9):
            angle = blade_index * math.tau / 9.0 + index * 0.37
            direction = Gf.Vec3d(math.cos(angle), math.sin(angle), 0.35)
            blade_between(
                f"/World/Vegetation/GroundFern_{index}/Blade_{blade_index}",
                tuple(origin), tuple(origin + direction * (0.45 + 0.18 * (blade_index % 2))),
                0.085, palm_leaf_dark,
            )

    for fern_index in range(32):
        x = RNG.uniform(-15.0, 15.0)
        y = RNG.uniform(-13.0, 13.0)
        if min(abs(x + 2.6), abs(x - 7.4)) > 0.9:
            ground_fern(fern_index, x, y)

    for rock_index in range(44):
        x, y = RNG.uniform(-15.5, 15.5), RNG.uniform(-13.5, 13.5)
        radius = RNG.uniform(0.07, 0.22)
        sphere(
            f"/World/Props/Rock_{rock_index}",
            (x, y, terrain_height(x, y) + radius * 0.38), radius, stone,
            (RNG.uniform(0.75, 1.45), RNG.uniform(0.70, 1.25), RNG.uniform(0.45, 0.85)),
        )

    # ------------------------------------------------------------------
    # Interactive six-legged inspection robot
    # ------------------------------------------------------------------
    # A genuine hexapod USD/URDF is not bundled with Isaac Sim on this PC, so
    # this is a kinematic six-legged scout rather than falsely labelling the
    # supplied four-legged Ant reference robot as a hexapod.  It still has a
    # proper 3D body, articulated-looking legs, collision-safe navigation,
    # WASD control, and an oil-palm proximity sensor.
    robot_root = UsdGeom.Xform.Define(stage, "/World/Props/OilPalmScout")
    robot_xform = UsdGeom.Xformable(robot_root.GetPrim())
    robot_translate = robot_xform.AddTranslateOp()
    robot_yaw = robot_xform.AddRotateZOp()
    robot_position = [-14.1, -12.0]
    robot_heading = math.radians(28.0)
    robot_translate.Set(Gf.Vec3d(robot_position[0], robot_position[1], terrain_height(*robot_position) + 0.76))
    robot_yaw.Set(math.degrees(robot_heading))

    chassis = cube(
        "/World/Props/OilPalmScout/Chassis", (0.0, 0.0, 0.0), (1.55, 0.92, 0.34), robot_paint,
    )
    UsdPhysics.CollisionAPI.Apply(chassis.GetPrim())
    cube("/World/Props/OilPalmScout/TopDeck", (0.08, 0.0, 0.24), (0.92, 0.62, 0.16), robot_joint)
    cylinder_between(
        "/World/Props/OilPalmScout/SensorMast", (0.42, 0.0, 0.20), (0.42, 0.0, 0.72), 0.07, robot_joint,
    )
    sphere("/World/Props/OilPalmScout/Lidar", (0.42, 0.0, 0.80), 0.17, sensor_lens, (1.0, 1.0, 0.42))
    sensor_light = UsdLux.SphereLight.Define(stage, "/World/Props/OilPalmScout/SignalLight")
    sensor_light.CreateRadiusAttr(0.11)
    sensor_light.CreateIntensityAttr(380.0)
    sensor_light.CreateColorAttr(Gf.Vec3f(0.12, 0.85, 1.0))
    transformed(sensor_light.GetPrim(), (0.42, 0.0, 0.86))

    leg_rotation_ops = []
    for leg_index, (hip_x, side_sign) in enumerate(
        ((0.57, -1.0), (0.0, -1.0), (-0.57, -1.0), (0.57, 1.0), (0.0, 1.0), (-0.57, 1.0))
    ):
        leg = UsdGeom.Xform.Define(stage, f"/World/Props/OilPalmScout/Leg_{leg_index}")
        leg_xform = UsdGeom.Xformable(leg.GetPrim())
        leg_xform.AddTranslateOp().Set(Gf.Vec3d(hip_x, side_sign * 0.42, -0.08))
        leg_rotation_ops.append((leg_xform.AddRotateZOp(), leg_index * math.pi / 3.0))
        knee = (0.08, side_sign * 0.44, -0.31)
        foot = (0.24, side_sign * 0.74, -0.72)
        cylinder_between(
            f"/World/Props/OilPalmScout/Leg_{leg_index}/Upper", (0.0, 0.0, 0.0), knee, 0.075, robot_joint,
        )
        cylinder_between(
            f"/World/Props/OilPalmScout/Leg_{leg_index}/Lower", knee, foot, 0.060, robot_joint,
        )
        sphere(f"/World/Props/OilPalmScout/Leg_{leg_index}/Foot", foot, 0.105, robot_joint, (1.25, 0.86, 0.52))

    # One palm has a floating yellow beacon.  The robot can detect every palm,
    # while this marked tree provides a clear inspection destination.
    target_palm = palm_records[10]
    target_x, target_y = float(target_palm["x"]), float(target_palm["y"])
    target_z = float(target_palm["z"]) + float(target_palm["height"]) + 0.55
    sphere("/World/Props/TargetPalmBeacon", (target_x, target_y, target_z), 0.22, painted_metal)
    target_light = UsdLux.SphereLight.Define(stage, "/World/Props/TargetPalmBeaconLight")
    target_light.CreateRadiusAttr(0.16)
    target_light.CreateIntensityAttr(650.0)
    target_light.CreateColorAttr(Gf.Vec3f(1.0, 0.62, 0.04))
    transformed(target_light.GetPrim(), (target_x, target_y, target_z))

    # A yellow survey marker makes the research plot and scale obvious when
    # the stage first opens.  It is also a useful future navigation goal.
    marker_x, marker_y = 2.6, -10.8
    marker_base = terrain_height(marker_x, marker_y)
    cylinder_between(
        "/World/Props/ResearchMarker/Post",
        (marker_x, marker_y, marker_base), (marker_x, marker_y, marker_base + 2.2), 0.06, painted_metal,
    )
    sphere("/World/Props/ResearchMarker/Beacon", (marker_x, marker_y, marker_base + 2.28), 0.14, painted_metal)

    # Warm low-angle sun plus blue sky fill gives visible shadows and depth in
    # Isaac Sim's RTX renderer without requiring an online HDR environment.
    sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/TropicalSun")
    sun.CreateIntensityAttr(2800.0)
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.83, 0.66))
    sun.AddRotateXYZOp().Set(Gf.Vec3f(28.0, -35.0, -25.0))
    sky = UsdLux.DomeLight.Define(stage, "/World/Lighting/SkyFill")
    sky.CreateIntensityAttr(420.0)
    sky.CreateColorAttr(Gf.Vec3f(0.42, 0.62, 1.0))

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)

    # The starting camera frames multiple rows and the access track.  It is a
    # normal Isaac Sim camera, so students can still orbit, pan, and inspect.
    camera = UsdGeom.Camera.Define(stage, "/World/Cameras/PlantationOverview")
    camera.CreateFocalLengthAttr(32.0)
    set_camera_view(
        eye=[22.0, -25.0, 16.5],
        target=[0.0, 0.0, 3.3],
        camera_prim_path="/OmniverseKit_Persp",
    )
    transformed(camera.GetPrim(), (22.0, -25.0, 16.5))

    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output))
    texture_count = sum(value is not None for value in (grass_diffuse, grass_normal, bark_diffuse, bark_normal))
    print(f"[SUCCESS] Oil-palm virtual environment exported to: {output}")
    print(f"[INFO] Created {len(tree_locations)} oil palms, a terrain mesh, PhysX collisions, and {texture_count}/4 local PBR textures.")
    print("[INFO] Click inside the viewport, then use W/S to drive and A/D to turn the six-legged palm scout.")
    print("[INFO] Drive to the yellow beacon; cyan = searching, orange = palm detected, green = target lock.")

    # A few initial updates let materials and texture streaming finish before
    # the user starts navigating.  In headless mode this is also the validation
    # path used by automated checks.
    class KeyboardControl:
        """Track only the WASD keys and leave other Isaac Sim keys alone."""

        key_names = {
            carb.input.KeyboardInput.W: "W",
            carb.input.KeyboardInput.A: "A",
            carb.input.KeyboardInput.S: "S",
            carb.input.KeyboardInput.D: "D",
        }

        def __init__(self) -> None:
            self.held: set[str] = set()
            self.input = carb.input.acquire_input_interface()
            self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self._on_key)

        def _on_key(self, event, *unused) -> bool:
            key_name = self.key_names.get(event.input)
            if key_name is None:
                return False
            if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
                self.held.add(key_name)
                return True
            if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self.held.discard(key_name)
                return True
            return False

        def close(self) -> None:
            self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)

    class ScoutDashboard:
        """A small in-app panel so the sensor result is visible, not console-only."""

        def __init__(self) -> None:
            self.window = ui.Window("OIL PALM SCOUT", width=350, height=170)
            with self.window.frame:
                with ui.VStack(spacing=7):
                    ui.Label("SIX-LEGGED INSPECTION ROBOT", style={"font_size": 18, "color": 0xFF65DBFF})
                    ui.Label("Click the viewport first", style={"font_size": 14})
                    ui.Label("W/S: drive   A/D: turn", style={"font_size": 14})
                    self.signal = ui.Label("SENSOR: STARTING", style={"font_size": 17})
                    self.range = ui.Label("Nearest palm: --", style={"font_size": 14})

        def update(self, signal: str, nearest_name: str, distance: float, colour: int) -> None:
            self.signal.text = f"SENSOR: {signal}"
            self.signal.style = {"font_size": 17, "color": colour}
            self.range.text = f"Nearest palm: {nearest_name}  |  {distance:.1f} m"

    keyboard = None if ARGS.headless else KeyboardControl()
    dashboard = None if ARGS.headless else ScoutDashboard()
    last_signal = ""
    frame_count = max(1, int(seconds * 60))
    try:
        for frame_index in range(frame_count):
            if not SIMULATION_APP.is_running():
                break

            held = keyboard.held if keyboard else set()
            turn = float("A" in held) - float("D" in held)
            robot_heading += turn * 1.75 / 60.0
            drive = float("W" in held) - float("S" in held)
            candidate = [
                robot_position[0] + math.cos(robot_heading) * drive * 2.15 / 60.0,
                robot_position[1] + math.sin(robot_heading) * drive * 2.15 / 60.0,
            ]

            # Do not let the kinematic robot pass through a trunk or leave the
            # plot.  This is the same safe navigation rule a later PhysX/ROS
            # controller would use as a high-level safety boundary.
            blocked = False
            if abs(candidate[0]) > 15.3 or abs(candidate[1]) > 13.3:
                blocked = True
            if any(math.hypot(candidate[0] - float(palm["x"]), candidate[1] - float(palm["y"])) < 1.05 for palm in palm_records):
                blocked = True
            if drive and not blocked:
                robot_position[:] = candidate

            robot_translate.Set(
                Gf.Vec3d(robot_position[0], robot_position[1], terrain_height(*robot_position) + 0.76)
            )
            robot_yaw.Set(math.degrees(robot_heading))
            gait_amount = 15.0 if drive and not blocked else 0.0
            for rotate_op, phase in leg_rotation_ops:
                rotate_op.Set(gait_amount * math.sin(frame_index * 0.18 + phase))

            nearest = min(
                palm_records,
                key=lambda palm: math.hypot(robot_position[0] - float(palm["x"]), robot_position[1] - float(palm["y"])),
            )
            nearest_distance = math.hypot(robot_position[0] - float(nearest["x"]), robot_position[1] - float(nearest["y"]))
            target_distance = math.hypot(robot_position[0] - target_x, robot_position[1] - target_y)
            if target_distance < 1.8:
                signal, colour, light_colour = "TARGET LOCK: OIL PALM", 0xFF62F78C, Gf.Vec3f(0.12, 1.0, 0.25)
            elif nearest_distance < 3.8:
                signal, colour, light_colour = "OIL PALM DETECTED", 0xFFFFB454, Gf.Vec3f(1.0, 0.38, 0.04)
            elif blocked and drive:
                signal, colour, light_colour = "OBSTACLE: TRUNK / BOUNDARY", 0xFFFF4D4D, Gf.Vec3f(1.0, 0.05, 0.05)
            else:
                signal, colour, light_colour = "SEARCHING", 0xFF65DBFF, Gf.Vec3f(0.12, 0.85, 1.0)
            sensor_light.GetColorAttr().Set(light_colour)
            if signal != last_signal:
                print(f"[SENSOR] {signal} | nearest={nearest['name']} | range={nearest_distance:.1f} m")
                last_signal = signal
            if dashboard and frame_index % 6 == 0:
                dashboard.update(signal, str(nearest["name"]), nearest_distance, colour)

            SIMULATION_APP.update()
    finally:
        if keyboard:
            keyboard.close()


if __name__ == "__main__":
    try:
        build_scene(ARGS.output.resolve(), ARGS.seconds)
    except Exception as error:
        carb.log_error(f"Oil-palm virtual environment failed: {error}")
        raise
    finally:
        SIMULATION_APP.close()
