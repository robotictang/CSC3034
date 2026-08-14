"""Create a procedural durian plantation in NVIDIA Isaac Sim.

Run with Isaac Sim's launcher:
    ~/isaacsim/python.sh src/files/isaac_durian_plantation.py

It creates two stylised durian trees and a farmer centred between them,
without relying on downloadable Omniverse assets.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

try:
    from isaacsim import SimulationApp
except ImportError:  # Isaac Sim 4.x compatibility
    try:
        from omni.isaac.kit import SimulationApp
    except ImportError:
        SimulationApp = None


def create_plantation(seconds: float, output: Path) -> None:
    """Build the USDA stage, export it, and keep it visible briefly."""
    app = SimulationApp({"headless": False})

    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")

    def make_material(name: str, colour: tuple[float, float, float]):
        material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def place(prim, position, scale=None):
        transform = UsdGeom.Xformable(prim)
        transform.AddTranslateOp().Set(Gf.Vec3d(*position))
        if scale:
            transform.AddScaleOp().Set(Gf.Vec3f(*scale))

    def bind(prim, look):
        UsdShade.MaterialBindingAPI(prim).Bind(look)

    def sphere(path, position, radius, look, scale=None):
        shape = UsdGeom.Sphere.Define(stage, path)
        shape.CreateRadiusAttr(radius)
        place(shape.GetPrim(), position, scale)
        bind(shape.GetPrim(), look)

    def cylinder(path, position, radius, height, look):
        shape = UsdGeom.Cylinder.Define(stage, path)
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(height)
        place(shape.GetPrim(), position)
        bind(shape.GetPrim(), look)

    bark = make_material("Bark", (0.26, 0.10, 0.03))
    leaves = make_material("Leaves", (0.04, 0.28, 0.06))
    # A ripe yellow-green colour deliberately contrasts with the dark canopy.
    durian = make_material("DurianSkin", (0.85, 0.72, 0.05))
    soil = make_material("Soil", (0.22, 0.09, 0.025))
    shirt = make_material("FarmerShirt", (0.06, 0.28, 0.72))
    trousers = make_material("FarmerTrousers", (0.03, 0.08, 0.22))
    skin = make_material("Skin", (0.56, 0.29, 0.14))
    hat = make_material("Hat", (0.72, 0.48, 0.13))

    ground = UsdGeom.Cube.Define(stage, "/World/PlantationSoil")
    ground.CreateSizeAttr(1.0)
    place(ground.GetPrim(), (0, 0, -0.15), (12, 9, 0.30))
    bind(ground.GetPrim(), soil)

    def durian_tree(name: str, x: float) -> None:
        UsdGeom.Xform.Define(stage, f"/World/{name}")
        cylinder(f"/World/{name}/Trunk", (x, 0, 2.1), 0.32, 4.2, bark)
        canopy_offsets = (
            (-0.75, 0, 4.4), (0.75, 0, 4.4), (0, -0.7, 4.6),
            (0, 0.7, 4.6), (0, 0, 5.15),
        )
        for index, (dx, dy, dz) in enumerate(canopy_offsets):
            sphere(f"/World/{name}/Canopy_{index}", (x + dx, dy, dz), 1.15,
                   leaves, (1.15, 0.9, 0.85))

        # Fruit clusters around the canopy edge so they are easy to see in the
        # viewport.  Each fruit has simple conical spikes.
        fruit_positions = (
            (-0.85, -1.45, 4.05),
            (0.72, -1.45, 4.25),
            (-0.18, -1.50, 4.90),
            (1.10, -0.25, 4.52),
            (-1.10, -0.30, 4.55),
            (0.12, 1.05, 4.20),
        )
        for fruit_index, (dx, dy, dz) in enumerate(fruit_positions):
            path = f"/World/{name}/Durian_{fruit_index}"
            centre = (x + dx, dy, dz)
            # The vertically elongated core and short woody stalk match the
            # distinctive teardrop silhouette of a durian.
            sphere(path + "/Core", centre, 0.42, durian, (0.95, 0.95, 1.35))
            cylinder(path + "/Stem", (x + dx, dy, dz + 0.72), 0.09, 0.42, bark)

            # Dense, outward-facing cones form the characteristic thorny skin.
            # Rotating each cone away from the fruit core avoids the flat,
            # upward-only spikes of the earlier stylised version.
            spike_index = 0
            for vertical, ring_radius in ((-0.42, 0.20), (-0.22, 0.36),
                                          (0.00, 0.42), (0.22, 0.36), (0.42, 0.20)):
                for segment in range(8):
                    angle = segment * math.tau / 8 + (spike_index % 2) * 0.18
                    direction = Gf.Vec3d(
                        ring_radius * math.cos(angle),
                        ring_radius * math.sin(angle),
                        vertical,
                    ).GetNormalized()
                    spike = UsdGeom.Cone.Define(stage, path + f"/Spike_{spike_index}")
                    spike.CreateRadiusAttr(0.085)
                    spike.CreateHeightAttr(0.28)
                    spike_transform = UsdGeom.Xformable(spike.GetPrim())
                    spike_transform.AddTranslateOp().Set(
                        Gf.Vec3d(*centre) + direction * 0.45
                    )
                    quaternion = Gf.Rotation(Gf.Vec3d(0, 0, 1), direction).GetQuat()
                    spike_transform.AddOrientOp().Set(
                        Gf.Quatf(quaternion.GetReal(), Gf.Vec3f(*quaternion.GetImaginary()))
                    )
                    bind(spike.GetPrim(), durian)
                    spike_index += 1

    durian_tree("DurianTree_Left", -3.2)
    durian_tree("DurianTree_Right", 3.2)

    # Farmer standing at the midpoint of the trees.
    farmer = "/World/Farmer"
    cylinder(farmer + "/Torso", (0, 0, 1.85), 0.34, 1.25, shirt)
    sphere(farmer + "/Head", (0, 0, 2.70), 0.26, skin)
    cylinder(farmer + "/LeftLeg", (-0.16, 0, 0.65), 0.12, 1.3, trousers)
    cylinder(farmer + "/RightLeg", (0.16, 0, 0.65), 0.12, 1.3, trousers)
    cylinder(farmer + "/LeftArm", (-0.43, 0, 1.88), 0.10, 1.05, skin)
    cylinder(farmer + "/RightArm", (0.43, 0, 1.88), 0.10, 1.05, skin)
    cylinder(farmer + "/HatBrim", (0, 0, 2.98), 0.43, 0.07, hat)
    cylinder(farmer + "/HatCrown", (0, 0, 3.10), 0.25, 0.25, hat)

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(0.8)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(35, -30, 20))
    UsdLux.DomeLight.Define(stage, "/World/SkyLight").CreateIntensityAttr(500.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output))
    print(f"[SUCCESS] Durian plantation exported to: {output}")
    print("[INFO] Trees: x=-3.2 and x=3.2; farmer: x=0 (between the trees).")

    for _ in range(max(1, int(seconds * 60))):
        if not app.is_running():
            break
        app.update()
    app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a durian plantation in Isaac Sim.")
    parser.add_argument("--seconds", type=float, default=30, help="Seconds to keep the stage open.")
    parser.add_argument("--output", type=Path, default=Path("durian_plantation.usda"), help="USD output path.")
    args = parser.parse_args()
    if SimulationApp is None:
        raise SystemExit("Isaac Sim not found. Run with Isaac Sim's python.sh or python.bat launcher.")
    create_plantation(args.seconds, args.output.resolve())
