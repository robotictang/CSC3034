# Newton Physics Engine examples

Newton is separate from NVIDIA Isaac Sim. These examples use the Newton
environment at `../.venv-newton` and the RTX 4000 GPU (`cuda:0`).

## Run all three verification examples

From the `CSC3034` folder:

```bash
bash newton_examples/run_headless_examples.sh
```

This runs the simulations without opening a window. It is useful for proving
that CUDA and the Newton installation work correctly.

## Show each example to the lecturer

Run one command at a time. Use `--viewer gl` to open the Newton viewer; close
the viewer window when you have captured a screenshot.

```bash
source .venv-newton/bin/activate

# Example 1: a double pendulum with revolute joints
python -m newton.examples basic_pendulum --device cuda:0 --viewer gl

# Example 2: sphere, box, cylinder, capsule and mesh collision shapes falling
# onto a ground plane
python -m newton.examples basic_shapes --device cuda:0 --viewer gl

# Example 3: different rigid-body joint types and their motion constraints
python -m newton.examples basic_joints --device cuda:0 --viewer gl
```

## What to include in your submission

For each example, record:

1. The command used and that it ran on `cuda:0`.
2. One screenshot from the Newton viewer.
3. A short explanation: pendulum = articulated rotation; shapes = gravity and
   collision response; joints = constrained rigid-body motion.
4. One observation, such as how the pendulum swings under gravity or how the
   dropped objects settle on the ground plane.

The examples are the official Newton examples distributed with Newton 1.5.0.

## Custom project: oil-palm forest patrol

`oil_palm_hexapod.py` is a custom Newton scene for this coursework. A
six-legged robot explores a stylised oil-palm forest. Click in the 3D window,
then use `W`/`S` to move and `A`/`D` to turn. The side panel and terminal
show a weak, strong, or target-lock signal when the robot approaches a palm.

```bash
python newton_examples/oil_palm_hexapod.py --device cuda:0 --viewer gl
```
