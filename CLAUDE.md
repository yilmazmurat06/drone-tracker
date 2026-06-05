# CLAUDE.md
## What this is

CPU-only, onboard air-to-air small-drone (2–6 px target) detection & tracking, in C++17/20.
Target hardware: 6-core ARM (Cortex-A78/A76). **No GPU, no heavy ML frameworks** (no
PyTorch/TF) — classic CV + hand-written Kalman/IMM. Hard latency budget: photon → steering
command **< 50 ms**. This is framed as a *tracking* problem, not classification (the system
arrives already cued at the target). The six design problems and their weak points are
documented in `initial-prompt.md`; `README.md` has the module map and status checklist.

## Commands

```bash
cmake -S . -B build           # configure (re-run after adding files/CMakeLists)
cmake --build build           # build
ctest --test-dir build --output-on-failure   # run all tests
ctest --test-dir build -R ring_buffer         # run one test by name (regex)
./build/tests/test_ring_buffer                # run a test binary directly (binaries under build/tests/)
./build/app/drone_tracker     # end-to-end synthetic-source demo (writes sample PNGs to ./samples/)
```

**OpenCV is optional at configure time.** If `find_package(OpenCV)` fails, CMake builds only
the core (`common` + `pipeline`) and the OpenCV-free tests (e.g. `ring_buffer`), and skips
`io/stabilization/detection/tracking/fusion/app`. Install with `brew install opencv` (macOS)
or `apt install libopencv-dev` (Ubuntu), then re-run `cmake -S . -B build`. Required OpenCV
components are listed in the root `CMakeLists.txt` `find_package` line — **add the component
there** when you start using a new OpenCV module (a missing component shows up as an
*undefined symbol at link time*, not a configure error).

`compile_commands.json` is symlinked at repo root → clangd works. Stale red squiggles
referencing `dtrack/...` headers usually mean compile_commands hasn't been regenerated; they
clear after `cmake -S . -B build`.

## Architecture

**Per-module static libraries.** Each top-level dir (`common`, `pipeline`, `io`,
`stabilization`, `detection`, `tracking`, `fusion`) is its own CMake library exposing headers
under `include/dtrack/<module>/` (the `dtrack/` prefix keeps include paths unambiguous:
`#include "dtrack/detection/detector.hpp"`). Namespaces mirror dirs: `dtrack::common`,
`dtrack::io`, etc.

**Header-only modules are `INTERFACE` libraries; they become `STATIC` when the first `.cpp`
is added.** Each module's `CMakeLists.txt` carries a comment showing the exact STATIC
conversion. `common` and `pipeline` are intentionally header-only.

**Interface-first.** Every module is defined by a pure-virtual interface (`ICameraSource`,
`IStabilizer`, `IDetector`, `IDiscriminator`, `ITracker`, `ITrackFusion`). The pipeline only
ever sees interfaces, so concrete implementations swap without touching wiring. Example: the
synthetic camera and a real/simulator camera are both `ICameraSource`; changing the source is
a one-line change in `app/main.cpp`.

**Pipeline = threaded stages joined by lock-free SPSC queues.** Read these together to
understand the core:
- `common/include/dtrack/common/ring_buffer.hpp` — `SpscRingBuffer<T>` (Single Producer
  Single Consumer, lock-free via `std::atomic` acquire/release). Has `push_overwrite`
  implementing **drop-oldest** back-pressure (real-time: keep the freshest frame, drop stale).
- `pipeline/include/dtrack/pipeline/stage.hpp` — `Stage<In, Out>` runs its own thread:
  `pop input → process() → emit output`. **To add a stage, subclass and implement only
  `process()`.** A `process()` returning `std::nullopt` emits nothing. Source stages (camera)
  have a `nullptr` input queue and use `In = common::Tick` (a sentinel "no input" type); they
  produce data each loop. Sink stages have a `nullptr` output queue.
- `pipeline/include/dtrack/pipeline/pipeline.hpp` — owns stages, starts them consumer→producer
  (reverse add-order) and stops them producer→consumer, so queues are never read before ready
  and never starved on shutdown. **Add stages in flow order** (source first).

**Data contract.** All stages speak the types in `common/include/dtrack/common/types.hpp`
(`Frame`/`FramePtr`, `ImuSample`, `StabilizedFrame`, `EgoMotion`, `Detection`, `Track`).
Frames flow as `shared_ptr<const Frame>` — moved between stages, never deep-copied. All
timestamps use the single `steady_clock` from `common/time.hpp`; camera and IMU sources share
a common `t0` so their stamps align (required for gyro↔frame matching and track-level fusion).

**IMU is pulled, not chained.** Despite the data-flow diagram showing an IMU thread, the IMU
is *not* a pipeline stage. The stabilizer holds an `IImuSource` and calls `drain()` to get the
samples covering each frame. `drain()` returns all samples accumulated since the last call.

**Synthetic source = ground truth.** `io/synthetic_scene.hpp` (`SceneModel`) is pure math (no
OpenCV, deterministic) modeling both the camera image motion and the IMU angular rate from the
*same* physical truth, so stabilization fusion solves a real problem. `SyntheticCameraSource`
renders it; `SyntheticImuSource` reads `ego_rate()` and adds bias-drift + noise. Because the
model is deterministic, tests assert detector/tracker accuracy against known target positions
(see `tests/test_synthetic_source.cpp`). `SyntheticImuSource::generate_until(t)` is the
deterministic, wall-clock-free path used by tests; `drain()` wraps it with real time.

**Simulator path.** `io/video_camera_source.hpp` (`VideoCameraSource`) wraps
`cv::VideoCapture` and is the entry point for a drone simulator (AirSim/Gazebo via RTSP/UDP or
a recorded `.mp4`) and for real V4L2 hardware. The project will be validated in a simulator
before flying on a real drone — keep camera/IMU implementations swappable behind their
interfaces.

## Conventions

- In-code comments are written in **Turkish** (the maintainer is Turkish-speaking and new to
  the domain). Match this: explain *why*, expand abbreviations on first use, keep prose
  approachable. Respond to the user in Turkish, building one module at a time.
- New work follows the flow order: `io → stabilization → detection → tracking → fusion`. Wire
  each new stage into `app/main.cpp` and validate it against the synthetic source before moving
  on.
- ** Main goal is continuous detection(getting the target in the box) and continuous tracking. This is the main goal. We need to make an EXCELENT pipeline**
- Make research of you need to search for the best-practice implementations of the problem solutions. This is unquestionable.
- Use mathemathics of necessary. 