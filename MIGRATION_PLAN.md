# Migration Plan: JuTrack.jl to TrackPad.jl

**Objective:** Migrate the legacy package `JuTrack.jl` to a new, high-performance architecture `TrackPad.jl`.
**Goals:** Enhanced usability, multiple AD support (Enzyme, TPSA), GPU integration, and Machine Learning readiness.
**Role of Legacy:** `JuTrack.jl` will be retained as the stable testbed and "ground truth" for validation.  The src of JuTrack.jl will be found in ../JuTrack.jl/

## Phase 1: Architectural Design & Foundation

The primary goal of this phase is to ensure the codebase is hardware and type agnostic, leveraging Julia's modern package ecosystem.

### 1.1 The Type System (Parametric Structs)

To support Automatic Differentiation (AD) and Truncated Power Series Algebra (TPSA), concrete types like `Float64` must be replaced with parametric types. We must establish a robust type hierarchy to allow dispatching on element types (e.g., `is_drift(::AbstractElement)`).

**Action:** Refactor element definitions to use a strictly parametric hierarchy.

**Abstract Base:** Define `AbstractElement` and subtypes like `AbstractMagnet`, `AbstractDrift`, `AbstractCavity`.

**Implementation:**

```julia
abstract type AbstractElement end

struct Quadrupole{T} <: AbstractElement
    L::T       # Length
    k1::T      # Normal Quadrupole strength
    tilt::T    # Rotation angle
end
# Constructor ensures promotion to common type T
Quadrupole(L, k1, tilt) = Quadrupole(promote(L, k1, tilt)...)
```

**Memory Layout:** Ensure structs are `isbits` types (immutable, no pointers) whenever possible to allow efficient GPU memory coalescing.

### 1.2 Package Extensions (Julia 1.9+, shall we do it?)

We will use Julia's "Weak Dependencies" to keep the core lightweight. Heavy dependencies (CUDA, Enzyme) will only load when imported by the user.

**Core (`src/`):** Contains pure physics, lattice definitions, and single-threaded CPU tracking.

**`Project.toml` Configuration:**

```toml
[weakdeps]
CUDA = "..."
Enzyme = "..."
Flux = "..."

[extensions]
TrackPadCUDAExt = "CUDA"
TrackPadEnzymeExt = "Enzyme"
```

**Extension Logic:**

*   **`TrackPadCUDAExt`:** Implements `KernelAbstractions.jl` backends for NVIDIA GPUs.
*   **`TrackPadEnzymeExt`:** Defines `EnzymeRules.forward` and `EnzymeRules.reverse` for specific physics elements if the automatic generation is suboptimal.

## Phase 2: Core Migration (Physics Porting)

### 2.1 Establishing Ground Truth

**Freeze JuTrack:** The master branch of `JuTrack.jl` is now the reference standard.

**Validation Suite:**

*   Create a "golden dataset" generator in `JuTrack`.
*   **Data Format:** Use HDF5 (`JLD2.jl`) to store full 6D phase space coordinates (x, px, y, py, z, δ) at the entrance and exit of every element in a standard test lattice (e.g., FODO, DBA).
*   **Precision:** Save as `Float64` to compare against `TrackPad` with a tolerance of `1e-14`.

### 2.2 Porting Integrators & Memory Management

**Allocation-Free Kernels:**

*   Port Symplectic Integrators (Drift-Kick-Drift, Yoshida 4th order) to `TrackPad`.
*   **Constraint:** Kernels must accept `StaticArrays` or view-based arguments. They must never allocate memory on the heap (no `zeros()`, no `[]`) inside the hot loop.

**Signature Standard:**

```julia
# Mutating in-place is critical for Enzyme and GPU
function track_element!(state::AbstractVector, elem::Quadrupole)
    # Update state[1]...state[6] in place
end
```

### 2.3 Lattice Construction & Parsing

**Intermediate Representation (IR):**

*   Decouple parsing from the physics struct. Create a `Lattice` struct that holds a flat `Vector{AbstractElement}`.

**Parsers:**

*   Implement a MAD-X parser that reads sequence files and converts them into the `TrackPad` IR.

*   Consider adopt AcceleratorLattice.jl?

### Lattice optics calculation.

* Migrate the lattice calculation functions from JuTrack
* Add 6-D optics by calculating the crab dispersion.

### Adding capability of particle generation
* Add a function of generating particle matching a given optics function.

## Phase 3: Advanced Feature Integration

### 3.1 Automatic Differentiation (AD) & TPSA

Implement a high-level API: `track(lattice, beam, method)`.

**Enzyme.jl:**

*   Focus on Forward Mode AD for optimization (finding magnet strengths `k` to minimize beam size).
*   **Implementation Detail:** Ensure the `track!` loop is annotated with `Enzyme.Duplicated` for variables we want gradients for, and `Enzyme.Const` for fixed parameters.
*   Handle "Active" variables (magnets) vs. "Passive" variables (drifts) efficiently.

**Fast first order AD**

* An faster way?

**TPSA (High-Order Maps):**

*   Homemade TPSA
*   Current Julia version of TPSA has bad memory allocation, need to be improved
*   Try to match the speed of gtpsa.jl
*   To adopt TPSA to GPU, may need to change the memory layout. 

### 3.2 GPU Integration

**KernelAbstractions.jl:** Use this library to write backend-agnostic kernels.

**Workflow:**

1.  **Kernel Definition:**
    ```julia
    @kernel function track_bunch_kernel!(bunch, lattice)
        idx = @index(Global)
        # Access particle: bunch[idx]
        # Loop over lattice elements
    end
    ```
2.  **Dispatch:** If the user passes a `CuArray` (CUDA) or `ROCArray` (ROCm), `TrackPad` detects the array type and launches the kernel on the GPU.
3.  **Memory Transfer:** Minimize host-device transfer. Move the entire lattice definition to GPU memory once before tracking begins.

### 3.3 Machine Learning (Differentiable Physics)

**Differentiable Layer:**

*   Wrap the `track` function in a `Lux.jl` or `Flux.jl` layer.
*   `y = model(x)` where `model` includes a physics step.

**Surrogate Modeling:** Use the differentiable physics engine to generate training data for a neural network surrogate, allowing for millisecond-scale feedback loops in control systems.

**Hybrid Optimization:** Optimize a loss function $L = L_{physics} + L_{NN}$, where the neural network corrects for physics effects not modeled in the symplectic integrator (e.g., complex space charge effects approximated by a neural net).

## Phase 4: Documentation & Usability

Docs will be built using `Documenter.jl` and `Literate.jl` to turn actual test scripts into readable tutorials.

**Quick Start:**

*   "Tracking a FODO cell in 5 minutes" (Code-first approach).
*   Installation guide handling the package extensions (e.g., "To use GPU, simply using CUDA").

**Migration Guide:**

*   A side-by-side comparison table of `JuTrack` functions vs. `TrackPad` equivalents.
*   A script to convert `JuTrack` saved lattices into `TrackPad` format.

**Topic Guides (The "Why" and "How"):**

*   **GPU Tracking:** Explaining batch sizes, memory limits, and when to use GPU vs CPU.
*   **Map Extraction:** How to use TPSA to extract chromaticity and resonance driving terms.
*   **Optimization:** A walkthrough of using Enzyme to match Twiss parameters.

**API Reference:**

*   Auto-generated from docstrings. Ensure every function has an "Arguments" and "Returns" section.
