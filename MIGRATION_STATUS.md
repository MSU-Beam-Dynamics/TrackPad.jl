# Migration Status

## Phase 1: Architectural Design & Foundation

### 1.1 The Type System (Parametric Structs)
- [x] Refactor element definitions to use a strictly parametric hierarchy.
- [x] Implement `AbstractElement` and subtypes.
- [x] Ensure structs are `isbits` types (mostly).
    - **Update:** Implemented `Adapt.jl` pattern to handle `name::Symbol` (CPU) -> `name::Nothing` (GPU) to satisfy `isbits` on device.
- [x] Use `StaticArrays` for all fixed-size vector/matrix fields.

### 1.2 Package Extensions
- [x] Add `Adapt` dependency.
- [ ] Add `KernelAbstractions` dependency (Planned).
- [ ] Configure `Project.toml` for extensions (Planned).

## Phase 2: Core Migration (Physics Porting)
- [ ] Establish Ground Truth (JuTrack validation).
- [ ] Port Integrators.

## Phase 3: Advanced Feature Integration
- [ ] Automatic Differentiation.
- [ ] GPU Integration.
- [ ] Machine Learning.

## Phase 4: Documentation & Usability
- [ ] Documentation.
