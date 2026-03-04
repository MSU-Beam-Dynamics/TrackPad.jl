"""
    TrackPadEnzymeExt

Enzyme extension for TrackPad.jl. Provides automatic differentiation rules
for particle tracking through accelerator elements.

This module is loaded automatically when `using Enzyme` is called alongside TrackPad.
"""
module TrackPadEnzymeExt

using TrackPad
using Enzyme

# TODO: Implement Enzyme-specific AD rules
# - EnzymeRules.forward for element tracking functions
# - EnzymeRules.reverse for gradient computation
# - Handle "Active" (magnets with variable k) vs "Passive" (drifts) efficiently

end # module TrackPadEnzymeExt
