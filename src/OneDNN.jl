module OneDNN

# stdlib
using LinearAlgebra: LinearAlgebra

# # julia ml
# import Zygote
# import Flux

using Flux: Flux
using MacroTools: MacroTools

# time how long some operations take
using TimerOutputs: TimerOutputs
const to = TimerOutputs.TimerOutput()

#####
##### deps
#####

# Include auto-generated wrapper for OneDNN
include("lib/lib.jl")
include("tiledarrays.jl")
using .TiledArrays: TiledArrays
include("utils.jl")
include("tiled.jl")
include("memory.jl")

# ops
include("ops/eltwise.jl")
include("ops/binary.jl")
include("ops/reorder.jl")
include("ops/matmul.jl")
include("ops/concat.jl")
include("ops/innerproduct.jl")
#
# # tracing
# include("tracer.jl")

# Just create a global engine and stream for everything to use for now.
const GLOBAL_ENGINE = Ref{Engine}()
const GLOBAL_STREAM = Ref{Stream}()

# Initialize the default Engine and Stream.
function __init__()
    GLOBAL_ENGINE[] = Engine(Lib.dnnl_cpu)
    return GLOBAL_STREAM[] = Stream(GLOBAL_ENGINE[])
end

global_engine() = GLOBAL_ENGINE[]
global_stream() = GLOBAL_STREAM[]

# #####
# ##### Flux compat
# #####
#
# # TODO: Turns out that updating is better parallelized across the layer dimension than the
# # within each parameter itself
# function Flux.Optimise.update!(o::Flux.Optimise.Descent, x::Memory, Δ::Memory)
#     # Make sure both objects have the same memory layout.
#     if memorydesc(x) != memorydesc(Δ)
#         error("Cannot update Memories with different descriptions")
#     end
#
#     x.array .= x.array .- (o.eta .* Δ.array)
#     return nothing
# end
#
# # # Apply the negative here so we can just add together in `update!`.
# # # This is because it appears that OneDNN is lacking a binary `-`
# # Flux.Optimise.apply!(o::Flux.Optimise.Descent, x, Δ::Memory) = linear!(Δ, -o.eta)
# #
# # # Expect `Memory` objects to already be negated from the `apply!` step.
# # function Flux.Optimise.update!(o::Flux.Optimise.Descent, x::Memory, Δ::Memory)
# #     return binary!(+, x, Flux.Optimise.apply!(o, x, Δ))
# end

end # module
