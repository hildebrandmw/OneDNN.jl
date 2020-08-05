module OneDNN

# stdlib
import LinearAlgebra

# julia ml
import Zygote
import Flux

# for primitive tracing
# note: primitive tracing ended up not working, the overhead from Cassette was too high.
import Cassette
import MacroTools

# time how long some operations take
import TimerOutputs
const to = TimerOutputs.TimerOutput()

#####
##### deps
#####

# Include auto-generated wrapper for OneDNN
include("lib/lib.jl")
include("utils.jl")
include("attributes.jl")

include("memory.jl")
include("execute.jl")

# ops
include("ops/eltwise.jl")
include("ops/binary.jl")
include("ops/reorder.jl")
include("ops/concat.jl")
include("ops/matmul.jl")
include("ops/innerproduct.jl")

# tracing
include("tracer.jl")

# Just create a global engine and stream for everything to use for now.
const GLOBAL_ENGINE = Ref{Engine}()
const GLOBAL_STREAM = Ref{Stream}()

# Initialize the default Engine and Stream.
function __init__()
    GLOBAL_ENGINE[] = Engine(Lib.dnnl_cpu)
    GLOBAL_STREAM[] = Stream(GLOBAL_ENGINE[])
end

global_engine() = GLOBAL_ENGINE[].handle
global_stream() = GLOBAL_STREAM[].handle

#####
##### Flux compat
#####

# Apply the negative here so we can just add together in `update!`.
# This is because it appears that OneDNN is lacking a binary `-`
Flux.Optimise.apply!(o::Flux.Optimise.Descent, x, Δ::Memory) = linear!(Δ, -o.eta)

# Expect `Memory` objects to already be negated from the `apply!` step.
function Flux.Optimise.update!(o::Flux.Optimise.Descent, x::Memory, Δ::Memory)
    return binary!(+, x, Flux.Optimise.apply!(o, x, Δ))
end

end # module
