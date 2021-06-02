module OneDNN

# stdlib
using LinearAlgebra: LinearAlgebra

# deps
using ChainRulesCore: ChainRulesCore
using Flux: Flux
using MacroTools: MacroTools
using Zygote: Zygote

# # time how long some operations take
# using TimerOutputs: TimerOutputs
# const to = TimerOutputs.TimerOutput()

#####
##### deps
#####

# Include auto-generated wrapper for OneDNN
include("lib/lib.jl")
include("wrap.jl")
include("tiledarrays.jl")
using .TiledArrays: TiledArrays
include("utils.jl")
include("tiled.jl")
include("memory.jl")

# ops
include("ops/simple.jl")
include("ops/matmul.jl")
include("ops/concat.jl")
include("ops/innerproduct.jl")

# include("placeholder.jl")

# Just create a global engine and stream for everything to use for now.
const GLOBAL_ENGINE = Ref{Engine}()
const GLOBAL_STREAM = Ref{Stream}()
const GLOBAL_THREADPOOL = Ref{Any}()

_get_in_parallel() = (Threads.threadid() != 1)
function _parallel_for(n::Cint, f::Ptr{Cvoid})
    Threads.@threads for i in Base.OneTo(n)
        Wrap.call_opaque(f, i-1, n)
    end
end

# Initialize the default Engine and Stream.
function __init__()
    GLOBAL_ENGINE[] = Engine(Lib.dnnl_cpu)

    # Create a thread pool.
    threadpool = Wrap.construct_threadpool(
        @cfunction(_get_in_parallel, Bool, ()),
        @cfunction(_parallel_for, Cvoid, (Cint, Ptr{Cvoid})),
        Threads.nthreads(),
    )
    GLOBAL_THREADPOOL[] = threadpool

    stream = Stream()
    @apicall dnnl_threadpool_interop_stream_create(
        stream,
        GLOBAL_ENGINE[],
        threadpool.cpp_object,
    )
    GLOBAL_STREAM[] = stream
    return nothing
end

global_engine() = GLOBAL_ENGINE[]
global_stream() = GLOBAL_STREAM[]

#####
##### Flux compat
#####

# TODO: Turns out that updating is better parallelized across the layer dimension than the
# within each parameter itself
function Flux.Optimise.update!(o::Flux.Optimise.Descent, x::Memory, Δ::Memory)
    # Make sure both objects have the same memory layout.
    mx = memorydesc(x)
    if mx != memorydesc(Δ)
        update_typed!(o, typed(x), typed(Δ))
        return nothing
    end

    xa = reshape(parent(x), size(x))
    Δa = reshape(parent(Δ), size(Δ))

    xa .= xa .- (o.eta .* Δa)
    return nothing
end

function update_typed!(o::Flux.Optimise.Descent, x::Memory, Δ::Memory)
    x .= x .- (o.eta .* Δ)
end

# # # Apply the negative here so we can just add together in `update!`.
# # # This is because it appears that OneDNN is lacking a binary `-`
# # Flux.Optimise.apply!(o::Flux.Optimise.Descent, x, Δ::Memory) = linear!(Δ, -o.eta)
# #
# # # Expect `Memory` objects to already be negated from the `apply!` step.
# # function Flux.Optimise.update!(o::Flux.Optimise.Descent, x::Memory, Δ::Memory)
# #     return binary!(+, x, Flux.Optimise.apply!(o, x, Δ))
# end

end # module
