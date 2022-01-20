module OneDNN

export Memory
export Conv, BatchNorm, Pooling

# stdlib
import LinearAlgebra
import Random

# temp local teps
import CachedArrays

# deps
import ChainRulesCore
import Flux
import MacroTools
import Polyester
import UnPack: @unpack
import Zygote
import ZygoteRules: ZygoteRules, _pullback, AContext, literal_getproperty, literal_getfield

function pullback_for_default_literal_getproperty(cx::AContext, x, ::Val{f}) where {f}
    return _pullback(cx, literal_getfield, x, Val{f}())
end

# Experimental - should probably not set this to `true`.
const SIMILAR_FOR_SCRATCHPAD = false

#####
##### deps
#####

# Include auto-generated wrapper for OneDNN
include("lib/lib.jl")
include("wrap.jl")
include("tiledarrays.jl")
using .TiledArrays: TiledArrays
include("utils.jl")
include("types.jl")
include("tiled.jl")
include("memory.jl")
include("bf16.jl")

# ops
include("ops/simple.jl")
include("ops/matmul.jl")
include("ops/batchnorm.jl")
include("ops/concat.jl")
include("ops/pool.jl")
include("ops/innerproduct.jl")
include("ops/convolution.jl")

# include("placeholder.jl")

# Just create a global engine and stream for everything to use for now.
const GLOBAL_ENGINE = Ref{Engine}()
const GLOBAL_STREAM = Ref{Stream}()
const GLOBAL_ATTRIBUTES = Ref{Attributes}()
const GLOBAL_THREADPOOL = Ref{Any}()

_get_in_parallel() = (Threads.threadid() != 1)
function _parallel_for(n::Cint, f::Ptr{Cvoid})
    #Threads.@threads for i in Base.OneTo(n)
    Polyester.@batch per = thread for i in Base.OneTo(n)
        Wrap.call_opaque(f, i - 1, n)
    end
    return nothing
end

# Initialize the default Engine and Stream.
function __init__()
    ### Engine
    engine = Engine(Lib.dnnl_cpu)
    GLOBAL_ENGINE[] = engine

    ### Thread Pool
    # The functions we create do not capture any variables, thus no need to preserve them
    # somewhere.
    threadpool = Wrap.construct_threadpool(
        @cfunction(_get_in_parallel, Bool, ()),
        @cfunction(_parallel_for, Cvoid, (Cint, Ptr{Cvoid})),
        Threads.nthreads(),
    )
    GLOBAL_THREADPOOL[] = threadpool

    ### Stream
    stream = Stream(engine)
    @apicall dnnl_threadpool_interop_stream_create(stream, engine, threadpool.cpp_object)
    GLOBAL_STREAM[] = stream

    ### Default Attributes
    GLOBAL_ATTRIBUTES[] = Attributes()

    return nothing
end

noattributes() = GLOBAL_ATTRIBUTES[]
global_engine() = GLOBAL_ENGINE[]
global_stream() = GLOBAL_STREAM[]

#####
##### Flux compat
#####

# # TODO: Turns out that updating is better parallelized across the layer dimension than the
# # within each parameter itself
function Flux.Optimise.update!(
    o::Flux.Optimise.Descent, x::Memory{T,N}, Δ::Memory{U,N}
) where {T,U,N}
    # Make sure both objects have the same memory layout.
    # TODO: Handle different types correctly.
    # mx = memorydesc(x)
    # mΔ = memorydesc(Δ)
    # if mx != mΔ
    #     error("Incompatible memory formats!")
    #     # # Enter the realm of the type unstable!
    #     # sz = logicalsize(mx, Val(N))
    #     # indexer_x = TiledArrays.TiledIndexer{layout(mx)}(sz, padded_size(mx, Val(N)))
    #     # indexer_Δ = TiledArrays.TiledIndexer{layout(mΔ)}(
    #     #     logicalsize(mΔ, Val(N)), padded_size(mΔ, Val(N))
    #     # )
    #     # update_typed!(o, parent(x), indexer_x, parent(Δ), indexer_Δ, CartesianIndices(sz))
    #     # return nothing
    # end

    eta = convert(Float32, o.eta)
    xa = vec(parent(x))
    Δa = vec(parent(Δ))
    xa .-= (eta .* Δa)
    return nothing
end

function Flux.Optimise.update!(
    o::Flux.Optimise.Descent, x::Memory{T,N}, ix, y::Memory{U,N}, iy
) where {T,U,N}
    px = parent(x)
    py = parent(y)
    eta = convert(Float32, o.eta)
    for i in eachindex(ix, iy)
        @inbounds(px[ix[i]] -= eta * py[iy[i]])
    end
    return nothing
end

#####
##### Mirrored
#####

function Flux.Optimise.update!(
    o::Flux.Optimise.Descent, x::Mirrored, Δ::Memory{T,N}
) where {T,N}
    eta = convert(Float32, o.eta)
    px = parent(x)
    pΔ = parent(Δ)
    for i in eachindex(pΔ)
        @inbounds px[i] -= eta * pΔ[i]
    end
    return nothing
end

function Flux.Optimise.update!(
    o::Flux.Optimise.Descent, x::Mirrored, ix, y::Memory{T,N}, iy
) where {T,N}
    px = parent(x)
    py = parent(y)
    eta = convert(Float32, o.eta)
    for i in eachindex(ix, iy)
        @inbounds(px[ix[i]] -= eta * py[iy[i]])
    end
    return nothing
end

#####
##### Zygote Compat
#####

Zygote.accum(x::AbstractArray, y::Memory) = +(Memory(x), y)
Zygote.accum(x::Memory, y::AbstractArray) = +(x, Memory(y))
Zygote.accum(x::Memory, y::Memory) = +(x, y)

# Special cases
function Zygote.accum(_x::SubArray, y::Memory)
    x = maybe_reorder(memorydesc(y), Memory(_x))
    binary!(+, x, x, y)
    return x
end

# function update_typed!(
#     o::Flux.Optimise.Descent,
#     x,
#     indexer_x::TiledArrays.TiledIndexer,
#     y,
#     indexer_y::TiledArrays.TiledIndexer,
#     iter,
# )
#     for i in iter
#         ix = TiledArrays.genindex(indexer_x, Tuple(i))
#         iy = TiledArrays.genindex(indexer_y, Tuple(i))
#         x[ix] -= o.eta * y[iy]
#     end
#     return nothing
# end

# function update_typed!(o::Flux.Optimise.Descent, x::Memory, Δ::Memory)
#     return x .= x .- (o.eta .* Δ)
# end

# # # Apply the negative here so we can just add together in `update!`.
# # # This is because it appears that OneDNN is lacking a binary `-`
# # Flux.Optimise.apply!(o::Flux.Optimise.Descent, x, Δ::Memory) = linear!(Δ, -o.eta)
# #
# # # Expect `Memory` objects to already be negated from the `apply!` step.
# # function Flux.Optimise.update!(o::Flux.Optimise.Descent, x::Memory, Δ::Memory)
# #     return binary!(+, x, Flux.Optimise.apply!(o, x, Δ))
# end

end # module
