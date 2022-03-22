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
import SIMD
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

# Just create a global engine and stream for everything to use for now.
const GLOBAL_ENGINE = Ref{Engine}()
const GLOBAL_STREAM = Ref{Stream}()
const GLOBAL_ATTRIBUTES = Ref{Attributes}()
const GLOBAL_THREADPOOL = Ref{Any}()

_get_in_parallel() = (Threads.threadid() != 1)
function _parallel_for(n::Cint, f::Wrap.dnnl_kernel)
    Polyester.@batch (per = thread) for i in Base.OneTo(n)
        Wrap.call(f, i - 1, n)
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
        @cfunction(_parallel_for, Cvoid, (Cint, Wrap.dnnl_kernelDereferenced)),
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

const TRANSLATION_DICT_LOCK = ReentrantLock()
const TRANSLATION_DICT = Dict{Tuple{MemoryDesc,MemoryDesc},SIMDPtrMap}()

# # TODO: Turns out that updating is better parallelized across the layer dimension than the
# # within each parameter itself
function Flux.Optimise.update!(
    o::Flux.Optimise.Descent, x::Memory{T,N}, Δ::Memory{U,N}
) where {T,U,N}
    mx = memorydesc(x)
    mΔ = memorydesc(Δ)
    if mx != mΔ
        translation = Base.@lock TRANSLATION_DICT_LOCK begin
            key = (mx, mΔ)
            _translation = get(TRANSLATION_DICT, key, nothing)
            if _translation === nothing
                ix = generate_linear_indices(x)
                iΔ = generate_linear_indices(Δ)
                _translation = simdcompress(T, iΔ, ix)
                TRANSLATION_DICT[key] = _translation
            end
            _translation
        end
        sgd!(parent(x), parent(Δ), translation, convert(Float32, o.eta))
    else
        # Layouts are the same - just need to apply elementwise.
        _sgd!(parent(x), parent(Δ), convert(Float32, o.eta))
    end

    return nothing
end

function _sgd!(x::AbstractArray{T}, Δ::AbstractArray{T}, eta::T) where {T}
    for i in eachindex(x, Δ)
        @inbounds(x[i] -= eta * Δ[i])
    end
    return x
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

end # module
