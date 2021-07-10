module OneDNN

# stdlib
using LinearAlgebra: LinearAlgebra

# deps
using ChainRulesCore: ChainRulesCore
using Flux: Flux
using MacroTools: MacroTools
using Polyester: Polyester
using Zygote: Zygote

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
    #Threads.@threads for i in Base.OneTo(n)
    Polyester.@batch per = thread for i in Base.OneTo(n)
        Wrap.call_opaque(f, i - 1, n)
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
        stream, GLOBAL_ENGINE[], threadpool.cpp_object
    )
    GLOBAL_STREAM[] = stream
    return nothing
end

global_engine() = GLOBAL_ENGINE[]
global_stream() = GLOBAL_STREAM[]

#####
##### Flux compat
#####

# # TODO: Turns out that updating is better parallelized across the layer dimension than the
# # within each parameter itself
function Flux.Optimise.update!(
    o::Flux.Optimise.Descent, x::Memory{T,N}, Δ::Memory{T,N}
) where {T,N}
    # Make sure both objects have the same memory layout.
    mx = memorydesc(x)
    mΔ = memorydesc(Δ)
    if mx != mΔ
        error("Incompatible memory formats!")
        # # Enter the realm of the type unstable!
        # sz = logicalsize(mx, Val(N))
        # indexer_x = TiledArrays.TiledIndexer{layout(mx)}(sz, padded_size(mx, Val(N)))
        # indexer_Δ = TiledArrays.TiledIndexer{layout(mΔ)}(
        #     logicalsize(mΔ, Val(N)), padded_size(mΔ, Val(N))
        # )
        # update_typed!(o, parent(x), indexer_x, parent(Δ), indexer_Δ, CartesianIndices(sz))
        # return nothing
    end

    xa = vec(parent(x))
    Δa = vec(parent(Δ))
    xa .= xa .- (o.eta .* Δa)
    return nothing
end

function Flux.Optimise.update!(
    o::Flux.Optimise.Descent, x::Memory{T,N}, ix, y::Memory{T,N}, iy
) where {T,N}
    px = parent(x)
    py = parent(y)
    eta = convert(T, o.eta)
    for i in eachindex(ix, iy)
        @inbounds(px[ix[i]] -= eta * py[iy[i]])
    end
    return nothing
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

function setup()
    diff_weights_dims = (1024, 1024)
    diff_bias_dims = (1024,)
    diff_dst_dims = (1024, 2^15)
    src_dims = (1024, 2^15)

    diff_dst = OneDNN.Memory(randn(Float32, diff_dst_dims))
    src = OneDNN.Memory(randn(Float32, src_dims))

    diff_bias_desc = memorydesc(Float32, diff_bias_dims, Lib.dnnl_a)
    inner_product_desc = Ref{Lib.dnnl_inner_product_desc_t}()
    @apicall dnnl_inner_product_backward_weights_desc_init(
        inner_product_desc,
        memorydesc(Float32, src_dims, dnnl_format_any()),
        memorydesc(Float32, diff_weights_dims, dnnl_format_any()),
        diff_bias_desc,
        memorydesc(Float32, diff_dst_dims, dnnl_format_any()),
    )

    primitive_desc = __PrimitiveDescriptor(
        inner_product_desc, noattributes(), global_engine(), noforward()
    )
    primitive = __Primitive(primitive_desc)

    diff_weights_desc_opt = query_md(primitive_desc, Lib.dnnl_query_diff_weights_md)
    diff_weights = similar(
        diff_dst, eltype(diff_dst), diff_weights_dims, diff_weights_desc_opt
    )

    diff_bias = similar(diff_dst, eltype(diff_dst), diff_bias_dims, diff_bias_desc)
    return (; primitive, primitive_desc, src, diff_bias, diff_weights, diff_dst)
end

end # module
