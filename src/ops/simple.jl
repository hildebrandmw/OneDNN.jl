#####
##### Reorder
#####

function reorder(md::MemoryDesc, from::Memory)
    to = similar(from, eltype(from), size(from), md)
    reorder!(to, from)
    return kernel_exit_hook(to)
end

function reorder!(to::Memory, from::Memory, attributes = noattributes())
    temp_primitive(
        Lib.dnnl_reorder_primitive_desc_create,
        from,
        global_engine(),
        to,
        global_engine(),
        attributes,
    ) do p, _
        execute!(p, @dnnl_args to from)
    end
    return to
end

# Pullback is simply the identity.
# Let downstream kernels decide if they want to reorder the sensitivities.
function ChainRulesCore.rrule(
    ::typeof(reorder), desc::MemoryDesc, from::Memory, format = Opaque
)
    to = reorder(desc, from, format)
    pullback = function reorder_pullback(Δ)
        return (ChainRules.NoTangent(), ChainRules.NoTangent(), Δ, ChainRules.NoTangent())
    end
    return to, pullback
end

#####
##### Eltwise
#####

eltwise(f::F, src::Memory) where {F} = eltwise(src, forward_expand(f)...)
eltwise(::typeof(identity), src::Memory) = src
function eltwise(
    src::Memory, kind::Lib.dnnl_alg_kind_t, alpha = one(Float32), beta = zero(Float32)
)
    # Keep similar format to source
    dst = similar(src)
    eltwise!(dst, src, kind, alpha, beta)
    return kernel_exit_hook(dst)
end

function eltwise!(
    dst::Memory,
    src::Memory,
    algo::Lib.dnnl_alg_kind_t,
    alpha = one(Float32),
    beta = zero(Float32),
)
    opdesc = Ref{Lib.dnnl_eltwise_desc_t}()
    @apicall dnnl_eltwise_forward_desc_init(opdesc, Inference(), algo, src, alpha, beta)

    temp_primitive(opdesc, noattributes(), global_engine(), noforward()) do primitive, _
        execute!(primitive, @dnnl_args dst src)
    end
    return dst
end

#####
##### Backward Eltwise
#####

eltwise_backward(::typeof(identity), diff_data::Memory, data::Memory) = diff_data
function eltwise_backward(f::F, diff_data::Memory, data::Memory) where {F}
    return eltwise_backward(diff_data, data, backward_expand(f)...)
end

function eltwise_backward(
    diff_data::Memory,
    data::Memory,
    kind::Lib.dnnl_alg_kind_t,
    alpha = one(Float32),
    beta = zero(Float32),
    use_dst_for_bwd::Bool = false,
)
    diff_src = similar(diff_data)
    eltwise_backward!(diff_src, diff_data, data, kind, alpha, beta, use_dst_for_bwd)
    return kernel_exit_hook(diff_src)
end

function eltwise_backward!(
    diff_src::Memory,
    diff_dst::Memory,
    data::Memory,
    kind::Lib.dnnl_alg_kind_t,
    alpha = one(Float32),
    beta = zero(Float32),
    use_dst_for_bwd::Bool = false,
)
    opdesc = Ref{Lib.dnnl_eltwise_desc_t}()
    @apicall dnnl_eltwise_backward_desc_init(opdesc, kind, diff_dst, data, alpha, beta)

    if use_dst_for_bwd
        dst = data
        args = @dnnl_args diff_src diff_dst dst
    else
        src = data
        args = @dnnl_args diff_src diff_dst src
    end

    temp_primitive(opdesc, noattributes(), global_engine(), noforward()) do p, _
        execute!(p, args)
    end
    return diff_src
end

#####
##### Convenience Wrappers.
#####

# Aliases
# N.B.: Must be kept insync with tests!
Base.identity(x::Memory) = x
const ELTWISE_ALIASES = [
    :(Base.abs), :(Flux.sigmoid), :(Base.sqrt), :(Flux.relu), :(Base.log)
]
for op in ELTWISE_ALIASES
    @eval $op(x::Memory) = eltwise($op, x)
end

# Conversion to OneDNN forward eltwise ops
struct Linear
    α::Float32
    β::Float32
end

forward_expand(::typeof(Base.abs)) = (Lib.dnnl_eltwise_abs, zero(Float32), zero(Float32))
forward_expand(x::Linear) = (Lib.dnnl_eltwise_linear, x.α, x.β)
function forward_expand(::typeof(Flux.sigmoid))
    return (Lib.dnnl_eltwise_logistic, zero(Float32), zero(Float32))
end
forward_expand(::typeof(Base.sqrt)) = (Lib.dnnl_eltwise_sqrt, zero(Float32), zero(Float32))
forward_expand(::typeof(Flux.relu)) = (Lib.dnnl_eltwise_relu, zero(Float32), zero(Float32))
forward_expand(::typeof(Base.log)) = (Lib.dnnl_eltwise_log, zero(Float32), zero(Float32))

# Conversion to OneDNN backward eltwise ops
function backward_expand(::typeof(Base.abs))
    return (Lib.dnnl_eltwise_abs, zero(Float32), zero(Float32), false)
end
backward_expand(x::Linear) = (Lib.dnnl_eltwise_linear, x.α, x.β, false)
function backward_expand(::typeof(Flux.sigmoid))
    return (Lib.dnnl_eltwise_logistic_use_dst_for_bwd, zero(Float32), zero(Float32), true)
end
function backward_expand(::typeof(Base.sqrt))
    return (Lib.dnnl_eltwise_sqrt, zero(Float32), zero(Float32), false)
end
function backward_expand(::typeof(Flux.relu))
    return (Lib.dnnl_eltwise_relu_use_dst_for_bwd, zero(Float32), zero(Float32), true)
end
function backward_expand(::typeof(Base.log))
    return (Lib.dnnl_eltwise_log, zero(Float32), zero(Float32), false)
end

# Can this elementwise op be fused into a previous operation?
# Only if the backward op can operate on the destination tensor only.
dst_for_bwd(f::F) where {F} = last(backward_expand(f))
canfuse(f::F) where {F} = dst_for_bwd(f)
canfuse(::typeof(identity)) = true

# Forward backprop definition to the main entry point.
for op in ELTWISE_ALIASES
    @eval function ChainRulesCore.rrule(::typeof($op), x::Memory)
        return ChainRulesCore.rrule(eltwise, $op, x)
    end
end

function ChainRulesCore.rrule(::typeof(eltwise), f::F, from::Memory) where {F}
    to = eltwise(f, from)
    data = dst_for_bwd(f) ? to : from
    function eltwise_pullback(Δ)
        diff_from = eltwise_backward(f, Δ, data)
        return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), diff_from)
    end
    return to, eltwise_pullback
end

#####
##### Binary
#####

# Note: No need to define `rrule`s here since higher level definitions should take
# care of this automatically.
function binary(f::F, src_0::Memory, src_1::Memory) where {F}
    return binary(src_0, src_1, binary_forward(f))
end
function binary(src_0::Memory, src_1::Memory, kind)
    dst = similar(src_0)
    binary!(dst, src_0, src_1, kind)
    return kernel_exit_hook(dst)
end

function binary!(f::F, dst::Memory, src_0::Memory, src_1::Memory) where {F}
    return binary!(dst, src_0, src_1, binary_forward(f))
end
function binary!(dst::Memory, src_0::Memory, src_1::Memory, kind)
    opdesc = Ref{Lib.dnnl_binary_desc_t}()
    @apicall dnnl_binary_desc_init(opdesc, kind, src_0, src_1, dst)

    args = @dnnl_args dst src_0 src_1
    temp_primitive(opdesc, noattributes(), global_engine(), noforward()) do p, _
        execute!(p, args)
    end
    return dst
end

Base.:+(a::Memory, b::Memory) = binary(+, a, b)
Base.:-(a::Memory, b::Memory) = binary(-, a, b)
Base.broadcasted(::typeof(+), a::Memory, b::Memory) = binary(+, a, b)
Base.broadcasted(::typeof(-), a::Memory, b::Memory) = binary(-, a, b)
Base.broadcasted(::typeof(*), a::Memory, b::Memory) = binary(*, a, b)
Base.broadcasted(::typeof(/), a::Memory, b::Memory) = binary(/, a, b)
Base.broadcasted(::typeof(min), a::Memory, b::Memory) = binary(min, a, b)
Base.broadcasted(::typeof(max), a::Memory, b::Memory) = binary(max, a, b)

binary_forward(::typeof(+)) = Lib.dnnl_binary_add
binary_forward(::typeof(*)) = Lib.dnnl_binary_mul
binary_forward(::typeof(-)) = Lib.dnnl_binary_sub
binary_forward(::typeof(/)) = Lib.dnnl_binary_div
binary_forward(::typeof(max)) = Lib.dnnl_binary_max
binary_forward(::typeof(min)) = Lib.dnnl_binary_min

#####
##### Optimized Format Conversion.
#####

maybe_reorder(md::MemoryDesc, x::Memory) = (md == memorydesc(x)) ? x : reorder(md, x)
function maybe_reorder(descriptor::PrimitiveDescriptor, x::Memory, key)
    return maybe_reorder(query_md(descriptor, key), x)
end

reorder_if_any(::Ptr{MemoryDesc}, ::PrimitiveDescriptor, x::Memory, key) = x
function reorder_if_any(md::MemoryDesc, desc::PrimitiveDescriptor, x::Memory, key)
    return isany(md) ? maybe_reorder(desc, x, key) : x
end

#####
##### Logsoftmax
#####

function logsoftmax(src::Memory{T,N}, axis::Integer) where {T,N}
    opdesc = Ref{Lib.dnnl_logsoftmax_desc_t}()
    @apicall dnnl_logsoftmax_forward_desc_init(opdesc, Inference(), src, N - axis)

    return temp_primitive(opdesc, noattributes(), global_engine(), noforward()) do p, _
        dst = similar(src)
        execute!(p, @dnnl_args dst src)
        return dst
    end
end

function logsoftmax_backward(diff_dst::Memory{T,N}, dst::Memory{T,N}, axis::Integer) where {T,N}
    opdesc = Ref{Lib.dnnl_logsoftmax_desc_t}()
    @apicall dnnl_logsoftmax_backward_desc_init(opdesc, diff_dst, dst, N - axis)

    return temp_primitive(opdesc, noattributes(), global_engine(), noforward()) do p, _
        diff_src = similar(diff_dst)
        execute!(p, @dnnl_args diff_dst dst diff_src)
        return diff_src
    end
end

function ChainRulesCore.rrule(::typeof(logsoftmax), src::Memory, axis)
    dst = logsoftmax(src, axis)
    function logsoftmax_pullback(diff_dst)
        diff_src = logsoftmax_backward(diff_dst, dst, axis)
        return (
            ChainRulesCore.NoTangent(),
            diff_src,
            ChainRulesCore.NoTangent(),
        )
    end
    return dst, logsoftmax_pullback
end
