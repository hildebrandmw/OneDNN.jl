struct Eltwise{F}
    primitive::Primitive
    output_description::MemoryDesc
    # Keep track of some construction parameters to allow for introspection.
    kind::Lib.dnnl_alg_kind_t
    alpha::Float32
    beta::Float32
end

Eltwise(f::F, src::MemoryOrDesc) where {F} = Eltwise(src, forward_expand(f)...)
# Note: formatter messes up this call.
#! format: off
function Eltwise(
    src::MemoryOrDesc,
    kind::Lib.dnnl_alg_kind_t,
    alpha = one(Float32),
    beta = zero(Float32)
)
#! format: on
    op_descriptor = Ref{Lib.dnnl_eltwise_desc_t}()

    # Note: no difference between training and inference for eltwise ops.
    @apicall dnnl_eltwise_forward_desc_init(
        op_descriptor, Lib.dnnl_forward_inference, kind, src, alpha, beta
    )

    primitive_descriptor = PrimitiveDescriptor(
        op_descriptor, noattributes(), global_engine(), noforward()
    )

    primitive = Primitive(primitive_descriptor)
    return Eltwise{layout(src)}(primitive, memorydesc(src), kind, alpha, beta)
end

function (op::Eltwise{F})(src::Memory{F}) where {F}
    dst = similar(src, eltype(src), size(src), op.output_description, Val(F))
    return op(dst, src)
end

function (op::Eltwise{F})(dst::Memory{F}, src::Memory{F}) where {F}
    args = @dnnl_args dst src
    execute!(op.primitive, args)
    return dst
end

#####
##### Backward Eltwise
#####

# OneDNN supports two kinds of element-wise backprop modes.
# One where the source (src) and destination difference (dst_diff) are used, and
# one where only the destination difference (dst_diff) is used.
#
# The type parameter `T` represents the layout for all tensors involved
# (OneDNN recommends this for best performance)
#
# The type parameter `B` is `true` if the destination tensor is used. Otherwise,
# the source densor is used.
struct EltwiseBackward{T,B}
    primitive::Primitive
    output_description::MemoryDesc
    # Keep track of some construction parameters to allow for introspection.
    kind::Lib.dnnl_alg_kind_t
    alpha::Float32
    beta::Float32
    dst_for_bwd::Bool
end

function EltwiseBackward(f::F, diff_data::MemoryOrDesc, data::MemoryOrDesc) where {F}
    return EltwiseBackward(diff_data, data, backward_expand(f)...)
end

# Note: formatter messes up this call.
function EltwiseBackward(
    diff_data::MemoryOrDesc,
    data::MemoryOrDesc,
    kind::Lib.dnnl_alg_kind_t,
    alpha = one(Float32),
    beta = zero(Float32),
    use_dst = false,
)
    op_descriptor = Ref{Lib.dnnl_eltwise_desc_t}()

    # Note: no difference between training and inference for eltwise ops.
    @apicall dnnl_eltwise_backward_desc_init(
        op_descriptor, kind, diff_data, data, alpha, beta
    )

    primitive_descriptor = PrimitiveDescriptor(
        op_descriptor, noattributes(), global_engine(), noforward()
    )

    primitive = Primitive(primitive_descriptor)
    T = layout(diff_data)
    return EltwiseBackward{T,use_dst}(
        primitive, memorydesc(diff_data), kind, alpha, beta, use_dst
    )
end

function (op::EltwiseBackward{T})(diff_dst::Memory{T}, src_or_dst::Memory{T}) where {T}
    diff_src = similar(
        diff_dst, eltype(diff_dst), size(diff_dst), op.output_description, Val(T)
    )
    return op(diff_src, diff_dst, src_or_dst)
end

# Source for backprop
function (op::EltwiseBackward{DD,false})(
    diff_src::Memory{DD}, diff_dst::Memory{DD}, src::Memory{S}
) where {DD,S}
    args = @dnnl_args diff_src diff_dst src
    execute!(op.primitive, args)
    return diff_src
end

# Destination for backprop
function (op::EltwiseBackward{DD,true})(
    diff_src::Memory{DD}, diff_dst::Memory{DD}, dst::Memory{S}
) where {DD,S}
    args = @dnnl_args diff_src diff_dst dst
    execute!(op.primitive, args)
    return diff_src
end

#####
##### Convenience Wrappers.
#####

struct Linear
    α::Float32
    β::Float32
end

forward_expand(::typeof(abs)) = (Lib.dnnl_eltwise_abs, zero(Float32), zero(Float32))
forward_expand(x::Linear) = (Lib.dnnl_eltwise_linear, x.α, x.β)
function forward_expand(::typeof(Flux.sigmoid))
    return (Lib.dnnl_eltwise_logistic, zero(Float32), zero(Float32))
end
forward_expand(::typeof(sqrt)) = (Lib.dnnl_eltwise_sqrt, zero(Float32), zero(Float32))
forward_expand(::typeof(Flux.relu)) = (Lib.dnnl_eltwise_relu, zero(Float32), zero(Float32))

function backward_expand(::typeof(abs))
    return (Lib.dnnl_eltwise_abs, zero(Float32), zero(Float32), false)
end
backward_expand(x::Linear) = (Lib.dnnl_eltwise_linear, x.α, x.β, false)
function backward_expand(::typeof(Flux.sigmoid))
    return (Lib.dnnl_eltwise_logistic_use_dst_for_bwd, zero(Float32), zero(Float32), true)
end
function backward_expand(::typeof(sqrt))
    return (Lib.dnnl_eltwise_sqrt, zero(Float32), zero(Float32), false)
end
function backward_expand(::typeof(Flux.relu))
    return (Lib.dnnl_eltwise_relu_use_dst_for_bwd, zero(Float32), zero(Float32), true)
end

# Can this elementwise op be fused into a previous operation?
# Only if the backward op can operate on the destination tensor only.
dst_for_bwd(f::F) where {F} = last(backward_expand(f))
canfuse(f::F) where {F} = dst_for_bwd(f)
canfuse(::typeof(identity)) = true
