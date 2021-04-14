struct InnerProduct{O,I,T<:MaybeReorder,W<:Memory,B<:Memory}
    primitive::Primitive
    output_descriptor::MemoryDesc
    outsize::Tuple{Int,Int}
    # Keep track of if we need to reorder the source tensor.
    reorder::T
    # Track the weights and biases.
    weights::W
    bias::B
end

function InnerProduct(
    f::F, src::MemoryOrDesc, weight::Memory, bias::Memory; kind = Lib.dnnl_forward_inference
) where {F}
    # Create propper attributes for the activation function `f`.
    attributes = Attributes()
    postops = PostOps()
    eltwise!(postops, f)
    append!(attributes, postops)
    return InnerProduct(src, weight, bias; kind, attributes)
end

# This primitive is going to act as a holder for the `weight` and `bias`.
# In other words, it's not going to be stateless like the other primitives.
#
# If this ends up being a problem, we can change it in the future.
function InnerProduct(
    src::MemoryOrDesc,
    weight::Memory,
    bias::Memory;
    kind = Lib.dnnl_forward_inference,
    attributes = noattributes(),
)
    # Calculate the dimensions of the output.
    dst_dims = (first(size(src)), size(bias, 1))

    # Create similar memory formats for `src`, `weight`, and `bias`, but with the
    # format tag set to "any" to let OneDNN choose the best memory format.
    src_desc = toany(memorydesc(src))
    weight_desc = toany(memorydesc(weight))
    dst_desc = memorydesc(eltype(src), dst_dims, dnnl_format_any())

    # Now, create the primitive like normal.
    op_descriptor = Ref{Lib.dnnl_inner_product_desc_t}()
    @apicall Lib.dnnl_inner_product_forward_desc_init(
        op_descriptor, kind, src_desc, weight_desc, bias, dst_desc
    )

    primitive_descriptor = PrimitiveDescriptor(
        op_descriptor, attributes, global_engine(), noforward()
    )

    # Get the optimal memory layouts.
    src_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_src_md)
    weight_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_weights_md)
    dst_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_dst_md)

    if weight_desc_opt != memorydesc(weight)
        op = Reorder(weight_desc_opt, memorydesc(weight))
        weight = op(weight)
    end

    if src_desc_opt != memorydesc(src)
        reorder = Reorder(src_desc_opt, memorydesc(src))
    else
        reorder = nothing
    end

    primitive = Primitive(primitive_descriptor)

    # Prepare type parameters
    O = layout(dst_desc_opt)
    I = layout(src_desc_opt)
    T = typeof(reorder)
    W = typeof(weight)
    B = typeof(bias)
    return InnerProduct{O,I,T,W,B}(primitive, dst_desc_opt, dst_dims, reorder, weight, bias)
end

function (op::InnerProduct{O,I})(src::Memory{I_}) where {O,I,I_}
    dst = similar(src, eltype(src), op.outsize, op.output_descriptor, Val(O))
    if I != I_
        src = op.reorder(src)
    end

    return op(dst, src)
end

function (op::InnerProduct{O,I})(dst::Memory{O}, src::Memory) where {O,I}
    args = @dnnl_args src op.weights op.bias dst
    execute!(op.primitive, args)
    return dst
end

#####
##### Backward Primitives
#####

# Backward Data
struct InnerProductBackwardData{DS,W,DD,WR<:MaybeReorder,DDR<:MaybeReorder}
    primitive::Primitive
    output_descriptor::MemoryDesc
    outsize::Tuple{Int,Int}
    # Possible reorderings
    weights_reorder::WR
    diff_dsc_reorder::DDR
end

function InnerProductBackwardData(
    diff_src_size::NTuple, weight::MemoryOrDesc, diff_dst::MemoryOrDesc
)
    diff_src_desc = memorydesc(eltype(diff_dst), diff_src_size, dnnl_format_any())
    weight_desc = toany(memorydesc(weight))
    diff_dst_desc = toany(memorydesc(diff_dst))

    op_descriptor = Ref{Lib.dnnl_inner_product_desc_t}()
    @apicall dnnl_inner_product_backward_data_desc_init(
        op_descriptor, diff_src_desc, weight_desc, diff_dst_desc
    )

    primitive_descriptor = PrimitiveDescriptor(
        op_descriptor, noattributes(), global_engine(), noforward()
    )

    # Check data formats
    diff_src_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_diff_src_md)
    weights_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_weights_md)
    diff_dst_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_diff_dst_md)

    if memorydesc(diff_dst) != diff_dst_desc_opt
        diff_dst_reorder = Reorder(memorydesc(diff_dst), diff_dst_desc_opt)
    else
        diff_dst_reorder = nothing
    end

    if memorydesc(weight) != weights_desc_opt
        weights_reorder = Reorder(weights_desc_opt, memorydesc(weight))
    else
        weights_reorder = nothing
    end

    primitive = Primitive(primitive_descriptor)

    # Prepare type parameters
    DS = layout(diff_src_desc_opt)
    W = layout(weights_desc_opt)
    DO = layout(diff_dst_desc_opt)
    WR = typeof(weights_reorder)
    DDR = typeof(diff_dst_reorder)
    return InnerProductBackwardData{DS,W,DO,WR,DDR}(
        primitive, diff_src_desc_opt, diff_src_size, weights_reorder, diff_dst_reorder
    )
end

function (op::InnerProductBackwardData{DS})(weight::Memory, diff_dst::Memory) where {DS}
    diff_src = similar(
        diff_dst, eltype(diff_dst), op.outsize, op.output_descriptor, Val(DS)
    )
    return op(diff_src, weight, diff_dst)
end

# At least one argument needs reordering.
function (op::InnerProductBackwardData{DS,W,DD})(
    diff_src::Memory{DS}, weight::Memory{W_}, diff_dst::Memory{DD_}
) where {DS,W,DD,W_,DD_}
    if W != W_
        weight = op.weights_reorder(weight)
    end
    if DD != DD_
        diff_dst = op.diff_dst_reorder(diff_dst)
    end
    return op(diff_src, weight, diff_dst)
end

function (op::InnerProductBackwardData{DS,W,DD})(
    diff_src::Memory{DS}, weights::Memory{W}, diff_dst::Memory{DD}
) where {DS,W,DD}
    args = @dnnl_args diff_src weights diff_dst
    execute!(op.primitive, args)
    return diff_src
end

# Backward Weights
struct InnerProductBackwardWeight{DW,S,DD,SR<:MaybeReorder,DDR<:MaybeReorder}
    primitive::Primitive
    weight_descriptor::MemoryDesc
    weight_size::Tuple{Int,Int}
    bias_descriptor::MemoryDesc
    bias_size::Int

    # Possible reorderings
    src_reorder::SR
    diff_dsc_reorder::DDR
end

function InnerProductBackwardWeight(
    diff_weights_size::NTuple,
    diff_bias_size::Integer,
    src::MemoryOrDesc,
    diff_dst::MemoryOrDesc,
)
    diff_weights_desc = memorydesc(eltype(src), diff_weights_size, dnnl_format_any())
    diff_bias_desc = memorydesc(eltype(src), (diff_bias_size,), Lib.dnnl_a)
    src_desc = toany(memorydesc(src))
    diff_dst_desc = toany(memorydesc(diff_dst))

    op_descriptor = Ref{Lib.dnnl_inner_product_desc_t}()
    @apicall dnnl_inner_product_backward_weights_desc_init(
        op_descriptor, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc
    )

    primitive_descriptor = PrimitiveDescriptor(
        op_descriptor, noattributes(), global_engine(), noforward()
    )

    # Check data formats
    src_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_src_md)
    diff_weight_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_diff_weights_md)
    diff_dst_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_diff_dst_md)

    if memorydesc(diff_dst) != diff_dst_desc_opt
        diff_dst_reorder = Reorder(memorydesc(diff_dst), diff_dst_desc_opt)
    else
        diff_dst_reorder = nothing
    end

    if memorydesc(src) != src_desc_opt
        src_reorder = Reorder(memorydesc(src), src_desc_opt)
    else
        src_reorder = nothing
    end

    primitive = Primitive(primitive_descriptor)

    # Prepare type parameters
    DW = layout(diff_weight_desc_opt)
    S = layout(src_desc_opt)
    DD = layout(diff_dst_desc_opt)
    SR = typeof(src_reorder)
    DDR = typeof(diff_dst_reorder)
    return InnerProductBackwardWeight{DW,S,DD,SR,DDR}(
        primitive,
        diff_weight_desc_opt,
        diff_weights_size,
        diff_bias_desc,
        diff_bias_size,
        src_reorder,
        diff_dst_reorder,
    )
end

function (op::InnerProductBackwardWeight{DW})(src::Memory, diff_dst::Memory) where {DW}
    diff_weights = similar(src, eltype(src), op.weight_size, op.weight_descriptor, Val(DW))

    diff_bias = similar(src, eltype(src), (op.bias_size,), op.bias_descriptor, Val((1,)))

    return op(diff_weights, diff_bias, src, diff_dst)
end

# At least one argument needs reordering.
function (op::InnerProductBackwardWeight{DW,S,DD})(
    diff_weights::Memory{DW},
    diff_bias::Memory{(1,)},
    src::Memory{S_},
    diff_dst::Memory{DD_},
) where {DW,S,DD,S_,DD_}
    if S != S_
        src = op.src_reorder(src)
    end
    if DD != DD_
        diff_dst = op.diff_dst_reorder(diff_dst)
    end
    return op(diff_weights, diff_bias, src, diff_dst)
end

function (op::InnerProductBackwardWeight{DW,S,DD})(
    diff_weights::Memory{DW}, diff_bias::Memory{(1,)}, src::Memory{S}, diff_dst::Memory{DD}
) where {DW,S,DD}
    args = @dnnl_args src diff_weights diff_bias diff_dst
    execute!(op.primitive, args)
    return (diff_weights, diff_bias)
end

# function inner_product_forward(src, weights, bias; kw...)
#     return inner_product_forward(memory(src), memory(weights), memory(bias); kw...)
# end
#
# function inner_product_forward(src::Memory, weights::Memory, bias::Memory; kw...)
#     dst_dims = (size(weights, 2), size(src, ndims(src)))
#     dst_eltype = out_eltype(eltype(src), eltype(weights))
#
#     dst_format = dnnl_format(Val{2}())
#     dst_desc = memorydesc(dst_eltype, dst_dims, dst_format)
#     dst = similar(src, dst_eltype, dst_dims, dst_desc)
#
#     inner_product_forward!(dst, src, weights, bias; kw...)
#     return dst
# end
#
# function inner_product_forward!(dst, src, weights, bias; kw...)
#     return inner_product_forward!(
#         memory(dst), memory(src), memory(weights), memory(bias); kw...
#     )
# end
# function inner_product_forward!(
#     dst::Memory,
#     src::Memory,
#     weights::Memory,
#     bias::Memory;
#     attributes = Ptr{Nothing}(),
#     kind = Lib.dnnl_forward_inference,
# )
#
#     # Op Descriptor
#     # construct a temporary memory descriptor for `src` which will allow us to optimize
#     # that format if needed.
#     temp_src_desc = memorydesc(eltype(src), size(src), Lib.dnnl_format_tag_any)
#
#     opd = Ref{Lib.dnnl_inner_product_desc_t}()
#     @apicall Lib.dnnl_inner_product_forward_desc_init(
#         opd,
#         kind,
#         Ref(temp_src_desc),
#         memorydesc_ptr(weights),
#         memorydesc_ptr(bias),
#         memorydesc_ptr(dst),
#     )
#
#     # primitive descriptor
#     pd = primitive_descriptor(opd, attributes, global_engine(), Ptr{Nothing}())
#
#     # Check if we need to reorder the `src` argument.
#     optimal_src_desc = Lib.dnnl_primitive_desc_query_md(pd, Lib.dnnl_query_src_md, 0)
#     if optimal_src_desc != memorydesc(src)
#         src = reorder(src, optimal_src_desc)
#     end
#
#     # primitive
#     args = [
#         arg(Lib.DNNL_ARG_SRC, src),
#         arg(Lib.DNNL_ARG_WEIGHTS, weights),
#         arg(Lib.DNNL_ARG_BIAS, bias),
#         arg(Lib.DNNL_ARG_DST, dst),
#     ]
#
#     p = primitive(pd)
#     execute!(p, args)
#
#     # cleanup
#     destroy(p, pd)
#     return nothing
# end
#
# #####
# ##### Backward Primitives
# #####
#
# function inner_product_backward_data!(diff_src::Memory, weights::Memory, diff_dst::Memory)
#     # op desc
#     opd = Ref{Lib.dnnl_inner_product_desc_t}()
#     @apicall Lib.dnnl_inner_product_backward_data_desc_init(
#         opd, memorydesc_ptr(diff_src), memorydesc_ptr(weights), memorydesc_ptr(diff_dst)
#     )
#
#     # primitive descriptor
#     pd = primitive_descriptor(opd, Ptr{Nothing}(), global_engine(), Ptr{Nothing}())
#
#     # primitive
#     args = [
#         arg(Lib.DNNL_ARG_DIFF_SRC, diff_src),
#         arg(Lib.DNNL_ARG_WEIGHTS, weights),
#         arg(Lib.DNNL_ARG_DIFF_DST, diff_dst),
#     ]
#
#     p = primitive(pd)
#     execute!(p, args)
#
#     # cleanup
#     destroy(p, pd)
#     return nothing
# end
#
# function inner_product_backward_weights!(
#     src::Memory, diff_weights::Memory, diff_bias::Memory, diff_dst::Memory
# )
#
#     # op desc
#     opd = Ref{Lib.dnnl_inner_product_desc_t}()
#     @apicall Lib.dnnl_inner_product_backward_weights_desc_init(
#         opd,
#         memorydesc_ptr(src),
#         memorydesc_ptr(diff_weights),
#         memorydesc_ptr(diff_bias),
#         memorydesc_ptr(diff_dst),
#     )
#
#     # primitive descriptor
#     pd = primitive_descriptor(opd, Ptr{Nothing}(), global_engine(), Ptr{Nothing}())
#
#     # primitive
#     args = [
#         arg(Lib.DNNL_ARG_SRC, src),
#         arg(Lib.DNNL_ARG_DIFF_WEIGHTS, diff_weights),
#         arg(Lib.DNNL_ARG_DIFF_BIAS, diff_bias),
#         arg(Lib.DNNL_ARG_DIFF_DST, diff_dst),
#     ]
#
#     p = primitive(pd)
#     execute!(p, args)
#
#     # cleanup
#     destroy(p, pd)
#     return nothing
# end
#
# #####
# ##### Dense
# #####
#
# # Like the Flux layer, but dispatches to OneDNN implementations.
# #
# # In particular, we need to make sure that we transpose the `W` matrix from the standard
# # Flux model to match the semantics between Julia's implementation and what OneDNN is
# # expecting.
# struct Dense{S<:Memory,T<:Memory,F}
#     W::S
#     b::T
#     σ::F
# end
#
# Dense(d::Flux.Dense) = Dense(memory(collect(transpose(d.W))), memory(d.b), d.σ)
#
# Flux.@functor Dense (W, b)
#
# function Dense(
#     in::Integer,
#     out::Integer,
#     σ = identity;
#     initW = Flux.glorot_normal,
#     initb = (x...) -> zeros(Float32, x),
# )
#
#     # Transpose W to begin with
#     W = memory(initW(in, out))
#     b = memory(initb(out))
#     return Dense(W, b, σ)
# end
#
# function (D::Dense)(x; attributes = nothing, kind = Lib.dnnl_forward_inference)
#     if attributes === nothing
#         # If no external attributes are passed, construct a post-op for the activation
#         # function.
#         postops = PostOps()
#
#         appendeltwise!(postops, D.σ)
#         attributes = Attributes()
#         add!(attributes, postops)
#     else
#         attributes = Ptr{Nothing}()
#     end
#
#     return inner_product_forward(x, D.W, D.b; attributes = attributes, kind = kind)
# end
#
# function Zygote._pullback(cx::Zygote.AContext, D::Dense, x)
#     mx = memory(x)
#
#     # can we compute the pullback of the activation function from the destination?
#     # If so, fuse it as a post op.
#     # Otherwise, we will need to pull it out.
#     if back_from_dst(D.σ)
#         # Activation will fuse like normal
#         out = D(mx; kind = Lib.dnnl_forward_training)
#     else
#         error("Can't yet support activations that can't be fused")
#     end
#
#     return out, function dense_pullback(Δ)
#         # Construct all the memory objects we will need.
#         diff_dst = memory(Δ)
#         diff_weights = similar(D.W)
#         diff_bias = similar(D.b)
#         diff_src = similar(mx)
#
#         # backprop the activation function.
#         diff_dst = backprop_eltwise_dst(D.σ, out, diff_dst)
#
#         # Do the backprop operations
#         inner_product_backward_data!(diff_src, D.W, diff_dst)
#         inner_product_backward_weights!(mx, diff_weights, diff_bias, diff_dst)
#
#         # Accumulate these parameters
#         Zygote.accum_param(cx, D.W, diff_weights)
#         Zygote.accum_param(cx, D.b, diff_bias)
#
#         return ((W = diff_weights, b = diff_bias, σ = nothing), diff_src)
#     end
# end
#
# # M = Chain(
# #     OneDNN.Dense(OneDNN.memory(collect(transpose(m[1].W))), OneDNN.memory(m[1].b), identity),
# #     OneDNN.Dense(OneDNN.memory(collect(transpose(m[2].W))), OneDNN.memory(m[2].b), identity),
# #     OneDNN.Dense(OneDNN.memory(collect(transpose(m[3].W))), OneDNN.memory(m[3].b), identity),
# #     OneDNN.Dense(OneDNN.memory(collect(transpose(m[4].W))), OneDNN.memory(m[4].b), identity),
# # )
#
# # For training, we need to keep track of primitive descriptors
# # Zygote.@adjoint function (D::Dense)(x)
# #
# # end
