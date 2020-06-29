inner_product_forward(x...; kw...) = inner_product_forward(memory.(x)...; kw...)
function inner_product_forward(src::Memory, weights::Memory, bias::Memory; kw...)
    dst_dims = (size(weights, 2), size(src, ndims(src)))
    dst_eltype = out_eltype(eltype(src), eltype(weights))

    dst_format = dnnl_format(Val{2}())
    dst_desc = memorydesc(dst_eltype, dst_dims, dst_format)
    dst = similar(src, dst_eltype, dst_dims, dst_desc)

    inner_product_forward!(dst, src, weights, bias; kw...)
    return dst
end

inner_product_forward!(args...; kw...) = inner_product_forward!(memory.(args)...; kw...)
function inner_product_forward!(
        dst::Memory,
        src::Memory,
        weights::Memory,
        bias::Memory;
        attributes = Ptr{Nothing}(),
        kind = Lib.dnnl_forward_inference,
    )
    # Op Descriptor
    # construct a temporary memory descriptor for `src` which will allow us to optimize
    # that format if needed.
    temp_src_desc = memorydesc(eltype(src), size(src), Lib.dnnl_format_tag_any)

    opd = Ref{Lib.dnnl_inner_product_desc_t}()
    @apicall Lib.dnnl_inner_product_forward_desc_init(
        opd,
        kind,
        Ref(temp_src_desc),
        memorydesc_ptr(weights),
        memorydesc_ptr(bias),
        memorydesc_ptr(dst),
    )

    # primitive descriptor
    pd = primitive_descriptor(
        opd,
        attributes,
        global_engine(),
        Ptr{Nothing}(),
    )

    # Check if we need to reorder the `src` argument.
    optimal_src_desc = Lib.dnnl_primitive_desc_query_md(pd, Lib.dnnl_query_src_md, 0)
    if optimal_src_desc != memorydesc(src)
        src = reorder(src, optimal_src_desc)
    end

    # primitive
    args = [
        arg(Lib.DNNL_ARG_SRC, src),
        arg(Lib.DNNL_ARG_WEIGHTS, weights),
        arg(Lib.DNNL_ARG_BIAS, bias),
        arg(Lib.DNNL_ARG_DST, dst),
    ]

    p = primitive(pd)
    execute!(p, args)

    # cleanup
    destroy(p, pd)
    return nothing
end

#####
##### Backward Primitives
#####

function inner_product_backward_data!(diff_src::Memory, weights::Memory, diff_dst::Memory)
    # op desc
    opd = Ref{Lib.dnnl_inner_product_desc_t}()
    @apicall Lib.dnnl_inner_product_backward_data_desc_init(
        opd,
        memorydesc_ptr(diff_src),
        memorydesc_ptr(weights),
        memorydesc_ptr(diff_dst),
    )

    # primitive descriptor
    pd = primitive_descriptor(
        opd,
        Ptr{Nothing}(),
        global_engine(),
        Ptr{Nothing}(),
    )

    # primitive
    args = [
        arg(Lib.DNNL_ARG_DIFF_SRC, diff_src),
        arg(Lib.DNNL_ARG_WEIGHTS, weights),
        arg(Lib.DNNL_ARG_DIFF_DST, diff_dst),
    ]

    p = primitive(pd)
    execute!(p, args)

    # cleanup
    destroy(p, pd)
    return nothing
end

function inner_product_backward_weights!(
        src::Memory,
        diff_weights::Memory,
        diff_bias::Memory,
        diff_dst::Memory
    )

    # op desc
    opd = Ref{Lib.dnnl_inner_product_desc_t}()
    @apicall Lib.dnnl_inner_product_backward_weights_desc_init(
        opd,
        memorydesc_ptr(src),
        memorydesc_ptr(diff_weights),
        memorydesc_ptr(diff_bias),
        memorydesc_ptr(diff_dst),
    )

    # primitive descriptor
    pd = primitive_descriptor(
        opd,
        Ptr{Nothing}(),
        global_engine(),
        Ptr{Nothing}(),
    )

    # primitive
    args = [
        arg(Lib.DNNL_ARG_SRC, src),
        arg(Lib.DNNL_ARG_DIFF_WEIGHTS, diff_weights),
        arg(Lib.DNNL_ARG_DIFF_BIAS, diff_bias),
        arg(Lib.DNNL_ARG_DIFF_DST, diff_dst)
    ]

    p = primitive(pd)
    execute!(p, args)

    # cleanup
    destroy(p, pd)
    return nothing

end

#####
##### Dense
#####

# Like the Flux layer, but dispatches to OneDNN implementations.
#
# In particular, we need to make sure that we transpose the `W` matrix from the standard
# Flux model to match the semantics between Julia's implementation and what OneDNN is
# expecting.
struct Dense{S <: Memory,T <: Memory,F}
    W::S
    b::T
    σ::F
end

Dense(d::Flux.Dense) = Dense(memory(collect(transpose(d.W))), memory(d.b), d.σ)

function Dense(
        in::Integer,
        out::Integer,
        σ = identity;
        initW = Flux.glorot_normal,
        initb = (x...) -> zeros(Float32, x)
    )

    # Transpose W to begin with
    W = memory(initW(in, out))
    b = memory(initb(out))
    return Dense(W, b, σ)
end

function (D::Dense)(x; attributes = nothing, kind = Lib.dnnl_forward_inference)
    if attributes === nothing
        # If no external attributes are passed, construct a post-op for the activation
        # function.
        postops = PostOps()
        appendeltwise!(postops, D.σ)
        attributes = Attributes()
        add!(attributes, postops)
    else
       attributes = Ptr{Nothing}()
    end

    return inner_product_forward(x, D.W, D.b; attributes = attributes, kind = kind)
end

function ZygoteRules._pullback(cx::ZygoteRules.AContext, D::Dense, x)
    mx = memory(x)

    # can we compute the pullback of the activation function from the destination?
    # If so, fuse it as a post op.
    # Otherwise, we will need to pull it out.
    if back_from_dst(D.σ)
        # Activation will fuse like normal
        out = D(mx; kind = Lib.dnnl_forward_training)
    else
        error("Can't yet support activations that can't be fused")
    end

    return out, Δ -> begin
        # Construct all the memory objects we will need.
        diff_dst = memory(Δ)
        diff_weights = similar(D.W)
        diff_bias = similar(D.b)
        diff_src = similar(mx)

        # backprop the activation function.
        backprop_eltwise(D.σ, diff_dst)

        # Do the backprop operations
        inner_product_backward_data!(diff_src, D.W, diff_dst)
        inner_product_backward_weights!(mx, diff_weights, diff_bias, diff_dst)

        return (
            (W = diff_weights, b = diff_bias, σ = nothing),
            diff_src,
        )
    end
end

# M = Chain(
#     OneDNN.Dense(OneDNN.memory(collect(transpose(m[1].W))), OneDNN.memory(m[1].b), identity),
#     OneDNN.Dense(OneDNN.memory(collect(transpose(m[2].W))), OneDNN.memory(m[2].b), identity),
#     OneDNN.Dense(OneDNN.memory(collect(transpose(m[3].W))), OneDNN.memory(m[3].b), identity),
#     OneDNN.Dense(OneDNN.memory(collect(transpose(m[4].W))), OneDNN.memory(m[4].b), identity),
# )


# For training, we need to keep track of primitive descriptors
# Zygote.@adjoint function (D::Dense)(x)
#
# end

