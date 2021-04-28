function innerproduct(
    _src::Memory,
    _weights::Memory,
    bias::Memory;
    kind = Lib.dnnl_forward_inference,
    attributes = noattributes(),
    # Allow for callers to maybe convert these descriptions to "any", allowing callers
    # to later convert their data to a better format.
    src_desc = memorydesc(_src),
    weights_desc = memorydesc(_weights),
    callback = (x...) -> nothing,
)
    dst_dims = (size(bias, 1), size(_src, 2))
    dst_desc = memorydesc(eltype(_src), dst_dims, dnnl_format_any())
    inner_product_desc = Ref{Lib.dnnl_inner_product_desc_t}()
    @apicall dnnl_inner_product_forward_desc_init(
        inner_product_desc, kind, _src, _weights, bias, dst_desc
    )

    return temp_primitive(
        inner_product_desc, attributes, global_engine(), noforward()
    ) do primitive, primitive_descriptor
        # Maybe the caller wants to know what the optimal weights format is.
        callback(primitive, primitive_descriptor)
        # Convert if needed.
        src = _src
        if isany(src_desc)
            src_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_src_md)
            if src_desc != src_desc_opt
                src = reorder(src_desc_opt, src)
            end
        end

        weights = _weights
        if isany(weights_desc)
            weights_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_weights_md)
            if weights_desc != weights_desc_opt
                weights = reorder(weights_desc_opt, weights)
            end
        end

        dst = similar(
            src,
            eltype(src),
            dst_dims,
            query_md(primitive_descriptor, Lib.dnnl_query_dst_md),
        )
        execute!(primitive, @dnnl_args src weights bias dst)
        return dst
    end
end

function innerproduct_backward_data(
    diff_src_dims::NTuple, weights::Memory, diff_dst::Memory
)
    diff_src_desc = memorydesc(eltype(diff_dst), diff_src_dims, dnnl_format_any())
    inner_product_desc = Ref{Lib.dnnl_inner_product_desc_t}()
    @apicall dnnl_inner_product_backward_data_desc_init(
        inner_product_desc, diff_src_desc, weights, diff_dst
    )

    return temp_primitive(
        inner_product_desc, noattributes(), global_engine(), noforward()
    ) do primitive, primitive_descriptor
        diff_src_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_diff_src_md)
        diff_src = similar(diff_dst, eltype(diff_dst), diff_src_dims, diff_src_desc_opt)
        execute!(primitive, @dnnl_args diff_src weights diff_dst)
        return diff_src
    end
end

function innerproduct_backward_weights(
    diff_weights_dims::NTuple, src::Memory, diff_dst::Memory
)
    diff_bias_dims = (diff_weights_dims[2],)
    diff_weights_desc = memorydesc(eltype(src), diff_weights_dims, dnnl_format_any())
    diff_bias_desc = memorydesc(eltype(src), diff_bias_dims, Lib.dnnl_a)
    inner_product_desc = Ref{Lib.dnnl_inner_product_desc_t}()

    @apicall dnnl_inner_product_backward_weights_desc_init(
        inner_product_desc, src, diff_weights_desc, diff_bias_desc, diff_dst
    )

    return temp_primitive(
        inner_product_desc, noattributes(), global_engine(), noforward()
    ) do primitive, primitive_descriptor
        diff_weights_desc_opt = query_md(
            primitive_descriptor, Lib.dnnl_query_diff_weights_md
        )
        diff_weights = similar(
            diff_dst, eltype(diff_dst), diff_weights_dims, diff_weights_desc_opt
        )
        diff_bias = similar(diff_dst, eltype(diff_dst), diff_bias_dims, diff_bias_desc)
        execute!(primitive, @dnnl_args diff_weights diff_bias src diff_dst)
        return (diff_weights, diff_bias)
    end
end

#####
##### Dense Layer
#####

mutable struct Dense{W<:Memory{Opaque},B<:Memory{Opaque},F}
    weights::W
    bias::B
    activation::F
    # Memory format optimization
    optimized_weights::Bool
    optimized_weights_desc::MemoryDesc
end

function Dense(weights, bias, activation)
    weights_memory = Memory(weights)
    return Dense(
        weights_memory, Memory(bias), activation, false, memorydesc(weights_memory)
    )
end

Dense(m::Flux.Dense) = Dense(OneDNN.Memory(transpose(m.weight)), OneDNN.Memory(m.bias), m.σ)

function (dense::Dense)(_src, fuse_activation = true)
    src = Memory(_src)
    attributes = Attributes()
    if fuse_activation
        postops = PostOps()
        eltwise!(postops, dense.activation)
        append!(attributes, postops)
    end

    if dense.optimized_weights == false
        callback = function weight_opt_callback(_, pd)
            dense.optimized_weights_desc = query_md(pd, Lib.dnnl_query_weights_md)
            return nothing
        end

        dst = innerproduct(
            src,
            dense.weights,
            dense.bias;
            attributes = attributes,
            weights_desc = toany(memorydesc(dense.weights)),
            callback = callback,
        )
        if dense.optimized_weights_desc != memorydesc(dense.weights)
            dense.weights = reorder(dense.weights_desc, dense.weights)
        end
        dense.optimized_weights = true
        return dst
    else
        dst = innerproduct(src, dense.weights, dense.bias; attributes = attributes)
        return dst
    end
end

function ChainRulesCore.rrule(dense::Dense, src::AbstractMatrix)
    # The result of `canfuse` is known at compile time, so Julia can optimize out the branch.
    return canfuse(dense.activation) ? rrule_fused(dense, src) : rrule_unfused(dense, src)
end

function rrule_fused(dense, _src)
    src = Memory(_src)
    src_size = size(src)
    dst = dense(src)
    pullback = function dense_fused_pullback(_diff_dst)
        # Maybe convert argument
        diff_dst = Memory(_diff_dst)
        # Reverse activation function.
        diff_dst_pre = eltwise_backward(dense.activation, diff_dst, dst)

        # Backprop innerproduct kernel
        diff_src = innerproduct_backward_data(src_size, dense.weights, diff_dst_pre)
        (diff_weights, diff_bias) = innerproduct_backward_weights(size(dense.weights), src, diff_dst_pre)
        return ((weights = diff_weights, bias = diff_bias), diff_src)
    end
    return dst, pullback
end

function rrule_unfused(dense, _src)
    src = Memory(_src)
    dst_pre = dense(src, false)
    dst = eltwise(dense.activation, dst_pre)
    src_size = size(src)
    pullback = function dense_pullback(_diff_dst)
        diff_dst = Memory(_diff_dst)
        diff_dst_pre = eltwise_backward(dense.activation, diff_dst, dst_pre)
        diff_src = innerproduct_backward_data(src_size, dense.weights, diff_dst_pre)
        (diff_weights, diff_bias) = innerproduct_backward_weights(size(dense.weights), src, diff_dst_pre)

        return ((weights = diff_weights, bias = diff_bias), diff_src)
    end
    return dst, pullback
end
