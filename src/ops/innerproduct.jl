_maybepromote(::Type{T}) where {T} = T
_maybepromote(::Type{T}) where {T<:AbstractFloat} = Float32

function innerproduct(
    _src::Memory{T},
    _weights::Memory{T},
    bias::Memory{T};
    kind = Inference(),
    attributes::Attributes = noattributes(),
    # Allow for callers to maybe convert these descriptions to "any", allowing callers
    # to later convert their data to a better format.
    src_md = memorydesc_ptr(_src),
    weights_md = memorydesc_ptr(_weights),
    callback = (x...) -> nothing,
) where {T}
    #@show (size(_src), size(_weights), size(bias))
    dst_dims = (size(bias, 1), size(_src, 2))
    dst_md = memorydesc(T, dst_dims, dnnl_format_any())
    opdesc = Ref{Lib.dnnl_inner_product_desc_t}()

    @apicall dnnl_inner_product_forward_desc_init(
        opdesc, kind, src_md, weights_md, bias, dst_md
    )

    return temp_primitive(opdesc, attributes, global_engine(), noforward()) do p, pd
        # Maybe the caller wants to know what the optimal weights format is.
        callback(p, pd)

        # Convert if needed.
        src = reorder_if_any(src_md, pd, _src, @query(src))
        weights = reorder_if_any(weights_md, pd, _weights, @query(weights))

        # Materialize destinations
        dst_md = query_md(pd, @query(dst))
        dst = similar(src, T, dst_dims, dst_md)

        # Execute primitive
        execute!(p, @dnnl_args src weights bias dst; descriptor = pd)
        return dst
    end
end

function innerproduct_backward_data(
    diff_src_dims::NTuple, _weights::Memory, _diff_dst::Memory{T}
) where {T}
    diff_src_md = memorydesc(T, diff_src_dims, dnnl_format_any())
    opdesc = Ref{Lib.dnnl_inner_product_desc_t}()
    @apicall dnnl_inner_product_backward_data_desc_init(
        opdesc, diff_src_md, toany(_weights), toany(_diff_dst)
    )

    return temp_primitive(opdesc, noattributes(), global_engine(), noforward()) do p, pd
        weights = maybe_reorder(pd, _weights, @query(weights))
        diff_dst = maybe_reorder(pd, _diff_dst, @query(diff_dst))

        diff_src_md = query_md(pd, @query(diff_src))
        diff_src = similar(diff_dst, T, diff_src_dims, diff_src_md)

        execute!(p, @dnnl_args diff_src weights diff_dst)
        return diff_src
    end
end

function innerproduct_backward_weights(
    diff_weights_dims::NTuple, _src::Memory{T}, _diff_dst::Memory
) where {T}
    diff_bias_dims = (diff_weights_dims[2],)
    diff_weights_md = memorydesc(_maybepromote(T), diff_weights_dims, dnnl_format_any())
    diff_bias_md = memorydesc(_maybepromote(T), diff_bias_dims)
    opdesc = Ref{Lib.dnnl_inner_product_desc_t}()
    @apicall dnnl_inner_product_backward_weights_desc_init(
        opdesc, toany(_src), diff_weights_md, diff_bias_md, toany(_diff_dst)
    )

    return temp_primitive(opdesc, noattributes(), global_engine(), noforward()) do p, pd
        #@show (diff_weights_dims, size(_src), size(_diff_dst))
        # Maybe convert arguments
        src = maybe_reorder(pd, _src, @query(src))
        diff_dst = maybe_reorder(pd, _diff_dst, @query(diff_dst))

        # Materialize destinations
        diff_weights_md = query_md(pd, @query(diff_weights))
        diff_weights = similar(
            diff_dst, _maybepromote(T), diff_weights_dims, diff_weights_md
        )
        diff_bias = similar(diff_dst, _maybepromote(T), diff_bias_dims, diff_bias_md)

        # Execute and return
        execute!(p, @dnnl_args diff_weights diff_bias src diff_dst)
        return (diff_weights, diff_bias)
    end
end

#####
##### Dense Layer
#####

mutable struct Dense{W<:Memory,B<:Memory,F}
    weights::W
    bias::B
    activation::F
    attributes::Attributes
    # Memory format optimization
    optimized_weights::Bool
end

function Dense(
    weights::AbstractMatrix{T},
    bias::AbstractVector{T},
    activation::F = identity;
    allocator = stdallocator,
) where {T,F}
    attributes = Attributes()
    postops = PostOps()
    eltwise!(postops, activation)
    append!(attributes, postops)

    return Dense(
        make_memory(weights, allocator),
        make_memory(bias, allocator),
        activation,
        attributes,
        false,
    )
end
function Dense(m::Flux.Dense; allocator = stdallocator)
    return Dense(transpose(m.weight), m.bias, m.σ; allocator)
end

Flux.@functor Dense (weights, bias)
function ZygoteRules._pullback(
    cx::AContext, ::typeof(literal_getproperty), x::Dense, ::Val{f}
) where {f}
    return pullback_for_default_literal_getproperty(cx, x, Val{f}())
end

function (dense::Dense{<:OneDNN.Memory{T}})(_src, fuse_activation = true) where {T}
    src = toeltype(T, Memory(_src))
    attributes = fuse_activation ? dense.attributes : noattributes()

    if dense.optimized_weights == false
        optimized_weights_desc = Ref{MemoryDesc}()
        function weight_opt_callback(_, pd)
            optimized_weights_desc[] = query_md(pd, Lib.dnnl_query_weights_md)
            return nothing
        end

        dst = innerproduct(
            src,
            dense.weights,
            dense.bias;
            attributes = attributes,
            src_md = toany(src),
            weights_md = toany(dense.weights),
            callback = weight_opt_callback,
        )

        dense.weights = maybe_reorder(optimized_weights_desc[], dense.weights)
        dense.optimized_weights = true
        return dst
    else
        return innerproduct(
            src, dense.weights, dense.bias; attributes = attributes, src_md = toany(src)
        )
    end
end

function ChainRulesCore.rrule(
    dense::Dense{<:OneDNN.Memory{T}}, _src::AbstractMatrix{U}, fuse_activation::Bool
) where {T,U}
    src = toeltype(T, Memory(_src))
    # The result of `canfuse` is known at compile time, so Julia can optimize out the branch.
    return if canfuse(dense.activation)
        rrule_fused(dense, src, true)
    else
        rrule_unfused(dense, src, false)
    end
end

function rrule_fused(dense::Dense{<:OneDNN.Memory{T}}, _src, _fuse_activation) where {T}
    src = Memory(_src)
    src_size = size(src)
    dst = dense(src, true)
    function dense_fused_pullback(_diff_dst)
        # Maybe convert argument
        diff_dst = toeltype(eltype(dst), Memory(_diff_dst))
        # Reverse activation function.
        diff_dst_pre = toeltype(T, eltwise_backward(dense.activation, diff_dst, dst))

        # Backprop innerproduct kernel
        diff_src = innerproduct_backward_data(src_size, dense.weights, diff_dst_pre)
        (diff_weights, diff_bias) = innerproduct_backward_weights(
            size(dense.weights), src, diff_dst_pre
        )
        return (
            ChainRulesCore.Tangent{typeof(dense)}(;
                weights = diff_weights, bias = diff_bias
            ),
            diff_src,
            ChainRulesCore.NoTangent(),
        )
    end
    return dst, dense_fused_pullback
end

function rrule_unfused(dense::T, _src, _fuse_activation) where {T<:Dense}
    src = Memory(_src)
    dst_pre = dense(src, false)
    dst = eltwise(dense.activation, dst_pre)
    src_size = size(src)
    function dense_pullback(_diff_dst)
        diff_dst = Memory(_diff_dst)
        diff_dst_pre = eltwise_backward(dense.activation, diff_dst, dst_pre)
        diff_src = innerproduct_backward_data(src_size, dense.weights, diff_dst_pre)
        (diff_weights, diff_bias) = innerproduct_backward_weights(
            size(dense.weights), src, diff_dst_pre
        )

        return (
            ChainRulesCore.Tangent{T}(; weights = diff_weights, bias = diff_bias),
            diff_src,
            ChainRulesCore.NoTangent(),
        )
    end
    return dst, dense_pullback
end
