function convolution(
    _src::Memory{T},
    _weights::Memory{T},
    bias::Memory,
    dims::Dims{N};
    kind = Inference(),
    algo = Lib.dnnl_convolution_auto,
    attributes = noattributes(),
    src_md = memorydesc_ptr(_src),
    weights_md = memorydesc_ptr(_weights),
    callback = (x...) -> nothing,
) where {T,N}
    dst_dims = output_size(size(_src), dims)
    dst_md = memorydesc(T, dst_dims, dnnl_format_any())

    opdesc = Ref{Lib.dnnl_convolution_desc_t}()
    @apicall dnnl_convolution_forward_desc_init(
        opdesc,
        kind,
        algo,
        src_md,
        weights_md,
        bias,
        dst_md,
        dims.strides,
        dims.padding,
        dims.padding,
    )

    return temp_primitive(opdesc, attributes, global_engine(), noforward()) do p, pd
        callback(p, pd)

        # Convert if needed.
        src = reorder_if_any(src_md, pd, _src, @query(src))
        weights = reorder_if_any(weights_md, pd, _weights, @query(weights))

        # Allocate destination
        dst_md = query_md(pd, @query(dst))
        dst = similar(src, T, dst_dims, dst_md)

        # Run kernel
        execute!(p, @dnnl_args src weights bias dst)
        if kind === Inference()
            return dst
        elseif kind === Training()
            return (; dst, forward = pd)
        end
    end
end

function convolution_backward_data(
    diff_src_dims::NTuple,
    _weights::Memory{T},
    _diff_dst::Memory{T},
    dims::Dims;
    forward,
    algo = Lib.dnnl_convolution_auto,
) where {T}
    diff_src_md = memorydesc(T, diff_src_dims, dnnl_format_any())
    opdesc = Ref{Lib.dnnl_convolution_desc_t}()
    @apicall dnnl_convolution_backward_data_desc_init(
        opdesc,
        algo,
        diff_src_md,
        toany(_weights),
        toany(_diff_dst),
        dims.strides,
        dims.padding,
        dims.padding,
    )

    return temp_primitive(opdesc, noattributes(), global_engine(), forward) do p, pd
        weights = maybe_reorder(pd, _weights, @query(weights))
        diff_dst = maybe_reorder(pd, _diff_dst, @query(diff_dst))

        diff_src_md = query_md(pd, @query(diff_src))
        diff_src = similar(diff_dst, T, diff_src_dims, diff_src_md)
        execute!(p, @dnnl_args diff_src weights diff_dst)
        return diff_src
    end
end

function convolution_backward_weights(
    diff_weights_dims::NTuple{N},
    _src::Memory{T},
    _diff_dst::Memory{T},
    dims::Dims;
    forward,
    algo = Lib.dnnl_convolution_auto,
) where {T,N}
    diff_bias_dims = (diff_weights_dims[N],)
    diff_weights_md = memorydesc(T, diff_weights_dims, dnnl_format_any())
    diff_bias_md = memorydesc(T, diff_bias_dims)

    opdesc = Ref{Lib.dnnl_convolution_desc_t}()
    @apicall dnnl_convolution_backward_weights_desc_init(
        opdesc,
        algo,
        toany(_src),
        diff_weights_md,
        diff_bias_md,
        toany(_diff_dst),
        dims.strides,
        dims.padding,
        dims.padding,
    )

    return temp_primitive(opdesc, noattributes(), global_engine(), forward) do p, pd
        # Maybe convert arguments
        src = maybe_reorder(pd, _src, @query(src))
        diff_dst = maybe_reorder(pd, _diff_dst, @query(diff_dst))

        # Materialize destinations
        diff_weights_md = query_md(pd, @query(diff_weights))
        diff_weights = similar(diff_dst, T, diff_weights_dims, diff_weights_md)
        diff_bias = similar(diff_dst, T, diff_bias_dims, diff_bias_md)

        execute!(p, @dnnl_args diff_weights diff_bias src diff_dst)
        return (diff_weights, diff_bias)
    end
end

## Convolutional layer
mutable struct Conv{W<:Memory,B<:Memory,F,N}
    weights::W
    bias::B
    activation::F
    dims::Dims{N}
    attributes::Attributes
    optimized_weights::Bool
end

function Conv(
    _weights::AbstractArray{T,N},
    activation = identity;
    stride = 1,
    padding = 0,
    dilation = 0,
) where {T,N}
    kernel = (size(_weights, 1), size(_weights, 2))
    stride = expand(Val(2), stride)
    dilation = expand(Val(2), dilation)
    padding = expand(Val(2), padding)

    dims = Dims{2}(kernel, stride, dilation, padding)
    weights = Memory(_weights)
    bias = similar(weights, size(weights, N))

    # Create attributes for fusing the postop.
    attributes = Attributes()
    postops = PostOps()
    eltwise!(postops, activation)
    append!(attributes, postops)
    return Conv(weights, bias, activation, dims, attributes, false)
end

# function Conv(
#     k::NTuple{N,Integer},
#     ch::Pair{Int32,Int32},
#     σ = identity;
#     init = glorot_uniform,
#     stride = 1,
#     padding = 0,
#     dilation = 1,
#     groups = 1,
#     weight = Flux.convfilter(k, (ch[1] + groups => ch[2]); init),
#     bias = true,
# ) where {N}
#     ## Using Flux to build weight filter. As they use the WHCN format, we need to change format so
#     ## NCHW
#     w_dims = length(size(_weights))
#     if w_dims > 3
#         weight = PermutedDimsArray(weight, (4, 3, 2, 1))
#     else
#         weight = PermutedDimsArray(weight, (3, 2, 1))
#     end
#     return Conv(weight, bias, σ; stride, padding, dilation, groups)
# end
#
# function Conv(m::Flux.Conv)
#     ## Filter weight dimensions include the input and output channels. This means that for
#     ## single dimension filters, we will have a maximum of 3 dimensions. Only in this instance,
#     ## do we not want to take the transpose.
#     weight = m.weight
#     w_dims = length(size(weight))
#     if w_dims > 3
#         weight = PermutedDimsArray(m.weight, (2, 1, 3, 4))
#     end
#
#     return Conv(OneDNN.Memory(weight), OneDNN.Memory(m.bias), m.σ)
# end

## Apparently, this specifies what to collect from training???
Flux.@functor Conv (weights, bias)

## This notation is a functor (adds functionality to struct
function (conv::Conv)(_src, fuse_activation = true; kw...)
    src = Memory(_src)
    attributes = fuse_activation ? conv.attributes : noattributes()

    if conv.optimized_weights == false
        # Prepare a callback to capture the optimized memory format.
        weights_md = Ref{MemoryDesc}()
        function weight_opt_callback(_, pd)
            weights_md[] = query_md(pd, @query(weights))
            return nothing
        end

        dst = convolution(
            src,
            conv.weights,
            conv.bias,
            conv.dims;
            attributes = attributes,
            src_md = toany(src),
            weights_md = toany(conv.weights),
            callback = weight_opt_callback,
            kw...,
        )

        conv.weights = maybe_reorder(weights_md[], conv.weights)
        conv.optimized_weights = true
        return dst
    else
        return convolution(
            src, conv.weights, conv.bias, conv.dims; attributes, src_md = toany(src), kw...
        )
    end
end

function ChainRulesCore.rrule(conv::Conv, src::AbstractArray, _fuse_activation = false)
    # The result of `canfuse` is known at compile time, so Julia can optimize out the branch.
    return canfuse(conv.activation) ? rrule_fused(conv, src) : rrule_unfused(conv, src)
end

function rrule_fused(conv::T, _src) where {T<:Conv}
    src = Memory(_src)
    src_size = size(src)
    nt = conv(src, true; kind = Training())
    @unpack dst, forward = nt
    function conv_fused_pullback(_diff_dst)
        # Maybe convert argument
        diff_dst = Memory(_diff_dst)
        # Reverse activation function.
        diff_dst_pre = eltwise_backward(conv.activation, diff_dst, dst)

        # Backprop convolution kernel
        diff_src = convolution_backward_data(
            src_size, conv.weights, diff_dst_pre, conv.dims; forward
        )
        (diff_weights, diff_bias) = convolution_backward_weights(
            size(conv.weights), src, diff_dst_pre, conv.dims; forward
        )

        return (
            ChainRulesCore.Tangent{T}(; weights = diff_weights, bias = diff_bias),
            diff_src,
            ChainRulesCore.NoTangent(),
        )
    end
    return dst, conv_fused_pullback
end

function rrule_unfused(conv::T, _src) where {T<:Conv}
    src = Memory(_src)
    nt = conv(src, false; kind = Training())
    dst_pre, forward = nt.dst, nt.forward
    dst = eltwise(conv.activation, dst_pre)
    src_size = size(src)
    function conv_pullback(_diff_dst)
        diff_dst = Memory(_diff_dst)
        diff_dst_pre = eltwise_backward(conv.activation, diff_dst, dst_pre)
        diff_src = convolution_backward_data(
            src_size, conv.weights, diff_dst_pre, conv.dims; forward
        )
        (diff_weights, diff_bias) = convolution_backward_weights(
            size(conv.weights), src, diff_dst_pre, conv.dims; forward
        )

        return (
            ChainRulesCore.Tangent{T}(; weights = diff_weights, bias = diff_bias),
            diff_src,
            ChainRulesCore.NoTangent(),
        )
    end
    return dst, conv_pullback
end

