## To allow for padding and strides of multiple dimensions
expand(N, i::Tuple) = i
expand(N, i::Int)   = ntuple(_ -> i, N)

"""
    convolution(
        _src::Memory,
        _weights::Memory,
        bias::Memory,
        out_channel::Int32,
        stride::T,
        padding::T;
        prop_kind    = Lib.dnnl_forward_inference, 
        alg_kind     = Lib.dnnl_convolution_direct,
        attributes   = noattributes(),
        src_desc     = memorydesc(_src),
        weights_desc = memorydesc(_weights),
        callback     = (x...) -> nothing,
    )

Generates the resulting convolution based on a given output.

# Examples
```jldoctest
julia>

julia>

julia>
```
"""
## NOTE: Input can only be 1D or 2D (not including channel and batch)
function convolution(
    _src::Memory,
    _weights::Memory, ## This is the filter size to use
    bias::Memory,
    stride::NTuple{N,Int},
    padding::NTuple{M,Int};
    prop_kind    = Lib.dnnl_forward_inference, 
    alg_kind     = Lib.dnnl_convolution_direct,
    attributes   = noattributes(),
    src_desc     = memorydesc(_src),
    weights_desc = memorydesc(_weights),
    callback     = (x...) -> nothing,
) where {N,M}
    ## Data Processing - Set `dst` dimensions ALWAYS HEIGHT THEN WIDTH
    ##  Is there a specific format??? NCHW (This is what onednn looks for, but the inverse for Flux)
    ##  net_dst_sizes = {BATCH, OC, CONV_OH, CONV_OW}
    ##  bias size is {OC}
    ##  strides is {stride, stride}
    ##  padding is {pad, pad}
    ##  weight_sizes is {0C, IC, 11, 11} LAST TWO ARE THE FILTER SIZE
    ##  * C and N are the exact same (Nope, channel is determined by passed in value)
    ##  * H and W can depend on padding. Is this provided at all? 

    ## Parameter dims depend on `src` dims
    w_dims = length(size(_weights))
    if w_dims > 3
        src_h, src_w         = (size(_src, 3), size(_src, 4))
        filter_h, filter_w   = (size(_weights, 3) ,size(_weights, 4))
        padding_t, padding_b = padding
        padding_l, padding_r = padding
    else
        src_h, src_w         = (1, size(_src, 3))
        filter_h, filter_w   = (1, size(_weights, 3))
        padding_t, padding_b = (padding[1], padding[1])
        padding_l, padding_r = (padding[1], padding[1])
    end

    ## Build layer output
    dst_depth        = size(_src, 1)
    dst_out_channel  = size(_weights, w_dims)
    dst_height       = ((src_h + padding_t + padding_b - filter_h) / stride[1]) + 1 
    dst_width        = ((src_w + padding_l + padding_r - filter_w) / stride[1]) + 1 ## Feeling lazy. Need to handle 
    dst_dims         = (dst_depth, dst_out_channel, Int(dst_height), Int(dst_width))
    dst_desc         = memorydesc(eltype(_src), dst_dims, dnnl_format_any())
    convolution_desc = Ref{Lib.dnnl_convolution_desc_t}()
    @apicall dnnl_convolution_forward_desc_init(
        convolution_desc, 
	prop_kind, 
	alg_kind, 
	src_desc, 
	weights_desc, 
	bias, 
	dst_desc, 
	stride, 
	padding, 
	padding 
    )

    return temp_primitive(
        convolution_desc, attributes, global_engine(), noforward()
    ) do primitive, primitve_descriptor
        callback(primitive, primitve_descriptor)

        # Convert if needed.
        src = _src
        if isany(src_desc)
            src = maybe_reorder(primitive_descriptor, src, Lib.dnnl_query_src_md)
	end

	weights = _weights
	if isany(weights_desc)
	    weights = maybe_reorder(
                primitve_descriptor, weights, Lib.dnnl_query_weights_md
	    )
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

function convolution_backward_data(                                                         
    diff_src_dims::NTuple, _weights::Memory, _diff_dst::Memory, stride, padding, alg_kind = Lib.dnnl_convolution_direct
)                                                                                            
    diff_src_desc = memorydesc(eltype(_diff_dst), diff_src_dims, dnnl_format_any())          
    conv_desc = Ref{Lib.dnnl_convolution_desc_t}
    @apicall dnnl_convolution_backward_data_desc_init(                                     
        conv_desc, alg_kind, diff_src_desc, toany(_weights), toany(_diff_dst), stride, padding, padding                 
    )                                                                                        
                                                                                             
    return temp_primitive(                                                                   
       conv_desc, noattributes(), global_engine(), noforward()                     
    ) do primitive, primitive_descriptor                                                     
        weights = maybe_reorder(primitive_descriptor, _weights, Lib.dnnl_query_weights_md)   
        diff_dst = maybe_reorder(                                                            
            primitive_descriptor, _diff_dst, Lib.dnnl_query_diff_dst_md                      
        )                                                                                    
                                                                                             
        diff_src_desc_opt = query_md(primitive_descriptor, Lib.dnnl_query_diff_src_md)       
        diff_src = similar(diff_dst, eltype(diff_dst), diff_src_dims, diff_src_desc_opt)     
        execute!(primitive, @dnnl_args diff_src weights diff_dst)                            
        return diff_src                                                                      
    end                                                                                      
end                                                                                          
                                                                                             
function convolution_backward_weights(                                                      
    diff_weights_dims::NTuple,                                                               
    _src::Memory,                                                                            
    _diff_dst::Memory,                                                                       
    stride,
    padding,
    alg_kind = Lib.dnnl_convolution_direct
)                                                                                            
    diff_bias_dims = (diff_weights_dims[2],)                                                 
    diff_weights_desc = memorydesc(eltype(_src), diff_weights_dims, dnnl_format_any())       
    diff_bias_desc = memorydesc(eltype(_src), diff_bias_dims, Lib.dnnl_a)                    
    conv_desc = Ref{Lib.dnnl_convolution_desc_t}()                                
                                                                                             
    @apicall dnnl_convolution_backward_weights_desc_init(                                  
        convolution_desc, 
        alg_kind, 
        toany(_src), 
        diff_weights_desc, 
        diff_bias_desc, 
        toany(_diff_dst), 
        stride, 
        padding, 
        padding                                                                                    
    )                                                                                        
                                                                                             
    return temp_primitive(                                                                   
        conv_desc, noattributes(), global_engine(), noforward()                     
    ) do primitive, primitive_descriptor                                                     
        # Maybe convert arguments                                                            
        src = maybe_reorder(primitive_descriptor, _src, Lib.dnnl_query_src_md)               
        diff_dst = maybe_reorder(                                                            
            primitive_descriptor, _diff_dst, Lib.dnnl_query_diff_dst_md                      
        )                                                                                    
                                                                                             
        # Materialize destinations                                                           
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

## Convolutional layer

mutable struct Conv{W<:Memory,B<:Memory,F,N,M}
    weights::W
    bias::B
    activation::F
    stride::NTuple{N,Int}
    padding::NTuple{M,Int}
    ##channels::Pair{Int32,Int32}
    dilation::NTuple{N,Int} ## Not sure of the purpose of this
    groups::Int
    optimized_weights::Bool
    optimized_weights_desc::MemoryDesc
end

## Expects weights to be in NCHW format 
function Conv(weights::AbstractArray{T,N}, bias = true, σ = identity; 
	      stride = 1, padding = 0, dilation = 1, groups = 1) where {T,N}
    ##stride   = expand(Val(N-2), stride)
    ##dilation = expand(Val(N-2), dilation)
    stride   = expand(2, stride)
    dilation = expand(2, dilation)
    padding  = Flux.calc_padding(Conv, padding, size(weights)[1:N-2], dilation, stride)
    bias     = Flux.create_bias(weights, bias, size(weights, N))
    weights_memory = Memory(weights)
    return Conv(
        weights_memory, Memory(bias), σ, stride, padding, dilation, groups, false, memorydesc(weights_memory)
    )
end

function Conv(k::NTuple{N,Integer}, ch::Pair{Int32,Int32}, σ = identity;
              init = glorot_uniform, stride = 1, padding = 0, dilation = 1, groups = 1,
              weight = Flux.convfilter(k, (ch[1] + groups => ch[2]); init), bias = true) where N
    ## Using Flux to build weight filter. As they use the WHCN format, we need to change format so
    ## NCHW
    w_dims = length(size(_weights))
    if w_dims > 3
        weight = PermutedDimsArray(weight, (4, 3, 2, 1)) 
    else
        weight = PermutedDimsArray(weight, (3, 2, 1)) 
    end 
    return Conv(
        weight, bias, σ; stride, padding, dilation, groups
    )
end

function Conv(m::Flux.Conv) 
    ## Filter weight dimensions include the input and output channels. This means that for
    ## single dimension filters, we will have a maximum of 3 dimensions. Only in this instance,
    ## do we not want to take the transpose.
    weight = m.weight
    w_dims   = length(size(weight))
    if w_dims > 3
        weight = PermutedDimsArray(m.weight, (2,1,3,4))
    end

    return Conv(OneDNN.Memory(weight), OneDNN.Memory(m.bias), m.σ)
end

## Apparently, this specifies what to collect from training???
Flux.@functor Conv (weights, bias)

## This notation is a functor (adds functionality to struct
function (conv::Conv)(_src, fuse_activation = true)
    src = Memory(_src)
    attributes = Attributes()
    if fuse_activation
        postops = PostOps()
	eltwise!(postops, conv.activation)
	append!(attributes, postops)
    end

    if conv.optimized_weights == false
        callback = function weight_opt_callback(_, pd)
	    conv.optimized_weights_desc = query_md(pd, Lib.dnnl_query_weights_md)
	    return nothing
	end

        dst = convolution(
            src,
            conv.weights,
            conv.bias,
            conv.stride,
            conv.padding;
            attributes = attributes,
            src_desc = toany(memorydesc(src)),
            weights_desc = toany(memorydesc(conv.weights)),
            callback = callback,
        )
	if conv.optimized_weights_desc != memorydesc(conv.weights)
	    conv.weights = reorder(conv.optimized_weights_desc, conv.weights)
	end
	conv.optimized_weights = true
	return dst
    else
        dst = convolution(src, conv.weights, conv.bias; attributes = attributes)
    end
end

function ChainRulesCore.rrule(conv::Conv, src::AbstractMatrix, fuse_activation::Bool)      
    # The result of `canfuse` is known at compile time, so Julia can optimize out the branch.
    return canfuse(conv.activation) ? rrule_fused(conv, src, fuse_activation) :            
           rrule_unfused(conv, src, fuse_activation)                                        
end                                                                                          
                                                                                             
function rrule_fused(conv::T, _src, _fuse_activation) where {T}                             
    src = Memory(_src)                                                                       
    src_size = size(src)                                                                     
    dst = conv(src, true)                                                                   
    pullback = function conv_fused_pullback(_diff_dst)                                      
        # Maybe convert argument                                                             
        diff_dst = Memory(_diff_dst)                                                         
        # Reverse activation function.                                                       
        diff_dst_pre = eltwise_backward(conv.activation, diff_dst, dst)                     
                                                                                             
        # Backprop convolution kernel                                                       
        diff_src = convolution_backward_data(src_size, conv.weights, diff_dst_pre, conv.stride, conv.padding)         
        (diff_weights, diff_bias) = convolution_backward_weights(                           
            size(conv.weights),                                                             
            src,                                                                             
            diff_dst_pre,                                                                    
            conv.stride,
            conv.padding
        )                                                                                    
                                                                                             
        return (                                                                             
            ChainRulesCore.Tangent{T}(; weights = diff_weights, bias = diff_bias),           
            diff_src,                                                                        
            ChainRulesCore.NoTangent(),                                                      
        )                                                                                    
    end                                                                                      
    return dst, pullback                                                                     
end                                                                                          
                                                                                             
function rrule_unfused(conv::T, _src, _fuse_activation) where {T}                           
    src = Memory(_src)                                                                       
    dst_pre = conv(src, false)                                                              
    dst = eltwise(conv.activation, dst_pre)                                                 
    src_size = size(src)                                                                     
    pullback = function conv_pullback(_diff_dst)                                            
        diff_dst = Memory(_diff_dst)                                                         
        diff_dst_pre = eltwise_backward(conv.activation, diff_dst, dst_pre)                 
        diff_src = convolution_backward_data(src_size, conv.weights, diff_dst_pre, conv.stride, conv.padding)         
        (                                                                                    
            diff_weights, diff_bias                                                          
        ) = convolution_backward_weights(size(conv.weights), src, diff_dst_pre, conv.stride, conv.padding)            
                                                                                             
        return (                                                                             
            ChainRulesCore.Tangent{T}(; weights = diff_weights, bias = diff_bias),           
            diff_src,                                                                        
            ChainRulesCore.NoTangent(),                                                      
        )                                                                                    
    end                                                                                      
    return dst, pullback                                                                     
end
