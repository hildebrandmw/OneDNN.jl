#####
##### Dimension Helpers
#####

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

_paddims(x::Tuple, y::Tuple) = (x..., y[(end - (length(y) - length(x) - 1)):end]...)

struct Dims{N}
    kernel::NTuple{N,Int}
    strides::NTuple{N,Int}
    dilation::NTuple{N,Int}
    padding::NTuple{N,Int}
end

function _output_size(dims::Dims, sz::Int, i::Int)
    @unpack kernel, strides, dilation, padding = dims
    numerator = sz + 2 * padding[i] - ((kernel[i] - 1) * (dilation[i] + 1) + 1)
    return div(numerator, strides[i]) + 1
end

function output_size(sz::NTuple{N,Int}, dims::Dims{M}) where {N,M}
    return _paddims(ntuple(i -> _output_size(dims, sz[i], i), Val(M)), sz)
end

# Keep these here for now I guess
abstract type AbstractKind end
struct Inference <: AbstractKind end
struct Training <: AbstractKind end

kind(::Inference) = Lib.dnnl_forward_inference
kind(::Training) = Lib.dnnl_forward_training

dnnl_convert(x::AbstractKind) = kind(x)

#####
##### Middle Layer
#####

function pooling_forward(
    src::Memory{T,N},
    dims::Dims;
    kind::AbstractKind = Inference(),
) where {T,N,M}
    dst_size = output_size(size(src), dims)
    @show dst_size
    dst_desc = memorydesc(T, dst_size, dnnl_format_any())

    pooling_desc = Ref{Lib.dnnl_pooling_v2_desc_t}()
    @apicall dnnl_pooling_v2_forward_desc_init(
        pooling_desc,
        kind,
        Lib.dnnl_pooling_max,
        src,
        dst_desc,
        dims.strides,
        dims.kernel,
        dims.dilation,
        dims.padding,
        dims.padding,
    )

    return temp_primitive(
        pooling_desc, noattributes(), global_engine(), noforward()
    ) do primitive, primitive_descriptor
        dst = similar(
            src, T, dst_size, query_md(primitive_descriptor, Lib.dnnl_query_dst_md)
        )

        if kind === Inference()
            execute!(primitive, @dnnl_args src dst)
            return dst
        else
            workspace_md = query_md(primitive_descriptor, Lib.dnnl_query_workspace_md)
            workspace = similar(src, UInt8, (Int(getbytes(workspace_md)),), workspace_md)
            execute!(primitive, @dnnl_args src dst workspace)
            return (; dst, workspace, forward = primitive_descriptor)
        end
    end
end

function pooling_backward(
    diff_dst::Memory{T,N},
    diff_src_desc::MemoryDesc,
    dims::Dims;
    workspace::Memory,
    forward = noforward(),
) where {T,N}
    pooling_desc = Ref{Lib.dnnl_pooling_v2_desc_t}()
    @apicall dnnl_pooling_v2_backward_desc_init(
        pooling_desc,
        Lib.dnnl_pooling_max,
        diff_src_desc,
        diff_dst,
        dims.strides,
        dims.kernel,
        dims.dilation,
        dims.padding,
        dims.padding,
    )

    return temp_primitive(
        pooling_desc, noattributes(), global_engine(), forward,
    ) do primitive, primitive_descriptor
        optimized_md = query_md(primitive_descriptor, Lib.dnnl_query_diff_src_md)
        diff_src = similar(
           diff_dst, T, logicalsize(optimized_md, Val(N)), optimized_md,
        )
        execute!(primitive, @dnnl_args diff_dst diff_src workspace)
        return diff_src
    end
end

#####
##### Higher Level API
#####

struct MaxPool{N}
    dims::Dims{N}
end

function MaxPool(kernel::NTuple{N,Integer}; strides = kernel, dilation = 0, padding = 0) where {N}
    strides = expand(Val(N), strides)
    dilation = expand(Val(N), dilation)
    padding = expand(Val(N), padding)
    return MaxPool(Dims(kernel, strides, dilation, padding))
end

function (pool::MaxPool)(src::AbstractArray; kw...)
    return pooling_forward(OneDNN.Memory(src), pool.dims; kw...)
end

function ChainRulesCore.rrule(pool::MaxPool, _src)
    src = Memory(_src)
    src_desc_any = toany(src)
    nt = pool(src; kind = Training())
    @unpack workspace, forward = nt

    function maxpool_pullback(_diff_dst)
        diff_dst = Memory(_diff_dst)
        diff_src = pooling_backward(
            diff_dst,
            src_desc_any,
            pool.dims;
            workspace,
            forward,
        )
        return (ChainRulesCore.NoTangent(), diff_src)
    end

    return nt.dst, maxpool_pullback
end

