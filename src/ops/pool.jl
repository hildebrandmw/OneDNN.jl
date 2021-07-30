#####
##### Middle Layer
#####

function pooling_forward(
    src::Memory{T,N},
    dims::Dims;
    kind::AbstractKind = Inference(),
    algo = Lib.dnnl_pooling_max,
    opdesc = Ref{Lib.dnnl_pooling_v2_desc_t}()
) where {T,N,M}
    dst_size = output_size(size(src), dims)
    dst_md = memorydesc(T, dst_size, dnnl_format_any())

    @apicall dnnl_pooling_v2_forward_desc_init(
        opdesc,
        kind,
        algo,
        src,
        dst_md,
        dims.strides,
        dims.kernel,
        dims.dilation,
        dims.padding,
        dims.padding,
    )

    return temp_primitive(opdesc, noattributes(), global_engine(), noforward()) do p, pd
        dst_md = query_md(pd, @query(dst))
        dst = similar(src, T, dst_size, dst_md)

        if kind === Inference()
            execute!(p, @dnnl_args src dst)
            return dst
        else
            workspace_md = query_md(pd, @query(workspace))
            workspace = similar(src, UInt8, (getbytes(workspace_md),), workspace_md)
            execute!(p, @dnnl_args src dst workspace)
            return (; dst, workspace, forward = pd)
        end
    end
end

function pooling_backward(
    _diff_dst::Memory{T,N},
    diff_src_md::MemoryDesc,
    dims::Dims;
    algo = Lib.dnnl_pooling_max,
    workspace::Memory,
    forward,
    opdesc = Ref{Lib.dnnl_pooling_v2_desc_t}()
) where {T,N}
    @apicall dnnl_pooling_v2_backward_desc_init(
        opdesc,
        algo,
        diff_src_md,
        toany(_diff_dst),
        dims.strides,
        dims.kernel,
        dims.dilation,
        dims.padding,
        dims.padding,
    )

    return temp_primitive(opdesc, noattributes(), global_engine(), forward) do p, pd
        diff_dst = maybe_reorder(pd, _diff_dst, @query(diff_dst))
        diff_src_md = query_md(pd, @query(diff_src))
        diff_src = similar(diff_dst, T, logicalsize(diff_src_md, Val(N)), diff_src_md)
        execute!(p, @dnnl_args diff_dst diff_src workspace)
        return diff_src
    end
end

#####
##### Higher Level API
#####

const Max = Lib.dnnl_pooling_max
const MeanInclude = Lib.dnnl_pooling_avg_include_padding
const MeanExclude = Lib.dnnl_pooling_avg_exclude_padding

struct Pooling{T,N}
    dims::Dims{N}
    opdesc::Base.RefValue{Lib.dnnl_pooling_v2_desc_t}
end

# Type aliases
const MaxPool{N} = Pooling{Max,N}
const MeanPool{N} = Pooling{MeanExclude,N}
const InclusiveMeanPool{N} = Pooling{MeanInclude,N}

function Pooling{T}(
    kernel::NTuple{N,Integer}; strides = kernel, dilation = 0, padding = 0
) where {T,N}
    strides = expand(Val(N), strides)
    dilation = expand(Val(N), dilation)
    padding = expand(Val(N), padding)
    opdesc = Ref{Lib.dnnl_pooling_v2_desc_t}()
    return Pooling{T,N}(Dims{N}(kernel, strides, dilation, padding), opdesc)
end

function (pool::Pooling{T})(src::AbstractArray; kw...) where {T}
    return pooling_forward(OneDNN.Memory(src), pool.dims; algo = T, pool.opdesc, kw...)
end

function ChainRulesCore.rrule(pool::Pooling{T}, _src) where {T}
    src = Memory(_src)
    src_md_any = toany(src)
    nt = pool(src; kind = Training())
    @unpack workspace, forward = nt

    function pooling_pullback(_diff_dst)
        diff_dst = Memory(_diff_dst)
        diff_src = pooling_backward(
            diff_dst, src_md_any, pool.dims; algo = T, workspace, forward, pool.opdesc
        )
        return (ChainRulesCore.NoTangent(), diff_src)
    end

    return nt.dst, pooling_pullback
end
