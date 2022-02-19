# TODO: Add more methods of "mul!" as they become applicable.
function matmul(
    weights::Memory{T},
    src::Memory{T};
    forward = noforward(),
    attributes = noattributes(),
) where {T}
    dst_dims = matmul_dst_dims(size(src), size(weights))
    dst_md = memorydesc(T, dst_dims, dnnl_format_any())

    opdesc = Ref{Lib.dnnl_matmul_desc_t}()
    @apicall dnnl_matmul_desc_init(opdesc, src, weights, Ptr{MemoryDesc}(), dst_md)

    return temp_primitive(opdesc, attributes, global_engine(), forward) do p, pd
        dst_md = query_md(pd, @query(dst))
        dst = similar(src, T, dst_dims, dst_md)
        execute!(p, @dnnl_args src weights dst)
        return dst
    end
end

function matmul!(
    dst::Memory,
    weights::Memory,
    src::Memory;
    forward = noforward(),
    attributes = noattributes(),
)
    opdesc = Ref{Lib.dnnl_matmul_desc_t}()
    @apicall dnnl_matmul_desc_init(opdesc, src, weights, Ptr{MemoryDesc}(), dst)

    temp_primitive(opdesc, attributes, global_engine(), forward) do p, _
        execute!(p, @dnnl_args src weights dst)
    end
    return dst
end

# Support batched matmul in the third dimension.
matmul_dst_dims(a, b) = matmul_dst_dims(logicalsize(a), logicalsize(b))
matmul_dst_dims(a::T, b::T) where {T<:Tuple{Int,Int}} = (b[1], a[2])
matmul_dst_dims(a::T, b::T) where {T<:Tuple{Int,Int,Int}} = (b[1], a[2], a[3])

Base.:*(src::Memory, weights::Memory) = matmul(src, weights)
