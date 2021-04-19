# TODO: Add more methods of "mul!" as they become applicable.
function matmul(
    src::Memory,
    weights::Memory;
    bias = nothing,
    forward = noforward(),
    attributes = noattributes(),
)
    dst_dims = matmul_dst_dims(size(src), size(weights))
    dst_eltype = promote_type(eltype(src), eltype(weights))
    dst_format = dnnl_format_any()
    dst_desc = memorydesc(dst_eltype, dst_dims, dst_format)

    matmul_desc = Ref{Lib.dnnl_matmul_desc_t}()
    @apicall dnnl_matmul_desc_init(matmul_desc, src, weights, Ptr{MemoryDesc}(), dst_desc)

    return temp_primitive(
        matmul_desc, attributes, global_engine(), forward
    ) do primitive, primitive_desc
        # Get the output format to create the destination.
        dst_desc_opt = query_md(primitive_desc, Lib.dnnl_query_dst_md)
        dst = similar(src, dst_eltype, dst_dims, dst_desc_opt)
        execute!(primitive, @dnnl_args src weights dst)
        return dst
    end
end

# Support batched matmul in the third dimension.
matmul_dst_dims(a, b) = matmul_dst_dims(logicalsize(a), logicalsize(b))
matmul_dst_dims(a::T, b::T) where {T<:Tuple{Int,Int}} = (a[1], b[2])
matmul_dst_dims(a::T, b::T) where {T<:Tuple{Int,Int,Int}} = (a[1], a[2], b[3])

Base.:*(src::Memory, weights::Memory) = matmul(src, weights)
