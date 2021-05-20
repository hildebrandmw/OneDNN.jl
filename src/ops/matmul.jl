# TODO: Add more methods of "mul!" as they become applicable.
function matmul(
    _weights,
    _src;
    bias = nothing,
    forward = noforward(),
    attributes = noattributes(),
)
    # Maybe convert arguments if they haven't been converted yet.
    src = Memory(_src)
    weights = Memory(_weights)

    dst_dims = matmul_dst_dims(size(src), size(weights))
    dst_format = dnnl_format_any()
    dst_desc = memorydesc(promote_type(eltype(src), eltype(weights)), dst_dims, dst_format)

    matmul_desc = Ref{Lib.dnnl_matmul_desc_t}()
    @apicall dnnl_matmul_desc_init(matmul_desc, src, weights, Ptr{MemoryDesc}(), dst_desc)

    return temp_primitive(
        matmul_desc, attributes, global_engine(), forward
    ) do primitive, primitive_desc
        # Get the output format to create the destination.
        dst_desc_opt = query_md(primitive_desc, Lib.dnnl_query_dst_md)
        # N.B.: Recompute destination eltype inside the closure to help type inference.
        dst = similar(src, promote_type(eltype(src), eltype(weights)), dst_dims, dst_desc_opt)
        execute!(primitive, @dnnl_args src weights dst)
        return dst
    end
end

function matmul!(
    _dst,
    _weights,
    _src;
    forward = noforward(),
    attributes = noattributes()
)
    dst = Memory(_dst)
    weights = Memory(_weights)
    src = Memory(_src)

    matmul_desc = Ref{Lib.dnnl_matmul_desc_t}()
    @apicall dnnl_matmul_desc_init(matmul_desc, src, weights, Ptr{MemoryDesc}(), dst)

    temp_primitive(
        matmul_desc, attributes, global_engine(), forward
 ) do primitive, primitive_desc
        execute!(primitive, @dnnl_args src weights dst)
    end
    return dst
end

# Support batched matmul in the third dimension.
matmul_dst_dims(a, b) = matmul_dst_dims(logicalsize(a), logicalsize(b))
matmul_dst_dims(a::T, b::T) where {T<:Tuple{Int,Int}} = (b[1], a[2])
matmul_dst_dims(a::T, b::T) where {T<:Tuple{Int,Int,Int}} = (b[1], a[2], a[3])

Base.:*(src::Memory, weights::Memory) = matmul(src, weights)
