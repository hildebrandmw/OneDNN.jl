struct MatMul{O,A,B,C}
    primitive::Primitive
    # Cached output description.
    output_description::MemoryDesc
end

function MatMul(
    a::MemoryOrDesc,
    b::MemoryOrDesc;
    bias = nothing,
    attributes = noattributes(),
    # forward pass primitive descriptor
    forward = noforward(),
)
    output_dims = matmul_destination_dims(logicalsize(a), logicalsize(b))
    # TODO: Get eltypes from MemoryDesc
    output_eltype = Float32
    output_format = dnnl_format_any()
    output_description = memorydesc(output_eltype, output_dims, output_format)

    # Need
    matmul_descriptor = Ref{Lib.dnnl_matmul_desc_t}()
    @apicall dnnl_matmul_desc_init(
        matmul_descriptor,
        a,
        b,
        Ptr{MemoryDesc}(),  # null pointer - no bias.
        output_description,
    )

    primitive_descriptor = PrimitiveDescriptor(
        matmul_descriptor, attributes, global_engine(), forward
    )

    primitive = Primitive(primitive_descriptor)

    # Get the output format.
    output_descriptor_ptr = Lib.dnnl_primitive_desc_query_md(
        primitive_descriptor, Lib.dnnl_query_dst_md, 0
    )

    desc = unsafe_load(output_descriptor_ptr)
    return MatMul{layout(desc),layout(a),layout(b),Nothing}(primitive, desc)
end

# Support batched matmul in the third dimension.
matmul_destination_dims(a, b) = matmul_destination_dims(logicalsize(a), logicalsize(b))
matmul_destination_dims(a::T, b::T) where {T<:Tuple{Int,Int}} = (a[1], b[2])
matmul_destination_dims(a::T, b::T) where {T<:Tuple{Int,Int,Int}} = (a[1], b[2], a[3])

function (op::MatMul{O,A,B,Nothing})(a::Memory{A}, b::Memory{B}) where {O,A,B}
    dst = similar(
        a, eltype(a), matmul_destination_dims(a, b), op.output_description, Val(O)
    )
    return op(dst, a, b)
end

function (op::MatMul{O,A,B,Nothing})(
    dst::Memory{O}, src::Memory{A}, weights::Memory{B}
) where {O,A,B}
    args = @dnnl_args src weights dst
    execute!(op.primitive, args)
    return dst
end
