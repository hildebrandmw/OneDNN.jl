struct Binary{O,A,B}
    primitive::Primitive
    output_description::MemoryDesc
end

Binary(f::F, a::MemoryOrDesc, b::MemoryOrDesc) where {F} = Binary(a, b, binary_forward(f))
function Binary(a::MemoryOrDesc, b::MemoryOrDesc, kind)
    desc_any = memorydesc(eltype(a), size(a), dnnl_format_any())
    op_desc = Ref{Lib.dnnl_binary_desc_t}()
    @apicall dnnl_binary_desc_init(op_desc, kind, a, b, desc_any)

    # Create the primitive descriptor
    primitive_descriptor = PrimitiveDescriptor(
        op_desc, noattributes(), global_engine(), noforward()
    )

    primitive = Primitive(primitive_descriptor)

    # Get the output format.
    desc = query_md(primitive_descriptor, Lib.dnnl_query_dst_md)
    return Binary{layout(desc),layout(a),layout(b)}(primitive, desc)
end

function (op::Binary{O,A,B})(a::Memory{A}, b::Memory{B}) where {O,A,B}
    o = similar(a, eltype(a), size(a), memorydesc(a), Val(O))
    return op(o, a, b)
end

function (op::Binary{O,A,B})(
    dst::Memory{O}, src_0::Memory{A}, src_1::Memory{B}
) where {O,A,B}
    args = @dnnl_args dst src_0 src_1
    execute!(op.primitive, args)
    return dst
end

binary_forward(::typeof(+)) = Lib.dnnl_binary_add
binary_forward(::typeof(*)) = Lib.dnnl_binary_mul
binary_forward(::typeof(-)) = Lib.dnnl_binary_sub
binary_forward(::typeof(/)) = Lib.dnnl_binary_div
binary_forward(::typeof(max)) = Lib.dnnl_binary_max
binary_forward(::typeof(min)) = Lib.dnnl_binary_min
