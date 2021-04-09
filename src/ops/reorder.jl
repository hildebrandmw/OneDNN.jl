"""
    Reorder{A,B}

Reorder memory objets from layout A to layout B.
"""
struct Reorder{A,B}
    primitive::Primitive
    # Track the size and memory description for the destination to allow allocation of
    # destination if it's not provided.
    output_description::MemoryDesc
end

const MaybeReorder = Union{Nothing,Reorder}

function Reorder(dst::MemoryOrDesc, src::MemoryOrDesc; attributes = noattributes())
    descriptor = PrimitiveDescriptor(
        Lib.dnnl_reorder_primitive_desc_create,
        src,
        global_engine(),
        dst,
        global_engine(),
        attributes,
    )
    primitive = Primitive(descriptor)
    return Reorder{layout(dst),layout(src)}(primitive, memorydesc(dst))
end

# Allocating Version
function (op::Reorder{A,B})(src::Memory{B}) where {A,B}
    dst = similar(src, eltype(src), size(src), op.output_description, Val(A))
    return op(dst, src)
end

# Inplace version
function (op::Reorder{A,B})(to::Memory{A}, from::Memory{B}) where {A,B}
    args = @dnnl_args to from
    execute!(op.primitive, args)
    return to
end
