# Restrict inputs to just `Memory` types.
# Default the output format to normal Julia column-major format
function reorder(
        src::Memory,
        format::Lib.dnnl_format_tag_t = dnnl_format(val_ndims(src))
    )

    # Construct the destination object
    dst_desc = memorydesc(eltype(src), size(src), format)
    return reorder(src, dst_desc)
end

reorder(src::Memory, desc::Ptr{MemoryDesc}) = reorder(src, unsafe_load(desc))
function reorder(src::Memory, desc::MemoryDesc)
    dst = similar(src, eltype(src), size(src), desc)
    reorder!(dst, src)
    return dst
end

function reorder!(dst::Memory, src::Memory)
    # primitive descriptor
    pd = primitive_descriptor(
        Lib.dnnl_reorder_primitive_desc_create,
        src,
        global_engine(),
        dst,
        global_engine(),
        Ptr{Nothing}(),
    )

    # primitive
    args = [
        arg(Lib.DNNL_ARG_FROM, src),
        arg(Lib.DNNL_ARG_TO, dst),
    ]

    p = primitive(pd)
    execute!(p, args)

    # cleanup
    destroy(p, pd)
    return nothing
end

