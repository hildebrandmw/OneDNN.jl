# Restrict inputs to just `Memory` types.
# Default the output format to normal Julia column-major format
function reorder(src::Memory, format = dnnl_format(val_ndims(src)))
    # Construct the destination object
    dst_desc = memorydesc(eltype(src), size(src), format)
    dst = similar(src, eltype(src), size(src), dst_desc)

    reorder!(dst, src)
    return dst
end

function reorder!(dst::Memory, src::Memory)
    reorder_desc = Ref{Lib.dnnl_primitive_desc_t}()
    @apicall Lib.dnnl_reorder_primitive_desc_create(
        reorder_desc,
        memorydesc_ptr(src),
        global_engine(),
        memorydesc_ptr(dst),
        global_engine(),
        Ptr{Nothing}(),
    )

    args = [
        arg(Lib.DNNL_ARG_FROM, src),
        arg(Lib.DNNL_ARG_TO, dst),
    ]

    execute!(reorder_desc, args)

    # Cleanup the primitive descriptor
    @apicall Lib.dnnl_primitive_desc_destroy(reorder_desc[])
    return nothing
end

