matmul_dst_dims(a, b) = matmul_dst_dims(size(a), size(b))
matmul_dst_dims(a::T, b::T) where {T <: NTuple} = (b[1], a[2], maybethird(a)...)

maybethird(::NTuple{2}) = ()
maybethird(x::NTuple{3}) = (x[3],)

out_eltype(::Type{T}...) where {T} = T
out_eltype(x...) = error("Different Eltypes! $(typeof.(x))")

# Switch order to account for column major to row-major ordering.
matmul(x...; kw...) = matmul(memory.(x)...; kw...)
function matmul(a::Memory, b::Memory; kw...)
    # Compute the size of the destinaion.
    dst_dims = matmul_dst_dims(size(a), size(b))
    dst_eltype = out_eltype(eltype(a), eltype(b))

    output_format = dnnl_format(val_ndims(a))
    dst_desc = memorydesc(dst_eltype, dst_dims, output_format)
    dst = similar(a, dst_eltype, dst_dims, dst_desc)

    matmul!(dst, a, b; kw...)
    return dst
end

matmul!(c, a, b; kw...) = matmul!(memory(c), memory(a), memory(b); kw...)
function matmul!(c::Memory, a::Memory, b::Memory; attributes = Ptr{Nothing}())
    # Op Description

    ### NOTE: very important - we swap the ordering of `a` and `b` to get behavior to match
    # for Julia's column major layout.

    matmul_d = Ref{Lib.dnnl_matmul_desc_t}()
    @apicall Lib.dnnl_matmul_desc_init(
        matmul_d,
        memorydesc_ptr(a),
        memorydesc_ptr(b),
        Ptr{Nothing}(),         # Currently don't support bias
        memorydesc_ptr(c),
    )

    # Primitive Description
    pd = primitive_descriptor(
        matmul_d,
        attributes,
        global_engine(),
        Ptr{Nothing}(),
    )

    # Primitive
    args = [
            arg(Lib.DNNL_ARG_SRC, a),
            arg(Lib.DNNL_ARG_WEIGHTS, b),
            arg(Lib.DNNL_ARG_DST, c),
    ]

    p = primitive(pd)
    execute!(p, args)

    # cleanup
    destroy(p, pd)
end

