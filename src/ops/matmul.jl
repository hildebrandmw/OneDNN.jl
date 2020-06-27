matmul_dst_dims(a, b) = matmul_dst_dims(size(a), size(b))
matmul_dst_dims(a::T, b::T) where {T <: NTuple} = (a[1], b[2], maybethird(a)...)

maybethird(::NTuple{2}) = ()
maybethird(x::NTuple{3}) = (x[3],)

out_eltype(::Type{T}...) where {T} = T
out_eltype(x, y) = error("Different Eltypes! $(typeof(x)) and $(typeof(y))")

function matmul(a, b; kw...)
    # Create `Memory` objects out of `a` and `b`.
    #
    # If `a` or `b` are already in instance `Memory`, this is just a no-op.
    a = memory(a)
    b = memory(b)

    # Compute the size of the destinaion.
    dst_dims = matmul_dst_dims(size(a), size(b))
    dst_eltype = out_eltype(eltype(a), eltype(b))

    output_format = dnnl_format(TransposeContext(), val_ndims(a))
    dst_desc = memorydesc(dst_eltype, dst_dims, output_format)
    dst = similar(a, dst_eltype, dst_dims, dst_desc)

    matmul!(dst, a, b; kw...)
    return dst
end

function matmul!(c, a, b; kw...)
    c = memory(c)
    a = memory(a)
    b = memory(b)

    # Construct the primitive descriptor
    matmul_d = Ref{Lib.dnnl_matmul_desc_t}()
    @apicall Lib.dnnl_matmul_desc_init(
        matmul_d,
        # Reverse the order of `a` and `b` to get the column-major behavior
        memorydesc_ptr(a),
        memorydesc_ptr(b),
        Ptr{Nothing}(),         # Currently don't support bias
        memorydesc_ptr(c),
    )

    args = [
            arg(Lib.DNNL_ARG_SRC, a),
            arg(Lib.DNNL_ARG_WEIGHTS, b),
            arg(Lib.DNNL_ARG_DST, c),
    ]

    # Create the primitive descriptor, then the primitive, then execute it.
    primitive_descriptor(x -> execute!(x, args), matmul_d; kw...)
    return nothing
end

