matmul_dst_dims(a, b) = matmul_dst_dims(size(a), size(b))
matmul_dst_dims(a::T, b::T) where {T <: NTuple} = (a[1], b[2], maybethird(a)...)

maybethird(::NTuple{2}) = ()
maybethird(x::NTuple{3}) = (x[3],)

out_eltype(::Type{T}...) where {T} = T
out_eltype(x, y) = error("Different Eltypes! $(typeof(x)) and $(typeof(y))")

function matmul(a, b)
    # Swap the order of `a` and `b`
    a = memorywrap(a)
    b = memorywrap(b)

    # Compute the size of the destinaion.
    dst_dims = matmul_dst_dims(size(a), size(b))
    dst_eltype = out_eltype(eltype(a), eltype(b))

    output_format = dnnl_format(OutputContext(), val_ndims(a))
    dst_desc = memorydesc(dst_eltype, dst_dims, output_format)
    dst = similar(a, dst_eltype, dst_dims, dst_desc)

    # null pointer for the bias for now.
    bias_desc = Ptr{Lib.dnnl_memory_desc_t}()

    # Construct the primitive descriptor
    matmul_d = Ref{Lib.dnnl_matmul_desc_t}()
    @apicall Lib.dnnl_matmul_desc_init(
        matmul_d,
        # Reverse the order of `a` and `b` to get the column-major behavior
        memorydesc(a),
        memorydesc(b),
        bias_desc,
        memorydesc(dst),
    )

    args = [
            arg(Lib.DNNL_ARG_SRC, a.memory),
            arg(Lib.DNNL_ARG_WEIGHTS, b.memory),
            arg(Lib.DNNL_ARG_DST, dst.memory),
    ]

    # Create the primitive descriptor, then the primitive, then execute it.
    primitive_descriptor(x -> execute!(x, args), matmul_d)
    return dst
end

