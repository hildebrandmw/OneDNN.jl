matmul_dst_dims(a, b) = matmul_dst_dims(size(a), size(b))
matmul_dst_dims(a::T, b::T) where {T <: NTuple} = (a[1], b[2], maybethird(a)...)

maybethird(::NTuple{2}) = ()
maybethird(x::NTuple{3}) = (x[3],)

out_eltype(::Type{T}, ::Type{T}) where {T} = T
out_eltype(x, y) = error("Different Eltypes! $(typeof(x)) and $(typeof(y))")

function matmul(a, b)
    # Swap the order of `a` and `b`
    a = memorywrap(a)
    b = memorywrap(b)

    # Compute the size of the destinaion.
    dst_dims = matmul_dst_dims(size(a), size(b))
    dst_eltype = out_eltype(eltype(a), eltype(b))

    dst_desc = memorydesc(dst_eltype, dst_dims, Lib.dnnl_ab)
    dst = similar(a, dst_eltype, dst_dims, dst_desc)

    # null pointer for the bias for now.
    bias_desc = Ptr{Lib.dnnl_memory_desc_t}()

    # Construct the primitive descriptor
    matmul_d = Ref{Lib.dnnl_matmul_desc_t}()
    @apicall Lib.dnnl_matmul_desc_init(
        matmul_d,
        memorydesc(a),
        memorydesc(b),
        bias_desc,
        memorydesc(dst),
    )

    # Now that we have the primitive descriptor, we create the primitive and invoke it.
    primitive_d = Ref{Lib.dnnl_primitive_desc_t}()
    @apicall Lib.dnnl_primitive_desc_create(
        primitive_d,
        matmul_d,
        Ptr{Nothing}(),
        GLOBAL_ENGINE[].handle,
        Ptr{Nothing}(),
    )

    # Create the primitive
    primitive = Ref{Lib.dnnl_primitive_t}()
    @apicall Lib.dnnl_primitive_create(primitive, primitive_d[])

    # Clean up primitive descriptor
    @apicall Lib.dnnl_primitive_desc_destroy(primitive_d[])

    # Call the primitive.
    args = [
            arg(Lib.DNNL_ARG_SRC, a.memory),
            arg(Lib.DNNL_ARG_WEIGHTS, b.memory),
            arg(Lib.DNNL_ARG_DST, dst.memory),
    ]

    @apicall Lib.dnnl_primitive_execute(
        primitive[],
        GLOBAL_STREAM[].handle,
        length(args),
        args,
    )

    # Wait for the op to complete.
    @apicall Lib.dnnl_stream_wait(GLOBAL_STREAM[].handle)
    return dst
end

