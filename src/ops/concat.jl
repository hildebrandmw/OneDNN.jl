function concat(A, dim)
    A = memory.(A)

    # Compute the size of the output
    nargs = length(A)

    # The prep code is largely based on the code in Julia's `_typed_vcat`.
    outputdim = sum(a -> size(a, dim), A)::Int
    outsize = ntuple(i -> i == dim ? outputdim : size(A[1], i), ndims(first(A)))
    dst_desc = memorydesc(eltype(A[1]), outsize, dnnl_format_any())
    concat_desc = Ref{Lib.dnnl_primitive_desc_t}()

    @apicall Lib.dnnl_concat_primitive_desc_create(
        concat_desc,
        Ref(dst_desc),
        length(A),
        dim - 1,
        memorydesc.(A),
        Ptr{Nothing}(),
        global_engine()
    )

    # Now that we've created the primitive, we need to query it to get the proper output
    # memory descriptor.
    dst_desc = Lib.dnnl_primitive_desc_query_md(
        concat_desc[],
        Lib.dnnl_query_dst_md,
        0
    )

    dst = similar(A[1], eltype(A[1]), outsize, unsafe_load(dst_desc))

    args = [arg(Lib.DNNL_ARG_MULTIPLE_SRC + i - 1, A[i]) for i in 1:length(A)]
    push!(args, arg(Lib.DNNL_ARG_DST, dst))

    execute!(concat_desc, args)
    return dst
end

