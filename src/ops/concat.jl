function concat(A::Vector{<:Memory}, dim) where {N}
    # Compute the size of the output
    # The prep code is largely based on the code in Julia's `_typed_vcat`.
    nargs = length(A)
    outputdim = sum(a -> size(a, dim), A)::Int
    outsize = ntuple(i -> i == dim ? outputdim : size(A[1], i), ndims(first(A)))

    # use `dnnl_format_any` and let the primitive descriptor decide which output format
    # to use.
    dst_desc = memorydesc(eltype(A[1]), outsize, dnnl_format_any())

    # primitive descriptor
    pd = primitive_descriptor(
        Lib.dnnl_concat_primitive_desc_create,
        Ref(dst_desc),
        length(A),
        ndims(A[1]) - dim,
        memorydesc.(A),
        Ptr{Nothing}(),
        global_engine(),
    )

     # Now that we've created the primitive, we need to query it to get the proper output
     # memory descriptor.
     dst_desc = Lib.dnnl_primitive_desc_query_md(
         pd,
         Lib.dnnl_query_dst_md,
         0
     )

    dst = similar(A[1], eltype(A[1]), outsize, unsafe_load(dst_desc))

    # primitive
    args = [arg(Lib.DNNL_ARG_MULTIPLE_SRC + i - 1, A[i]) for i in 1:length(A)]
    push!(args, arg(Lib.DNNL_ARG_DST, dst))

    p = primitive(pd)
    execute!(p, args)

    # cleanup
    destroy(p, pd)
    return dst
end

#####
##### Slicing
#####

# map` seems to be having Julia issue #15276 is problems when keeping track of where
# we are indexing to create views.
#
# As such, we have to build this `Slicer` struct below in order to give inference
# some help.
mutable struct Slicer{T,N,A <: AbstractArray{T,N},F}
    current_index::Int
    concat_dim::Int
    captured_array::A
    f::F
end

function (S::Slicer{T,N})(sz) where {T,N}
    current_index = S.current_index
    range = current_index:(current_index + sz - 1)
    inds = ntuple(i -> i == S.concat_dim ? range : 1:size(S.captured_array, i), Val{N}())
    S.current_index += sz
    return S.f(view(S.captured_array, inds...))
end

#####
##### concat backprop
#####

Zygote.@adjoint function concat(A::Vector{<:Memory{T,N}}, dim::Integer) where {T,N}
    # Just capture the sizes instead of the original arguments.
    # Potentially holds on to less memory that way.
    sizes = size.(A, dim)
    y = concat(A, dim)

    # For the adjoint, construct views of Δ
    # - If Δ is trivially materializable, than this will be quick.
    # - If it isn't, well that constructing sub memories will probably be difficult anyways.
    function concat_adjoint(Δ)
        # Take views in the Julia domain.
        #
        # This may have a performance penalty if the layout of Δ is wierd ... but it's
        # likely to have a performance penalty in that case anyways ...
        Δmaterialized = materialize(Δ)
        base = 1
        f = Slicer(1, dim, materialize(Δ), memory)
        δA = map(f, sizes)
        return (δA, nothing)
    end

    return (y, concat_adjoint)
end

