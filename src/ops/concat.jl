struct Concat{O,A,N}
    primitive::Primitive
    output_descriptor::MemoryDesc
    # Keep track of the expected number of arguments for validation.
    output_size::NTuple{N,Int}
    nargs::Int
    argbuffer::Arguments{Vector{Lib.dnnl_exec_arg_t}}
end

function Concat(A::Vector{T}, dim) where {T<:MemoryOrDesc}
    nargs = length(A)
    outputdim = sum(a -> logicalsize(a)[dim], A)

    a = first(A)
    lsize = logicalsize(a)
    vndims = Val(ndims(a))
    outputdim = sum(a -> size(a, dim), A)
    outsize = ntuple(i -> (i == dim) ? outputdim : lsize[i], vndims)

    # Let OneDNN choose the best output memory format.
    desc_any = memorydesc(eltype(a), outsize, dnnl_format_any())
    primitive_descriptor = PrimitiveDescriptor(
        Lib.dnnl_concat_primitive_desc_create,
        Ref(desc_any),
        nargs,
        dim - 1,  # convert from index-1 to index-0
        map(memorydesc, A),
        noattributes(),
        global_engine(),
    )

    primitive = Primitive(primitive_descriptor)

    # Query output format
    output_descriptor_ptr = Lib.dnnl_primitive_desc_query_md(
        primitive_descriptor, Lib.dnnl_query_dst_md, 0
    )
    output_descriptor = unsafe_load(output_descriptor_ptr)

    return Concat{layout(output_descriptor),layout(a),length(outsize)}(
        primitive,
        output_descriptor,
        outsize,
        nargs,
        # Create Arguments vector to be one longer than the number of source arguments
        # to account for the destination argument.
        Arguments(Vector{Lib.dnnl_exec_arg_t}(undef, nargs + 1)),
    )
end

function (op::Concat{O,A})(src::AbstractVector{<:Memory{A}}) where {O,A}
    _src = first(src)
    dst = similar(_src, eltype(eltype(src)), op.output_size, op.output_descriptor, Val(O))
    return op(dst, src)
end

function (op::Concat{O,A})(dst::Memory{O}, src::AbstractVector{<:Memory{A}}) where {O,A}
    # Use the existing argument buffer in `op`.
    # NOTE: This doesn't allow for concurrent execution, but that should be okay.
    argbuffer = op.argbuffer
    resize!(argbuffer, length(src) + 1)
    for (i, a) in enumerate(src)
        argbuffer[i] = dnnl_arg(Lib.DNNL_ARG_MULTIPLE_SRC + i - 1, a)
    end
    argbuffer[end] = dnnl_arg(Lib.DNNL_ARG_DST, dst)
    execute!(op.primitive, argbuffer)
    return dst
end

# function concat(A::Vector{<:Memory}, dim) where {N}
#     # Compute the size of the output
#     # The prep code is largely based on the code in Julia's `_typed_vcat`.
#     nargs = length(A)
#     outputdim = sum(a -> size(a, dim), A)::Int
#     outsize = ntuple(i -> i == dim ? outputdim : size(A[1], i), ndims(first(A)))
#
#     # use `dnnl_format_any` and let the primitive descriptor decide which output format
#     # to use.
#     dst_desc = memorydesc(eltype(A[1]), outsize, dnnl_format_any())
#
#     # primitive descriptor
#     pd = primitive_descriptor(
#         Lib.dnnl_concat_primitive_desc_create,
#         Ref(dst_desc),
#         length(A),
#         ndims(A[1]) - dim,
#         memorydesc.(A),
#         Ptr{Nothing}(),
#         global_engine(),
#     )
#
#      # Now that we've created the primitive, we need to query it to get the proper output
#      # memory descriptor.
#      dst_desc = Lib.dnnl_primitive_desc_query_md(
#          pd,
#          Lib.dnnl_query_dst_md,
#          0
#      )
#
#     dst = similar(A[1], eltype(A[1]), outsize, unsafe_load(dst_desc))
#
#     # primitive
#     args = [arg(Lib.DNNL_ARG_MULTIPLE_SRC + i - 1, A[i]) for i in 1:length(A)]
#     push!(args, arg(Lib.DNNL_ARG_DST, dst))
#
#     p = primitive(pd)
#     execute!(p, args)
#
#     # cleanup
#     destroy(p, pd)
#     return dst
# end
#
# #####
# ##### Slicing
# #####
#
# # map` seems to be having Julia issue #15276 is problems when keeping track of where
# # we are indexing to create views.
# #
# # As such, we have to build this `Slicer` struct below in order to give inference
# # some help.
# mutable struct Slicer{T,N,A <: AbstractArray{T,N},F}
#     current_index::Int
#     concat_dim::Int
#     captured_array::A
#     f::F
# end
#
# function (S::Slicer{T,N})(sz) where {T,N}
#     current_index = S.current_index
#     range = current_index:(current_index + sz - 1)
#     inds = ntuple(i -> i == S.concat_dim ? range : 1:size(S.captured_array, i), Val{N}())
#     S.current_index += sz
#     return S.f(view(S.captured_array, inds...))
# end
#
# #####
# ##### concat backprop
# #####
#
# Zygote.@adjoint function concat(A::Vector{<:Memory{T,N}}, dim::Integer) where {T,N}
#     # Just capture the sizes instead of the original arguments.
#     # Potentially holds on to less memory that way.
#     sizes = size.(A, dim)
#     y = concat(A, dim)
#
#     # For the adjoint, construct views of Δ
#     # - If Δ is trivially materializable, than this will be quick.
#     # - If it isn't, well that constructing sub memories will probably be difficult anyways.
#     function concat_adjoint(Δ)
#         # Take views in the Julia domain.
#         #
#         # This may have a performance penalty if the layout of Δ is wierd ... but it's
#         # likely to have a performance penalty in that case anyways ...
#         Δmaterialized = materialize(Δ)
#         base = 1
#         f = Slicer(1, dim, materialize(Δ), memory)
#         δA = map(f, sizes)
#         return (δA, nothing)
#     end
#
#     return (y, concat_adjoint)
# end
#
