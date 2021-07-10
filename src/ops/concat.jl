function concat(A::Vector{T}, dim::Integer) where {T<:Memory}
    nargs = length(A)
    front = first(A)
    dims = size(front)
    vndims = Val(ndims(front))
    outputdim = sum(i -> size(i, dim), A)
    outsize = ntuple(i -> (i == dim) ? outputdim : dims[i], vndims)
    desc_any = memorydesc(eltype(front), outsize, dnnl_format_any())
    return temp_primitive(
        Lib.dnnl_concat_primitive_desc_create,
        desc_any,
        nargs,
        ndims(front) - dim,
        map(memorydesc, A),
        noattributes(),
        global_engine(),
    ) do primitive, primitive_descriptor
        dst_desc = query_md(primitive_descriptor, Lib.dnnl_query_dst_md)
        dst = similar(front, eltype(front), outsize, dst_desc)
        argbuffer = Arguments(Vector{Lib.dnnl_exec_arg_t}(undef, nargs + 1))
        for (i, _a) in enumerate(A)
            argbuffer[i] = dnnl_arg(Lib.DNNL_ARG_MULTIPLE_SRC + i - 1, _a)
        end
        argbuffer[end] = dnnl_arg(Lib.DNNL_ARG_DST, dst)
        execute!(primitive, argbuffer)
        return dst
    end
end

#####
##### Slicing
#####

# TODO: See if the #15276 style problems still exist in Julia 1.6.
#
# map` seems to be having Julia issue #15276 is problems when keeping track of where
# we are indexing to create views.
#
# As such, we have to build this `Slicer` struct below in order to give inference
# some help.
mutable struct Slicer{T,N,A <: AbstractArray{T,N}}
    current_index::Int
    concat_dim::Int
    captured_array::A
end

function (S::Slicer{T,N})(sz) where {T,N}
    current_index = S.current_index
    range = current_index:(current_index + sz - 1)
    inds = ntuple(i -> i == S.concat_dim ? range : 1:size(S.captured_array, i), Val{N}())
    S.current_index += sz
    return view(S.captured_array, inds...)
end

#####
##### concat backprop
#####

function ChainRulesCore.rrule(::typeof(concat), A::Vector{T}, dim::Integer) where {T <: Memory}
    # Capture sizes for reconstruction.
    lengths = size.(A, dim)
    dst = concat(A, dim)

    function concat_pullback(Δ)
        f = Slicer(1, dim, materialize(Δ))
        δA = map(f, lengths)
        return (ChainRulesCore.NoTangent(), δA, ChainRulesCore.NoTangent())
    end
    return dst, concat_pullback
end

