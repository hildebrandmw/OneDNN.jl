# Map data types from Julia to DNNL
dnnl_type(::Type{Float16}) = Lib.dnnl_f16
dnnl_type(::Type{Float32}) = Lib.dnnl_f32
dnnl_type(::Type{Int32}) = Lib.dnnl_s32
dnnl_type(::Type{Int8}) = Lib.dnnl_s8
dnnl_type(::Type{UInt8}) = Lib.dnnl_u8

# Try to forward Numbers to their underlying type
dnnl_type(x::T) where {T<:Number} = dnnl_type(T)
dnnl_type(::Type{T}) where {T} = error("No DNNL type for type $T")
dnnl_type(::T) where {T} = error("No DNNL type for $T")

# Automatic argument conversion.
dnnl_convert(::Type{T}) where {T} = dnnl_type(T)

# Make a DIMS array
# NOTE: The OneDNN C-API expects a pointer, so we can't just pass a tuple.
# We either need to pass an array, or a Ref{Tuple}.
# Hwere, we choose to do the latter.
function dnnl_dims(x::NTuple{N,Int64}) where {N}
    # Need to reverse the order of dimensions because C/C++ is row-major while Julia
    # is column major.
    #
    # This leads to all kinds of translation schenanigans ...
    f(i) = (i <= length(x)) ? Lib.dnnl_dim_t(x[i]) : zero(Lib.dnnl_dim_t)
    return ntuple(i -> f(i), Val(Lib.DNNL_MAX_NDIMS))
end
dnnl_dims(x::NTuple{Lib.DNNL_MAX_NDIMS,Int64}) = x

dnnl_dims(x::AbstractArray) = dnnl_dims(strides(x))
dnnl_dims() = ntuple(i -> zero(Int64), Val(Lib.DNNL_MAX_NDIMS))
dnnl_dims(::Tuple{}) = dnnl_dims()

# Automatically wrap in a "Ref" when calling an API function.
# This is almost always passed by reference.
# If it's ever passed by value, then we'll have to add an escape hatch.
dnnl_convert(x::NTuple{N,Int64}) where {N} = Ref(dnnl_dims(x))

# Formats
default_strides(size::Tuple{Vararg{Int,N}}) where {N} = Base.size_to_strides(1, size...)
dnnl_format_any() = Lib.dnnl_format_tag_any

#####
##### Memory Desc
#####

const MemoryDesc = Lib.dnnl_memory_desc_t
memorydesc(x::MemoryDesc) = x
function Base.cconvert(::Type{Ptr{Lib.dnnl_memory_desc_t}}, x::MemoryDesc)
    return Base.cconvert(Ptr{Lib.dnnl_memory_desc_t}, Ref(x))
end

# Specialize on typed dimensions to make typestable
logicalsize(md::MemoryDesc) = reverse(md.dims[1:(md.ndims)])
logicalsize(md::MemoryDesc, v::Val) = reverse(ntuple(i -> md.dims[i], v))
function Base.strides(md::MemoryDesc, v::Val)
    return reverse(ntuple(i -> md.format_desc.blocking.strides[i], v))
end
Base.ndims(md::MemoryDesc) = md.ndims

function Base.show(io::IO, md::MemoryDesc)
    @nospecialize

    ndims = md.ndims
    size = logicalsize(md)
    data_type = md.data_type
    padded_dims = md.padded_dims[1:ndims]
    format_kind = md.format_kind
    padded_offsets = md.padded_offsets[1:ndims]

    # Additional information if the format is "blocked"
    extra_format = ""
    if format_kind == Lib.dnnl_blocked
        blocking_desc = md.format_desc.blocking
        num_inner_blocks = blocking_desc.inner_nblks
        extra_format = """
                strides: $(blocking_desc.strides[1:ndims])
                num inner blocks: $(num_inner_blocks)
                inner blocks: $(blocking_desc.inner_blks[1:num_inner_blocks])
                inner indexes: $(blocking_desc.inner_idxs[1:num_inner_blocks])
        """
    end

    str = """
    OneDNN Memory Description
        ndims: $ndims
        size: $size
        datatype: $data_type
        format kind: $format_kind
    $extra_format
        padded dims: $padded_dims
        padded offsets: $padded_offsets
    """
    print(io, str)
    return nothing
end

#####
##### Memory Desc constructors in all their glory!
#####

memorydesc(x::AbstractArray{T}) where {T} = memorydesc(T, size(x), strides(x))
function memorydesc(
    datatype, dims::NTuple{N,Int}, strides::NTuple{N,Int} = default_strides(dims)
) where {N}
    handle = Ref{MemoryDesc}()
    @apicall dnnl_memory_desc_init_by_strides(
        handle, N, reverse(dims), datatype, reverse(strides)
    )
    return unwrap_ref(handle)
end

# convenience creation by tag.
function memorydesc(datatype, dims::NTuple{N,Int}, tag::Lib.dnnl_format_tag_t) where {T,N}
    handle = Ref{MemoryDesc}()
    @apicall dnnl_memory_desc_init_by_tag(handle, N, reverse(dims), datatype, tag)
    return unwrap_ref(handle)
end

toany(a::MemoryDesc) = memorydesc(a.data_type, logicalsize(a), dnnl_format_any())
isany(a::MemoryDesc) = a.format_kind == Lib.dnnl_format_kind_any

function Base.:(==)(a::MaybeRef{MemoryDesc}, b::MaybeRef{MemoryDesc})
    return Bool(Lib.dnnl_memory_desc_equal(wrap_ref(a), wrap_ref(b)))
end

getbytes(a::MaybeRef{MemoryDesc}) = Lib.dnnl_memory_desc_get_size(wrap_ref(a))

#####
##### Memory
#####

# Bridge into TiledArrays
vlayout(x) = Val(layout(x))
layout(::Val{N}) where {N} = ntuple(identity, Val(N))
layout(::Array{T,N}) where {T,N} = layout(Val(N))
layout(x::LinearAlgebra.Transpose) = reverse(layout(parent(x)))

# Often, we don't want to specialize on the layout of the array, instead allowing oneDNN
# to work on its own
struct Opaque end

mutable struct Memory{L,T,N,A<:AbstractArray{T}} <: AbstractArray{T,N}
    # The underlying array that is supplying the data.
    array::A
    offset::Int

    # Derived `Memory` objects will have their dimension stripped away since this information
    # is not maintained in a type stable manner.
    #
    # Furthermore, depending on the format, the size of A could actually be bigger than
    # the product of the logical dimensions.
    #
    # Instead, we maintain their logical dims separately.
    logicalsize::NTuple{N,Int}

    # Memory object from DNNL
    memory::Lib.dnnl_memory_t

    # Inner constructor to ensure finalizers.
    function Memory{L}(
        array::A, offset, dims::NTuple{N,Int}, memory::Lib.dnnl_memory_t
    ) where {L,T,N,A<:AbstractArray{T}}
        object = new{L,T,N,A}(array, convert(Int64, offset), dims, memory)
        finalizer(object) do obj
            Lib.dnnl_memory_destroy(obj.memory)
        end
        return object
    end
end
layout(::Memory{L}) where {L} = L
layout(x::Opaque) = x

# TODO: This is kind of a sloppy definition ...
function Base.convert(::Type{Memory{L,T,N,A}}, x::Memory{L,T,N,B}) where {L,T,N,A,B}
    _array = x.array
    return Memory{L}(
        convert(A, _array), x.offset, x.logicalsize, creatememory(_array, memorydesc(x))
    )
end

toany(x::Memory) = toany(memorydesc(x))

logicalsize(x::Memory) = size(x)
Base.strides(x::Memory{L,T,N}) where {L,T,N} = strides(memorydesc(x), Val(N))

Base.parent(x::Memory) = x.array
arraytype(::Memory{L,T,N,A}) where {L,T,N,A} = A

Base.show(io::IO, x::Memory{Opaque}) = print(io, "Opaque Memory $(logicalsize(x))")
Base.show(io::IO, ::MIME"text/plain", x::Memory{Opaque}) = show(io, x)

# for creating OneDNN arguments
@inline access_pointer(x, offset, context) = pointer(x, offset)
function setptr!(x::Memory{<:Any,T}, context::AccessContext = Reading()) where {T}
    ptr = access_pointer(x.array, x.offset, context)
    @apicall dnnl_memory_set_data_handle_v2(x.memory, ptr, global_stream())
end

function Base.cconvert(::Type{Lib.dnnl_memory_t}, x::Memory)
    setptr!(x)
    return x.memory
end

function Base.cconvert(::Type{Ptr{Lib.dnnl_memory_t}}, x::Memory)
    setptr!(x)
    return Ptr{Lib.dnnl_memory_t}(Base.pointer_from_objref(x) + fieldoffset(typeof(x), 3))
end
Base.cconvert(::Type{Ptr{Lib.dnnl_memory_desc_t}}, x::Memory) = memorydesc_ptr(x)

# For constructing DNNL arguments.
function dnnl_exec_arg(x::Memory, context::AccessContext = Reading())
    setptr!(x, context)
    return x.memory
end

# Try to remove as many layers of wrapping around `A` as possible.
# Since all of the dimension and layout information will be stored in the OneDNN
# `memorydesc`, we don't need to hold onto it on the Julia level, which can potentially
# cause down-stream type instabilities.
function Memory(A::AbstractArray)
    return Memory{Opaque}(ancestor(A), _offset(A), size(A), creatememory(A))
end

_offset(::AbstractArray) = one(Int64)
_offset(x::Base.SubArray) = Base.first_index(x)

Memory(M::Memory) = M

function ChainRulesCore.rrule(::Type{<:Memory}, x)
    return (Memory(x), Δ -> (ChainRulesCore.NoTangent(), Δ))
end

# Get to the ultimate parent.
ancestor(x::Array) = x
ancestor(x) = ancestor(parent(x))

function creatememory(A::AbstractArray, desc = memorydesc(A))
    memory = Ref{Lib.dnnl_memory_t}()
    @apicall dnnl_memory_create(memory, desc, global_engine(), A)
    return memory[]
end

function typed(M::Memory{Opaque})
    A = parent(M)
    desc = memorydesc(M)
    return Memory{layout(desc)}(A, M.offset, size(M), creatememory(A, desc))
end

# Convenience method for creating destination memories from a source memory.
Base.size(M::Memory) = M.logicalsize
Base.eltype(M::Memory{L,T}) where {L,T} = T

striptiles(_::TiledArrays.Tile, x...) = striptiles(x...)
striptiles(x::Int, y...) = (x, y...)

Base.@propagate_inbounds function Base.getindex(
    M::Memory{L,T,N}, I::Vararg{Int,N}
) where {L,T,N}
    @boundscheck checkbounds(M, I...)
    # TODO: this is wrong
    _strides = ntuple(i -> strides(M)[invperm(striptiles(L...))[i]], Val(N))
    return getindex(M.array, M.offset + TiledArrays.getoffset(Val(L), size(M), I, _strides))
end

function Base.getindex(::Memory{Opaque}, I::Vararg{Int,N}) where {N}
    return error("Cannot index opaque memory formats")
end

Base.@propagate_inbounds function Base.setindex!(
    M::Memory{L,T,N}, v, I::Vararg{Int,N}
) where {L,T,N}
    @boundscheck checkbounds(M, I...)
    _strides = ntuple(i -> strides(M)[invperm(striptiles(L...))[i]], Val(N))
    return setindex!(
        M.array, v, M.offset + TiledArrays.getoffset(Val(L), size(M), I, _strides)
    )
end

function Base.setindex!(::Memory{Opaque}, v, I::Vararg{Int,N}) where {N}
    return error("Cannot index opaque memory formats")
end

memorydesc(M::Memory) = unsafe_load(memorydesc_ptr(M))
function memorydesc_ptr(M::Memory)
    md = Ref{Ptr{MemoryDesc}}()
    @apicall Lib.dnnl_memory_get_memory_desc(M.memory, md)
    return unwrap_ref(md)
end

#####
##### Lazy Transpose
#####

# General idea: swap the dims and strides.
# TODO: Need to validate that this is a blocked layout with no tiling ...
function Base.adjoint(M::Memory{Opaque,T,2}) where {T}
    dims = size(M)
    strides = Base.strides(memorydesc(M), Val(2))

    reversed_dims = reverse(dims)
    desc = memorydesc(T, reversed_dims, reverse(strides))
    memory = creatememory(parent(M), desc)
    return Memory{Opaque}(parent(M), M.offset, reversed_dims, memory)
end

function Base.permutedims(M::Memory{Opaque,T,N}, perm::NTuple{N,Int}) where {T,N}
    dims = size(M)
    strides = Base.strides(memorydesc(M), Val(N))
    dims_permuted = unsafe_permute(dims, perm)
    strides_permuted = unsafe_permute(strides, perm)

    desc = memorydesc(T, dims_permuted, strides_permuted)
    memory = creatememory(parent(M), desc)
    return Memory{Opaque}(parent(M), M.offset, dims_permuted, memory)
end

function unsafe_permute(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N}
    return ntuple(i -> @inbounds(a[@inbounds b[i]]), Val(N))
end

#####
##### Construct more memories!!
#####

function Base.similar(
    M::Memory,
    ::Type{T} = eltype(M),
    dims::NTuple{N,Int} = size(M),
    desc::MemoryDesc = memorydesc(M),
    # Final argument not type stable in the general case since MemoryDesc are opaque.
    # However, provide this argument so Ops and create type-stable copies.
    format::Union{Type{Opaque},Val} = layout(M),
) where {T,N}
    # Number of bytes to allocate.
    # For now, we just do this to check that nothing funny is going on.
    # Otherwise, we will have to get a little creative for the memory formats that require
    # slightly more than normal storage space.
    expected = Int(getbytes(desc))
    if format !== Opaque
        nelements = TiledArrays.fullsize(format, dims)
        @assert sizeof(T) * nelements == expected
    end

    # Allocate the output array.
    # This will be allocated as just a plain vector with dimensions padded with ones so it
    # has the same dimension as the wrapped "Memory"
    padded_dims = (div(expected, sizeof(T)), ntuple(_ -> 1, Val(N - 1))...)
    out = similar(M.array, T, padded_dims)

    # Since we specifically created this array, the offset will always start atone.
    return Memory{_rm_val(format)}(out, 1, dims, creatememory(out, desc))
end

_rm_val(x) = x
_rm_val(::Val{T}) where {T} = T

materialize(x::AbstractArray, args...; kw...) = x
function materialize(
    M::Memory{L,T,N}, format::Val = vlayout(Val(N)); allowreorder = true
) where {L,T,N}
    # Check if this memory is already in the requested layout.
    # If so, return the underlying array.
    desired_strides = TiledArrays.dimstrides(format, logicalsize(M))
    actual_strides = strides(M)

    # In order to return the underlying object, we need to ensure that:
    # 1. The length of the wrapped object is the same as the length of the Memory.
    # This helps handle views correctly.
    #
    # 2. Strides are the same[
    if length(parent(M)) == length(M) && desired_strides == actual_strides
        return reshape(parent(M), logicalsize(M))
    end

    if !allowreorder
        msg = """
        Expected strides: $desired_strides.
        Found strides: $actual_strides.
        """
        throw(ArgumentError(msg))
    end

    desc = memorydesc(T, logicalsize(M), desired_strides)
    return reshape(parent(reorder(desc, M)), logicalsize(M))
end

function ChainRulesCore.rrule(
    ::typeof(materialize), x, args::Vararg{Any,N}; kw...
) where {N}
    return materialize(x, args...; kw...),
    Δ -> (ChainRulesCore.NoTangent(), Δ, ntuple(_ -> ChainRulesCore.NoTangent()(), Val(N)))
end
