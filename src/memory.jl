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
logicalsize(md::MemoryDesc) = md.dims[1:(md.ndims)]
logicalsize(md::MemoryDesc, v::Val) = ntuple(i -> md.dims[i], v)
Base.strides(md::MemoryDesc, v::Val) = ntuple(i -> md.format_desc.blocking.strides[i], v)
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
    @apicall dnnl_memory_desc_init_by_strides(handle, N, dims, datatype, strides)
    return unwrap_ref(handle)
end

# convenience creation by tag.
function memorydesc(datatype, dims::NTuple{N,Int}, tag::Lib.dnnl_format_tag_t) where {T,N}
    handle = Ref{MemoryDesc}()
    @apicall dnnl_memory_desc_init_by_tag(handle, N, dims, datatype, tag)
    return unwrap_ref(handle)
end

toany(a::MemoryDesc) = memorydesc(a.data_type, logicalsize(a), dnnl_format_any())

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
        A::U, dims::NTuple{N,Int}, memory::Lib.dnnl_memory_t
    ) where {L,T,U<:AbstractArray{T},N}
        object = new{L,T,N,U}(A, dims, memory)
        finalizer(object) do obj
            Lib.dnnl_memory_destroy(obj.memory)
        end
        return object
    end
end
layout(::Memory{L}) where {L} = L
layout(x::Opaque) = x

logicalsize(x::Memory) = size(x)
Base.parent(x::Memory) = x.array

Base.show(io::IO, x::Memory{Opaque}) = print(io, "Opaque Memory $(logicalsize(x))")
Base.show(io::IO, ::MIME"text/plain", x::Memory{Opaque}) = show(io, x)

# for creating OneDNN arguments
Base.cconvert(::Type{Lib.dnnl_memory_t}, x::Memory) = x.memory
function Base.cconvert(::Type{Ptr{Lib.dnnl_memory_t}}, x::Memory)
    return Ptr{Lib.dnnl_memory_t}(Base.pointer_from_objref(x) + fieldoffset(typeof(x), 3))
end
Base.cconvert(::Type{Ptr{Lib.dnnl_memory_desc_t}}, x::Memory) = memorydesc_ptr(x)

# For constructing DNNL arguments.
dnnl_exec_arg(x::Memory) = x.memory

# Try to remove as many layers of wrapping around `A` as possible.
# Since all of the dimension and layout information will be stored in the OneDNN
# `memorydesc`, we don't need to hold onto it on the Julia level, which can potentially
# cause down-stream type instabilities.
Memory(A::AbstractArray) = Memory{Opaque}(A, size(A), creatememory(A))
Memory(M::Memory) = M

function creatememory(A::AbstractArray, desc = memorydesc(A))
    memory = Ref{Lib.dnnl_memory_t}()
    @apicall dnnl_memory_create(memory, desc, global_engine(), A)
    return memory[]
end

function typed(M::Memory{Opaque})
    A = parent(M)
    desc = memorydesc(M)
    return Memory{layout(desc)}(A, size(M), creatememory(A, desc))
end

# Convenience method for creating destination memories from a source memory.
Base.size(M::Memory) = M.logicalsize
Base.eltype(M::Memory{L,T}) where {L,T} = T

Base.@propagate_inbounds function Base.getindex(
    M::Memory{L,T,N}, I::Vararg{Int,N}
) where {L,T,N}
    @boundscheck checkbounds(M, I...)
    return getindex(M.array, TiledArrays.getoffset(Val(L), size(M), I) + 1)
end

function Base.getindex(::Memory{Opaque}, I::Vararg{Int,N}) where {N}
    return error("Cannot index opaque memory formats")
end

Base.@propagate_inbounds function Base.setindex!(
    M::Memory{L,T,N}, v, I::Vararg{Int,N}
) where {L,T,N}
    @boundscheck checkbounds(M, I...)
    return setindex!(M.array, v, TiledArrays.getoffset(Val(L), size(M), I) + 1)
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
    # This will be allocated as just a plain vector.
    out = similar(M.array, T, div(expected, sizeof(T)))
    return Memory{_rm_val(format)}(out, dims, creatememory(out, desc))
end

_rm_val(x) = x
_rm_val(::Val{T}) where {T} = T

function materialize(
    M::Memory{L,T,N}, format::Val = vlayout(Val(N)), allowreorder = true
) where {L,T,N}
    # Check if this memory is already in the requested layout.
    # If so, return the underlying array.
    desired_strides = TiledArrays.dimstrides(format, logicalsize(M))
    actual_strides = strides(memorydesc(M), Val(N))
    if desired_strides == actual_strides
        return reshape(parent(M), logicalsize(M))
    end

    if !allowreorder
        msg = """
        Expected strides: $desired_strides.
        Found strides: $actual_strides.
        """
    end

    desc = memorydesc(T, logicalsize(M), desired_strides)
    return reshape(parent(reorder(desc, M)), logicalsize(M))
end
