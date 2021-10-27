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
dnnl_convert(x::NTuple{N,MemoryDesc}) where {N} = Ref(x)

memorydesc(x::MemoryDesc) = x
function Base.cconvert(::Type{Ptr{Lib.dnnl_memory_desc_t}}, x::MemoryDesc)
    return Base.cconvert(Ptr{Lib.dnnl_memory_desc_t}, Ref(x))
end

# Specialize on typed dimensions to make typestable
logicalsize(md::MemoryDesc) = reverse(md.dims[1:min(12, md.ndims)])
logicalsize(md::MemoryDesc, v::Val) = reverse(ntuple(i -> md.dims[i], v))
function Base.strides(md::MemoryDesc, v::Val)
    return reverse(ntuple(i -> md.format_desc.strides[i], v))
end

function padded_size(md::MemoryDesc, v::Val)
    return reverse(ntuple(i -> md.padded_dims[i], v))
end
Base.ndims(md::MemoryDesc) = md.ndims

function Base.show(io::IO, md::MemoryDesc)
    @nospecialize

    # Perform the "min" check to handle corner cases where the memory layout is too exotic
    # for this simple showing method.
    ndims = min(md.ndims, 12)
    size = logicalsize(md)
    data_type = md.data_type
    padded_dims = reverse(md.padded_dims[1:ndims])
    format_kind = md.format_kind
    padded_offsets = md.padded_offsets[1:ndims]

    # Additional information if the format is "blocked"
    extra_format = ""
    if format_kind == Lib.dnnl_blocked
        #blocking_desc = md.format_desc.blocking
        blocking_desc = md.format_desc
        num_inner_blocks = min(12, blocking_desc.inner_nblks)
        extra_format = """
                strides: $(blocking_desc.strides[1:ndims])
                num inner blocks: $(num_inner_blocks)
                inner blocks: $(reverse(blocking_desc.inner_blks[1:num_inner_blocks]))
                inner indexes: $(reverse(blocking_desc.inner_idxs[1:num_inner_blocks]))
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
function memorydesc(
    datatype, dims::NTuple{N,Int}, tag::Union{Lib.dnnl_format_tag_t,UInt32}
) where {T,N}
    handle = Ref{MemoryDesc}()
    @apicall dnnl_memory_desc_init_by_tag(handle, N, reverse(dims), datatype, tag)
    return unwrap_ref(handle)
end

toany(a::MemoryDesc) = memorydesc(a.data_type, logicalsize(a), dnnl_format_any())

isany(a::Ptr{MemoryDesc}) = isany(unsafe_load(a))
isany(a::MemoryDesc) = a.format_kind == Lib.dnnl_format_kind_any

function Base.:(==)(a::MaybeRef{MemoryDesc}, b::MaybeRef{MemoryDesc})
    return Bool(Lib.dnnl_memory_desc_equal(wrap_ref(a), wrap_ref(b)))
end

getbytes(a::MaybeRef{MemoryDesc}) = signed(Lib.dnnl_memory_desc_get_size(wrap_ref(a)))

#####
##### Memory
#####

# Often, we don't want to specialize on the layout of the array, instead allowing oneDNN
# to work on its own
struct Memory{T,N,A<:AbstractArray{T}} <: AbstractArray{T,N}
    # The underlying array that is supplying the data.
    array::A
    offset::Int

    # Keep around some information about size and padding.
    logicalsize::NTuple{N,Int}

    # Memory object from DNNL
    memory::MemoryPtr
end

function Memory(
    array::A, offset, dims::NTuple{N,Int}, memory::MemoryPtr
) where {T,N,A<:AbstractArray{T}}
    return Memory{T,N,A}(array, convert(Int64, offset), dims, memory)
end

function Base.convert(::Type{Memory{T,N,A}}, x::Memory{T,N,B}) where {T,N,A,B}
    return Memory(convert(A, x.array), x.offset, x.logicalsize, x.memory)
end

Base.sizeof(x::Memory) = getbytes(memorydesc_ptr(x))
toany(x::Memory) = toany(memorydesc(x))

logicalsize(x::Memory) = size(x)
Base.strides(x::Memory{T,N}) where {T,N} = strides(memorydesc(x), Val(N))
padded_size(x::Memory{T,N}) where {T,N} = padded_size(memorydesc(x), Val(N))

Base.parent(x::Memory) = x.array
function ChainRulesCore.rrule(::typeof(Base.parent), x::Memory)
    return parent(x), Δ -> (ChainRulesCore.NoTangent(), Δ)
end

arraytype(::Memory{T,N,A}) where {T,N,A} = A

function Base.show(io::IO, x::Memory)
    print(io, "Opaque Memory $(logicalsize(x))")
    x.offset != 1 && print(io, " - SubArray")
end
Base.show(io::IO, ::MIME"text/plain", x::Memory) = show(io, x)

# for creating OneDNN arguments
@inline access_pointer(x, offset, context) = pointer(x, offset)
function setptr!(x::Memory{T}, context::AccessContext = Reading()) where {T}
    ptr = access_pointer(x.array, x.offset, context)
    @apicall dnnl_memory_set_data_handle_v2(x.memory, ptr, global_stream())
end

function Base.cconvert(
    ::Type{T}, x::Memory
) where {T<:Union{Lib.dnnl_memory_t,Ptr{Lib.dnnl_memory_t}}}
    setptr!(x)
    return Base.cconvert(T, x.memory)
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
    return Memory(ancestor(A), _offset(A), size(A), MemoryPtr(A))
end

_offset(::AbstractArray) = one(Int64)
_offset(x::Base.SubArray) = Base.first_index(x)

Memory(M::Memory) = M

function ChainRulesCore.rrule(::Type{<:Memory}, x)
    return (Memory(x), Δ -> (ChainRulesCore.NoTangent(), Δ))
end

# Convenience method for creating destination memories from a source memory.
Base.size(M::Memory) = M.logicalsize
Base.eltype(M::Memory{T}) where {T} = T

function Base.getindex(::Memory, I::Vararg{Int,N}) where {N}
    return error("Cannot index opaque memory formats")
end

function Base.setindex!(::Memory, v, I::Vararg{Int,N}) where {N}
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
function Base.adjoint(M::Memory{T,2}) where {T}
    dims = size(M)
    strides = Base.strides(memorydesc(M), Val(2))

    reversed_dims = reverse(dims)
    desc = memorydesc(T, reversed_dims, reverse(strides))
    memory = MemoryPtr(parent(M), desc)
    return Memory(parent(M), M.offset, reversed_dims, memory)
end

function Base.permutedims(M::Memory{T,N}, perm::NTuple{N,Int}) where {T,N}
    dims = size(M)
    strides = Base.strides(memorydesc(M), Val(N))
    dims_permuted = unsafe_permute(dims, perm)
    strides_permuted = unsafe_permute(strides, perm)

    desc = memorydesc(T, dims_permuted, strides_permuted)
    memory = MemoryPtr(parent(M), desc)
    return Memory(parent(M), M.offset, dims_permuted, memory)
end

function unsafe_permute(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N}
    return ntuple(i -> @inbounds(a[@inbounds b[i]]), Val(N))
end

#####
##### Construct more memories!!
#####

function Base.similar(
    x::Memory{U,M},
    ::Type{T} = eltype(x),
    dims::NTuple{N,Int} = size(x),
    desc::MemoryDesc = (M == N && U === T) ? memorydesc(x) : memorydesc(T, dims),
) where {U,T,M,N}
    # Number of bytes to allocate.
    # Since OneDNN is free to reorder and pad, we need to explicitly ask it.
    bytes = getbytes(desc)

    # Allocate the output array.
    # This will be allocated as just a plain vector with dimensions padded with ones so it
    # has the same dimension as the wrapped "Memory"
    padded_dims = (div(bytes, sizeof(T)), ntuple(_ -> 1, Val(N - 1))...)
    out = similar(x.array, T, padded_dims)

    # Since we specifically created this array, the offset will always start atone.
    return Memory(out, 1, dims, MemoryPtr(out, desc))
end

materialize(x::AbstractArray, args...; kw...) = x
function materialize(M::Memory{T,N}; allowreorder = true) where {T,N}
    # Check if this memory is already in the requested layout.
    # If so, return the underlying array.
    desired_strides = default_strides(logicalsize(M))
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
    Δ -> (ChainRulesCore.NoTangent(), Δ, ntuple(_ -> ChainRulesCore.NoTangent(), Val(N)))
end

#####
##### Reshape
#####

function Base.reshape(memory::Memory{T}, dims::NTuple{N,Int}) where {T,N}
    md = Ref{MemoryDesc}()
    @apicall dnnl_memory_desc_reshape(md, memory, N, Ref(reverse(dims)))
    new_memory = MemoryPtr(parent(memory), md)
    return Memory(parent(memory), memory.offset, dims, new_memory)
end

#####
##### Bridge to TiledArrays
#####

function generate_linear_indices(memory::Memory{T,N}) where {T,N}
    md = OneDNN.memorydesc(memory)
    _layout = layout(md)
    _size = logicalsize(md, Val(N))
    _padded_size = padded_size(md, Val(N))
    return generate_linear_indices(_layout, _size, _padded_size)
end
