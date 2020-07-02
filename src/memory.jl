# Map data types from Julia to DNNL
dnnl_type(::Type{Float16}) = Lib.dnnl_f16
dnnl_type(::Type{Float32}) = Lib.dnnl_f32
dnnl_type(::Type{Int32}) = Lib.dnnl_s32
dnnl_type(::Type{Int8}) = Lib.dnnl_s8
dnnl_type(::Type{UInt8}) = Lib.dnnl_u8

# Try to forward Numbers to their underlying type
dnnl_type(x::T) where {T <: Number} = dnnl_type(T)
dnnl_type(::Type{T}) where {T} = error("No DNNL type for type $T")
dnnl_type(::T) where {T} = error("No DNNL type for $T")

# Make a DIMS array
#
# These arrays are constant size in DNNL, but are apparently passed around as pointers,
# so we just use normal Julia Arrays on this side, let `cconvert` convert them to pointers
# for all the `ccall` operations and profit!
function dnnl_dims(x::NTuple{N,Int64}) where {N}
    # Need to reverse the order of dimensions because C/C++ is row-major while Julia
    # is column major.
    f(i) = (i <= length(x)) ? Lib.dnnl_dim_t(x[N+1-i]) : zero(Lib.dnnl_dim_t)
    return [f(i) for i in 1:Lib.DNNL_MAX_NDIMS]
end

dnnl_dims() = zeros(Lib.dnnl_dim_t, Lib.DNNL_MAX_NDIMS)
dnnl_dims(::Tuple{}) = dnnl_dims()

dnnl_format_any() = Lib.dnnl_format_tag_any

# Since Julia is column major instead of row major, we have to permute the last two
# dimensions in the format tag by default.
dnnl_format(::Val{1}) = Lib.dnnl_a
dnnl_format(::Val{2}) = Lib.dnnl_ab
dnnl_format(::Val{3}) = Lib.dnnl_abc
dnnl_format(::Val{4}) = Lib.dnnl_abcd

dnnl_format(x::AbstractArray{T,N}) where {T,N} = dnnl_format(Val{N}())

#####
##### Memory Desc
#####

const MemoryDesc = Lib.dnnl_memory_desc_t
MemoryDesc(x::MemoryDesc) = x

logicalsize(md::MemoryDesc) = md.dims[1:md.ndims]

# Specialize on typed dimensions to make typestable
logicalsize(md::MemoryDesc, v::Val) = ntuple(i -> md.dims[i], v)

function Base.show(io::IO, md::MemoryDesc)
    ndims = md.ndims
    size = logicalsize(md)
    data_type = md.data_type
    padded_dims = md.padded_dims[1:ndims]
    format_kind = md.format_kind
    padded_offsets = md.padded_offsets[1:ndims]

    str = """
    OneDNN Memory Description
        ndims: $ndims
        size: $size
        datatype: $data_type
        format kind: $format_kind

        padded dims: $padded_dims
        padded_offsets: $padded_offsets
    """
    println(io, str)
    return nothing
end

#####
##### Memory Desc constructors in all their glory!
#####

memorydesc(x::AbstractArray{T}) where {T} = memorydesc(T, size(x), dnnl_format(x))

memorydesc(x::LinearAlgebra.Transpose) = memorydesc(eltype(x), size(x), Lib.dnnl_ba)

# batched transpose
function memorydesc(x::Base.PermutedDimsArray{T,3,(2,1,3)}) where {T}
    return memorydesc(T, size(x), Lib.dnnl_acb)
end

function memorydesc(
        ::Type{T},
        dims::NTuple{N,Int},
        format = dnnl_format(Val{N}())
    ) where {T,N}

    handle = Ref{MemoryDesc}()
    @apicall Lib.dnnl_memory_desc_init_by_tag(
        handle,
        N,
        dnnl_dims(dims),
        dnnl_type(T),
        format,
    )

    return MemoryDesc(handle[])
end

Base.:(==)(a::Ref{MemoryDesc}, b::Ref{MemoryDesc}) = Bool(Lib.dnnl_memory_desc_equal(a, b))
Base.:(==)(a::MemoryDesc, b::MemoryDesc) = (Ref(a) == Ref(b))

getbytes(a::Ref{MemoryDesc}) = Lib.dnnl_memory_desc_get_size(a)
getbytes(a::MemoryDesc) = getbytes(Ref(a))

#####
##### Memory
#####

# As always, make this mutable so we can finalize the C pointers we are holding onto.
mutable struct Memory{T,N,A <: AbstractArray{T}} <: AbstractArray{T,N}
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

    function Memory(
            A::U,
            dims::NTuple{N,Int},
            memory::Lib.dnnl_memory_t
           ) where {T, U <: AbstractArray{T},N}

        object = new{T,N,U}(A, dims, memory)
        finalizer(object) do obj
            Lib.dnnl_memory_destroy(obj.memory)
        end
        return object
    end
end

memory(A::AbstractArray) = Memory(A, size(A), creatememory(A))
memory(M::Memory, x...) = M

function creatememory(A::AbstractArray, desc::Lib.dnnl_memory_desc_t = memorydesc(A))
    handle = Ref{Lib.dnnl_memory_t}()
    @apicall Lib.dnnl_memory_create(
        handle,
        Ref(desc),
        GLOBAL_ENGINE[].handle,
        convert(Ptr{Cvoid}, pointer(A))
    )
    return handle[]
end

# Convenience method for creating destionation memories from a source memory.
Base.size(M::Memory) = M.logicalsize
Base.eltype(M::Memory{T}) where {T} = T
Base.getindex(M::Memory, i::Int) = error("Cannot `getindex` directly from a Memory")
Base.setindex!(M::Memory, v, i::Int) = error("Cannot `setindex!` directly to a Memory")

Base.display(M::Memory) = show(stdout, M)
Base.show(io::IO, M::Memory{T}) where {T} = print(io, "OneDNN.Memory{$T} with logical size: $(size(M))")

val_ndims(M::Memory{A,N}) where {A,N} = Val{N}()
Base.ndims(M::Memory{A,N}) where {A,N} = N

memorydesc(M::Memory) = unsafe_load(memorydesc_ptr(M))
function memorydesc_ptr(M::Memory)
    md = Ref{Ptr{MemoryDesc}}()
    @apicall Lib.dnnl_memory_get_memory_desc(M.memory, md)

    # Unsafe load is a little sketchy, but since `MemoryDesc` is immutable, we should
    # be fine.
    return md[]
end

getdata(x::Memory) = x.array
getdata(x::AbstractArray) = x

# for creating OneDNN arguments
toarg(x::Memory) = x.memory

function Base.similar(
        M::Memory,
        ::Type{T} = eltype(M),
        dims::NTuple{N,Int} = size(M),
        desc::MemoryDesc = memorydesc(M),
    ) where {T,N}

    # Number of bytes to allocate.
    # For now, we just do this to check that nothing funny is going on.
    # Otherwise, we will have to get a little creative for the memory formats that require
    # slightly more than normal storage space.
    bytes_to_allocate = Int(getbytes(desc))
    @assert bytes_to_allocate >= sizeof(T) * prod(dims)

    # If the bytes to allocate is larger than the product of the dims, then we need
    # to extrude some dimension of our underlying array.
    extrude_dims = extrude(T, dims, bytes_to_allocate)

    # Allocate the output array.
    out = similar(M.array, T, extrude_dims)

    return Memory(out, dims, creatememory(out, desc))
end

# Make sure we allocate a large enough array to hold all the results.
#
# It's okay for it to be a little bit larger.
function extrude(::Type{T}, dims, bytes_to_allocate) where {T}
    (bytes_to_allocate == sizeof(T) * prod(dims)) && return dims

    # For now, just extrude the last dimension.
    first = leading(dims...)
    last_dimension = 1 + div(bytes_to_allocate - 1, sizeof(T) * prod(first))
    return (first..., last_dimension)
end

leading(x, y, z...) = (x, leading(y, z...)...)
leading(x, y) = (x,)
leading(x) = ()

#####
##### Converting `Memory` types to normal Arrays
#####

# General `Memory` types can have arbitrary layouts.
# If we're trying to get out a normal Julia array out of a wrapped `Memory` type, we may
# have to perform a reorder before returing the final object.
materialize(x::AbstractArray) = x
function materialize(M::Memory{T,N}) where {T,N}
    # If the format of this array is already in a standard format, just return a reshaped
    # version of the underlying array.
    #
    # Otherwise, perform a specific reordering before materializing.
    if memorydesc(T, size(M)) == memorydesc(M)
        @assert size(M.array) == size(M)
        return M.array
    end

    # Use the `reorder` op.

    # preallocate the destination so it has the correct dimensions.
    dst_array = similar(M.array, eltype(M.array), size(M))
    dst_desc = memorydesc(dst_array)

    dst = Memory(dst_array, size(dst_array), creatememory(dst_array, dst_desc))
    reorder!(dst, M)

    # Unpack the array and reshape.
    return dst.array
end

Zygote.@adjoint materialize(x) = materialize(x), Δ -> (Δ,)
