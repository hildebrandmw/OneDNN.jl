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

# We swap the first two dimensions because that's the format OneDNN is expecting.
swapleading(a, b, x...) = (b, a, x...)
swapleading(a) = (a,)

swapdims(i) = (i <= 2) ? xor(i, one(i) << 1) : i

# Make a DIMS array
#
# These arrays are constant size in DNNL, but are apparently passed around as pointers,
# so we just use normal Julia Arrays on this side, let `cconvert` convert them to pointers
# for all the `ccall` operations and profit!
function dnnl_dims(x::NTuple{N,Int64}) where {N}
    # Need to reverse the order of dimensions because C/C++ is row-major while Julia
    # is column major.
    y = swapleading(x...)
    f(i) = (i <= length(x)) ? Lib.dnnl_dim_t(y[N+1-i]) : zero(Lib.dnnl_dim_t)
    return [f(i) for i in 1:Lib.DNNL_MAX_NDIMS]
end

dnnl_dims() = zeros(Lib.dnnl_dim_t, Lib.DNNL_MAX_NDIMS)
dnnl_dims(::Tuple{}) = dnnl_dims()

# Canonical formats
#
# Since Julia is column major, we have to reverse the order
abstract type AbstractFormatContext end
struct DefaultContext <: AbstractFormatContext end
struct TransposeContext <: AbstractFormatContext end

dnnl_format_any() = Lib.dnnl_format_tag_any
# Since Julia is column major instead of row major, we have to permute the last two
# dimensions in the format tag by default.
dnnl_format(::DefaultContext, ::Val{1}) = Lib.dnnl_a
dnnl_format(::DefaultContext, ::Val{2}) = Lib.dnnl_ba
dnnl_format(::DefaultContext, ::Val{3}) = Lib.dnnl_acb
dnnl_format(::DefaultContext, ::Val{4}) = Lib.dnnl_abdc

dnnl_format(v::Val) = dnnl_format(DefaultContext(), v)

dnnl_format(context, x::Union{DenseArray{T,N}, Base.ReshapedArray{T,N}}) where {T,N} = dnnl_format(context, Val{N}())
dnnl_format(x::DenseArray{T,N}) where {T,N} = dnnl_format(DefaultContext(), Val{N}())

dnnl_format(::TransposeContext, ::Val{1}) = Lib.dnnl_a
dnnl_format(::TransposeContext, ::Val{2}) = Lib.dnnl_ab
dnnl_format(::TransposeContext, ::Val{3}) = Lib.dnnl_abc
dnnl_format(::TransposeContext, ::Val{4}) = Lib.dnnl_abcd

#####
##### Memory Desc
#####

const MemoryDesc = Lib.dnnl_memory_desc_t

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

function memorydesc(
        x::Union{DenseArray{T}, Base.ReshapedArray{T}},
        context = DefaultContext()
    ) where {T}

    return memorydesc(T, size(x), dnnl_format(context, x))
end

function memorydesc(x::LinearAlgebra.Transpose)
    return memorydesc(x.parent, TransposeContext())
end

# batched transpose
function memorydesc(x::Base.PermutedDimsArray{T,3,(2,1,3)}) where {T}
    format = dnnl_format(TransposeContext(), Val{3}())
    return memorydesc(T, size(x), format)
end

function memorydesc(::Type{T}, dims::NTuple{N,Int}, format = dnnl_format(Val{N}())) where {T,N}
    handle = Ref{MemoryDesc}()
    @apicall Lib.dnnl_memory_desc_init_by_tag(
        handle,
        N,
        dnnl_dims(dims),
        dnnl_type(T),
        format,
    )

    return handle[]
end

Base.:(==)(a::Ref{MemoryDesc}, b::Ref{MemoryDesc}) = Bool(Lib.dnnl_memory_desc_equal(a, b))
Base.:(==)(a::MemoryDesc, b::MemoryDesc) = (Ref(a) == Ref(b))

getbytes(a::Ref{MemoryDesc}) = Lib.dnnl_memory_desc_get_size(a)
getbytes(a::MemoryDesc) = getbytes(Ref(a))

#####
##### Memory
#####

# As always, make this mutable so we can finalize the C pointers we are holding onto.
mutable struct Memory{A <: AbstractArray,N}
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
        ) where {N,U <: AbstractArray}

        object = new{U,N}(A, dims, memory)
        finalizer(object) do obj
            Lib.dnnl_memory_destroy(obj.memory)
        end
        return object
    end
end

memory(A::AbstractArray) = Memory(A, size(A), creatememory(A))
memory(M::Memory) = M

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
Base.size(M::Memory, i) = size(M)[i]

Base.eltype(M::Memory) = eltype(M.array)
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
        desc::MemoryDesc = memorydesc(M)
    ) where {T,N}

    # Number of bytes to allocate.
    # For now, we just do this to check that nothing funny is going on.
    # Otherwise, we will have to get a little creative for the memory formats that require
    # slightly more than normal storage space.
    bytes_to_allocate = Int(getbytes(desc))
    @assert bytes_to_allocate >= sizeof(T) * prod(dims)

    # Allocate the output array.
    # Since for a general OneDNN format, the number of bytes required might be greater than
    # the product of the dimensions.
    #
    # Therefore, we use a linear vector as the actual storage.
    out = similar(
        M.array,
        T,
        (div(bytes_to_allocate, sizeof(T)),)
    )

    return Memory(out, dims, creatememory(out, desc))
end

#####
##### Converting `Memory` types to normal Arrays
#####

# General `Memory` types can have arbitrary layouts.
# If we're trying to get out a normal Julia array out of a wrapped `Memory` type, we may
# have to perform a reorder before returing the final object.
function materialize(M::Memory{A,N}) where {A,N}
    # Use the `reorder` op.
    dst = reorder(M, dnnl_format(Val{N}()))

    # Unpack the array and reshape.
    return reshape(dst.array, size(dst))
end
