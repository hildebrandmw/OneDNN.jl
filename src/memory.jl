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
    return [f(i) for i in 1:12]
end

dnnl_dims() = zeros(Lib.dnnl_dim_t, 12)
dnnl_dims(::Tuple{}) = dnnl_dims()

# Canonical formats
#
# Since Julia is column major, we have to reverse the order
dnnl_format(::Val{1}) = Lib.dnnl_a
dnnl_format(::Val{2}) = Lib.dnnl_ba
dnnl_format(::Val{3}) = Lib.dnnl_acb
dnnl_format(::Val{4}) = Lib.dnnl_abdc

dnnl_format(x::DenseArray{T,N}) where {T,N} = dnnl_format(Val{N}())

# Create a memory descriptor for a DenseArray.
memorydesc(x::DenseArray{T,N}) where {T,N} = memorydesc(T, size(x), dnnl_format(x))
function memorydesc(::Type{T}, dims::NTuple{N,Int}, format = dnnl_format(Val{N}())) where {T,N}
    handle = Ref{Lib.dnnl_memory_desc_t}()
    @apicall Lib.dnnl_memory_desc_init_by_tag(
        handle,
        N,
        dnnl_dims(dims),
        dnnl_type(T),
        format,
    )

    return handle[]
end

function Base.:(==)(a::Ref{Lib.dnnl_memory_desc_t}, b::Ref{Lib.dnnl_memory_desc_t})
    return Bool(Lib.dnnl_memory_desc_equal(a, b))
end

getsize(a::Ref{Lib.dnnl_memory_desc_t}) = Lib.dnnl_memory_desc_get_size(a)

# As always, make this mutable so we can finalize the C pointers we are holding onto.
mutable struct Memory{A <: AbstractArray}
    # The underlying array that is supplying the data.
    array::A

    # Memory object from DNNL
    memory::Lib.dnnl_memory_t

    function Memory(A::T, memory::Lib.dnnl_memory_t) where {T <: AbstractArray}
        object = new{T}(A, memory)
        finalizer(object) do obj
            Lib.dnnl_memory_destroy(obj.memory)
        end
        return object
    end
end

Memory(A::AbstractArray) = Memory(A, creatememory(A))

function creatememory(A::DenseArray, desc::Lib.dnnl_memory_desc_t = memorydesc(A))
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
Base.size(M::Memory) = size(M.array)
Base.eltype(M::Memory) = eltype(M.array)

function memorydesc(M::Memory)
    md = Ref{Ptr{Lib.dnnl_memory_desc_t}}()
    @apicall Lib.dnnl_memory_get_memory_desc(M.memory, md)
    return md[]
end

memorywrap(M::Memory) = M
memorywrap(A::AbstractArray) = Memory(A)
getdata(x::Memory) = x.array
getdata(x::AbstractArray) = x

function Base.similar(
        M::Memory,
        ::Type{T},
        dims::NTuple{N,Int},
        desc::Lib.dnnl_memory_desc_t
    ) where {T,N}

    # Number of bytes to allocate.
    # For now, we just do this to check that nothing funny is going on.
    # Otherwise, we will have to get a little creative for the memory formats that require
    # slightly more than normal storage space.
    bytes_to_allocate = Int(Lib.dnnl_memory_desc_get_size(Ref(desc)))
    @assert bytes_to_allocate == prod(dims) * sizeof(T)
    out = similar(M.array, T, dims)
    return Memory(out, creatememory(out, desc))
end

# # Find the canonical enum representation for format `T`.
# #
# # Implement this as an @generated function to allow us to use the format as a type
# # parameter to auto-dectect necessary format conversions.
# @generated canonical_format(::Val{T}) where {T} = :(Lib.$(findname(T)))
# function findname(v)
#     for (name, val) in Lib.CEnum.name_value_pairs(Lib.dnnl_format_tag_t)
#         val == v && return name
#     end
# end

