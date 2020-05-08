module OneDNN

# Include auto-generated wrapper for OneDNN
include("lib/lib.jl")
include("utils.jl")

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
function dnnldims(x::NTuple{N,Int64}) where {N}
    # Need to reverse the order of dimensions because C/C++ is row-major while Julia
    # is column major.
    f(i) = (i <= length(x)) ? Lib.dnnl_dim_t(x[N+1-i]) : zero(Lib.dnnl_dim_t)
    return [f(i) for i in 1:12]
end

dnnldims() = zeros(Lib.dnnl_dim_t, 12)
dnnldims(::Tuple{}) = dnnldims()

# Create a memory descriptor for a DenseArray.
function memorydesc(x::DenseArray{T,N}) where {T,N}
    handle = Ref{Lib.dnnl_memory_desc_t}()
    dims = dnnldims(size(x))
    datatype = dnnl_type(T)

    # Break point for inspection
    @apicall Lib.dnnl_memory_desc_init_by_tag(
        handle,
        N,
        dims,
        datatype,
        Lib.dnnl_abcd,
    )

    return handle[]
end

# As always, make this mutable so we can finalize the C pointers we are holding onto.
mutable struct Memory{A <: AbstractArray}
    # The underlying array that is supplying the data.
    array::A

    # Memory object from DNNL
    memory::Lib.dnnl_memory_t
    engine::Engine

    function Memory(A::AbstractArray, engine)
        handle = Ref{Lib.dnnl_memory_t}()
        desc = memorydesc(A)
        ret = Lib.dnnl_memory_create(
            handle,
            Ref(desc),
            engine.handle,
            convert(Ptr{Cvoid}, pointer(A))
        )

        buf = new{typeof(A)}(A, handle[], engine)
        finalizer(buf) do x
            Lib.dnnl_memory_destroy(x.memory)
        end
        return buf
    end
end

# Convenience method for creating destionation memories from a source memory.
Base.copy(M::Memory) = Memory(similar(M.array), M.engine)

end # module
