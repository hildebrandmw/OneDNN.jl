module OneDNN

# Include auto-generated wrapper for OneDNN
include("lib/lib.jl")

# Map data types from Julia to DNNL
dnnl_type(::Type{Float16}) = Lib.dnnl_f16
dnnl_type(::Type{Float32}) = Lib.dnnl_f32
dnnl_type(::Type{Int32}) = Lib.dnnl_s32
dnnl_type(::Type{Int8}) = Lib.dnnl_s8
dnnl_type(::Type{UInt8}) = Lib.dnnl_u8

# Try to forward Numbers to their underlying type
dnnl_type(x::T) where {T <: Number} = dnnl_type(T)
dnnl_type(::T) where {T} = error("No DNNL type for $T")

# Make a DIMS array
#
# These arrays are constant size in DNNL
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
    dnnlstrides = dnnldims(strides(x))

    # Break point for inspection
    ret = Lib.dnnl_memory_desc_init_by_strides(
        handle,
        N,
        dims,
        datatype,
        dnnlstrides,
    )

    return handle[]
end

# As always, make this mutable so we can finalize the C pointers we are holding onto.
mutable struct MemoryBuffer{A <: AbstractArray}
    # The underlying array that is supplying the data.
    array::A

    # Memory object from DNNL
    memory::Lib.dnnl_memory_t

    function MemoryBuffer(A::AbstractArray, engine)
        handle = Ref{Lib.dnnl_memory_t}()
        desc = memorydesc(A)
        ret = Lib.dnnl_memory_create(
            handle,
            Ref(desc),
            engine.handle,
            convert(Ptr{Cvoid}, pointer(A))
        )

        buf = new{typeof(A)}(A, handle[])
        finalizer(buf) do x
            Lib.dnnl_memory_destroy(x.memory)
        end
        return buf
    end
end

#####
##### Global Stuff
#####

# Should only need to create one engine and stream and hold onto these as "singletons"

# Construct an execution engine.
mutable struct Engine
    handle::Lib.dnnl_engine_t

    # Inner constructor
    function Engine(kind::Lib.dnnl_engine_kind_t, index = 0)
        handle = Ref(Lib.dnnl_engine_t())
        Lib.dnnl_engine_create(handle, kind, index)
        engine = new(handle[])

        # Cleanup when destroyed
        finalizer(engine) do x
            Lib.dnnl_engine_destroy(x.handle)
        end
        return engine
    end
end

# Execution Stream to run on an engine.
mutable struct Stream
    handle::Lib.dnnl_stream_t

    function Stream(engine::Engine)
        handle = Ref(Lib.dnnl_stream_t())
        Lib.dnnl_stream_create(handle, engine.handle, Lib.dnnl_stream_default_flags)
        stream = new(handle[])

        # Cleanup when destroyed
        finalizer(stream) do x
            Lib.dnnl_stream_destroy(x.handle)
        end

        return stream
    end
end

end # module
