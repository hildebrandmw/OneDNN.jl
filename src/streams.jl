abstract type AbstractThreadPool end
needs_registration(::T) where {T<:AbstractThreadPool} = needs_registration(T)
needs_registration(::Type{<:AbstractThreadPool}) = true

struct OneDNNThreadPool <: AbstractThreadPool end
needs_registration(::Type{OneDNNThreadPool}) = false

# Use Julia Threads to execute functions.
_get_in_parallel() = (Threads.threadid() != 1)
function _parallel_for(n::Cint, f::Wrap.dnnl_kernel)
    Polyester.@batch (per = thread) for i in Base.OneTo(n)
        Wrap.call(f, i - 1, n)
    end
    return nothing
end

struct JuliaNativeThreadPool <: AbstractThreadPool
    handle::Wrap.OneDNNThreadpoolAllocated
    # Inner constructor just because
    function JuliaNativeThreadPool()
        handle = Wrap.construct_threadpool(
            @cfunction(_get_in_parallel, Bool, ()),
            @cfunction(_parallel_for, Cvoid, (Cint, Wrap.dnnl_kernelDereferenced)),
            Threads.nthreads(),
        )
        return new(handle)
    end
end
unwrap(pool::JuliaNativeThreadPool) = pool.handle.cpp_object

#####
##### ExecutionStream
#####

# Use to execute OneDNN primitives
abstract type AbstractExecutionStream end
struct GlobalExecutionStream <: AbstractExecutionStream end
getengine(::GlobalExecutionStream) = getengine(default_execution_stream())
getstream(::GlobalExecutionStream) = getstream(default_execution_stream())

struct ExecutionStream{U,T<:AbstractThreadPool} <: AbstractExecutionStream
    engine::Engine{U}
    stream::Stream
    threadpool::T

    function ExecutionStream(
        engine::Engine{U}, stream::Stream, threadpool::T
    ) where {U,T<:AbstractThreadPool}
        if needs_registration(threadpool)
            @apicall dnnl_threadpool_interop_stream_create(
                stream, engine, unwrap(threadpool)
            )
        end
        return new{U,T}(engine, stream, threadpool)
    end
end

getengine(exec::ExecutionStream) = exec.engine
getstream(exec::ExecutionStream) = exec.stream
