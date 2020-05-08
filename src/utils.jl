macro apicall(expr)
    if expr.head != :call
        error("@apicall needs to be a function call")
    end

    # The transformation is pretty simple.
    # We just grab the return value and compare it with the Enum values that the C-API
    # returns.
    #
    # If it's anything other than "success", then we throw an error :D
    return quote
        ret = $(esc(expr))
        if ret != Lib.dnnl_success
            error("DNNL Failure: $ret")
        end
        ret
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
        @apicall Lib.dnnl_engine_create(handle, kind, index)
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
        @apicall Lib.dnnl_stream_create(handle, engine.handle, Lib.dnnl_stream_default_flags)
        stream = new(handle[])

        # Cleanup when destroyed
        finalizer(stream) do x
            Lib.dnnl_stream_destroy(x.handle)
        end

        return stream
    end
end
