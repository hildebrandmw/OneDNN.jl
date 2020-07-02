#####
##### Primitive Descriptor
#####

# Hook to allow types to be converted into the correct value passing to OneDNN.
pd_lower(x) = x
pd_lower(attr::Attributes) = attr.val[]
pd_lower(m::Memory) = memorydesc_ptr(m)

struct PrimitiveDescriptor
    ptr::Lib.dnnl_primitive_desc_t
end
Base.unsafe_convert(::Type{Ptr{Nothing}}, pd::PrimitiveDescriptor) = pd.ptr

# Hooko to allow overloading with Cassette.
primitive_descriptor(args...) = primitive_descriptor(Lib.dnnl_primitive_desc_create, args...)
function primitive_descriptor(f::Function, args...)
    TimerOutputs.@timeit to "Primitive Descriptors" begin
        pd = Ref{Lib.dnnl_primitive_desc_t}()
        @apicall f(pd, pd_lower.(args)...)
    end
    return PrimitiveDescriptor(pd[])
end

function destroy(x::PrimitiveDescriptor)
    @apicall Lib.dnnl_primitive_desc_destroy(x)
    return nothing
end
destroy(x...) = destroy.(x)

#####
##### Primitive
#####

struct Primitive
    ptr::Lib.dnnl_primitive_t
end
Base.unsafe_convert(::Type{Ptr{Nothing}}, p::Primitive) = p.ptr

function primitive(pd::PrimitiveDescriptor)
    TimerOutputs.@timeit to "Primitives" begin
        p = Ref{Lib.dnnl_primitive_t}()
        @apicall Lib.dnnl_primitive_create(p, pd)
    end
    return Primitive(p[])
end

function execute!(p::Primitive, args)
    @apicall Lib.dnnl_primitive_execute(
        p.ptr,
        global_stream(),
        length(args),
        args,
    )

    @apicall Lib.dnnl_stream_wait(global_stream())
    return nothing
end

destroy(p::Primitive) = @apicall Lib.dnnl_primitive_destroy(p)

