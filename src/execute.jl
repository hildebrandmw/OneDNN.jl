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

# Many primitives just use `Lib.dnnl_primitive_desc_create`, so we apply that as a default
# creation function.
#
# However, many others, (like convolution, matrix multiplication, etc) have their own
# primitive descriptor creation functions.
#
# So, we allow the primitive creation function to be passed as well.
primitive_descriptor(args...) = primitive_descriptor(Lib.dnnl_primitive_desc_create, args...)
function primitive_descriptor(f::F, args...) where {F <: Function}
    pd = Ref{Lib.dnnl_primitive_desc_t}()
    @apicall f(pd, pd_lower.(args)...)
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
    p = Ref{Lib.dnnl_primitive_t}()
    @apicall Lib.dnnl_primitive_create(p, pd)
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

