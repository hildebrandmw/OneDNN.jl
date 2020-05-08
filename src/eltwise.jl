# TODO: Add type parameters to this thing.
mutable struct Primitive
    handle::Lib.dnnl_primitive_t

    function Primitive(handle::Lib.dnnl_primitive_t)
        p = new(handle)
        finalizer(p) do x
            Lib.dnnl_primitive_destroy(x.handle)
        end
        return p
    end
end

# Perform the call??
(P::Primitive)(x::DenseArray) = P(Memory(x))

function (P::Primitive)(src::Memory)
    dst = copy(src)

    args = [
        Lib.dnnl_exec_arg_t(Lib.DNNL_ARG_SRC, src.memory),
        Lib.dnnl_exec_arg_t(Lib.DNNL_ARG_DST, dst.memory),
    ]

    @apicall Lib.dnnl_primitive_execute(
        P.handle,
        GLOBAL_STREAM[].handle,
        length(args),
        args
    )

    return dst
end

# Wrappers for element-wise operations.
algokind(::typeof(NNlib.relu)) = Lib.dnnl_eltwise_relu

# Make mutable for finalizers?
mutable struct Initializer{F}
    f::F
    # The initialized primitive will be stored here for retrieval.
    primitive::Union{Nothing, Primitive}
end

# TODO: Allow `Memory` objects to be passed in here for format propogation.
function instantiate(I::Initializer, x::AbstractArray{T,N}) where {T,N}
    md = memorydesc(x)

    # Create an op descriptor
    #
    # TODO: Dispatch here based on `I.f`
    descriptor = Ref{Lib.dnnl_eltwise_desc_t}()
    @apicall Lib.dnnl_eltwise_forward_desc_init(
        descriptor,
        Lib.dnnl_forward_inference,
        algokind(I.f),
        Ref(md),
        zero(Float32),
        zero(Float32),
    )

    # Now that we have the descriptor, we create the primitive descriptor
    primitive_descriptor = Ref{Lib.dnnl_primitive_desc_t}()
    @apicall Lib.dnnl_primitive_desc_create(
        primitive_descriptor,
        descriptor,
        Ptr{Nothing}(),     # Null Pointer for Attribute
        GLOBAL_ENGINE[].handle,
        Ptr{Nothing}(),     # No forward primitive.
    )

    # Instantiate the primitive
    primitive = Ref{Lib.dnnl_primitive_t}()
    @apicall Lib.dnnl_primitive_create(primitive, primitive_descriptor[])

    # Cleanup the Primitive Descriptor
    @apicall Lib.dnnl_primitive_desc_destroy(primitive_descriptor[])

    # Return the primitive
    return Primitive(primitive[])
end
