# Make mutable for finalizers?
mutable struct Initializer{F}
    dispatcher::F
    # The initialized primitive will be stored here for retrieval.
    primitive::Union{Nothing, Primitive}

    # Allow arbitrary metadata to be attached to a initializer to help when setting up
    # for backward passes.
    meta::Any
end
Initializer(f) = Initializer(dispatcher(f), nothing, nothing)

dispatcher(I::Initializer) = I.dispatcher

isinitialized(I::Initializer) = !isnothing(I.primitive)
unpack(I::Initializer) = I.primitive

hasmeta(I::Initializer) = !isnothing(I.meta)
getmeta(I::Initializer) = I.meta

# TODO: Allow `Memory` objects to be passed in here for format propogation.
function (I::Initializer)(x...)
    # Create an op descriptor
    #
    # TODO: Dispatch here based on `I.f`
    op_descriptor = descriptor(dispatcher(I), x...)

    # Now that we have the descriptor, we create the primitive descriptor
    primitive_descriptor = Ref{Lib.dnnl_primitive_desc_t}()
    @apicall Lib.dnnl_primitive_desc_create(
        primitive_descriptor,
        op_descriptor,
        # TODO:  Use dispatcher for Attributes as well.
        Ptr{Nothing}(),     # Null Pointer for Attribute
        GLOBAL_ENGINE[].handle,
        # TODO: How do we differentiate between inference and training before Zygote
        # executes?
        #
        # Maybe we have to run both the inference pass and actual pass using Zygote.
        Ptr{Nothing}(),     # No forward primitive.
    )

    # Instantiate the primitive
    primitive = Ref{Lib.dnnl_primitive_t}()
    @apicall Lib.dnnl_primitive_create(primitive, primitive_descriptor[])

    # Cleanup the Primitive Descriptor
    @apicall Lib.dnnl_primitive_desc_destroy(primitive_descriptor[])

    # Invoke the primitive and return the results.
    I.primitive = Primitive(dispatcher(I), primitive[])
    return I.primitive(x...)
end

#####
##### Op Descripter Dispatch
#####

function descriptor(f::AbstractEltwiseOp, x)
    op_descriptor = Ref{Lib.dnnl_eltwise_desc_t}()
    @apicall Lib.dnnl_eltwise_forward_desc_init(
        op_descriptor,
        Lib.dnnl_forward_inference,
        algokind(f),
        Ref(memorydesc(x)),
        zero(Float32),
        zero(Float32),
    )

    return op_descriptor
end

# If no output is given, assume it's similar to one of the inputs.
descriptor(f::AbstractBinaryOp, x, y) = descriptor(f, similar(y), x, y)
function descriptor(f::AbstractBinaryOp, z, x, y)
    op_descriptor = Ref{Lib.dnnl_binary_desc_t}()
    @apicall Lib.dnnl_binary_desc_init(
        op_descriptor,
        algokind(f),
        Ref(memorydesc(x)),
        Ref(memorydesc(y)),
        Ref(memorydesc(z)),
    )

    return op_descriptor
end
