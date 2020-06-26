# Given a primitive descriptor, create a primitive, excute it, and then destroy the primitive.
execute!(primitive_desc::Ref, args) = execute!(primitive_desc[], args)

function execute!(primitive_desc::Lib.dnnl_primitive_desc_t, args)
    # The primitive is an output parameters of the c function call
    primitive = Ref{Lib.dnnl_primitive_t}()
    @apicall Lib.dnnl_primitive_create(primitive, primitive_desc)

    # Execute the primitive and wait for the results
    @apicall Lib.dnnl_primitive_execute(
        primitive[],
        global_stream(),
        length(args),
        args
    )

    @apicall Lib.dnnl_stream_wait(GLOBAL_STREAM[].handle)

    # Destroy the primitive and return
    @apicall Lib.dnnl_primitive_destroy(primitive[])
    return nothing
end

function primitive_descriptor(f, args...; attributes = Ptr{Nothing}())
    # Construct the primitive descriptor from the arguments
    primitive_desc = Ref{Lib.dnnl_primitive_desc_t}()
    @apicall Lib.dnnl_primitive_desc_create(
        primitive_desc,
        args...,
        attributes,
        global_engine(),
        Ptr{Nothing}(),
    )

    # Pass the constructed primitive descriptor to the function.
    f(primitive_desc[])

    # Cleanup the primitive descriptor
    @apicall Lib.dnnl_primitive_desc_destroy(primitive_desc[])
    return nothing
end

