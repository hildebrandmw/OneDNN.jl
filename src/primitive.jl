# TODO:
# Include input sizes and memory tags for validation and maybe automatic format conversion.
# Include memory descriptors directoy in the primitive to avoid reallocation every time
#   we make a new call?
mutable struct Primitive{T,N}
    dispatch::T
    handle::Lib.dnnl_primitive_t

    # The formats of the inputs and outputs.
    # Used to check if we need to do any format conversion.
    #
    # Keep the memory descriptions as Ref's to avoid allocating a new Ref every time
    # we need one.
    #
    # The memory descriptor pointers belong to the C/C++ `dnnl_primitive_t` object,
    # so we just hold raw pointers to them.
    formats::NTuple{N,Ptr{Lib.dnnl_memory_desc_t}}

    # Pre-allocate argument vector to cut down on total allocations.
    args::Vector{Lib.dnnl_exec_arg_t}

    # Inner constructor so we can be pedantic about attaching finalizers.
    function Primitive(dispatch::T, handle::Lib.dnnl_primitive_t) where {T}
        # Get the input and output formats.
        formats = query_formats(dispatch, handle)

        p = new{T,length(formats)}(dispatch, handle, formats, Lib.dnnl_exec_arg_t[])
        finalizer(p) do x
            Lib.dnnl_primitive_destroy(x.handle)
        end
        return p
    end
end

function invoke!(P::Primitive)
    @apicall Lib.dnnl_primitive_execute(
        P.handle,
        GLOBAL_STREAM[].handle,
        length(P.args),
        P.args
    )

    # Wait for the op to complete.
    @apicall Lib.dnnl_stream_wait(GLOBAL_STREAM[].handle)
    return nothing
end

arg(x...) = Lib.dnnl_exec_arg_t(x...)

getdesc(P::Primitive) = getdesc(P.handle)
function getdesc(handle::Lib.dnnl_primitive_t)
    # Get the primitive description.
    #
    # Make this a `Ptr` because we do not own the memory that is being returned by
    # this function.
    pd = Ref{Lib.dnnl_primitive_desc_t}()
    @apicall Lib.dnnl_primitive_get_primitive_desc(handle, pd)
    return pd[]
end

"""
    query_formats(dispatch::AbstractDispatch, primitive::Lib.dnnl_primitive_t)

Return a tuple of `Ref`s for all the relevant Input/Output memories for `dispatch`.

Dispatch types extend this behavior by defining
```
queries(dispatch)
```
Which returns a tuple (Lib.dnnl_query_t, index) pairs for all the relevant I/O for
`dispatch`.
"""
function query_formats(T::AbstractDispatch, primitive::Lib.dnnl_primitive_t)
    pd = getdesc(primitive)

    return map(queries(T)) do (what, index)
        ptr = Lib.dnnl_primitive_desc_query_md(pd, what, index)
        if iszero(convert(UInt, ptr))
            error("Query returned a Null ptr!")
        end
        return ptr
    end
end

#####
##### Element-wise Ops
#####

# Just look at the input and output formats.
function queries(::AbstractEltwiseOp)
    return (
        (Lib.dnnl_query_src_md, 0),
        (Lib.dnnl_query_dst_md, 0),
    )
end

(P::Primitive{<:AbstractEltwiseOp})(x::DenseArray) = P(memorywrap(x))
function (P::Primitive{<:AbstractEltwiseOp})(src::Memory)
    # Similar pattern for all elementwise-ops
    dst = copy(src)

    # Setup primitive arguments
    resize!(P.args, 2)
    P.args[1] = arg(Lib.DNNL_ARG_SRC, src.memory)
    P.args[2] = arg(Lib.DNNL_ARG_DST, dst.memory)

    invoke!(P)
    return dst
end

#####
##### Binary Ops
#####

function queries(::AbstractBinaryOp)
    return (
        (Lib.dnnl_query_src_md, 0),
        (Lib.dnnl_query_src_md, 1),
        (Lib.dnnl_query_dst_md, 0),
    )
end

(P::Primitive{<:AbstractBinaryOp})(x...) = P(memorywrap.(x)...)
function (P::Primitive{<:AbstractBinaryOp})(a::Memory, b::Memory)
    dst = copy(a)

    # Setup Primitive arguments
    resize!(P.args, 3)
    P.args[1] = arg(Lib.DNNL_ARG_SRC_0, a.memory)
    P.args[2] = arg(Lib.DNNL_ARG_SRC_1, b.memory)
    P.args[3] = arg(Lib.DNNL_ARG_DST, dst.memory)

    invoke!(P)
    return dst
end

#####
##### Convolutions
#####

# (P::Primitive{<:Convolution})(x...) = P(memorywrap.(x))
# function (P::Primitive{<:Convolution})(x::Memory, weights::Memory, bias::Memory)
#     # TODO: Formats!!
#     dst = copy(x)
#
#     resize!(P.args, 4)
#     P.args[1] = arg(Lib.DNNL_ARG_SRC, x.memory)
#     P.args[2] = arg(Lib.DNNL_ARG_WEIGHTS, weights.memory)
#     P.args[3] = arg(Lib.DNNL_ARG_BIAS, bias.memory)
#     P.args[4] = arg(Lib.DNNL_DEST, dst.memory)
#
#     invoke!(P)
#     return dst
# end
