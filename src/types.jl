struct InnerConstructor end
macro wrap_type(julia_name, c_name, destructor)
    lower_constructor_name = Symbol("_$julia_name")

    # Automatically add the "Lib" prefix if required.
    if isa(c_name, Symbol)
        c_name = :(Lib.$c_name)
    end

    return esc(
        quote
            # Type definition
            mutable struct $julia_name
                handle::$c_name
                $julia_name(::InnerConstructor) = new($c_name())
            end

            # Use a trick of Lower and Higher constructeors.
            # Lower constructors should have the name `_$julia_name` and not attach
            # finalizers.
            #
            # The higher constructor will simply forward to the lower constructor but
            # attach a finalizer before returning.
            function $julia_name(args...)
                x = $lower_constructor_name(args...)
                attach_finalizer!(x)
                return x
            end

            # Finalizer
            destroy(x::$julia_name) = @apicall $destructor(x)
            attach_finalizer!(x::$julia_name) = finalizer(destroy, x)

            # Conversion functions
            Base.unsafe_convert(::Type{$c_name}, x::$julia_name) = x.handle
            function Base.unsafe_convert(::Type{Ptr{$c_name}}, x::$julia_name)
                return Base.unsafe_convert(Ptr{$c_name}, Base.pointer_from_objref(x))
            end
        end,
    )
end

destroy(x, y...) = (destroy(x); destroy(y...))
destroy(x) = error("Define `destroy` for type $(typeof(x))")

#####
##### Engine
#####

@wrap_type Engine dnnl_engine_t dnnl_engine_destroy
# Lower Constructor
function _Engine(kind::Lib.dnnl_engine_kind_t, index = 0)
    engine = Engine(InnerConstructor())
    @apicall dnnl_engine_create(engine, kind, index)
    return engine
end

@wrap_type Stream dnnl_stream_t dnnl_stream_destroy
# Lower Constructor
function _Stream(engine::Engine)
    stream = Stream(InnerConstructor())
    @apicall dnnl_stream_create(stream, engine, Lib.dnnl_stream_default_flags)
    return stream
end

@wrap_type Attributes dnnl_primitive_attr_t dnnl_primitive_attr_destroy
# Lower Constructor
function _Attributes()
    attributes = Attributes(InnerConstructor())
    @apicall dnnl_primitive_attr_create(attributes)
    return attributes
end

noattributes() = Attributes()

@wrap_type PostOps dnnl_post_ops_t dnnl_post_ops_destroy
function _PostOps()
    postops = PostOps(InnerConstructor())
    @apicall dnnl_post_ops_create(postops)
    return postops
end

@wrap_type PrimitiveDescriptor dnnl_primitive_desc_t dnnl_primitive_desc_destroy
# Lower Constructor
# Automatically apply "Lib.dnnl_primitive_desc_create" if the first argument is
# not already a function.
function _PrimitiveDescriptor(args::Vararg{Any,N}) where {N}
    return _PrimitiveDescriptor(Lib.dnnl_primitive_desc_create, args...)
end

function _PrimitiveDescriptor(f::F, args::Vararg{Any,N}) where {F<:Function,N}
    descriptor = PrimitiveDescriptor(InnerConstructor())
    # Configure scratchpad mode to be "user" so we can apply our own scratchpads.
    attributes = _get_attributes(args...)
    @apicall dnnl_primitive_attr_set_scratchpad_mode(
        attributes, Lib.dnnl_scratchpad_mode_user
    )
    @apicall f(descriptor, args...)
    return descriptor
end

function Base.copy(x::PrimitiveDescriptor)
    y = PrimitiveDescriptor(InnerConstructor())
    @apicall dnnl_primitive_desc_clone(y, x)
    attach_finalizer!(y)
    return y
end

noforward() = Ptr{Lib.dnnl_primitive_desc}()

@wrap_type Primitive dnnl_primitive_t dnnl_primitive_destroy
# Lower Constructor
function _Primitive(descriptor::PrimitiveDescriptor)
    primitive = Primitive(InnerConstructor())
    @apicall dnnl_primitive_create(primitive, descriptor)
    return primitive
end

@wrap_type MemoryPtr dnnl_memory_t dnnl_memory_destroy
# Lower Constructor
function _MemoryPtr(A::AbstractArray, desc = memorydesc(A))
    return _MemoryPtr(convert(Ptr{Nothing}, pointer(A)), desc)
end

function _MemoryPtr(ptr::Ptr{Nothing}, desc)
    memory = MemoryPtr(InnerConstructor())
    @apicall dnnl_memory_create(memory, desc, global_engine(), ptr)
    return memory
end

# Define an extra conversion function.
@inline function Base.convert(
    ::Type{T}, memory::MemoryPtr
) where {T<:MaybePtr{Lib.dnnl_memory_t}}
    return Base.unsafe_convert(T, memory)
end

#####
##### Functions on Types
#####

function eltwise!(postops::PostOps, f::F, scale = 1) where {F}
    @apicall dnnl_post_ops_append_eltwise(postops, scale, forward_expand(f)...)
    return nothing
end

# Specialize identity to do nothing
eltwise!(postops::PostOps, ::typeof(identity), scale = 1) = nothing

# Attach Post Ops to Attributes
function Base.append!(a::Attributes, p::PostOps)
    @apicall Lib.dnnl_primitive_attr_set_post_ops(a, p)
end

appendsum!(p::PostOps, scale = 1) = @apicall dnnl_post_ops_append_sum(p, scale)

# Utility for finding attributes from a list of arguments
# TODO: Elide attaching a finalizer?
_get_attributes(x, y...) = _get_attributes(y...)
_get_attributes(x::Attributes, y...) = x
_get_attributes() = error("No attributes found!")

function query_md(descriptor::PrimitiveDescriptor, kind, index = 0)
    # TODO: Do we need to perform a null check here?
    ptr = Lib.dnnl_primitive_desc_query_md(descriptor, kind, index)
    return unsafe_load(ptr)
end

SCRATCHPAD = UInt8[]

function execute!(primitive::Primitive, _args; descriptor = nothing, wait = true)
    # Get the primitive descriptor if it isn't passed in alread.
    if descriptor === nothing
        _descriptor = PrimitiveDescriptor(InnerConstructor())
        @apicall dnnl_primitive_get_primitive_desc(primitive, _descriptor)
    else
        _descriptor = decriptor
    end

    # Create a scratchpad
    md = query_md(_descriptor, Lib.dnnl_query_scratchpad_md)
    bytes = getbytes(md)
    if bytes > length(SCRATCHPAD)
        resize!(SCRATCHPAD, bytes)
    end
    _memory_ptr = MemoryPtr(SCRATCHPAD, md)
    args = append(_args, Lib.dnnl_exec_arg_t(Lib.DNNL_ARG_SCRATCHPAD, _memory_ptr))

    # Finally, call the primitive
    @apicall dnnl_primitive_execute(primitive, global_stream(), length(args), args)
    wait && @apicall(dnnl_stream_wait(global_stream()))
    return nothing
end

# Automatically apply recursively to tuples.
kernel_exit_hook(x) = x
kernel_exit_hook(x::Union{<:Tuple, <:NamedTuple}) = map(kernel_exit_hook, x)

function temp_primitive(f::F, args::Vararg{Any,N}) where {F,N}
    desc = _PrimitiveDescriptor(args...)
    primitive = _Primitive(desc)
    ret = f(primitive, desc)
    ret_wrapped = wrap_tuple(ret)
    maybe_destroy(walk_results(ret_wrapped, primitive))
    maybe_destroy(walk_results(ret_wrapped, desc))
    return kernel_exit_hook(ret)
end

wrap_tuple(x) = (x,)
wrap_tuple(x::Tuple) = x
wrap_tuple(x::NamedTuple) = Tuple(x)

maybe_destroy(::Tuple{}) = nothing
maybe_destroy(x) = destroy(x)

walk_results(x::Tuple, y::Tuple{}) = ()
walk_results(x::Tuple, y) = walk_results(Base.tail(x), maybe_finalize(x[1], y))
walk_results(x::Tuple{}, y) = y
walk_results(x::Tuple{}, y::Tuple{}) = y

maybe_finalize(x, y) = y
function maybe_finalize(x::T, y::T) where {T}
    println("Attaching Finalizer!")
    attach_finalizer!(y)
    return ()
end

