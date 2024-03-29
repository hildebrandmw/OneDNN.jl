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
    @apicall dnnl_primitive_attr_set_scratchpad_mode(
        attributes, Lib.dnnl_scratchpad_mode_user
    )
    return attributes
end

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
    @apicall dnnl_primitive_attr_set_post_ops(a, p)
end

appendsum!(p::PostOps, scale = 1) = @apicall dnnl_post_ops_append_sum(p, scale)
function query_md(
    descriptor::Union{PrimitiveDescriptor,Lib.dnnl_primitive_desc_t}, kind, index = 0
)
    # TODO: Do we need to perform a null check here?
    ptr = Lib.dnnl_primitive_desc_query_md(descriptor, kind, index)
    return unsafe_load(ptr)
end

macro query(sym)
    fullsym = Symbol("dnnl_query_$(sym)_md")
    return :(Lib.$fullsym)
end

#####
##### We manage our own global scratchpad here for now.
#####

const SCRATCHPAD = UInt8[]
const SCRATCHPAD_MEMORY = Ref{Lib.dnnl_memory_t}()

function unsafe_primitive_descriptor(primitive::Primitive)
    _pd = Ref{Lib.dnnl_primitive_desc_t}()
    @apicall dnnl_primitive_get_primitive_desc(primitive, _pd)
    return _pd[]
end

function execute!(primitive::Primitive, args::Tuple{<:AbstractArray,<:Arguments}; kw...)
    return execute!(primitive, args[2]; scratchpad_source = args[1], kw...)
end

function execute!(
    primitive::Primitive,
    _args;
    scratchpad_source = nothing,
    descriptor = nothing,
    wait = true,
)
    # Get the primitive descriptor if it isn't passed in alread.
    if descriptor === nothing
        _descriptor = unsafe_primitive_descriptor(primitive)
    else
        _descriptor = descriptor
    end

    # Create a scratchpad
    md = query_md(_descriptor, Lib.dnnl_query_scratchpad_md)
    bytes = getbytes(md)

    if scratchpad_source === nothing && bytes > length(SCRATCHPAD)
        resize!(SCRATCHPAD, bytes)
    end

    if scratchpad_source === nothing
        scratchpad = SCRATCHPAD
    else
        scratchpad = similar(scratchpad_source, UInt8, bytes)
    end

    GC.@preserve scratchpad begin
        @apicall dnnl_memory_create(
            SCRATCHPAD_MEMORY, md, global_engine(), pointer(scratchpad)
        )
        args = append(
            _args, Lib.dnnl_exec_arg_t(Lib.DNNL_ARG_SCRATCHPAD, SCRATCHPAD_MEMORY[])
        )

        # Finally, call the primitive
        @apicall dnnl_primitive_execute(primitive, global_stream(), length(args), args)
        @apicall dnnl_memory_destroy(SCRATCHPAD_MEMORY[])
        wait && @apicall(dnnl_stream_wait(global_stream()))
    end
    return nothing
end

# Automatically apply recursively to tuples.
kernel_exit_hook(x) = x
kernel_exit_hook(x::Union{<:Tuple,<:NamedTuple}) = map(kernel_exit_hook, x)

function temp_primitive(f::F, args::Vararg{Any,N}) where {F,N}
    desc = _PrimitiveDescriptor(args...)
    primitive = _Primitive(desc)
    ret = f(primitive, desc)

    # If `desc` or `primitive` show up in the returned items from `f`, then attach a
    # finalizer and don't eagerly destroy.
    #
    # If they don't show up, then assume it is safe to eagerly destroy them.
    wrapped = tuplewrap(ret)
    cleanup(desc, wrapped)
    cleanup(primitive, wrapped)
    return kernel_exit_hook(ret)
end

cleanup(x, ys::Tuple) = recursive_any(i -> i === x, ys) ? attach_finalizer!(x) : destroy(x)

tuplewrap(x) = (x,)
tuplewrap(x::Tuple) = x
tuplewrap(x::NamedTuple) = Tuple(x)

# Slightly better type inference then the generic `any`.
recursive_any(f::F, x::Tuple) where {F} = f(x[1]) ? true : recursive_any(f, Base.tail(x))
recursive_any(f::F, x::Tuple{}) where {F} = false
