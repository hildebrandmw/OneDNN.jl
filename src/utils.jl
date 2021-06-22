# Expected Transformation
#
# [Lib.]dnnl_f(a,b,c) -> Lib.dnnl_f(dnnl_convert(a), dnnl_convert(b), dnnl_convert(c))
#
# Pretty straightforward.
# Note that `dnnl_convert` should return valid GC tracked objects and not raw pointers
# because the results of `dnnl_convert` are passed through the
# `Base.cconvert` -> `Base.unsafe_convert` chain.
macro apicall(expr)
    if expr.head != :call
        error("Only call `@apicall` on function calls")
    end

    # Prefix "Lib." in front of the function call.
    # However, sometimes the function to call is passed as a higher order function.
    # Thus, we only implicitly attach "Lib" is the function name starts with "dnnl".
    fname = expr.args[1]
    if isa(fname, Symbol)
        if startswith(string(fname), "dnnl")
            fname = :(Lib.$fname)
        else
            fname = :($(esc(fname)))
        end
    end

    # Escape and convert each of the arguments.
    args = expr.args[2:end]
    for i in eachindex(args)
        # Handle splats.
        arg = args[i]
        if isa(arg, Expr) && arg.head == :...
            args[i] = :(_dnnl_convert($(esc(arg.args[1]))...)...)
        else
            args[i] = :(dnnl_convert($(esc(args[i]))))
        end
    end

    return quote
        status = $fname($(args...))
        if status != Lib.dnnl_success
            error("DNNL Failure: $status")
        end
        status
    end
end

#####
##### Runtime Arguments
#####

# In general, we want there to have a static length.
# However, for primitives like Concat that may have many args, using a Vector makes more
# sense.
const TupleOrVector{T} = Union{NTuple{N,T},AbstractVector{T}} where {N}
struct Arguments{T<:TupleOrVector{Lib.dnnl_exec_arg_t}}
    args::T
end
Arguments(args::Lib.dnnl_exec_arg_t...) = Arguments(args)

Base.length(a::Arguments) = length(a.args)

# Note: `Base.cconvert` should return valid Julia objects and not pointers.
# We only really use `cconvert` for the `Arguments` based types.
# Everything else needs to go through `Base.unsafe_convert` for the GC protection.
function Base.cconvert(::Type{Ptr{Lib.dnnl_exec_arg_t}}, x::Arguments{<:NTuple})
    return Ref(x.args)
end

function Base.cconvert(::Type{Ptr{Lib.dnnl_exec_arg_t}}, x::Arguments{<:AbstractVector})
    return x.args
end

Base.resize!(x::Arguments{<:AbstractVector}, i) = resize!(x.args, i)
Base.setindex!(x::Arguments{<:AbstractVector}, v, i::Integer) = setindex!(x.args, v, i)
Base.lastindex(x::Arguments) = lastindex(x.args)

# Note - these definitions aren't strictly necessary, but useful for CachedArrays at
# the moment so I'm keeping them.
abstract type AccessContext end
struct Reading <: AccessContext end
struct Writing <: AccessContext end
struct Unknown <: AccessContext end

function dnnl_arg(x, y, context::AccessContext = Reading())
    return Lib.dnnl_exec_arg_t(x, dnnl_exec_arg(y, context))
end
function dnnl_exec_arg(y::T, context) where {T}
    error("Define `dnnl_exec_arg` for type $(T)!")
end

# In the macro below, add support for converting expressions like `:(a.b)` to just
# plain `b`
getsym(x::Symbol) = string(x)
getsym(x::QuoteNode) = getsym(x.value)
getsym(x::Expr) = getsym(last(x.args))

# General transformation provided by this macro:
#
# `@dnnl_args src dst`
#
# becomes
#
# ```
# (OneDNN).Arguments(
#   (OneDNN).dnnl_arg(Lib.DNNL_ARG_SRC, src, (OneDNN).Reading()),
#   (OneDNN).dnnl_arg(Lib.DNNL_ARG_DST, dst, (OneDNN).Writing()),
# )
# ```
#
# The argument symbols, (`src` and `dst` in the above example) get converted to
# `Lib.DNNL_ARG_SRC` and `Lib.DNNL_ARG_DST` respectively.
#
# We also try to automatically determine whether an object is being written or read
# based on the name of the argument.
#
# As far as I can tell, this should be pretty reliable, but we'll see.
#
# This requires that functions using this macro follow the same naming conventions as the
# OneDNN C-API, but that's probably good practiced anyways.

const _R = :(Reading())
const _W = :(Writing())
const CONTEXT_MAP = [
    "DIFF_SRC"      => _W,
    "DIFF_WEIGHTS"  => _W,
    "DIFF_BIAS"     => _W,
    "DIFF_DST"      => _R,
    "SRC"           => _R,
    "WEIGHTS"       => _R,
    "BIAS"          => _R,
    "DST"           => _W,
    "FROM"          => _R,
    "TO"            => _W,
]

macro dnnl_args(syms...)
    exprs = map(syms) do sym
        # Handle direct interpolation.
        if isa(sym, Expr) && sym.head == :$
            newarg = gensym()
            @assert length(sym.args) == 1
            return :($(esc(sym.args[1])))
        end

        dnnl_arg_enum = "DNNL_ARG_$(uppercase(getsym(sym)))"
        # Determing the context
        context = nothing
        for (str, quot) in CONTEXT_MAP
            if occursin(str, dnnl_arg_enum)
                context = quot
                break
            end
        end

        if context === nothing
            error("Unknown DNNL Argument: $dnnl_arg_enum")
        end

        return :(dnnl_arg(Lib.$(Symbol(dnnl_arg_enum)), $(esc(sym)), $context))
    end
    return :(Arguments($(exprs...)))
end

# Default implementation
dnnl_convert(x) = x
_dnnl_convert(x, y...) = (dnnl_convert(x), _dnnl_convert(y...)...)
_dnnl_convert(x) = (dnnl_convert(x),)

const MaybeRef{T} = Union{T,Ref{<:T}}
const MaybePtr{T} = Union{T,Ptr{<:T}}
wrap_ref(x::Ref) = x
wrap_ref(x) = Ref(x)
unwrap_ref(x::Ref) = x[]
unwrap_ref(x) = x

# Convenience for "noattributes".
noattributes() = Ptr{Lib.dnnl_primitive_attr}()
# No corresponding forward propagation primitive.
noforward() = Ptr{Lib.dnnl_primitive_desc}()

#####
##### Engine
#####

mutable struct Engine
    handle::Lib.dnnl_engine_t
    Engine() = new(Lib.dnnl_engine_t())
end

function Engine(kind::Lib.dnnl_engine_kind_t, index = 0)
    engine = Engine()
    @apicall dnnl_engine_create(engine, kind, index)
    attach_finalizer!(engine)
    return engine
end

function attach_finalizer!(engine::Engine)
    finalizer(engine) do x
        @apicall dnnl_engine_destroy(x)
    end
end

Base.unsafe_convert(::Type{Lib.dnnl_engine_t}, engine::Engine) = engine.handle
function Base.unsafe_convert(::Type{Ptr{Lib.dnnl_engine_t}}, engine::Engine)
    return Base.unsafe_convert(Ptr{Lib.dnnl_engine_t}, Base.pointer_from_objref(engine))
end

#####
##### Stream
#####

mutable struct Stream
    handle::Lib.dnnl_stream_t
    Stream() = new(Lib.dnnl_stream_t())
end

function Stream(engine::Engine)
    stream = Stream()
    @apicall dnnl_stream_create(stream, engine, Lib.dnnl_stream_default_flags)
    attach_finalizer!(stream)
    return stream
end

function attach_finalizer!(stream::Stream)
    finalizer(stream) do x
        @apicall dnnl_stream_destroy(x)
    end
end

Base.unsafe_convert(::Type{Lib.dnnl_stream_t}, stream::Stream) = stream.handle
function Base.unsafe_convert(::Type{Ptr{Lib.dnnl_stream_t}}, stream::Stream)
    return Base.unsafe_convert(Ptr{Lib.dnnl_stream_t}, Base.pointer_from_objref(stream))
end

#####
##### Primitive Descriptor
#####

# Hook to allow types to be converted into the correct value passing to OneDNN.
mutable struct PrimitiveDescriptor
    ptr::Lib.dnnl_primitive_desc_t
    PrimitiveDescriptor() = new(Lib.dnnl_primitive_desc_t())
end

destroy(desc::PrimitiveDescriptor) = @apicall dnnl_primitive_desc_destroy(desc)
Base.unsafe_convert(::Type{Lib.dnnl_primitive_desc_t}, x::PrimitiveDescriptor) = x.ptr
function Base.unsafe_convert(::Type{Ptr{Lib.dnnl_primitive_desc_t}}, x::PrimitiveDescriptor)
    return Base.unsafe_convert(Ptr{Lib.dnnl_primitive_desc_t}, Base.pointer_from_objref(x))
end

# Many primitives just use `Lib.dnnl_primitive_desc_create`, so we apply that as a default
# creation function.
#
# However, many others, (like convolution, matrix multiplication, etc) have their own
# primitive descriptor creation functions.
#
# So, we allow the primitive creation function to be passed as well.
function PrimitiveDescriptor(args::Vararg{Any,N}) where {N}
    return PrimitiveDescriptor(Lib.dnnl_primitive_desc_create, args...)
end

function PrimitiveDescriptor(f::F, args::Vararg{Any,N}) where {F<:Function,N}
    descriptor = __PrimitiveDescriptor(f, args...)
    finalizer(destroy, descriptor)
    return descriptor
end

function __PrimitiveDescriptor(args::Vararg{Any,N}) where {N}
    return __PrimitiveDescriptor(Lib.dnnl_primitive_desc_create, args...)
end

function __PrimitiveDescriptor(f::F, args::Vararg{Any,N}) where {F<:Function,N}
    descriptor = PrimitiveDescriptor()
    @apicall f(descriptor, args...)
    return descriptor
end

function query_md(descriptor::PrimitiveDescriptor, kind, index = 0)
    # TODO: Do we need to perform a null check here?
    ptr = Lib.dnnl_primitive_desc_query_md(descriptor, kind, index)
    return unsafe_load(ptr)
end

#####
##### Primitive
#####

mutable struct Primitive
    ptr::Lib.dnnl_primitive_t
    Primitive() = new(Lib.dnnl_primitive_t())
end

destroy(primitive::Primitive) = @apicall dnnl_primitive_destroy(primitive)
Base.unsafe_convert(::Type{Lib.dnnl_primitive_t}, x::Primitive) = x.ptr
function Base.unsafe_convert(::Type{Ptr{Lib.dnnl_primitive_t}}, x::Primitive)
    return Base.unsafe_convert(Ptr{Lib.dnnl_primitive_t}, Base.pointer_from_objref(x))
end

function Primitive(descriptor::PrimitiveDescriptor)
    primitive = __Primitive(descriptor)
    finalizer(destroy, primitive)
    return primitive
end

function __Primitive(descriptor::PrimitiveDescriptor)
    primitive = Primitive()
    @apicall dnnl_primitive_create(primitive, descriptor)
    return primitive
end

function execute!(primitive::Primitive, args; wait = true)
    @apicall dnnl_primitive_execute(primitive, global_stream(), length(args), args)
    wait && @apicall(dnnl_stream_wait(global_stream()))
    return nothing
end

# Automatically apply recursively to tuples.
kernel_exit_hook(x) = x
kernel_exit_hook(x::Tuple) = map(kernel_exit_hook, x)

function temp_primitive(f::F, args::Vararg{Any,N}) where {F,N}
    desc = __PrimitiveDescriptor(args...)
    primitive = __Primitive(desc)
    ret = f(primitive, desc)
    destroy(primitive)
    destroy(desc)
    return kernel_exit_hook(ret)
end

#####
##### Attributes and Post Ops
#####

# Attributes
mutable struct Attributes
    ptr::Lib.dnnl_primitive_attr_t

    # inner constructor to ensure a finalizer is attached.
    function Attributes()
        val = new(Lib.dnnl_primitive_attr_t())
        @apicall dnnl_primitive_attr_create(val)
        attach_finalizer!(val)
        return val
    end
end

function attach_finalizer!(attributes::Attributes)
    finalizer(attributes) do x
        @apicall dnnl_primitive_attr_destroy(x)
    end
end

Base.unsafe_convert(::Type{Lib.dnnl_primitive_attr_t}, x::Attributes) = x.ptr
function Base.unsafe_convert(::Type{Ptr{Lib.dnnl_primitive_attr_t}}, x::Attributes)
    return Base.unsafe_convert(Ptr{Lib.dnnl_primitive_attr_t}, Base.pointer_from_objref(x))
end

# PostOps
mutable struct PostOps
    ptr::Lib.dnnl_post_ops_t

    # Inner constructor to ensure a finalizer is attached.
    function PostOps()
        val = new(Lib.dnnl_post_ops_t())
        @apicall dnnl_post_ops_create(val)
        attach_finalizer!(val)
        return val
    end
end

function attach_finalizer!(postops::PostOps)
    finalizer(postops) do x
        @apicall dnnl_post_ops_destroy(x)
    end
end

Base.unsafe_convert(::Type{Lib.dnnl_post_ops_t}, x::PostOps) = x.ptr
function Base.unsafe_convert(::Type{Ptr{Lib.dnnl_post_ops_t}}, x::PostOps)
    return Base.unsafe_convert(Ptr{Lib.dnnl_post_ops_t}, Base.pointer_from_objref(x))
end

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

#####
##### Wrapper for `Lib.dnnl_memory_t`
#####

mutable struct MemoryPtr
    handle::Lib.dnnl_memory_t
    MemoryPtr() = new(Lib.dnnl_memory_t())
end

function MemoryPtr(A::AbstractArray, desc = memorydesc(A))
    return MemoryPtr(convert(Ptr{Nothing}, pointer(A)), desc)
end

function MemoryPtr(ptr::Ptr{Nothing}, desc)
    memory = MemoryPtr()
    @apicall dnnl_memory_create(memory, desc, global_engine(), ptr)
    attach_finalizer!(memory)
    return memory
end

function attach_finalizer!(memory::MemoryPtr)
    finalizer(memory) do x
        @apicall dnnl_memory_destroy(x)
    end
end

Base.unsafe_convert(::Type{Lib.dnnl_memory_t}, memory::MemoryPtr) = memory.handle
function Base.unsafe_convert(::Type{Ptr{Lib.dnnl_memory_t}}, memory::MemoryPtr)
    return Base.unsafe_convert(Ptr{Lib.dnnl_memory_t}, Base.pointer_from_objref(memory))
end

@inline function Base.convert(
    ::Type{T}, memory::MemoryPtr
) where {T<:Union{Lib.dnnl_memory_t,Ptr{Lib.dnnl_memory_t}}}
    return Base.unsafe_convert(T, memory)
end

