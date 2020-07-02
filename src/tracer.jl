#####
##### CircularBuffer
#####

# For cycling through stored elements
mutable struct CircularBuffer{T}
    buffer::Vector{T}
    next_index::Int
end

CircularBuffer{T}() where {T} = CircularBuffer{T}(T[], 1)

Base.push!(cb::CircularBuffer, x) = push!(cb.buffer, x)
function Base.getindex(cb::CircularBuffer)
    v = cb.buffer[cb.next_index]
    cb.next_index = (cb.next_index == length(cb.buffer)) ? 1 : (cb.next_index + 1)
    return v
end

reset!(cb::CircularBuffer) = (cb.next_index = 1)
unwrap(cb::CircularBuffer) = cb.buffer

#####
##### Cache
#####

mutable struct Cache
    descriptors::CircularBuffer{PrimitiveDescriptor}
    primitives::CircularBuffer{Primitive}
end

function Cache()
    obj = Cache(
        CircularBuffer{PrimitiveDescriptor}(),
        CircularBuffer{Primitive}(),
    )

    # Clean up all dangling descriptors and primitives
    finalizer(obj) do _obj
        destroy.(unwrap(obj.descriptors))
        destroy.(unwrap(obj.primitives))
    end

    return obj
end

descriptor(x::Cache) = x.descriptors[]
primitive(x::Cache) = x.primitives[]

#####
##### Contexts
#####

# The idea is to use two contexts - one to record primitive descriptors/primitives and
# one to replay them.

### Record
Cassette.@context RecordCtx

# When we find a primitive_descriptor or primitive, cache it.
#
# Also need to keep it from getting destroyed
function Cassette.overdub(ctx::RecordCtx, ::typeof(primitive_descriptor), args...)
    pd = primitive_descriptor(args...)
    push!(ctx.metadata.descriptors, pd)
    return pd
end

function Cassette.overdub(ctx::RecordCtx, ::typeof(primitive), args...)
    p = primitive(args...)
    push!(ctx.metadata.primitives, p)
    return p
end

Cassette.overdub(ctx::RecordCtx, ::typeof(destroy), args...) = nothing

function record(f, args...; kw...)
    ctx = RecordCtx(metadata = Cache())
    Cassette.overdub(ctx, f, args...; kw...)
    return ReplayCtx(metadata = ctx.metadata)
end

### Replay
Cassette.@context ReplayCtx

replay(ctx, f, args...; kw...) = Cassette.overdub(ctx, f, args...; kw...)

Cassette.overdub(ctx::ReplayCtx, ::typeof(primitive_descriptor), args...) = descriptor(ctx.metadata)
Cassette.overdub(ctx::ReplayCtx, ::typeof(primitive), args...) = primitive(ctx.metadata)
Cassette.overdub(ctx::ReplayCtx, ::typeof(destroy), args...) = nothing

#####
##### @notrace
#####

# Overdubbing a lot of code can be unnecessarily expensive.
# Putting @notrace f(a, b::Int, c) will not trace through `f`.
#
# This is to (for example) allow things like CachedArrays, random initialize etc. to still
# run quickly.

function named(arg)
    if MacroTools.isexpr(arg, :(::)) && length(arg.args) == 1
        return :($(gensym())::$(arg.args[1]))
    else
        return arg
    end
end
typeless(x) = MacroTools.postwalk(x -> MacroTools.isexpr(x, :(::), :kw) ? x.args[1] : x, x)
isvararg(x) = MacroTools.isexpr(x, :(::)) && MacroTools.namify(x.args[2]) == :Vararg

# Turns `:a` into `a = a` while leaving trailing variadics unchanged.
duplicate(x) = MacroTools.isexpr(x, :(...)) ? x : Expr(:kw, x, x)

macro notrace(expr)
    # turn the function stub into a complete function definition so we can use
    # MacroTool's "splitdef".
    def = MacroTools.splitdef(:($expr = nothing))

    # This is copied from ZygoteRules.jl

    # Process the function names.
    #
    # This takes care of normal functions [f(a,b,c)] as well as structs and such like
    # (d::Dense)(a,b,c)
    name = def[:name]
    f, T = MacroTools.isexpr(name, :(::)) ?
        (length(name.args) == 1 ? (esc(gensym()), esc(name.args[1])) : esc.(name.args)) :
        (esc(gensym()), :(Core.Typeof($(esc(name)))))

    # Pull out arguments - possibly generating names for them.
    # After processing, gather the argument names for processing on the inner function call.
    args = named.(def[:args])
    argnames = Any[typeless(arg) for arg in args]

    # Handle variadic final arguments
    if !isempty(args) && isvararg(args[end])
        argnames[end] = :($(argnames[end])...,)
    end

    # Slurp up "where" parameters
    Ts = def[:whereparams]

    # Deal with keyword arguments.
    kwargs = def[:kwargs]
    kwargnames = Any[duplicate(typeless(kw)) for kw in kwargs]

    # Manage macro hygeine
    args = esc.(args)
    argnames = esc.(argnames)
    Ts = esc.(Ts)
    kwargs = esc.(kwargs)
    kwargnames = esc.(kwargnames)

    return quote
        function Cassette.overdub(::RecordCtx, $f::$T, $(args...); $(kwargs...)) where {$(Ts...)}
            return $f($(argnames...); $(kwargnames...))
        end
        function Cassette.overdub(::ReplayCtx, $f::$T, $(args...); $(kwargs...)) where {$(Ts...)}
            return $f($(argnames...); $(kwargnames...))
        end
    end
end

#####
##### For type inference.
#####

@notrace memory(x...)
@notrace memorydesc(x...)
