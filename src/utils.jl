# Expected Transformation
#
# [Lib.]dnnl_f(a,b,c) -> Lib.dnnl_f(dnnl_convert(a), dnnl_convert(b), dnnl_convert(c))
#
# Pretty straightforward.
# Note that `dnnl_convert` should return valid GC tracked objects and not raw pointers
# because the results of `dnnl_convert` are passed through the
# `Base.cconvert` -> `Base.unsafe_convert` chain.
macro apicall(expr)
    # Split into a partial implementation for testing purposes.
    expr = _apicall_partial_impl(expr)
    return quote
        status = $expr
        if status != Lib.dnnl_success
            error("DNNL Failure: $status")
        end
        status
    end
end

function _apicall_partial_impl(expr)
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
    return :($fname($(args...)))
end

# Get to the ultimate parent.
# Strategy: keep calling `parent` until `typeof(parent(x)) == typeof(x)`.
ancestor(x::AbstractArray) = ancestor(x, parent(x))
ancestor(x::AbstractArray, y::AbstractArray) = ancestor(y, parent(y))
ancestor(x::T, ::T) where {T<:AbstractArray} = x

#####
##### Runtime Arguments
#####

# In the macro below, add support for converting expressions like `:(a.b)` to just
# plain `b`
getsym(x::Symbol) = string(x)
getsym(x::QuoteNode) = getsym(x.value)
getsym(x::Expr) = getsym(last(x.args))

# Note - these definitions aren't strictly necessary, but useful for CachedArrays at
# the moment so I'm keeping them.
abstract type AccessContext end
struct Reading <: AccessContext end
struct Writing <: AccessContext end
struct Unknown <: AccessContext end

function dnnl_arg(x, y, context::AccessContext = Reading())
    return Lib.dnnl_exec_arg_t(x, dnnl_exec_arg(y, context))
end

Base.length(::Lib.dnnl_exec_arg_t) = 1
Base.iterate(x::Lib.dnnl_exec_arg_t, s = (x, nothing)) = s

# Handle multiple arguments passed as a single item.
function dnnl_arg(x, y::Union{AbstractArray{<:AbstractArray}, Tuple}, context::AccessContext = Reading())
    return map(enumerate(y)) do (i, _y)
        Lib.dnnl_exec_arg_t(x + (i - 1), dnnl_exec_arg(_y, context))
    end
end

function dnnl_exec_arg(y::T, context) where {T}
    return error("Define `dnnl_exec_arg` for type $(T)!")
end

# In general, we want there to have a static length.
# However, for primitives like Concat that may have many args, using a Vector makes more
# sense.
const TupleOrVector{T} = Union{NTuple{N,T},AbstractVector{T}} where {N}
struct Arguments{T<:TupleOrVector{Lib.dnnl_exec_arg_t}}
    args::T
end

# construct `Arguments` directly.
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
Base.getindex(x::Arguments, i) = getindex(x.args, i)
Base.lastindex(x::Arguments) = lastindex(x.args)

append(x::Arguments{<:Tuple}, arg::Lib.dnnl_exec_arg_t) = Arguments(x.args..., arg)
append(x::Arguments{<:Tuple}, y::Arguments{<:Tuple}) = Arguments(x.args..., y.args...)
append(x::Arguments{<:AbstractVector}, arg::Lib.dnnl_exec_arg_t) = push!(x.args, arg)
append(x::Tuple{<:AbstractArray,<:Arguments}, y) = (x[1], append(x[2], y[2]))

#####
##### Construction Utilities
#####

# General transformation provided by this macro:
#
# `@dnnl_args src dst`
#
# becomes
#
# ```
# (OneDNN).make_args(
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
# OneDNN C-API, but that's probably good practice anyways.
const _R = :(Reading())
const _W = :(Writing())
const CONTEXT_MAP = [
    "DIFF_SRC" => _W,
    "DIFF_WEIGHTS" => _W,
    "DIFF_BIAS" => _W,
    "DIFF_DST" => _R,
    "SRC" => _R,
    "MULTIPLE_SRC" => _R,
    "WEIGHTS" => _R,
    "BIAS" => _R,
    "DST" => _W,
    "FROM" => _R,
    "TO" => _W,
    "SCALE_SHIFT" => _R,
    "DIFF_SFALE_SHIFT" => _W,
    "MEAN" => _R,
    "VARIANCE" => _R,
    "WORKSPACE" => _W,
]

macro dnnl_args(syms...)
    extra_arg = Any[]
    exprs = map(syms) do sym
        # Handle direct interpolation.
        if isa(sym, Expr)
            if sym.head == :$
                newarg = gensym()
                @assert length(sym.args) == 1
                return :($(esc(sym.args[1])))
            else
                error("Unknown expression: $sym")
            end
        end

        # Handle scratchpad generation.
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
        if length(extra_arg) == 0
            push!(extra_arg, esc(sym))
        end

        return :(dnnl_arg(Lib.$(Symbol(dnnl_arg_enum)), $(esc(sym)), $context))
    end

    # Capture one argument to forward into `execute!` if we're using `similar` for
    # scratchpad allocation.
    if SIMILAR_FOR_SCRATCHPAD
        return :((ancestor($(extra_arg[1])), make_args($(exprs...))))
    else
        return :(make_args($(exprs...)))
    end
end

# Generic path, if any of the args is a vector, then result will also be a vector.
# Otherwise, we have tuple arguments.
function make_args(args...)
    return Arguments(_hasarray(args...) ? _vcat(args) : _flatcat(args...))
end

_hasarray() = false
_hasarray(x, y...) = _hasarray(y...)
_hasarray(x::AbstractArray, y...) = true

_vcat(args; init = Lib.dnnl_exec_arg_t[]) = reduce(append!, args; init)

_flatcat(x, y...) = (x, _flatcat(y...)...)
_flatcat(x::Tuple, y...) = (_flatcat(x...)..., _flatcat(y...)...)
_flatcat() = ()

#####
##### Conversion Hooks
#####

dnnl_convert(x) = x
_dnnl_convert(x, y...) = (dnnl_convert(x), _dnnl_convert(y...)...)
_dnnl_convert(x) = (dnnl_convert(x),)

const MaybeRef{T} = Union{T,Ref{<:T}}
const MaybePtr{T} = Union{T,Ptr{<:T}}
wrap_ref(x::Ref) = x
wrap_ref(x) = Ref(x)
unwrap_ref(x::Ref) = x[]
unwrap_ref(x) = x

#####
##### Dimension Helpers
#####

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

_paddims(x::Tuple, y::Tuple) = (x..., y[(end - (length(y) - length(x) - 1)):end]...)

struct Dims{N}
    kernel::NTuple{N,Int}
    strides::NTuple{N,Int}
    dilation::NTuple{N,Int}
    padding::NTuple{N,Int}
end

function _output_size(dims::Dims, sz::Int, i::Int)
    @unpack kernel, strides, dilation, padding = dims
    numerator = sz + 2 * padding[i] - ((kernel[i] - 1) * (dilation[i] + 1) + 1)
    return div(numerator, strides[i]) + 1
end

function output_size(sz::NTuple{N,Int}, dims::Dims{M}) where {N,M}
    return _paddims(ntuple(i -> _output_size(dims, sz[i], i), Val(M)), sz)
end

function tuple_replace(x::NTuple{N,Int}, v, ::Val{I}) where {N,I}
    return ntuple(i -> (i == I) ? v : x[i], Val(N))
end

#####
##### Kind Types
#####

# Move the primitive kind enums into the type domain to give the Julia compiler better
# type-level information when the return type depends on things like inference vs traiing.

abstract type AbstractKind end
abstract type AbstractForwardKind <: AbstractKind end
abstract type AbstractBackwardKind <: AbstractKind end

struct Inference <: AbstractForwardKind end
struct Training <: AbstractForwardKind end
struct Backward <: AbstractBackwardKind end

kind(::Inference) = Lib.dnnl_forward_inference
kind(::Training) = Lib.dnnl_forward_training
kind(::Backward) = Lib.dnnl_backward

dnnl_convert(x::AbstractKind) = kind(x)
