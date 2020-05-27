# Use Mjlonir to trace the IR of a function.
# Find primitives that can map to DNNL kernels - generate those kernels and then modify the
# IR for those kernels.
using Mjolnir: Mjolnir, @abstract, Const, Shape, Partial, AType
using IRTools: IRTools

struct DNNLPrimitives end

DNNLContext() = Mjolnir.Multi(DNNLPrimitives(), Mjolnir.Basic(), Mjolnir.Numeric())

# Convert Flux models into Shapes and Partials
#
# Note: This is copied over from XLA.jl
tracetype(x::Number) = typeof(x)
tracetype(x::Tuple) = Mjolnir.ptuple(tracetype.(x)...)
tracetype(x::Array{<:Number}) = Mjolnir.Shape{typeof(x)}(size(x))

# Fancy trick - capture the type of the wrapped object in the type field of a Mjolnir.Partial,
# and return a shadow struct in the form of a NamedTuple
function tracetype(x)
    (isbits(x) && nfields(x) == 0) && return Mjolnir.Const(x)
    return Mjolnir.Partial{typeof(x)}((;
        map(f -> f => tracetype(getfield(x, f)), fieldnames(typeof(x)))...)
    )
end

####
#### Compiler
####

trace(Ts...) = Mjolnir.trace(DNNLContext(), tracetype.(Ts)...)

# TODO: Remove constant arguments??
function compile(ir::IRTools.Inner.IR)
    return IRTools.func(ir)
end

#####
##### Utils
#####

# Find the type of an IRTools Variable
function vartype(ir::IRTools.Inner.IR, var::IRTools.Variable)
    # Is this an argument?
    i = findfirst(isequal(var), IRTools.arguments(ir))
    isnothing(i) || return IRTools.argtypes(ir)[i]

    # Not an argument, find the defining statement.
    return ir[var].type
end

#####
##### Passes
#####

# Transform Certain IR Patterns to OneDNN Primitives
function dense_replacement(ir::IRTools.Inner.IR)
    map!(ir) do expr
        # only look at expressions that are Expr types.
        # for all other types, just do the identity.
        isa(expr, Expr) || return expr

        # Match falls to variables
        if expr.head == :call && expr.args[1] isa IRTools.Inner.Variable
            # Check the type of the variable. If it is a call to Dense, replace it.
            if vartype(ir, expr.args[1]) isa Partial{<:Flux.Dense}
                # TODO: Construct a primitive implementing Dense from OneDNN
                println("Found a Dense!")
            end
        end
        return expr
    end
end

#####
##### abstract
#####

# This is copied over from the `Mjolnir` repo
# The default implementation of broadcasted computed `A` below as the return type of
# `Base.broadcast` instead of `Base.broadcasted`.
#
# This resulted in `materialize` getting elided because it wasn't being fed a
# `Base.Broadcasted` object.
#
# Note the we need the corresponding `Base.materialize` definition implemented below.
@abstract DNNLPrimitives function Broadcast.broadcasted(style::Broadcast.AbstractArrayStyle, f, args...)
    A = Core.Compiler.return_type(Base.broadcasted, Tuple{Mjolnir.widen(style),Mjolnir.widen(f),Mjolnir.widen.(args)...})
    if f isa Const && args isa Tuple{Vararg{Const}}
        return Const(broadcast(f.value, map(x -> x.value, args)...))
    elseif args isa Tuple{Vararg{Union{Const,Shape,AType{<:Number}}}} && !(args isa Tuple{Vararg{AType{<:Number}}})
        return Shape{A}(Broadcast.broadcast_shape(size.(args)...))
    else
        return A
    end
end

@abstract DNNLPrimitives function Base.materialize(bc::Base.Broadcast.Broadcasted)
    A = Core.Compiler.return_type(Base.materialize, Tuple{Mjolnir.widen(bc)})
    if bc isa Const
        return Const(materialize(bc.value))
    elseif bc isa Shape
        return Mjolnir.Shape{A}(bc.size)
    else
        return A
    end
end

#####
##### Compiler Library
#####

@abstract DNNLPrimitives function Base.:*(A::U, B::U) where {T, U <: AbstractArray{T,2}}
    # For now, assume A and B are Shapes
    @assert A.size[2] == B.size[1]
    return Shape{U}((A.size[1], B.size[2]))
end

@abstract DNNLPrimitives function Base.:*(A::AbstractArray{T,2}, B::AbstractArray{T,1}) where {T}
    # For now, assume A and B are Shapes
    @assert A.size[2] == B.size[1]
    return Shape{Array{T,1}}((A.size[1],))
end

# Dense
@abstract DNNLPrimitives function (layer::L)(x::U) where {L <: Flux.Dense, U <: AbstractArray{T}}
    # For now - assume that everything isa Shape
    @assert layer.value.W isa Shape
    @assert layer.value.b isa Shape
    @assert x isa Shape

    # Determine the number of output rows
    output_rows = layer.value.b.size[1]

    if x isa Shape{<:AbstractVector}
        output_shape = (output_rows,)
    elseif x isa Shape{<:AbstractMatrix}
        output_shape = (output_rows, x.size[2])
    else
        error()
    end

    return Shape{U}(output_shape)
end

