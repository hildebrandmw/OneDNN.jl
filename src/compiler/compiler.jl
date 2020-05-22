# Use Mjlonir to trace the IR of a function.
# Find primitives that can map to DNNL kernels - generate those kernels and then modify the
# IR for those kernels.
import Mjolnir
import Mjolnir: @abstract, Const, Shape, Partial, AType


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
    return Mjolnir.Shape{A}(bc.size)
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

