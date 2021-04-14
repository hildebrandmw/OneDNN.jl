# The general idea is to perform memory format propagation using a two stage process.
# The first uses generic multiple dispatch to lazily initialize the OneDNN primitives.
# Then, these primitives can be extracted to construct a concretely typed pipeline.

mutable struct Placeholder{F}
    # The function to invoke when called.
    f::Any
    # Default to nothing if there is no initialized primitive yet.
    op::Any

    #-- Constructors
    # TODO: This is a little hacky ...
    Placeholder(::Type{F}, f) where {F} = new{F}(f, nothing)
end

const MaybePlaceholder{T} = Union{T,Placeholder{<:T}}

function (placeholder::Placeholder)(args...)
    # Wait until runtime arguments are recieved to finish initializing the `op`.
    if placeholder.op === nothing
        placeholder.op = placeholder.f(args...)
    end
    return placeholder.op(args...)
end

extract(placeholder::Placeholder) = placeholder.op

_op(placeholder::Placeholder) = placeholder.op
_op(x) = x

#####
##### Eltwise
#####

function Placeholder{Eltwise}(f)
    constructor = function EltwiseConstructor(src::Memory)
        return Eltwise(f, src)
    end
    return Placeholder(Eltwise, constructor)
end

function Placeholder{EltwiseBackward}(f)
    constructor = function EltwiseBackwardConstructor(diff_dst::Memory, src_or_dst::Memory)

        return EltwiseBackward(f, diff_dst, src_or_dst)
    end
    return Placeholder(EltwiseBackward, constructor)
end

struct EltwisePair{A<:MaybePlaceholder{Eltwise},B<:MaybePlaceholder{EltwiseBackward},F}
    forward::A
    backward::B
    f::F
end

function EltwisePair(f)
    forward = Placeholder{Eltwise}(f)
    backward = Placeholder{EltwiseBackward}(f)
    return EltwisePair(forward, backward, f)
end

function extract(x::EltwisePair{<:Placeholder,<:Placeholder})
    return EltwisePair(extract(x.forward), extract(x.backward), x.f)
end

function ChainRulesCore.rrule(f::EltwisePair, src::Memory)
    dst = f.forward(src)
    data = dst_for_bwd(f.f) ? dst : src
    pullback = function EltwisePullback(diff_dst::Memory)
        op = f.backward
        diff_src = op(diff_dst, data)
        return (ChainRulesCore.Zero(), diff_src)
    end
    return (dst, pullback)
end

#####
##### InnerProduct
#####

function Placeholder{InnerProduct}(weights, bias, activation = identity)
    constructor = function InnerProductConstructor(src::Memory)
        return InnerProduct(activation, src, weights, bias)
    end
    return Placeholder(InnerProduct, constructor)
end

function Placeholder{InnerProductBackwardData}(src_size)
    constructor =
        function InnerProductBackwardDataConstructor(weight::Memory, diff_dst::Memory)
            return InnerProductBackwardData(src_size, weight, diff_dst)
        end
    return Placeholder(InnerProductBackwardData, constructor)
end

function Placeholder{InnerProductBackwardWeight}(weights_size, bias_size)
    constructor =
        function InnerProductBackwardDataConstructor(src::Memory, diff_dst::Memory)
            return InnerProductBackwardWeight(weights_size, bias_size, src, diff_dst)
        end
    return Placeholder(InnerProductBackwardWeight, constructor)
end

mutable struct Dense{
    FW<:MaybePlaceholder{InnerProduct},
    # Defer initialization in the case of a placeholder.
    # `BD` must be typed to `Any` during the placeholder phase in order to transition from
    # `nothing` to an actual `Placeholder`.
    BD,
    BW<:MaybePlaceholder{InnerProductBackwardWeight},
    A<:MaybePlaceholder{EltwiseBackward},
}
    forward::FW
    backward_data::BD
    backward_weights::BW
    activation_backward::A
end

function Dense(weights::Memory, bias::Memory; activation = identity)
    # TODO: At the moment, the actibations we really care about can all be computed from
    # the `dst_diff` tensor, so initially we'll just support that case.
    #
    # At some point, it will probably be worth revisiting ...
    @assert canfuse(activation)
    forward = Placeholder{InnerProduct}(weights, bias, activation)
    backward_data = nothing
    backward_weights = Placeholder{InnerProductBackwardWeight}(size(weights), length(bias))
    activation_backward = Placeholder{EltwiseBackward}(activation)
    return Dense{typeof(forward),Any,typeof(backward_weights),typeof(activation_backward)}(
        forward, backward_data, backward_weights, activation_backward
    )
end

function extract(op::Dense)
    return Dense(
        _op(op.forward),
        _op(op.backward_data),
        _op(op.backward_weights),
        _op(op.activation_backward),
    )
end

function ChainRulesCore.rrule(f::Dense, src::Memory)
    forward = f.forward
    dst = forward(src)

    # Defer initialization of the lazy initializion (wrap your head around that one ...)
    # Until we know the size of the forward data.
    if isa(forward, Placeholder)
        f.backward_data = Placeholder{InnerProductBackwardData}(size(src))
    end

    pullback = function DensePullback(diff_dst::Memory)
        weights = _op(f.forward).weights
        diff_dst_act = f.activation_backward(diff_dst, dst)

        diff_src = ChainRulesCore.@thunk(f.backward_data(weights, diff_dst_act))
        df = ChainRulesCore.@thunk begin
            dw, db = f.backward_weights(src, diff_dst_act)
            # Build up NamedTuple with a similar struct to `f`.
            (forward = (weights = dw, bias = db),)
        end
        return (df, diff_src)
    end
    return dst, pullback
end
