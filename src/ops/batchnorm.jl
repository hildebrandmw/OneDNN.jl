# TODO: Nice API for all the different forms for scale and shift?
batchnorm_flags() = Lib.dnnl_use_scaleshift
batchnorm_flags(::typeof(identity)) = batchnorm_flags()
batchnorm_flags(::typeof(Flux.relu)) = batchnorm_flags() | Lib.dnnl_fuse_norm_relu

function batchnorm_inference(
    src::Memory,
    scale_shift::Memory;
    epsilon = 1f-10,
    attributes = noattributes(),
)
    opdesc = Ref{Lib.dnnl_batch_normalization_desc_t}()
    @apicall dnnl_batch_normalization_forward_desc_init(
        opdesc, Inference(), src, epsilon, batchnorm_flags(),
    )

    return temp_primitive(opdesc, attributes, global_engine(), noforward()) do p, pd
        dst = similar(src)
        execute!(p, @dnnl_args src scale_shift dst)
        return dst
    end
end

function batchnorm_training(
    src::Memory,
    scale_shift::Memory,
    activation::F = identity;
    epsilon = 1f-10,
    attributes = noattributes(),
) where {F<:Union{typeof(identity),typeof(Flux.relu)}}
    flags = batchnorm_flags(activation)
    opdesc = Ref{Lib.dnnl_batch_normalization_desc_t}()
    @apicall dnnl_batch_normalization_forward_desc_init(
        opdesc, Training(), src, epsilon, flags
    )

    return temp_primitive(opdesc, attributes, global_engine(), noforward()) do p, pd
        dst = similar(src)
        mean = similar(src, eltype(src), (size(src, ndims(src) - 1),))
        variance = similar(mean)

        # Fuse the Relu activation function is possible.
        if activation === Base.identity
            execute!(p, @dnnl_args src scale_shift dst mean variance)
            return (; dst, mean, variance, workspace = nothing, forward = noforward())
        elseif activation === Flux.relu
            workspace = similar(src)
            execute!(p, @dnnl_args src scale_shift dst mean variance workspace)
            return (; dst, mean, variance, workspace, forward = pd)
        end
    end
end

function batchnorm_backward(
    diff_dst::Memory,
    src::Memory,
    scale_shift::Memory,
    mean::Memory,
    variance::Memory;
    forward = noforward(),
    epsilon = 1f-10,
    attributes = noattributes(),
    workspace = nothing,
)
    flags = (workspace === nothing) ? batchnorm_flags(identity) : batchnorm_flags(Flux.relu)
    opdesc = Ref{Lib.dnnl_batch_normalization_desc_t}()
    src = maybe_reorder(memorydesc(diff_dst), src)

    @apicall dnnl_batch_normalization_backward_desc_init(
        opdesc, Lib.dnnl_backward, diff_dst, src, epsilon, flags
    )

    return temp_primitive(opdesc, attributes, global_engine(), forward) do p, pd
        diff_src = similar(diff_dst)
        diff_scale_shift = similar(scale_shift)
        args = @dnnl_args diff_dst src mean variance scale_shift diff_src diff_scale_shift
        # Maybe append on the workspace as well.
        if workspace !== nothing
            args = append(args, @dnnl_args workspace)
        end
        execute!(p, args)
        return (; diff_src, diff_scale_shift)
    end
end

#####
##### Higher level API
#####

struct BatchNorm{T<:OneDNN.Memory,F}
    scale_shift::T
    activation::F
    epsilon::Float32
    attributes::Attributes
end

function BatchNorm(scale_shift::AbstractMatrix, activation = identity; epsilon = 1f-5)
    attributes = Attributes()
    postops = PostOps()
    eltwise!(postops, activation)
    append!(attributes, postops)
    return BatchNorm(OneDNN.Memory(scale_shift), activation, epsilon, attributes)
end

function (bn::BatchNorm)(x)
    return batchnorm_inference(Memory(x), bn.scale_shift; bn.epsilon, bn.attributes)
end

function set_scale_shift!(x::AbstractArray{T,2}) where {T}
    for i in eachindex(x)
        x[i] = randn(T)
    end
    return nothing
end

_batchnorm_fusable(::Any) = false
_batchnorm_fusable(::typeof(identity)) = true
_batchnorm_fusable(::typeof(Flux.relu)) = true

function ChainRulesCore.rrule(bn::BatchNorm, src::Memory)
    return if _batchnorm_fusable(bn.activation)
        batchnorm_rrule_fused(bn, src)
    else
        batchnorm_rrule_unfused(bn, src)
    end
end

function batchnorm_rrule_fused(bn::BatchNorm, src::Memory)
    nt = batchnorm_training(src, bn.scale_shift, bn.activation; bn.epsilon)

    function batchnorm_fused_pullback(diff_data::Memory)
        Δ = batchnorm_backward(
            diff_data,
            src,
            bn.scale_shift,
            nt.mean,
            nt.variance;
            nt.forward,
            bn.epsilon,
            nt.workspace,
        )
        return ((; scale_shift = Δ.diff_scale_shift), Δ.diff_src)
    end

    return (nt.dst, batchnorm_fused_pullback)
end

function batchnorm_rrule_unfused(bn::BatchNorm, src::Memory)
    nt = batchnorm_training(src, bn.scale_shift, identity; bn.epsilon)
    eltwise_output, eltwise_pullback = ChainRulesCore.rrule(eltwise, bn.activation, nt.dst)

    function batchnorm_unfused_pullback(_diff_data::Memory)
        # Reverse the eltwise function.
        diff_data = eltwise_pullback(_diff_data)[3]
        Δ = batchnorm_backward(
            diff_data, src, bn.scale_shift, nt.mean, nt.variance; bn.epsilon
        )
        return ((; scale_shift = Δ.diff_scale_shift), Δ.diff_src)
    end

    return (eltwise_output, batchnorm_unfused_pullback)
end
