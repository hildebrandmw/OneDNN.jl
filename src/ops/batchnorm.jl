# TODO: Nice API for all the different forms for scale and shift?
batchnorm_flags(::typeof(identity)) = Lib.dnnl_use_scaleshift
batchnorm_flags(::typeof(Flux.relu)) = Lib.dnnl_use_scaleshift | Lib.dnnl_fuse_norm_relu

function execute_batchnorm(
    ::typeof(identity),
    primitive::Primitive,
    desc::PrimitiveDescriptor,
    src,
    scale_shift,
    dst,
    mean,
    variance,
)
    execute!(primitive, @dnnl_args src scale_shift dst mean variance)
    return (; dst, mean, variance, workspace = nothing, forward = noforward())
end

function execute_batchnorm(
    ::typeof(Flux.relu),
    primitive::Primitive,
    desc::PrimitiveDescriptor,
    src,
    scale_shift,
    dst,
    mean,
    variance,
)
    workspace = similar(src)
    execute!(primitive, @dnnl_args src scale_shift dst mean variance workspace)
    return (; dst, mean, variance, workspace, forward = desc)
end

function batchnorm_forward_training(
    src::Memory,
    scale_shift::Memory,
    activation::F = identity;
    epsilon = 1f-10,
    attributes = noattributes(),
) where {F<:Union{typeof(identity),typeof(Flux.relu)}}
    batch_normalization_desc = Ref{Lib.dnnl_batch_normalization_desc_t}()

    flags = batchnorm_flags(activation)
    @apicall dnnl_batch_normalization_forward_desc_init(
        batch_normalization_desc, Lib.dnnl_forward_training, src, epsilon, flags
    )

    return temp_primitive(
        batch_normalization_desc, attributes, global_engine(), noforward()
    ) do primitive, primitive_desc
        dst = similar(src)

        # Allocate mean and variance tensors as well.
        md = memorydesc(eltype(src), (size(src, ndims(src) - 1),), Lib.dnnl_a)
        mean = similar(src, eltype(src), (size(src, ndims(src) - 1),), md)
        variance = similar(mean)
        return execute_batchnorm(
            activation, primitive, primitive_desc, src, scale_shift, dst, mean, variance
        )
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
    batch_normalization_desc = Ref{Lib.dnnl_batch_normalization_desc_t}()
    flags = (workspace === nothing) ? batchnorm_flags(identity) : batchnorm_flags(Flux.relu)
    @apicall dnnl_batch_normalization_backward_desc_init(
        batch_normalization_desc, Lib.dnnl_backward, diff_dst, src, epsilon, flags
    )

    return temp_primitive(
        batch_normalization_desc, attributes, global_engine(), forward
    ) do primitive, primitive_desc
        diff_src = similar(src)
        diff_scale_shift = similar(scale_shift)
        args = @dnnl_args diff_dst src mean variance scale_shift diff_src diff_scale_shift
        # Maybe append on the workspace as well.
        if workspace !== nothing
            args = append(args, @dnnl_args workspace)
        end
        execute!(primitive, args)
        return (; diff_src, diff_scale_shift)
    end
end

#####
##### Higher level API
#####

struct BatchNorm{T<:OneDNN.Memory,F}
    scale_shift::T
    epsilon::Float32
    activation::F
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
    nt = batchnorm_forward_training(src, bn.scale_shift, bn.activation; bn.epsilon)

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
    nt = batchnorm_forward_training(src, bn.scale_shift, identity; bn.epsilon)
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
