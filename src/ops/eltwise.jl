# Element wise ops.

#function eltwise(kind, src, algo, alpha, beta)
#    src = memory(src)
#end

#####
##### Compatibility layer from Julia eltwise functions to OneDNN eltwise functions
#####

algkind(::typeof(Flux.relu)) = (Lib.dnnl_eltwise_relu, zero(Float32), zero(Float32))
algkind(::typeof(Flux.sigmoid)) = (Lib.dnnl_eltwise_logistic, zero(Float32), zero(Float32))

algkind_back_dst(::typeof(Flux.relu)) = (Lib.dnnl_eltwise_relu_use_dst_for_bwd, zero(Float32), zero(Float32))
algkind_back_dst(::typeof(Flux.sigmoid)) = (Lib.dnnl_eltwise_logistic_use_dst_for_bwd, zero(Float32), zero(Float32))

# Can the backprop for this function be computed from the dst tensor?
# If so, we can fuse this on the forward pass.
back_from_dst(::Any) = false
back_from_dst(::typeof(Flux.relu)) = true
back_from_dst(::typeof(Flux.sigmoid)) = true
back_from_dst(::typeof(identity)) = true

#####
##### Backprop
#####

backprop_eltwise_dst(f, dst, Δ) = backprop_eltwise_dst(f, memory(dst), memory(Δ))
function backprop_eltwise_dst(f, dst::Memory, diff_dst::Memory)
    diff_src = similar(dst)

    # op descriptor
    opd = Ref{Lib.dnnl_eltwise_desc_t}()
    kind, alpha, beta = algkind_back_dst(f)
    @apicall Lib.dnnl_eltwise_backward_desc_init(
        opd,
        kind,
        memorydesc_ptr(diff_dst),
        memorydesc_ptr(dst),
        alpha,
        beta,
    )

    # primitive descriptor
    pd = primitive_descriptor(opd, Ptr{Nothing}(), global_engine(), Ptr{Nothing}())

    # primitive
    args = [
        arg(Lib.DNNL_ARG_DST, dst),
        arg(Lib.DNNL_ARG_DIFF_DST, diff_dst),
        arg(Lib.DNNL_ARG_DIFF_SRC, diff_src),
    ]

    p = primitive(pd)
    execute!(p, args)

    # cleanup
    destroy(p, pd)

    return diff_src
end

backprop_eltwise_dst(::typeof(identity), dst::Memory, diff_dst::Memory) = diff_dst
