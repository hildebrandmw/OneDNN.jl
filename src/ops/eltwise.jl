# Element wise ops.

#function eltwise(kind, src, algo, alpha, beta)
#    src = memory(src)
#end

#####
##### Traits
#####

algkind(::typeof(Flux.relu)) = (Lib.dnnl_eltwise_relu, zero(Float32), zero(Float32))
algkind(::typeof(Flux.sigmoid)) = (Lib.dnnl_eltwise_logistic, zero(Float32), zero(Float32))

# Can the backprop for this function be computed from the dst tensor?
# If so, we can fuse this on the forward pass.
back_from_dst(::Any) = false
back_from_dst(::typeof(Flux.relu)) = true
back_from_dst(::typeof(Flux.sigmoid)) = true
back_from_dst(::typeof(identity)) = true

#####
##### Backprop
#####

backprop_eltwise(::typeof(identity), x) = x
