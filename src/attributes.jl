#####
##### Attributes
#####

struct Attributes
    val::Ref{Lib.dnnl_primitive_attr_t}
end
Base.unsafe_convert(::Type{Ptr{Nothing}}, x::Attributes) = x.val[]

function Attributes()
    val = Ref{Lib.dnnl_primitive_attr_t}()
    @apicall Lib.dnnl_primitive_attr_create(val)

    finalizer(val) do _val
        @apicall Lib.dnnl_primitive_attr_destroy(_val[])
    end

    return Attributes(val)
end

#####
##### Scale
#####

function setscale!(attr::Attributes, scale::Number)
    @apicall Lib.dnnl_primitive_attr_set_output_scales(
        attr,
        # Only one scale - applied to the whole output tensor
        1,
        0,
        Ref(convert(Float32, scale)),
    )
    return nothing
end

#####
##### PostOps
#####

struct PostOps
    val::Ref{Lib.dnnl_post_ops_t}
end
Base.unsafe_convert(::Type{Ptr{Nothing}}, x::PostOps) = x.val[]

function PostOps()
    val = Ref{Lib.dnnl_post_ops_t}()
    @apicall Lib.dnnl_post_ops_create(val)

    finalizer(val) do _val
        @apicall Lib.dnnl_post_ops_destroy(_val[])
    end

    return PostOps(val)
end

function appendsum!(P::PostOps, scale = one(Float32))
    @apicall Lib.dnnl_post_ops_append_sum(P.val[], scale)
    return nothing
end

appendeltwise!(P::PostOps, ::typeof(identity), scale = one(Float32)) = nothing
function appendeltwise!(
        P::PostOps,
        f,
        scale = Float32(1.0),
    )

    @apicall Lib.dnnl_post_ops_append_eltwise(P, scale, algkind(f)...)
    return nothing
end

# Things that require both `Attributes` and `PostOps`
add!(a::Attributes, p::PostOps) = @apicall Lib.dnnl_primitive_attr_set_post_ops(a, p)
