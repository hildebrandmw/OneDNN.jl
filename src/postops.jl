#####
##### Attributes and Post ops
#####

struct Attributes
    val::Ref{Lib.dnnl_primitive_attr_t}
end

function Attributes()
    val = Ref{Lib.dnnl_primitive_attr_t}
    @apicall Lib.dnnl_primitive_attr_create(val)

    finalizer(val) do _val
        @apicall Lib.dnnl_primitive_attr_destroy(_val[])
    end

    return Attributes(val)
end


struct PostOps
    val::Ref{Lib.dnnl_post_ops_t}
end

function PostOps()
    val = Ref{dnnl_ops_t}()
    @apicall Lib.dnnl_post_ops_create(po)

    finalizer(val) do _val
        @apicall Lib.dnnl_post_ops_destroy(_val[])
    end

    return PostOps(val)
end

function appendsum!(P::PostOps, scale = one(Float32))
    @apicall Lib.dnnl_post_ops_append_sum(P.val[], scale)
    return nothing
end

set_postops!(a::Attributes, p::PostOps) = @apicall Lib.dnnl_primitive_attr_set_post_ops(a, p)
