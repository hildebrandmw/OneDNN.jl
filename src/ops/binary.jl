#####
##### forwards
#####

binary_algkind(::typeof(+)) = Lib.dnnl_binary_add
binary_algkind(::typeof(*)) = Lib.dnnl_binary_mul
binary_algkind(::typeof(max)) = Lib.dnnl_binary_max
binary_algkind(::typeof(min)) = Lib.dnnl_binary_min

# operator overloading
Base.:+(a::Memory, b::Memory) = binary(+, a, b)
Base.:*(a::Memory, b::Memory) = binary(*, a, b)
Base.max(a::Memory, b::Memory) = binary(max, a, b)
Base.min(a::Memory, b::Memory) = binary(min, a, b)

# hijack broadcasting as well
Base.broadcasted(::typeof(+), a::Memory, b::Memory) = binary(+, a, b)
Base.broadcasted(::typeof(+), a::Memory, b::AbstractArray) = binary(+, a, memory(b))

Base.broadcasted(::typeof(*), a::Memory, b::Memory) = binary(*, a, b)
Base.broadcasted(::typeof(min), a::Memory, b::Memory) = binary(min, a, b)
Base.broadcasted(::typeof(max), a::Memory, b::Memory) = binary(max, a, b)



#####
##### implementation
#####

binary(f::F, src0, src1) where {F} = binary(f, memory(src0), memory(src1))
function binary(f::F, src0::Memory, src1::Memory) where {F}
    # Construct a destination descriptor based on the source
    dst_desc_temp = memorydesc(eltype(src0), size(src0), dnnl_format_any())

    # op descriptor
    od = Ref{Lib.dnnl_binary_desc_t}()
    @apicall Lib.dnnl_binary_desc_init(
        od,
        binary_algkind(f),
        memorydesc_ptr(src0),
        memorydesc_ptr(src1),
        Ref(dst_desc_temp),
    )

    # primitive descriptor
    pd = primitive_descriptor(
        od,
        Ptr{Nothing}(),
        global_engine(),
        Ptr{Nothing}()
    )

    # query pd for output format.
    dst_desc = Lib.dnnl_primitive_desc_query_md(pd, Lib.dnnl_query_dst_md, 0)
    dst = similar(src0, eltype(src0), size(src0), unsafe_load(dst_desc))

    # primitive
    args = [
        arg(Lib.DNNL_ARG_SRC_0, src0),
        arg(Lib.DNNL_ARG_SRC_1, src1),
        arg(Lib.DNNL_ARG_DST, dst),
    ]

    p = primitive(pd)
    execute!(p, args)

    # cleanup
    destroy(p, pd)
    return dst
end
