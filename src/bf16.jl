struct BFloat16 <: AbstractFloat
    val::UInt16
end

const MAGIC = Float32(1 + 2^16 * eps(Float32) / 2)
Base.convert(::Type{Float32}, x::BFloat16) = reinterpret(Float32, UInt32(x.val) << 16)
Base.convert(::Type{BFloat16}, x::Float32) = BFloat16(UInt16(reinterpret(UInt32, x * MAGIC) >> 16))
Base.convert(::Type{BFloat16}, x::Float64) = convert(BFloat16, convert(Float32, x))

BFloat16(x::Float32) = convert(BFloat16, x)

Base.zero(::Type{BFloat16}) = BFloat16(zero(UInt16))
dnnl_type(::Type{BFloat16}) = Lib.dnnl_bf16

Base.promote_rule(::Type{BFloat16}, ::Type{Float32}) = Float32

# This definition is a little weird, but should work just fine for now.
Base.:+(a::BFloat16, b::BFloat16) = convert(Float32, a) + convert(Float32, b)
Base.:-(a::BFloat16, b::BFloat16) = convert(Float32, a) - convert(Float32, b)
#Random.randn(::Type{BFloat16}) = convert(BFloat16, randn(Float32))

Base.eps(x::BFloat16) = 2^16 * convert(Float32, x)
Base.eps(::Type{BFloat16}) = 2^16 * eps(Float32)
LinearAlgebra.norm(x::BFloat16) = LinearAlgebra.norm(convert(Float32, x))
Base.abs2(x::BFloat16) = Base.abs2(convert(Float32, x))

#####
##### Extended Memory
#####

# Keeping the weights as BF16 has nice performance implications, but doesn't tend to
# converge as quickly as Float32 training.
#
# This struct will mirror the wrapped BFloat16 memory with another BFloat16 that will hold
# the lower mantissa bits for better numerical properties.
struct Mirrored{A <: AbstractArray{BFloat16}, B <: AbstractArray{UInt16}}
    base::A
    mantissa::B
end

function Mirrored(base::AbstractArray{BFloat16})
    mantissa = similar(base, UInt16)
    return Mirrored(base, mantissa)
end

function Base.parent(x::Mirrored)
    return Mirrored(parent(x.base), parent(x.mantissa))
end

Base.eachindex(x::Mirrored) = eachindex(x.base, b.mantissa)

Base.@propagate_inbounds function Base.getindex(x::Mirrored, i::Int)
    (hi, lo) = (x.base[i], x.mantissa[i])
    return reinterpret(Float32, UInt32(hi.val) << 16 + lo)
end

Base.@propagate_inbounds function Base.setindex!(x::Mirrored, v::Float32, i::Int)
    vint = reinterpret(UInt32, v)
    hi = UInt16(vint >> 16)
    lo = UInt16(vint & typemax(UInt16))
    (x.base[i], x.mantissa[i]) = (hi, lo)
end

