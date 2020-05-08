abstract type AbstractDispatch end

# Element-wise Operations
abstract type AbstractEltwiseOp <: AbstractDispatch end

struct Relu <: AbstractEltwiseOp end
struct Tanh <: AbstractEltwiseOp end

algokind(::Relu) = Lib.dnnl_eltwise_relu
algokind(::Tanh) = Lib.dnnl_eltwise_tanh

#####
##### Binary Operations
#####
abstract type AbstractBinaryOp <: AbstractDispatch end

struct BinaryAdd <: AbstractBinaryOp end
struct BinaryMul <: AbstractBinaryOp end
struct BinaryMax <: AbstractBinaryOp end
struct BinaryMin <: AbstractBinaryOp end

algokind(::BinaryAdd) = Lib.dnnl_binary_add
algokind(::BinaryMul) = Lib.dnnl_binary_mul
algokind(::BinaryMax) = Lib.dnnl_binary_max
algokind(::BinaryMin) = Lib.dnnl_binary_min

#####
##### Convolutions
#####

struct ConvolutionForward{T,N} <: AbstractDispatch
    kernel_size::NTuple{N,Int64}
    padding::NTuple{4,Int64}
end

algokind(::ConvolutionForward) = Lib.dnnl_convolution_auto

#####
##### Forwarding Methods
#####

# Get a dispatcher for normal functions.
dispatcher(x::AbstractDispatch) = x

# Eltwise
dispatcher(::typeof(NNlib.relu)) = Relu()
dispatcher(::typeof(Base.tanh)) = Tanh()

# Binary
#
# TODO: Should really hijack the broadcasted version of these ...
dispatcher(::typeof(Base.:+)) = BinaryAdd()
dispatcher(::typeof(Base.:*)) = BinaryMul()
dispatcher(::typeof(Base.max)) = BinaryMax()
dispatcher(::typeof(Base.min)) = BinaryMin()

