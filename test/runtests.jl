using OneDNN

# stdlib
using Test
using Random
import Statistics

# deps
import CEnum
import Flux
import ProgressMeter
using Zygote

#include("tiled.jl")
include("utils.jl")
include("memory.jl")
include("ops/simple.jl")
include("ops/matmul.jl")
include("ops/batchnorm.jl")
include("ops/innerproduct.jl")
include("ops/pool.jl")
include("ops/concat.jl")
include("ops/convolution.jl")

# include("bf16.jl")
