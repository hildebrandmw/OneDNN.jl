using OneDNN

# stdlib
using Test
using Random
import Statistics

# for testing equivalence
# using Flux, Zygote
import CEnum
using Flux: Flux
import ProgressMeter
using Zygote

include("tiled.jl")
include("utils.jl")
include("memory.jl")
include("ops/simple.jl")
include("ops/batchnorm.jl")
include("ops/matmul.jl")
include("ops/innerproduct.jl")
include("ops/pool.jl")
include("ops/concat.jl")

