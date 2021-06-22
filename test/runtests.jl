using OneDNN

# stdlib
using Test
using Random

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
include("ops/matmul.jl")
include("ops/innerproduct.jl")
include("ops/concat.jl")
