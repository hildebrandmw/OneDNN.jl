using OneDNN

# stdlib
using Test
using Random

# for testing equivalence
# using Flux, Zygote
using Flux: Flux
using Zygote

include("tiled.jl")
include("utils.jl")
include("memory.jl")

include("ops/eltwise.jl")
include("ops/binary.jl")
include("ops/concat.jl")
include("ops/matmul.jl")
include("ops/innerproduct.jl")
