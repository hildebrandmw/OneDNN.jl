# OneDNN

A Julia wrapper for OneDNN.

## Example

The code snippets below show how to use the functionality of OneDNN.jl.

```julia
julia> using OneDNN

# Construct a standard Julia array.
julia> x = randn(Float32, 100, 100);

# Wrap the Julia array in a `OneDNN.Memory` type.
# This will allow OneDNN to use the memory owned by Julia.
# `Memory` is opaque because the OneDNN library can have exotic
# memory layouts.
julia> y = Memory(x)
Opaque Memory (100, 100)

# Perform element-wise computation.
julia> z = abs.(y)
Opaque Memory (100, 100)

# To make the memory accessible by julia, use `OneDNN.materialize`.
julia> OneDNN.materialize(z)
100×100 Matrix{Float32}
...

julia> OneDNN.materialize(z) == abs.(x)
true
```

# Exposed Primitives

### Elementwise

The following element wise primitive are exposed for both forward and backward propagation.
More can be added easily if desired.

* `Linear(alpha, beta)`: Scale each entry `x in X` by `alpha * x + beta`.
* `Base.abs`: Absolute value.
* `Flux.sigmoid`: Sigmoid activation function.
* `Base.sqrt`: Elementwise square root.
* `Flux.relu`: Relu activation function.
* `Base.log`: Elementwiase natural logarithm.

Low level eltwise functions can be accessed using the function
```julia
OneDNN.eltwise(src::OneDNN.Memory, kind::OneDNN.Lib.dnnl_alg_kind_t, alpha = one(Float32), beta = zero(Float32))
```
Where `src` is the source memory tensor, `kind` is the C enum exposed by the OneDNN C header, and `alpha`/`beta` are scaling parameters.

### Binary

The following binary primitives are exposed for both forward and backward propagation.

* `+`: Elementwise addition of two `Memory`s.
* `-`: Elementwise subtraction of two `Memory`s.
* `*`: Elementwise multiplication.
* `/`: Elementwise division.
* `min`: Elementwise min.
* `max`: Elementwise min.

### Normalization

Both `softmax` and `logsoftmax` are exposed.
```julia
julia> using OneDNN

julia> x = rand(Float32, 5)
5-element Vector{Float32}:
 0.84481
 0.5697744
 0.6269949
 0.72741866
 0.5301513

julia> OneDNN.materialize(OneDNN.softmax(Memory(x), 1))
5-element Vector{Float32}:
 0.23905747
 0.18157493
 0.19226773
 0.21257876
 0.17452104
```

### Dense Layers

Dense layers can be constructed directly or taken from existing `Flux.Dense` layers.

```julia
julia> using Flux, Zygote, OneDNN

# Construct a simple MLP
julia> flux = Chain(
    Dense(10 => 20, Flux.relu),
    Dense(20 => 40, Flux.relu),
    Dense(40 => 10, Flux.sigmoid),
)
Chain(
  Dense(10 => 20, relu),                # 220 parameters
  Dense(20 => 40, relu),                # 840 parameters
  Dense(40 => 10, σ),                   # 410 parameters
)                   # Total: 6 arrays, 1_470 parameters, 6.117 KiB.

julia> onednn = Chain(OneDNN.Dense(flux)...)
Chain(
  Dense(10 => 20, relu),                # 220 parameters
  Dense(20 => 40, relu),                # 840 parameters
  Dense(40 => 10, σ),                   # 410 parameters
)                   # Total: 6 arrays, 1_470 parameters, 6.562 KiB.

# Forward pass works for both
julia> x = randn(Float32, 10, 16)

julia> isapprox(OneDNN.materialize(onednn(x)), flux(x))
true

# We can also compute backwards passes
julia> y, pullback = Zygote._pullback(onednn, x)
(Opaque Memory (10, 16), ∂(λ))

julia> pullback(y)
...
```
As a technical detail, activations will be fused where possible.
Also, note that we don't need to convert `x` to a `OneDNN.Memory` when passing it into a dense layer.

### Convolution Layers

Convolution layers function much like dense layers, except that OneDNN's convolution is really mapped
to Flux's cross-correlation.
```julia
julia> flux = Chain(
       Flux.CrossCor((3, 3), 10 => 20, identity),
       Flux.CrossCor((5, 5), 20 => 40, Flux.sigmoid),
       Flux.CrossCor((2, 2), 40 => 10, Flux.relu),
       )
Chain(
  CrossCor((3, 3), 10 => 20),           # 1_820 parameters
  CrossCor((5, 5), 20 => 40, σ),        # 20_040 parameters
  CrossCor((2, 2), 40 => 10, relu),     # 1_610 parameters
)                   # Total: 6 arrays, 23_470 parameters, 92.664 KiB.

julia> onednn = Chain(OneDNN.Conv.(flux)...)
Chain(
  Conv((3, 3), 10 => 20),               # 1_820 parameters
  Conv((5, 5), 20 => 40, σ),            # 20_040 parameters
  Conv((2, 2), 40 => 10, relu),         # 1_610 parameters
)                   # Total: 6 arrays, 23_470 parameters, 133.953 KiB

julia> x = randn(Float32, 32, 32, 10, 32);

# Again, we don't need to wrap `x` in a `OneDNN.Memory` when passing it to `onednn`.
# This is handled automatically.
julia> isapprox(OneDNN.materialize(onednn(x)), flux(x))
true

# Also, reverse passes can be constructed.
julia> y, pullback = Zygote._pullback(onednn, x)
(Opaque Memory (25, 25, 10, 32), ∂(λ))

julia> pullback(y)
...
```

### Concatenation

Multiple memories can be concatenated together, complete with the corresponding backwards pass.
```julia
julia> using OneDNN

julia> x, y, z = ntuple(_-> randn(Float32, 2, 2), Val(3));

julia> X, Y, Z = OneDNN.Memory.((x, y, z))
(Opaque Memory (2, 2), Opaque Memory (2, 2), Opaque Memory (2, 2))

julia> OneDNN.materialize(OneDNN.concat((X, Y, Z), 1)) == vcat(x, y, z)
true

julia> OneDNN.materialize(OneDNN.concat((X, Y, Z), 2)) == hcat(x, y, z)
true
```

### Pooling

Forward and backwards pooling (max and mean) are supported.
The compatibility with Flux is not yet super nice for these layers unfortunately.

```julia
julia> using OneDNN

julia> pool = OneDNN.MaxPool((3, 3); stride = 1, pad = 1);

julia> x = randn(Float32, 12, 12, 16, 16);

julia> pool(x);
Opaque Memory (12, 12, 16, 16)
```

### Batch Normalization

The batch-normalization primitive is supported using the scale-shift technique.
The array `scale_shift` is a 2-D array with per-channel scaling factors in the first column and per-channel shift factors in the second column.
```julia
julia> using OneDNN

julia> scale_shift = hcat(2 .* ones(Float32, 4), 0.1f0 .* ones(Float32, 4))
4×2 Matrix{Float32}:
 2.0  0.1
 2.0  0.1
 2.0  0.1
 2.0  0.1

julia> bn = OneDNN.BatchNorm(scale_shift);

julia> x = ones(Float32, 4, 4)
4×4 Matrix{Float32}:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0

julia> OneDNN.materialize(bn(x))
  0.000116 seconds (4 allocations: 176 bytes)
4×4 Matrix{Float32}:
 0.1  0.1  0.1  0.1
 0.1  0.1  0.1  0.1
 0.1  0.1  0.1  0.1
 0.1  0.1  0.1  0.1
```
