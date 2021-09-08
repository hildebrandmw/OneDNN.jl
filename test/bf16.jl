@testset "Testing BF16" begin
    # Note: use `abs` to serve as an example of an activation function that cannot be fused
    # with the `innerproduct` kernel.
    activations = [identity, Flux.relu, Flux.sigmoid, abs]
    BFloat16 = OneDNN.BFloat16
    for σ in activations
        @show σ
        m = Flux.Dense(128, 256, σ)
        M32 = OneDNN.Dense(m)
        M16 = OneDNN.Dense(
            convert.(BFloat16, transpose(m.weight)),
            convert.(BFloat16, m.bias),
            σ,
        )

        x32 = randn(Float32, 128, 128)
        y = m(x32)

        X32 = OneDNN.Memory(x32)
        X16 = OneDNN.Memory(convert.(BFloat16, x32))

        @test M32.optimized_weights == false
        @test M16.optimized_weights == false

        Y32 = M32(X32)
        Y16 = M16(X16)
        # Did the automatic weight transformation take place?
        @test M32.optimized_weights
        @test M16.optimized_weights

        @test isapprox(y, OneDNN.materialize(Y32))
        @test isapprox(y, OneDNN.materialize(Y16))

        # Ensure inference works correctly
        @inferred M32(X32)
        @inferred M16(X16)

        # Now - try backprop
        y, back_jl = Zygote._pullback(m, x32)
        Y32, back_dnnl32 = Zygote._pullback(M32, X32)
        Y16, back_dnnl16 = Zygote._pullback(M16, X16)

        @test isa(Y32, OneDNN.Memory)
        @test isa(Y16, OneDNN.Memory)

        @test isapprox(y, OneDNN.materialize(Y32))
        @test isapprox(y, OneDNN.materialize(Y16))

        @inferred Zygote._pullback(M32, X32)
        @inferred Zygote._pullback(M16, X16)

        dy = Float32(0.125) * randn(Float32, size(y))
        DY32 = OneDNN.Memory(dy)
        DY16 = OneDNN.Memory(convert.(BFloat16, dy))

        grads_jl = back_jl(dy)
        grads_dnnl32 = back_dnnl32(DY32)
        grads_dnnl16 = back_dnnl16(DY16)

        @test isa(grads_jl, Tuple)
        @test isa(grads_dnnl32, Tuple)
        @test isa(grads_dnnl16, Tuple)

        # N.B.: Notice the difference in the plurality of the `weights` field ...
        # In the OneDNN code, we try to stay with the plural version since that's what the
        # oneDNN source code generally does.
        @test isapprox(
            grads_jl[1].weight, transpose(OneDNN.materialize(grads_dnnl32[1].weights))
        )
        @test isapprox(
            grads_jl[1].weight, transpose(OneDNN.materialize(grads_dnnl16[1].weights))
        )

        @test isapprox(grads_jl[1].bias, OneDNN.materialize(grads_dnnl32[1].bias))
        @test isapprox(grads_jl[1].bias, OneDNN.materialize(grads_dnnl16[1].bias))
        @test isapprox(grads_jl[2], OneDNN.materialize(grads_dnnl32[2]))
        @test isapprox(grads_jl[2], OneDNN.materialize(grads_dnnl16[2]))
        @inferred back_dnnl32(DY32)
        @inferred back_dnnl16(DY16)
    end
end
