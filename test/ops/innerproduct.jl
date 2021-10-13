@testset "Testing Inner Product" begin
    # Note: use `abs` to serve as an example of an activation function that cannot be fused
    # with the `innerproduct` kernel.
    activations = [identity, Flux.relu, Flux.sigmoid, abs]
    for σ in activations
        @show σ
        m = Flux.Dense(128, 256, σ)
        M = OneDNN.Dense(m)

        x = randn(Float32, 128, 128)
        y = m(x)

        X = OneDNN.Memory(x)
        @test M.optimized_weights == false
        Y = M(X)
        # Did the automatic weight transformation take place?
        @test M.optimized_weights
        @test isapprox(y, OneDNN.materialize(Y))

        # Ensure inference works correctly
        @inferred M(X)

        # Now - try backprop
        y, back_jl = Zygote._pullback(m, x)
        Y, back_dnnl = Zygote._pullback(M, X)
        @test isa(Y, OneDNN.Memory)
        @test isapprox(y, OneDNN.materialize(Y))
        @inferred Zygote._pullback(M, X)
        out, back = Zygote._pullback(M, X)
        @inferred back(out)

        dy = Float32(0.125) * randn(Float32, size(y))
        DY = OneDNN.Memory(dy)

        grads_jl = back_jl(dy)
        grads_dnnl = back_dnnl(DY)
        @test isa(grads_jl, Tuple)
        @test isa(grads_dnnl, Tuple)

        # N.B.: Notice the difference in the plurality of the `weights` field ...
        # In the OneDNN code, we try to stay with the plural version since that's what the
        # oneDNN source code generally does.
        @test isapprox(
            grads_jl[1].weight, transpose(OneDNN.materialize(grads_dnnl[1].weights))
        )
        @test isapprox(grads_jl[1].bias, OneDNN.materialize(grads_dnnl[1].bias))
        @test isapprox(grads_jl[2], OneDNN.materialize(grads_dnnl[2]))
        @inferred back_dnnl(DY)
    end
end

