@testset "Testing Eltwise" begin
    # Testing normal eltwise functions

    ## Linear
    f = (x, α, β) -> (α .* x) .+ β
    x = rand(Float32, 10, 10)

    # non-mutating
    Y = OneDNN.linear(x)
    @test isapprox(OneDNN.materialize(Y), f(x, Float32(1.0), Float32(0.0)))

    Y = OneDNN.linear(x, 2.0, -10.0)
    @test isapprox(OneDNN.materialize(Y), f(x, 2.0, -10.0))

    # mutating
    X = OneDNN.memory(x)
    Y = similar(X)

    OneDNN.linear!(Y, X, -5, 20)
    @test isapprox(OneDNN.materialize(Y), f(x, -5, 20))

    # inplace - copy `x` because memory will just wrap it otherwise.
    X = OneDNN.memory(copy(x))
    OneDNN.linear!(X, -1)
    @test isapprox(OneDNN.materialize(X), f(x, -1, 0))

    #####
    ##### Backprop
    #####

    # Does backprop work correctly.
    activation_functions = [
        identity,
        Flux.relu,
        Flux.sigmoid,
    ]

    for fn in activation_functions
        # Use Zygote to generate the pullback
        f = x -> fn.(x)

        x = randn(Float32, 10, 10)
        y, back = Zygote._pullback(f, x)

        # Generate the expected result from Zygote.
        # Then, use the backwards kernels from OneDNN and check for equality.
        expected = back(x)[2]

        val = OneDNN.materialize(OneDNN.backprop_eltwise_dst(fn, y, x))
        @test isapprox(expected, val)
    end
end
