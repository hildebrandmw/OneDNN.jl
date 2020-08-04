@testset "Testing Binary" begin
    x = randn(Float32, 10, 10)
    y = randn(Float32, 10, 10)

    X = OneDNN.memory(x)
    Y = OneDNN.memory(y)

    # Just your basic ops
    @test isapprox(x + y, OneDNN.materialize(X + Y))
    @test isapprox(x .+ y, OneDNN.materialize(X .+ Y))
    @test isapprox(x .* y, OneDNN.materialize(X .* Y))
    @test isapprox(x .* y, OneDNN.materialize(X .* Y))
    @test isapprox(max.(x, y), OneDNN.materialize(max(X, Y)))
    @test isapprox(max.(x, y), OneDNN.materialize(max.(X, Y)))
    @test isapprox(min.(x, y), OneDNN.materialize(min.(X, Y)))
    @test isapprox(min.(x, y), OneDNN.materialize(min.(X, Y)))

    # Test the inplace versions
    #
    # This implementation over-writes the `src0` argument (in this case, `X`)
    for op in [+, *, min, max]
        X = OneDNN.memory(copy(x))
        Y = OneDNN.memory(copy(y))

        OneDNN.binary!(op, X, Y)
        @test isapprox(OneDNN.materialize(X), broadcast(op, x, y))
    end
end
