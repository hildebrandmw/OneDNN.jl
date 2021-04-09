@testset "Testing Binary" begin
    x = randn(Float32, 10, 10)
    y = randn(Float32, 10, 10)

    X = OneDNN.Memory(x)
    Y = OneDNN.Memory(y)

    fns = [+, -, *, /, max, min]
    for f in fns
        @show f
        z = f.(x, y)

        op = OneDNN.Binary(f, X, Y)
        Z = op(X, Y)
        @test isapprox(z, Z)
    end
end
