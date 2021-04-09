@testset "Testing Eltwise" begin
    # Testing normal eltwise functions

    ## Linear
    f = (x, α, β) -> (α .* x) .+ β
    x = rand(Float32, 10, 10)
    X = OneDNN.Memory(x)
    op = OneDNN.Eltwise(OneDNN.Linear(1, 2), X)

    Y = op(X)
    @test isapprox(Y, f(x, 1, 2))

    # Try mutating version
    Y .= zero(eltype(Y))
    @test all(iszero, Y)
    op(Y, X)
    @test isapprox(Y, f(x, 1, 2))

    # Backprop
    ff(x) = 0.5 .* x .+ 2
    y, back = Zygote._pullback(ff, x)
    z = ones(Float32, size(y))
    Z = OneDNN.Memory(z)
    op_back = OneDNN.EltwiseBackward(OneDNN.Linear(0.5, 2), Y, Z)

    expected = back(z)[2]
    result = op_back(Z, X)
    @test isapprox(result, expected)

    #####
    ##### Standard Functions
    #####

    fns = [abs, sqrt]
    x = rand(Float32, 100, 100)
    X = OneDNN.Memory(x)
    for f in fns
        @show f
        F(x) = f.(x)
        y, back = Zygote._pullback(F, x)

        op_forward = OneDNN.Eltwise(f, X)
        Y = op_forward(X)
        @test isapprox(y, Y)

        dy = rand(Float32, size(y))
        dY = OneDNN.Memory(dy)
        op_backward = OneDNN.EltwiseBackward(f, dY, Y)

        dX = op_backward(dY, X)
        dx = back(dy)[2]
        @test isapprox(dx, dX)
    end

    #####
    ##### The `use_bwd_for_dst` functions.
    #####

    fns = [Flux.relu, Flux.sigmoid]
    x = rand(Float32, 100, 100)
    X = OneDNN.Memory(x)
    for _f in fns
        @show _f
        f(x) = _f.(x)
        y, back = Zygote._pullback(f, x)

        op_forward = OneDNN.Eltwise(_f, X)
        Y = op_forward(X)
        @test isapprox(y, Y)

        dy = rand(Float32, size(y))
        dY = OneDNN.Memory(dy)
        op_backward = OneDNN.EltwiseBackward(_f, dY, Y)

        dX = op_backward(dY, Y)
        dx = back(dy)[2]
        @test isapprox(dx, dX)
    end
end
