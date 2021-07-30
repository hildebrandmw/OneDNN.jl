@testset "Testing Simple Ops" begin
    @testset "Testing Eltwise" begin
        # Testing normal eltwise functions

        ## Linear
        f = (x, α, β) -> (α .* x) .+ β
        x = rand(Float32, 10, 10)
        X = OneDNN.Memory(x)
        linear = OneDNN.Linear(1, 2)
        Y = OneDNN.eltwise(linear, X)
        Z = OneDNN.materialize(Y)
        @test isapprox(Z, f(x, 1, 2))

        # Try mutating version
        Z .= zero(eltype(Z))
        @test all(iszero, Z)
        OneDNN.eltwise!(OneDNN.Memory(Z), X, OneDNN.forward_expand(linear)...)
        @test isapprox(Z, f(x, 1, 2))

        # Backprop
        ff(x) = 0.5 .* x .+ 2
        y, back = Zygote._pullback(ff, x)
        z = ones(Float32, size(y))
        Z = OneDNN.Memory(z)
        result = OneDNN.eltwise_backward(OneDNN.Linear(0.5, 2), Z, X)

        expected = back(z)[2]
        @test isapprox(OneDNN.materialize(result), expected)

        #####
        ##### Aliased Functions
        #####

        fns = [abs, Flux.sigmoid, sqrt, Flux.relu, log]
        x = rand(Float32, 100, 100)
        X = OneDNN.Memory(x)
        for f in fns
            @show f
            F(x) = f.(x)
            y, back_expected = Zygote._pullback(F, x)
            Y, back_result = Zygote._pullback(OneDNN.eltwise, f, X)

            @test isapprox(OneDNN.materialize(Y), y)
            @inferred Zygote._pullback(OneDNN.eltwise, f, X)

            dy = rand(Float32, size(y))
            dY = OneDNN.Memory(dy)

            dX = back_result(dY)[3]
            dx = back_expected(dy)[2]
            @test isapprox(dx, OneDNN.materialize(dX))
            @inferred back_result(dY)

            # Make sure the alias rules work as well.
            YY, back_result_2 = Zygote._pullback(f, X)
            @test isapprox(OneDNN.materialize(YY), y)
            dXX = back_result_2(dY)[3]
            @test isapprox(dx, OneDNN.materialize(dXX))
            @inferred back_result_2(dY)
        end
    end

    @testset "Testing Binary" begin
        x = randn(Float32, 100, 100)
        y = randn(Float32, 100, 100)

        X = OneDNN.Memory(x)
        Y = OneDNN.Memory(y)

        fns = [+, -, *, /, max, min]
        for f in fns
            @show f
            z = f.(x, y)
            Z = f.(X, Y)
            @test isa(Z, OneDNN.Memory)
            @test isapprox(z, OneDNN.materialize(Z))
        end
    end
end
