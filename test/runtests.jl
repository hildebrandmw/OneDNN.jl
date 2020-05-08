using OneDNN
using NNlib

# stdlib
using Test
using Random

@testset "OneDNN.jl" begin
    x = randn(Float32, 10, 10, 10, 10)

    # Elementwise Ops
    ops = [
        NNlib.relu,
        Base.tanh
    ]

    for op in ops
        I = OneDNN.Initializer(op)
        reference = op.(x)

        @test OneDNN.isinitialized(I) == false
        @test isapprox(OneDNN.getdata(I(x)), reference)
        @test OneDNN.isinitialized(I) == true

        # Pull out the primitive from inside I
        f = OneDNN.unpack(I)
        @test isapprox(OneDNN.getdata(f(x)), reference)
    end

    # Binary Ops
    ops = [
        +,
        *,
        min,
        max
    ]

    x = randn(Float32, 10, 10, 10, 10)
    y = randn(Float32, 10, 10, 10, 10)

    for op in ops
        I = OneDNN.Initializer(op)
        reference = op.(x, y)

        @test OneDNN.isinitialized(I) == false
        @test isapprox(OneDNN.getdata(I(x,y)), reference)
        @test OneDNN.isinitialized(I) == true

        # Pull out the primitive from inside I
        f = OneDNN.unpack(I)
        @test isapprox(OneDNN.getdata(f(x,y)), reference)
    end
end
