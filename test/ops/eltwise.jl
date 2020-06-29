@testset "Testing Eltwise" begin
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
