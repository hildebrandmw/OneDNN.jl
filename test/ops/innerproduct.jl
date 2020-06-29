@testset "Testing Innerproduct" begin
    # In theory, the inner product should be equivalent to Flux's `dense` layer.
    # For the moment, we'll just use the `identity` activation function, but eventually
    # I'd like to have activation functions incorporated.

    # Construct a normal Flux Dense layer
    # Extract the weights and bias to test against `innerproduct`.
    m = Dense(10, 5, identity)

    weights = m.W
    bias = m.b
    src = rand(Float32, 10, 10)

    expected = m(src)
    result = OneDNN.materialize(OneDNN.inner_product_forward(src, transpose(weights), bias))
    @test isapprox(expected, result)

    # Now, we start seeing if backprop is working correctly.
    m = Dense(10, 5, identity)
    M = OneDNN.dense_from_flux(m)

    x = rand(Float32, 10, 10)
    @test isapprox(m(x), OneDNN.materialize(M(x)))

    y_baseline, back_baseline = Zygote._pullback(m, x)
    y_test, back_test = Zygote._pullback(M, x)

    z = m(x)

    # The result of the forward pass should still be the same.
    @test isapprox(y_baseline, OneDNN.materialize(y_test))

    # do we propogate the weights correctly?
    grads_baseline = back_baseline(z)
    grads_test = back_test(z)

    # Make sure the structure of the pullbacks is the same.
    @test isa(grads_baseline, Tuple)
    @test isa(grads_test, Tuple)
    @test length(grads_baseline) == length(grads_test) == 2

    @test isa(grads_baseline[1], NamedTuple)
    @test isa(grads_test[1], NamedTuple)

    # Now, start inspecting the results of the pullbacks are the same.
    @test isapprox(grads_baseline[1].W, transpose(OneDNN.materialize(grads_test[1].W)))
    @test isapprox(grads_baseline[1].b, OneDNN.materialize(grads_test[1].b))
    @test grads_baseline[1].σ == grads_test[1].σ
    @test isapprox(grads_baseline[2], OneDNN.materialize(grads_test[2]))

    #####
    ##### Chain
    #####

    m = Chain(
        Dense(10, 5, identity),
        Dense(5, 3, identity),
    )

    M = Chain(
        OneDNN.dense_from_flux(m[1]),
        OneDNN.dense_from_flux(m[2]),
    )

    @test isapprox(m(x), OneDNN.materialize(M(x)))

    y_baseline, back_baseline = Zygote._pullback(m, x)
    y_test, back_test = Zygote._pullback(M, x)

    z = m(x)

    # The result of the forward pass should still be the same.
    @test isapprox(y_baseline, OneDNN.materialize(y_test))

    # do we propogate the weights correctly?
    grads_baseline = back_baseline(z)
    grads_test = back_test(z)

    # Make sure the structure of the pullbacks is the same.
    @test isa(grads_baseline, Tuple)
    @test isa(grads_test, Tuple)
    @test length(grads_baseline) == length(grads_test) == 2

    # Now, start inspecting the results of the pullbacks are the same.
    @test isapprox(grads_baseline[1].layers[1].W, transpose(OneDNN.materialize(grads_test[1].layers[1].W)))
    @test isapprox(grads_baseline[1].layers[1].b,           OneDNN.materialize(grads_test[1].layers[1].b))
    @test isapprox(grads_baseline[1].layers[2].W, transpose(OneDNN.materialize(grads_test[1].layers[2].W)))
    @test isapprox(grads_baseline[1].layers[2].b,           OneDNN.materialize(grads_test[1].layers[2].b))
    @test isapprox(grads_baseline[2], OneDNN.materialize(grads_test[2]))
end
