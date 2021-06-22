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

        dy = Float32(0.125) * randn(Float32, size(y))
        DY = OneDNN.Memory(dy)

        grads_jl = back_jl(dy)
        grads_dnnl = back_dnnl(DY)
        @test isa(grads_jl, Tuple)
        @test isa(grads_dnnl, Tuple)

        # N.B.: Notice the difference in the plurality of the `weights` field ...
        # In the OneDNN code, we try to stay with the plural version since that's what the
        # oneDNN source code generally does.
        @test isapprox(grads_jl[1].weight, transpose(OneDNN.materialize(grads_dnnl[1].weights)))
        @test isapprox(grads_jl[1].bias, OneDNN.materialize(grads_dnnl[1].bias))
        @test isapprox(grads_jl[2], OneDNN.materialize(grads_dnnl[2]))
        @inferred back_dnnl(DY)
    end

    # #####
    # ##### InnerProductBackwardData
    # #####

    # m = Flux.Dense(10, 5, identity)
    # weights = m.weight
    # bias = m.bias
    # x = randn(Float32, 10, 10)

    # y, back = Zygote._pullback(m, x)
    # dy = randn(Float32, 5, 10)
    # grads = back(dy)
    # dx = dy[2]

    # # Grads should be a 2-tuple. The first element is a NamedTuple holding the
    # # gradients for `m`, the seconds should just be a matrix with the gradients for `x`.
    # @test length(grads) == 2
    # dx = grads[2]

    # W = OneDNN.Memory(weights)
    # DY = OneDNN.Memory(transpose(dy))
    # op = OneDNN.InnerProductBackwardData(size(transpose(x)), W, DY)
    # DX = op(W, DY)
    # @test isapprox(dx, transpose(DX))

    # #####
    # ##### InnerProductBackwardWeights
    # #####

    # m = Flux.Dense(10, 5, identity)
    # weights = m.weight
    # bias = m.bias
    # x = randn(Float32, 10, 10)

    # y, back = Zygote._pullback(m, x)
    # dy = randn(Float32, 5, 10)
    # grads = back(dy)
    # dw = grads[1].weight
    # db = grads[1].bias

    # W = OneDNN.Memory(weights)
    # B = OneDNN.Memory(bias)
    # X = OneDNN.Memory(transpose(x))
    # DY = OneDNN.Memory(transpose(dy))
    # op = OneDNN.InnerProductBackwardWeight(size(weights), length(bias), X, DY)
    # DW, DB = op(X, DY)
    # @test isapprox(dw, DW)
    # @test isapprox(db, DB)
end

# @testset "Testing Innerproduct" begin
#     # In theory, the inner product should be equivalent to Flux's `dense` layer.
#     # For the moment, we'll just use the `identity` activation function, but eventually
#     # I'd like to have activation functions incorporated.
#
#     # Construct a normal Flux Dense layer
#     # Extract the weights and bias to test against `innerproduct`.
#     m = Dense(10, 5, identity)
#
#     weights = m.W
#     bias = m.b
#     src = rand(Float32, 10, 10)
#
#     expected = m(src)
#     result = OneDNN.materialize(OneDNN.inner_product_forward(src, transpose(weights), bias))
#     @test isapprox(expected, result)
#
#     activation_functions = [identity, Flux.relu, Flux.sigmoid]
#
#     # Now, we start seeing if backprop is working correctly.
#     for fn in activation_functions
#         m = Dense(10, 5, fn)
#         M = OneDNN.Dense(m)
#
#         x = rand(Float32, 10, 10)
#         @test isapprox(m(x), OneDNN.materialize(M(x)))
#
#         y_baseline, back_baseline = Zygote._pullback(m, x)
#         y_test, back_test = Zygote._pullback(M, x)
#
#         z = m(x)
#
#         # The result of the forward pass should still be the same.
#         @test isapprox(y_baseline, OneDNN.materialize(y_test))
#
#         # do we propogate the weights correctly?
#         grads_baseline = back_baseline(z)
#         grads_test = back_test(z)
#
#         # Make sure the structure of the pullbacks is the same.
#         @test isa(grads_baseline, Tuple)
#         @test isa(grads_test, Tuple)
#         @test length(grads_baseline) == length(grads_test) == 2
#
#         @test isa(grads_baseline[1], NamedTuple)
#         @test isa(grads_test[1], NamedTuple)
#
#         # Now, start inspecting the results of the pullbacks are the same.
#         @test isapprox(grads_baseline[1].W, transpose(OneDNN.materialize(grads_test[1].W)))
#         @test isapprox(grads_baseline[1].b, OneDNN.materialize(grads_test[1].b))
#         @test grads_baseline[1].σ == grads_test[1].σ
#         @test isapprox(grads_baseline[2], OneDNN.materialize(grads_test[2]))
#     end
#
#     #####
#     ##### Chain
#     #####
#
#     for fn in activation_functions
#         m = Chain(Dense(10, 5, fn), Dense(5, 3, fn))
#
#         M = Chain(OneDNN.Dense(m[1]), OneDNN.Dense(m[2]))
#
#         x = rand(Float32, 10, 10)
#         @test isapprox(m(x), OneDNN.materialize(M(x)))
#
#         y_baseline, back_baseline = Zygote._pullback(m, x)
#         y_test, back_test = Zygote._pullback(M, x)
#
#         z = m(x)
#
#         # The result of the forward pass should still be the same.
#         @test isapprox(y_baseline, OneDNN.materialize(y_test))
#
#         # do we propogate the weights correctly?
#         grads_baseline = back_baseline(z)
#         grads_test = back_test(z)
#
#         # Make sure the structure of the pullbacks is the same.
#         @test isa(grads_baseline, Tuple)
#         @test isa(grads_test, Tuple)
#         @test length(grads_baseline) == length(grads_test) == 2
#
#         # Now, start inspecting the results of the pullbacks are the same.
#         @test isapprox(
#             grads_baseline[1].layers[1].W,
#             transpose(OneDNN.materialize(grads_test[1].layers[1].W)),
#         )
#         @test isapprox(
#             grads_baseline[1].layers[1].b, OneDNN.materialize(grads_test[1].layers[1].b)
#         )
#         @test isapprox(
#             grads_baseline[1].layers[2].W,
#             transpose(OneDNN.materialize(grads_test[1].layers[2].W)),
#         )
#         @test isapprox(
#             grads_baseline[1].layers[2].b, OneDNN.materialize(grads_test[1].layers[2].b)
#         )
#         @test isapprox(grads_baseline[2], OneDNN.materialize(grads_test[2]))
#     end
# end
