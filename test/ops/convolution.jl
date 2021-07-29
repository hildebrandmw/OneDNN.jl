@testset "Testing Convolution" begin
    activations = [identity, Flux.relu, Flux.sigmoid, abs]
    kernel_sizes = [(1, 1), (3, 3)]
    paddings = [0, 1]
    strides = [1, 2, 3]

    iter = Iterators.product(activations, kernel_sizes, paddings, strides)
    for tup in iter
        @show tup
        # Unpack
        activation = tup[1]
        kernel_size = tup[2]
        padding = tup[3]
        stride = tup[4]

        w = randn(Float32, kernel_size..., 16, 16)
        b = randn(Float32, size(w, ndims(w)))
        x = randn(Float32, 16, 16, 16, 8)
        X = OneDNN.Memory(x)

        # Just another reminder - OneDNN's convolution is actually cross-correlation.
        conv_flux = Flux.CrossCor(
            w, b, activation; stride = stride, pad = padding,
        )
        conv_onednn = OneDNN.Conv(conv_flux)

        # See if the internal weight gets optimized properly.
        @test conv_onednn.optimized_weights == false
        y_pre = conv_onednn(X)
        @test conv_onednn.optimized_weights == true
        y_post = conv_onednn(X)
        y_ref = conv_flux(x)
        @test isapprox(y_ref, OneDNN.materialize(y_pre))
        @test isapprox(y_ref, OneDNN.materialize(y_post))

        # Ensure type inference works correctly
        @inferred conv_onednn(X)

        # Now - try backprop
        out_onednn, back_onednn = Zygote._pullback(conv_onednn, X)
        out_flux, back_flux = Zygote._pullback(conv_flux, x)

        isa(out_onednn, OneDNN.Memory)
        isapprox(out_flux, OneDNN.materialize(out_onednn))
        @inferred Zygote._pullback(conv_onednn, X)

        dx_onednn = back_onednn(out_onednn)
        dx_flux = back_flux(out_flux)

        @test isa(dx_onednn, Tuple)
        @test isa(dx_flux, Tuple)

        @test isapprox(dx_flux[1].weight, OneDNN.materialize(dx_onednn[1].weights))
        @test isapprox(dx_flux[1].bias, OneDNN.materialize(dx_onednn[1].bias))
        @test isapprox(dx_flux[2], OneDNN.materialize(dx_onednn[2]))

        # Type inference
        @inferred back_onednn(out_onednn)
    end
end

