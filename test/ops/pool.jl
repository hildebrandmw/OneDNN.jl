@testset "Testing Pooling" begin
    kernel_sizes = [(2, 2), (3, 3), (7, 7)]
    paddings = [0, 1, 3, 5]
    strides = [1, 2, 3]

    # Occaisonally, OneDNN and Flux will disagree on which index is the "most popular"
    # among its filter window.
    #
    # There seem to be cases where either one or the other returns slightly incorrect
    # answers, based on two inputs that are pretty close.
    #
    # So maybe this has to do to some weird rouding issue? I'm not sure.
    #
    # For testing purposes, we just make sure to introduce enough delta between the
    # elements in the array so this confusion doesn't occus.
    Random.seed!(12345)
    x = zeros(Float32, 8, 8, 8, 4)
    nums = Set(eachindex(x))
    for i in eachindex(x)
        v = rand(nums)
        x[i] = v
        delete!(nums, v)
    end
    @test isempty(nums)

    X = OneDNN.Memory(x)

    iter = Iterators.product(kernel_sizes, paddings, strides)
    for _tup in iter
        kernel = _tup[1]
        padding = _tup[2]
        stride = _tup[3]

        @show (kernel, padding, stride)

        # Test on both MaxPool and MeanPool
        fn_pairs = [
            (OneDNN.MaxPool, Flux.MaxPool), (OneDNN.InclusiveMeanPool, Flux.MeanPool)
        ]
        for (fn_onednn, fn_flux) in fn_pairs
            pool_onednn = fn_onednn(kernel; padding = padding, strides = stride)
            pool_flux = fn_flux(kernel; pad = padding, stride = stride)

            # inference
            @test isapprox(OneDNN.materialize(pool_onednn(X)), pool_flux(x))

            # training
            out_onednn, back_onednn = Zygote._pullback(pool_onednn, X)
            out_flux, back_flux = Zygote._pullback(pool_flux, x)

            @test isapprox(OneDNN.materialize(out_onednn), out_flux)

            dx_onednn = back_onednn(out_onednn)
            dx_flux = back_flux(out_flux)

            @test dx_onednn[1] === dx_flux[1] === nothing
            @test isapprox(OneDNN.materialize(dx_onednn[2]), dx_flux[2])
        end
    end
end
