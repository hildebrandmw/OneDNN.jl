@testset "Testing Batchnorm" begin
    @testset "Testing Forward Training 2D" begin
        _src = randn(Float32, 100, 100)
        _scale_shift = randn(Float32, size(_src, 1), 2)

        src = OneDNN.Memory(_src)
        scale_shift = OneDNN.Memory(_scale_shift)

        # Run the OneDNN model
        nt = OneDNN.batchnorm_training(src, scale_shift; epsilon = 1f-5)

        # Some sanity checks.
        @test isapprox(OneDNN.materialize(nt.mean), Statistics.mean.(eachrow(_src)))
        @test isapprox(
            OneDNN.materialize(nt.variance),
            Statistics.var.(eachrow(_src); corrected = false),
        )

        # Create the Flux.jl model for comparison purposes.
        bn_flux = Flux.BatchNorm(
            size(_src, 1); affine = true, track_stats = false, ϵ = 1f-5
        )

        bn_flux.γ .= view(_scale_shift, :, 1)
        bn_flux.β .= view(_scale_shift, :, 2)

        expected = bn_flux(_src)
        @test isapprox(OneDNN.materialize(nt.dst), expected)
    end

    @testset "Testing Forward Training 4D" begin
        _src = randn(Float32, 10, 10, 10, 10)
        _scale_shift = randn(Float32, size(_src, 3), 2)

        src = OneDNN.Memory(_src)
        scale_shift = OneDNN.Memory(_scale_shift)

        # Run the OneDNN model
        nt = OneDNN.batchnorm_training(src, scale_shift; epsilon = 1f-5)

        # Some sanity checks.
        dims = (1, 2, 4)
        @test isapprox(
            OneDNN.materialize(nt.mean), view(Statistics.mean(_src; dims), 1, 1, :, 1)
        )
        @test isapprox(
            OneDNN.materialize(nt.variance),
            view(Statistics.var(_src; dims, corrected = false), 1, 1, :, 1),
        )

        # Create the Flux.jl model for comparison purposes.
        bn_flux = Flux.BatchNorm(
            size(_src, 3); affine = true, track_stats = false, ϵ = 1f-5
        )

        bn_flux.γ .= view(_scale_shift, :, 1)
        bn_flux.β .= view(_scale_shift, :, 2)

        expected = bn_flux(_src)
        @test isapprox(OneDNN.materialize(nt.dst), expected)
    end

    @testset "Testing Forward and Backward Pair" begin
        _scale_shift = randn(Float32, 100, 2)
        _src = randn(Float32, 100, 100)
        epsilon = 1f-5

        scale_shift = OneDNN.Memory(_scale_shift)
        src = OneDNN.Memory(_src)

        activations = [identity, Flux.relu, Flux.sigmoid, abs]
        for fn in activations
            @show fn
            bn_onednn = OneDNN.BatchNorm(scale_shift, fn; epsilon = epsilon)

            bn_flux = Flux.BatchNorm(
                size(_src, 1), fn; affine = true, track_stats = false, ϵ = 1f-5
            )
            bn_flux.γ .= view(_scale_shift, :, 1)
            bn_flux.β .= view(_scale_shift, :, 2)

            # Test results and pullbacks
            out_onednn, back_onednn = Zygote._pullback(bn_onednn, src)
            out_flux, back_flux = Zygote._pullback(bn_flux, _src)

            @test isapprox(OneDNN.materialize(out_onednn), out_flux)

            diff_onednn = back_onednn(out_onednn)
            diff_flux = back_flux(out_flux)
            @test length(diff_onednn) == length(diff_flux) == 2

            # Compare `diff_src`.
            # TODO: Figure out why there's such a difference between OneDNN and Flux ...
            @test isapprox(OneDNN.materialize(diff_onednn[2]), diff_flux[2]; rtol = 0.21)

            # scale and shift
            diff_scale_shift = OneDNN.materialize(diff_onednn[1].scale_shift)
            @test isapprox(diff_flux[1][].γ, view(diff_scale_shift, :, 1))
            @test isapprox(diff_flux[1][].β, view(diff_scale_shift, :, 2))
        end
    end
end
