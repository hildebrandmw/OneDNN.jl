@testset "Testing Concat" begin
    x, y, z = randn.(Float32, (10, 20, 30), 10)
    X, Y, Z = OneDNN.Memory.((x, y, z))

    # vcat
    expected = vcat(x, y, z)
    arg = [X, Y, Z]
    op = OneDNN.Concat(arg, 1)
    result = op(arg)
    @test isa(result, OneDNN.Memory)
    @test result == expected

    # hcat
    x, y, z = randn.(Float32, 10, (10, 20, 30))
    X, Y, Z = OneDNN.Memory.((x, y, z))

    # vcat
    expected = hcat(x, y, z)
    arg = [X, Y, Z]
    op = OneDNN.Concat(arg, 2)
    result = op(arg)
    @test result == expected

    # #####
    # ##### Pullbacks
    # #####

    # function tester(f, fargs, g, gargs; fkw...)
    #     expected, back_expected = Zygote._pullback(f, fargs...; fkw...)

    #     Δ = randn(Float32, size(expected))
    #     grads_expected = back_expected(Δ)

    #     result, back = Zygote._pullback(g, gargs...)
    #     @test OneDNN.materialize(result) == expected

    #     # two flavors, one with a `Memory` passed in, and one with just a normal julia
    #     # array.
    #     #
    #     # Note that the structures of the gradients are different because `vcat` is variadic,
    #     # and thus gets a tuple entry for each argument.
    #     #
    #     # OneDNN.concat, hoever, takes an array, so the sensitivity to the array input comes
    #     # back as an array.
    #     grads_result = back(OneDNN.memory(Δ))
    #     @test OneDNN.materialize(grads_result[2][1]) == grads_expected[2]
    #     @test OneDNN.materialize(grads_result[2][2]) == grads_expected[3]
    #     @test OneDNN.materialize(grads_result[2][3]) == grads_expected[4]
    #     @test grads_result[3] === nothing

    #     grads_result = back(Δ)
    #     @test OneDNN.materialize(grads_result[2][1]) == grads_expected[2]
    #     @test OneDNN.materialize(grads_result[2][2]) == grads_expected[3]
    #     @test OneDNN.materialize(grads_result[2][3]) == grads_expected[4]
    #     @test grads_result[3] === nothing

    #     # Test inference.
    #     @inferred back(OneDNN.memory(Δ))
    #     @inferred back(Δ)
    # end

    # # vcat
    # x, y, z = randn.(Float32, (10, 20, 30), 10)
    # X, Y, Z = OneDNN.memory.((x, y, z))
    # tester(vcat, (x, y, z), OneDNN.concat, ([X, Y, Z], 1))

    # # hcat
    # x, y, z = randn.(Float32, 10, (10, 20, 30))
    # X, Y, Z = OneDNN.memory.((x, y, z))
    # tester(hcat, (x, y, z), OneDNN.concat, ([X, Y, Z], 2))

    # ### 3d

    # # vcat
    # x, y, z = randn.(Float32, (10, 20, 30), 10, 20)
    # X, Y, Z = OneDNN.memory.((x, y, z))
    # tester(vcat, (x, y, z), OneDNN.concat, ([X, Y, Z], 1))

    # # hcat
    # x, y, z = randn.(Float32, 10, (10, 20, 30), 20)
    # X, Y, Z = OneDNN.memory.((x, y, z))
    # tester(hcat, (x, y, z), OneDNN.concat, ([X, Y, Z], 2))
end
