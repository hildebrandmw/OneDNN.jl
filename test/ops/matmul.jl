@testset "Testing MatMul" begin
    # First, test normal matmul.
    a = rand(Float32, 10, 20)
    b = rand(Float32, 20, 10)
    A = OneDNN.Memory(a)
    B = OneDNN.Memory(b)

    Z = OneDNN.materialize(OneDNN.matmul(A, B))
    z = a * b
    @test size(Z) == (size(a, 1), size(b, 2))
    @test isapprox(Z, z)

    Z = OneDNN.materialize(OneDNN.matmul(
        OneDNN.Memory(transpose(b)), OneDNN.Memory(transpose(a))
    ))
    @test isapprox(transpose(Z), z)

    # Do pullbacks work?
    # Note: `rrule` definition lives in ChainRuies.jl under `Base/arraymath.jl`.
    x = randn(Float32, 100, 100)
    y = randn(Float32, 100, 100)
    X = OneDNN.Memory(x)
    Y = OneDNN.Memory(y)

    Z, back_dnnl = Zygote._pullback(*, X, Y)
    z, back_jl = Zygote._pullback(*, x, y)
    @test isapprox(OneDNN.materialize(Z), z)

    dz = Float32(0.125) * randn(Float32, size(z))
    DZ = OneDNN.Memory(dz)

    grads_jl = back_jl(dz)
    grads_dnnl = back_dnnl(DZ)

    @test length(grads_jl) == length(grads_dnnl) == 3
    @test grads_jl[1] === grads_dnnl[1] === nothing
    @test isapprox(grads_jl[2], OneDNN.materialize(grads_dnnl[2]))
    @test isapprox(grads_jl[3], OneDNN.materialize(grads_dnnl[3]))
end

# # Test applying some post ops
# x = rand(Float32, 2, 2)
# y = rand(Float32, 2, 2)
# z = rand(Float32, 2, 2)

# # expected result
# expected = (x * y) + (transpose(z) * x)

# out = OneDNN.matmul(y, x)

# postops = OneDNN.PostOps()
# OneDNN.appendsum!(postops)
# attr = OneDNN.Attributes()
# OneDNN.add!(attr, postops)
# OneDNN.matmul!(out, x, transpose(z); attributes = attr)

# result = OneDNN.materialize(out)
# @test isapprox(result, expected)

# # Test scaling
# x = randn(Float32, 5, 5)
# y = randn(Float32, 5, 5)
# attr = OneDNN.Attributes()
# OneDNN.setscale!(attr, 2)
# z = OneDNN.matmul(y, x; attributes = attr)

# @test isapprox(OneDNN.materialize(z), 2 * x * y)
