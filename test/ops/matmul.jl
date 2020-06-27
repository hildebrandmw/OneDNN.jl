@testset "Testing MatMul" begin
    # First, test normal matmul.

    a = rand(Float32, 4, 2)
    b = rand(Float32, 2, 3)

    c = OneDNN.matmul(a, b)
    @test isa(c, OneDNN.Memory)
    @test size(c) == (size(a, 1), size(b, 2))

    # Note: the underlying Julia array in `c` is just a vector because in general,
    # derived memory formats may require more memory than just a product of the dimensions.
    @test ndims(c) == 2

    # Does multiplication actuall return the correct result?
    @test isapprox(OneDNN.materialize(c), a * b)

    # Test applying some post ops
    x = rand(Float32, 2, 2)
    y = rand(Float32, 2, 2)
    z = rand(Float32, 2, 2)

    # expected result
    expected = (x * y) + (transpose(z) * x)

    out = OneDNN.matmul(x,y)

    postops = OneDNN.PostOps()
    OneDNN.appendsum!(postops)
    attr = OneDNN.Attributes()
    OneDNN.add!(attr, postops)
    OneDNN.matmul!(out, transpose(z), x; attributes = attr)

    result = OneDNN.materialize(out)
    @test isapprox(result, expected)
end
