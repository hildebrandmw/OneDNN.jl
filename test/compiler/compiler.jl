@testset "Testing Compiler" begin
    # Make sure that type conversion works correctly.
    x = rand(Float32, 2, 2)
    @test OneDNN.tracetype(x) == Mjolnir.Shape{typeof(x)}(size(x))
end
