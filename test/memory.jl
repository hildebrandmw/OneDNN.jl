@testset "Testing Memory" begin
    @test OneDNN.dnnl_type(Float16) == OneDNN.Lib.dnnl_f16
    @test OneDNN.dnnl_type(Float32) == OneDNN.Lib.dnnl_f32
    @test OneDNN.dnnl_type(Int32) == OneDNN.Lib.dnnl_s32
    @test OneDNN.dnnl_type(Int8) == OneDNN.Lib.dnnl_s8
    @test OneDNN.dnnl_type(UInt8) == OneDNN.Lib.dnnl_u8

    @test OneDNN.dnnl_convert(Float16) == OneDNN.Lib.dnnl_f16
    @test OneDNN.dnnl_convert(Float32) == OneDNN.Lib.dnnl_f32
    @test OneDNN.dnnl_convert(Int32) == OneDNN.Lib.dnnl_s32
    @test OneDNN.dnnl_convert(Int8) == OneDNN.Lib.dnnl_s8
    @test OneDNN.dnnl_convert(UInt8) == OneDNN.Lib.dnnl_u8

    # Test conversion from an instance of a type
    @test OneDNN.dnnl_type(one(Float32)) == OneDNN.Lib.dnnl_f32

    # Dims
    # "12 dimensions is enough for anyone"
    dnnl_dims = OneDNN.dnnl_dims
    x = (1, 2, 3)
    @test dnnl_dims(x) == (1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    @test dnnl_dims() == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    @test dnnl_dims() == dnnl_dims(())

    x = rand(Float32, 1, 2, 3)
    _strides = strides(x)
    @test dnnl_dims(x) == (_strides..., 0, 0, 0, 0, 0, 0, 0, 0, 0)
end

@testset "Testing Views" begin
    x = rand(Float32, 40, 40)
    y = rand(Float32, 40, 40)

    vx = view(x, 1, :)
    VX = OneDNN.Memory(vx)
    @test OneDNN.typed(VX) == vx
    @test OneDNN.materialize(VX) == vx

    vy = view(x, :, 1)
    VY = OneDNN.Memory(vy)
    @test OneDNN.typed(VY) == vy
    @test OneDNN.materialize(VY) == vy

    Z = VX + VY
    @test OneDNN.typed(Z) == vx .+ vy
    @test OneDNN.materialize(Z) == vx .+ vy

    # 2D views
    vx = view(x, 1:5, 2:11)
    VX = OneDNN.Memory(vx)
    @test OneDNN.typed(VX) == vx
    @test OneDNN.materialize(VX) == vx

    vy = view(y, 20:30, :)
    VY = OneDNN.Memory(vy)
    @test OneDNN.typed(VY) == vy
    @test OneDNN.materialize(VY) == vy
end

# For CachedArrays style functionality, it's important that the "pointer" used for the
# "DNNL.memory" object can be updated if the underlying storage gets changed.
@testset "Testing Swapping Memory" begin
    x = ones(Float32, 10, 10)
    y = 2 .* ones(Float32, 10, 10)
    X = OneDNN.Memory(x)
    Y = OneDNN.Memory(y)

    Z = X + Y
    @test all(isequal(3), OneDNN.typed(Z))

    # Swap out the array and ensure the change is propagated.
    x2 = zeros(Float32, 10, 10)
    X.array = x2
    Z = X + Y
    @test all(isequal(2), OneDNN.typed(Z))

    # Try with views
    x = rand(Float32, 20, 20)
    va = view(x, 11:20, 11:20)
    vb = view(x, 1:10, 11:20)
    VA = OneDNN.Memory(va)
    VB = OneDNN.Memory(vb)

    Z = VA + VB
    @test isapprox(va .+ vb, OneDNN.typed(Z))
    y = rand(Float32, 20, 20)
    VA.array = y
    Z = VA + VB
    @test isapprox(view(y, 11:20, 11:20) .+ vb, OneDNN.typed(Z))
end

# @testset "Testing Memory" begin
#     #####
#     ##### Memory Descriptors
#     #####
#
#     @test OneDNN.dnnl_type(Float16) == OneDNN.Lib.dnnl_f16
#     @test OneDNN.dnnl_type(Float32) == OneDNN.Lib.dnnl_f32
#     @test OneDNN.dnnl_type(Int32) == OneDNN.Lib.dnnl_s32
#     @test OneDNN.dnnl_type(Int8) == OneDNN.Lib.dnnl_s8
#     @test OneDNN.dnnl_type(UInt8) == OneDNN.Lib.dnnl_u8
#
#     # Test conversion from an instance of a type
#     @test OneDNN.dnnl_type(one(Float32)) == OneDNN.Lib.dnnl_f32
#
#     # Construct Dimensions
#     # We pass dimensions as 12 long Int64 arrays to OneDNN
#     sz = (1,2,3)
#     dims = OneDNN.dnnl_dims(sz)
#     @test length(dims) == OneDNN.Lib.DNNL_MAX_NDIMS
#     @test dims[1:length(sz)] == collect(reverse(sz))
#     @test all(iszero, dims[length(sz)+1:OneDNN.Lib.DNNL_MAX_NDIMS])
#     @test all(iszero, OneDNN.dnnl_dims())
#     @test all(iszero, OneDNN.dnnl_dims(()))
#
#     x = rand(Float32, 2, 3)
#     md = OneDNN.memorydesc(x)
#
#     # OneDNN stores size information in reverse order from normal Julia
#     @test OneDNN.logicalsize(md) == reverse(size(x))
#     @test md.data_type == OneDNN.Lib.dnnl_f32
#
#     # Test equality
#     @test md == OneDNN.memorydesc(x)
#     @test md != OneDNN.memorydesc(rand(Float32, 3, 2))
#
#     #####
#     ##### Unwrapping Wrapper Types
#     #####
#
#     x = randn(Float32, 10, 20)
#     @test OneDNN.toparent(x) === x
#     @test OneDNN.toparent(transpose(x)) === x
#     @test OneDNN.toparent(reshape(transpose(x), :)) === x
#
#     # Views must be materialized in order to pass to OneDNN
#     #@test OneDNN.toparent(view(x, 1:2, 1:2)) == x
#
#     # batched transpose
#     x = randn(Float32, 5, 10, 15)
#     p = PermutedDimsArray(x, (2,1,3))
#     @test OneDNN.toparent(p) === x
#     @test OneDNN.toparent(reshape(p, :)) === x
#
#     # Views must be materialized in order to pass to OneDNN
#     #@test OneDNN.toparent(view(x, 1:2, 1:2, :)) === x
#
#     #####
#     ##### Extruding Dimensions
#     #####
#
#     @test OneDNN.leading(1) == ()
#     @test OneDNN.leading(1, 2) == (1,)
#     @test OneDNN.leading(1, 2, 3) == (1, 2)
#     @test OneDNN.leading(1, 2, 3, 4) == (1, 2, 3)
#     @test OneDNN.leading(1, 2, 3, 4,5) == (1, 2, 3, 4)
#
#     @test OneDNN.extrude(Float32, (10,), 4 * 12) == (12,)
#     @test OneDNN.extrude(Float32, (12,), 4 * 12) == (12,)
#     @test OneDNN.extrude(Float32, (12,), 4 * 13) == (13,)
#
#     @test OneDNN.extrude(Float32, (10, 10), 4 * 100) == (10, 10)
#     @test OneDNN.extrude(Float32, (10, 10), 4 * 101) == (10, 11)
#     @test OneDNN.extrude(Float32, (10, 10), 4 * 104) == (10, 11)
#
#     @test OneDNN.extrude(Float32, (2, 2, 2), 4 * 9) == (2, 2, 3)
#
#     #####
#     ##### Memory
#     #####
#
#     x = rand(Float32, 5, 10)
#     X = OneDNN.memory(x)
#
#     # The `memory_t` types in OneDNN return references to a cached MD.
#     # Make sure they still end up as equal to when we create one from scratch.
#     @test OneDNN.memorydesc(X) == OneDNN.memorydesc(x)
#     @test OneDNN.getdata(X) === x
#
#     @test size(X) == size(x)
#     @test eltype(X) == eltype(x)
#
#     @test OneDNN.val_ndims(X) == Val{2}()
#
#     Y = similar(X)
#     @test size(Y) == size(X)
#     @test OneDNN.memorydesc(X) == OneDNN.memorydesc(Y)
#
#     # Do view work correctly?
#     x = rand(Float32, 5, 10)
#     vx = view(x, 1:2, :)
#     VX = OneDNN.memory(vx)
#     @test size(VX) == size(vx)
#     @test OneDNN.materialize(VX) == vx
#
#     #####
#     ##### Transposes
#     #####
#
#     x = rand(Float32, 10, 5)
#     X = OneDNN.memory(transpose(x))
#     @test size(X) == (5, 10)
#     XX = OneDNN.materialize(X)
#     @test XX == transpose(x)
#
#     # batched transpose
#     x = PermutedDimsArray(rand(Float32, 10, 20, 30), (2,1,3))
#     X = OneDNN.memory(x)
#     @test size(X) == (20, 10, 30)
#     XX = OneDNN.materialize(X)
#
#     # the resulting array should be element wise equal, but since we told it the layout
#     # was different, it should be an entirely new array.
#     @test XX == x
#     @test XX !== x
# end
#
# @testset "Testing Views" begin
#     #x = rand(Float32, 10, 10)
#     #X = OneDNN.memory(x)
#
#     #@test OneDNN.materialize(view(X, :, 1)) == view(x, :, 1)
#     # Julia returns a vector ... OneDNN returns an array
#     #@test OneDNN.materialize(view(X, 1, :)) == view(x, 1, :)
# end
#
