@testset "Testing Memory" begin
    #####
    ##### Memory Descriptors
    #####

    @test OneDNN.dnnl_type(Float16) == OneDNN.Lib.dnnl_f16
    @test OneDNN.dnnl_type(Float32) == OneDNN.Lib.dnnl_f32
    @test OneDNN.dnnl_type(Int32) == OneDNN.Lib.dnnl_s32
    @test OneDNN.dnnl_type(Int8) == OneDNN.Lib.dnnl_s8
    @test OneDNN.dnnl_type(UInt8) == OneDNN.Lib.dnnl_u8

    # Test conversion from an instance of a type
    @test OneDNN.dnnl_type(one(Float32)) == OneDNN.Lib.dnnl_f32

    # Construct Dimensions
    # We pass dimensions as 12 long Int64 arrays to OneDNN
    sz = (1,2,3)
    dims = OneDNN.dnnl_dims(sz)
    @test length(dims) == OneDNN.Lib.DNNL_MAX_NDIMS
    @test dims[1:length(sz)] == collect(reverse(sz))
    @test all(iszero, dims[length(sz)+1:OneDNN.Lib.DNNL_MAX_NDIMS])
    @test all(iszero, OneDNN.dnnl_dims())
    @test all(iszero, OneDNN.dnnl_dims(()))

    x = rand(Float32, 2, 3)
    md = OneDNN.memorydesc(x)

    # OneDNN stores size information in reverse order from normal Julia
    @test OneDNN.logicalsize(md) == reverse(size(x))
    @test md.data_type == OneDNN.Lib.dnnl_f32

    # Test equality
    @test md == OneDNN.memorydesc(x)
    @test md != OneDNN.memorydesc(rand(Float32, 3, 2))

    #####
    ##### Extruding Dimensions
    #####

    @test OneDNN.leading(1) == ()
    @test OneDNN.leading(1, 2) == (1,)
    @test OneDNN.leading(1, 2, 3) == (1, 2)
    @test OneDNN.leading(1, 2, 3, 4) == (1, 2, 3)
    @test OneDNN.leading(1, 2, 3, 4,5) == (1, 2, 3, 4)

    @test OneDNN.extrude(Float32, (10,), 4 * 12) == (12,)
    @test OneDNN.extrude(Float32, (12,), 4 * 12) == (12,)
    @test OneDNN.extrude(Float32, (12,), 4 * 13) == (13,)

    @test OneDNN.extrude(Float32, (10, 10), 4 * 100) == (10, 10)
    @test OneDNN.extrude(Float32, (10, 10), 4 * 101) == (10, 11)
    @test OneDNN.extrude(Float32, (10, 10), 4 * 104) == (10, 11)

    @test OneDNN.extrude(Float32, (2, 2, 2), 4 * 9) == (2, 2, 3)

    #####
    ##### Memory
    #####

    x = rand(Float32, 5, 10)
    X = OneDNN.memory(x)

    # The `memory_t` types in OneDNN return references to a cached MD.
    # Make sure they still end up as equal to when we create one from scratch.
    @test OneDNN.memorydesc(X) == OneDNN.memorydesc(x)
    @test OneDNN.getdata(X) === x

    @test size(X) == size(x)
    @test eltype(X) == eltype(x)

    @test OneDNN.val_ndims(X) == Val{2}()

    Y = similar(X)
    @test size(Y) == size(X)
    @test OneDNN.memorydesc(X) == OneDNN.memorydesc(Y)

    #####
    ##### Transposes
    #####

    x = rand(Float32, 10, 5)
    X = OneDNN.memory(transpose(x))
    @test size(X) == (5, 10)
    XX = OneDNN.materialize(X)
    @test XX == transpose(x)
end

