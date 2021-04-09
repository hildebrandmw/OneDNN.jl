### helpful utils
# Used for removing LineNumberNodes from generated expressions for equality comparison
# purposes.
using MacroTools: MacroTools

@testset "Utils" begin
    TiledArrays = OneDNN.TiledArrays
    ### cdiv
    @test TiledArrays.cdiv(Int32(10), Int64(9)) == 2
    @test TiledArrays.cdiv(10, 10) == 1
    @test TiledArrays.cdiv(10, 11) == 1

    ### Test the "head" utility function.
    @test TiledArrays.head(1) == ()
    @test TiledArrays.head(1, 2) == (1,)
    @test TiledArrays.head(1, 2, 3) == (1, 2)
    @test TiledArrays.head(1, 2, 3, 4) == (1, 2, 3)
    @test TiledArrays.head(1, 2, 3, 4, 5) == (1, 2, 3, 4)
    @test TiledArrays.head(1, 2, 3, 4, 5, 6) == (1, 2, 3, 4, 5)
end

@testset "Validation" begin
    TiledArrays = OneDNN.TiledArrays

    # First, a normal layout should work fine.
    validate = TiledArrays.validate
    Tile = TiledArrays.Tile
    @test validate((1,))
    @test validate((1, 2))
    @test validate((1, 2, 3))
    @test validate((1, 2, 3, 4))

    # Permutations should be fine
    @test validate((2, 1))
    @test validate((2, 3, 1))
    @test validate((4, 2, 3, 1))

    # Negative indices should throw an error.
    @test_throws Exception validate((-1,))
    @test_throws Exception validate((1, 2, -3))

    # Skipping an index should be an error
    @test_throws Exception validate((2,))
    @test_throws Exception validate((1, 3))

    # Correct usage of tiling.
    @test validate((Tile(3, 8), 2, 1, 3))
    # Nested Tiling
    @test validate((Tile(3, 8), 2, Tile(3, 2), 1, 3))
    # Multiple Tiling
    @test validate((Tile(3, 8), Tile(1, 8), 2, 1, 3))
    # Multiple Tiling with Nesting
    @test validate((Tile(3, 8), Tile(1, 8), 2, Tile(3, 8), 1, 3))

    # Incorrect usage.
    # Tile after int.
    @test_throws Exception validate((Tile(3, 8), 2, 1, 3, 4, Tile(3, 2)))
    # Multiple Ints
    @test_throws Exception validate((1, 1))
    @test_throws Exception validate((1, Tile(2, 2), 1, 2))
    # Tile with no int
    @test_throws Exception validate((Tile(3, 8), 1, 2))
    # Tile with negative index
    @test_throws Exception validate((Tile(-2, 8), 1, 2))
end

@testset "Testing Splitting" begin
    TiledArrays = OneDNN.TiledArrays
    Tile = TiledArrays.Tile

    # Now that'd we've tested the code generation logic ... lets test the ACTUAL logic.
    vlayout = Val((2, 1, 3))
    @test TiledArrays.splitindex(vlayout, (10, 20, 30)) == (20, 10, 30) .- 1

    vlayout = Val((Tile(3, 8), 2, 1, 3))
    @test TiledArrays.splitindex(vlayout, (1, 1, 1)) == (0, 0, 0, 0)
    @test TiledArrays.splitindex(vlayout, (1, 1, 2)) == (1, 0, 0, 0)
    @test TiledArrays.splitindex(vlayout, (1, 1, 8)) == (7, 0, 0, 0)
    # Hit the tile barrier
    @test TiledArrays.splitindex(vlayout, (1, 1, 9)) == (0, 0, 0, 1)
    @test TiledArrays.splitindex(vlayout, (1, 1, 16)) == (7, 0, 0, 1)
    @test TiledArrays.splitindex(vlayout, (1, 1, 17)) == (0, 0, 0, 2)

    @test TiledArrays.splitindex(vlayout, (2, 3, 8)) == (7, 3 - 1, 2 - 1, 0)
    @test TiledArrays.splitindex(vlayout, (2, 3, 9)) == (0, 3 - 1, 2 - 1, 1)

    # Multiple levels of tiling
    vlayout = Val((Tile(3, 8), 2, Tile(3, 2), 1, 3))
    @test TiledArrays.splitindex(vlayout, (1, 1, 1)) == (0, 0, 0, 0, 0)
    @test TiledArrays.splitindex(vlayout, (1, 1, 2)) == (1, 0, 0, 0, 0)
    @test TiledArrays.splitindex(vlayout, (1, 1, 8)) == (7, 0, 0, 0, 0)
    @test TiledArrays.splitindex(vlayout, (1, 1, 9)) == (0, 0, 1, 0, 0)
    @test TiledArrays.splitindex(vlayout, (1, 1, 16)) == (7, 0, 1, 0, 0)
    @test TiledArrays.splitindex(vlayout, (1, 1, 17)) == (0, 0, 0, 0, 1)
    @test TiledArrays.splitindex(vlayout, (1, 1, 24)) == (7, 0, 0, 0, 1)
    @test TiledArrays.splitindex(vlayout, (1, 1, 25)) == (0, 0, 1, 0, 1)
end
