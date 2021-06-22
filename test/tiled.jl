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

    ### strip_leading_tiles
    Tile = OneDNN.TiledArrays.Tile
    strip_leading_tiles = OneDNN.TiledArrays.strip_leading_tiles
    x = (1, 2, 3)
    @test strip_leading_tiles(x) == x

    x = (Tile(1, 2), 3, 4, 5)
    @test strip_leading_tiles(x) == (3, 4, 5)

    x = (Tile(1, 2), Tile(3, 4), 5, 6)
    @test strip_leading_tiles(x) == (5, 6)

    x = (Tile(1, 2), Tile(3, 4), 5, Tile(6, 7), 8)
    @test strip_leading_tiles(x) == (5, Tile(6, 7), 8)

    ### adjust_for_padding
    adjust_for_padding = OneDNN.TiledArrays.adjust_for_padding
    size = (13, 17, 512)
    padded_size = (16, 32, 512)
    @test adjust_for_padding(size, padded_size, (0, 0, 0)) == (0, 0, 0)
    @test adjust_for_padding(size, padded_size, (1, 1, 1)) == (1, 1, 1)
    @test adjust_for_padding(size, padded_size, (12, 17, 10)) == (12, 32, 10)
    @test adjust_for_padding(size, padded_size, (13, 17, 10)) == (16, 32, 10)
    @test adjust_for_padding(size, padded_size, (14, 16, 512)) == (16 + 1, 16, 512)
    @test adjust_for_padding(size, padded_size, (27, 18, 513)) == (2 * 16 + 1, 32 + 1, 513)
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
    _splitindex(vlayout, I) = TiledArrays.splitindex(vlayout, I .- one(Int))

    # Now that'd we've tested the code generation logic ... lets test the ACTUAL logic.
    vlayout = Val((2, 1, 3))
    @test _splitindex(vlayout, (10, 20, 30)) == (20, 10, 30) .- 1

    vlayout = Val((Tile(3, 8), 2, 1, 3))
    @test _splitindex(vlayout, (1, 1, 1)) == (0, 0, 0, 0)
    @test _splitindex(vlayout, (1, 1, 2)) == (1, 0, 0, 0)
    @test _splitindex(vlayout, (1, 1, 8)) == (7, 0, 0, 0)
    # Hit the tile barrier
    @test _splitindex(vlayout, (1, 1, 9)) == (0, 0, 0, 1)
    @test _splitindex(vlayout, (1, 1, 16)) == (7, 0, 0, 1)
    @test _splitindex(vlayout, (1, 1, 17)) == (0, 0, 0, 2)

    @test _splitindex(vlayout, (2, 3, 8)) == (7, 3 - 1, 2 - 1, 0)
    @test _splitindex(vlayout, (2, 3, 9)) == (0, 3 - 1, 2 - 1, 1)

    # Multiple levels of tiling
    vlayout = Val((Tile(3, 8), 2, Tile(3, 2), 1, 3))
    @test _splitindex(vlayout, (1, 1, 1)) == (0, 0, 0, 0, 0)
    @test _splitindex(vlayout, (1, 1, 2)) == (1, 0, 0, 0, 0)
    @test _splitindex(vlayout, (1, 1, 8)) == (7, 0, 0, 0, 0)
    @test _splitindex(vlayout, (1, 1, 9)) == (0, 0, 1, 0, 0)
    @test _splitindex(vlayout, (1, 1, 16)) == (7, 0, 1, 0, 0)
    @test _splitindex(vlayout, (1, 1, 17)) == (0, 0, 0, 0, 1)
    @test _splitindex(vlayout, (1, 1, 24)) == (7, 0, 0, 0, 1)
    @test _splitindex(vlayout, (1, 1, 25)) == (0, 0, 1, 0, 1)
end

@testset "Testing Layout Conversion" begin
    # The goal here is to test that we're doing the correct thing when converting from
    # OneDNN layouts for ALL of the layout types in OneDNN.
    #
    # So, we need to:
    #
    # 1. Generate a candidate set of arrays from 1 to 12 dimensions.
    #    Note that we need to have kind of weird dimensionality on these arrays to make
    #    sure we deal with OneDNN's padding correctly.
    #
    # 2. Iterate through all the unique format tags available, converting the arrays of
    #    the correct dimension to that format.
    #
    # 3. Use our handy-dandy format conversion tool to pull out the appropriate indices,
    #    collect the Memory's parent using a view with these indices, and check if a
    #    reshaped version of the view matches the original array.

    # Make the candidate set of arrays.
    # Construct 5 arrays of each dimension.
    arrays_per_dim = 5
    cdiv(a, b) = ceil(Int, a / b)

    # Characters for determining the number of dimensions.
    # Just go up to 4 dims to keep this somewhat reasonable.
    # Keep the length of `chars` to 5 so we can detect if there are 6 or more dimensions
    # and bail.
    chars = ['a', 'b', 'c', 'd', 'e']
    # Drop the first 5 characters to avoid the leading "dnnl_".
    getndims(x::Symbol) = something(findlast(in(lowercase(string(x))[5:end]), chars))

    function rand_in_range(lower, upper, ndims)
        return randn(
            Float32, (rand(cdiv(lower, 4 * j):cdiv(upper, 4 * j)) for j = 1:ndims)...
        )
    end

    dimension_to_array = Dict(
        map(1:length(chars)) do i
            # Cut down upper bound to try to conserve memory consumption.
            return i => [rand_in_range(13, 512, i) for _ = 1:arrays_per_dim]
        end
    )
    @show Base.summarysize(dimension_to_array)

    # Create initial "Memory" objects so we don't always have to be recreating these.
    dimension_to_memory = Dict(k => OneDNN.Memory.(v) for (k, v) in dimension_to_array)

    # Formats to ignore
    ignore = [:dnnl_format_tag_undef, :dnnl_format_tag_any]
    exit_on = :dnnl_format_tag_last
    seen = Set{UInt32}()

    # Formats that have failed the round-trip test.
    broken_formats = Set([:dnnl_ABc4a4b, :dnnl_ABcd4a4b])

    failed_formats = Set{Symbol}()
    translation_failed = Set{Symbol}()
    meter = ProgressMeter.Progress(Int(OneDNN.Lib.dnnl_format_tag_last), 1)
    for (name, value) in CEnum.name_value_pairs(OneDNN.Lib.dnnl_format_tag_t)
        ProgressMeter.next!(meter)
        # Bookkeeping ...
        in(name, ignore) && continue
        name == exit_on && break
        in(value, seen) && error("Already seen: $name => $(Int(value))")
        push!(seen, value)

        ndims = getndims(name)
        ndims === lastindex(chars) && continue

        @test in(ndims, 1:length(chars))

        arrays = dimension_to_array[ndims]
        memories = dimension_to_memory[ndims]

        for (array, memory) in zip(arrays, memories)
            new_md = OneDNN.memorydesc(
                Float32, size(memory), OneDNN.Lib.dnnl_format_tag_t(value)
            )
            reordered = OneDNN.reorder(new_md, memory)

            # Can we at least round-trip `materialize`.
            passed = (array == OneDNN.materialize(reordered))

            # Some formats were broken when these tests were written.
            # Here, we put in a test to see if they remain broken in the future.
            if !passed
                push!(failed_formats, name)
                continue
            end

            # Now, we assume that everything is working correctly.
            @test passed

            indices = OneDNN.generate_linear_indices(reordered)
            formatted = reshape(view(parent(reordered), indices), size(array))
            translation_passed = (array == formatted)
            @test translation_passed
            if !translation_passed
                push!(translation_failed, name)
            end
        end
    end
    ProgressMeter.finish!(meter)
    @test failed_formats == broken_formats
    if !isempty(translation_failed)
        @show translation_failed
    end
end
