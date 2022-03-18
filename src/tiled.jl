#####
##### Parse `dnnl_blocking_desc_t`
#####

# Define a long dispatch chain to unwrap types until we finally get to the final
# blocking description.

layout(x::Ref{Lib.dnnl_memory_desc_t}) = layout(unwrap_ref(x))
function layout(memory_desc::Lib.dnnl_memory_desc_t)
    # Get the blocking desription.
    # We use this in conjunction with the `memory_desc_t` to reconstruct the format.
    blocking_desc = _blocking_desc(memory_desc)
    ndims = memory_desc.ndims

    # Construct the inner tilings
    tiles = TiledArrays.Tile[]
    inner_num_blocks = blocking_desc.inner_nblks
    inner_blks = reverse(blocking_desc.inner_blks[1:inner_num_blocks])
    #inner_idxs = blocking_desc.inner_idxs[1:inner_num_blocks]
    inner_idxs = ndims .- reverse(blocking_desc.inner_idxs[1:inner_num_blocks])
    for i in 1:inner_num_blocks
        # Convert from index-0 to index-1.
        push!(tiles, TiledArrays.Tile(inner_idxs[i], inner_blks[i]))
    end
    # Order the indices.
    _strides = collect(reverse(blocking_desc.strides[1:ndims]))
    outer = sortperm(_strides)
    return (tiles..., outer...)
end

function _blocking_desc(x::Lib.dnnl_memory_desc_t)
    if x.format_kind != Lib.dnnl_blocked
        @show x.format_kind
        error("Wrong Format Kind!")
    end
    return x.format_desc
    #return _blocking_desc(x.format_desc)
end

# TODO: Fix this weird "ANONYMOUS1" thing caused by the Union ...
#_blocking_desc(x::Lib.__JL_Ctag_89) = x.blocking

function generate_linear_indices(layout::TiledArrays.MemoryLayout, size, padded_size)
    indexer = TiledArrays.TiledIndexer{layout}(size, padded_size)
    return generate_linear_indices(indexer, size)
end

function generate_linear_indices(indexer::TiledArrays.TiledIndexer, size)
    return vec([TiledArrays.genindex(indexer, Tuple(I)) for I in CartesianIndices(size)])
end
