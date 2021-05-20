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
    inner_blks = blocking_desc.inner_blks
    inner_idxs = blocking_desc.inner_idxs
    for i in 1:(blocking_desc.inner_nblks)
        # Convert from index-0 to index-1.
        push!(tiles, TiledArrays.Tile(inner_idxs[i] + 1, inner_blks[i]))
    end
    # Order the indices.
    _strides = collect(blocking_desc.strides[1:ndims])
    outer = reverse(sortperm(_strides))
    return (tiles..., outer...)
end

function _blocking_desc(x::Lib.dnnl_memory_desc_t)
    if x.format_kind != Lib.dnnl_blocked
        @show x.format_kind
        error("Wrong Format Kind!")
    end
    return _blocking_desc(x.format_desc)
end

# TODO: Fix this weird "ANONYMOUS1" thing caused by the Union ...
_blocking_desc(x::Lib.ANONYMOUS1_format_desc) = x.blocking
