# OneDNN

# Eltwise



## Submemory and Julia Views

**NOTE**: This is outdated now.

Getting OneDNN `sub_memory_desc` and Julia views to cooperate is kind of a pain.
My general approach is that a `Memory` can be constructed from a Julia `SubArray` as long as

1. That SubArray wraps an appropriately low level type such as a Julia `Array` or a
    `ReshapedArray` of some `DenseArray` (reshaping doesn't change the layout of the
    underlying object.

    That is, we can't yet have a `SubArray{T,N,<:LinearAlgebra.Transpose}` or
    `SubArray{T,N,<:PermutedDimsArray}` because that just gets way to confusing to keep
    track of and does not provide sufficient benefit to me at the moment to go through the
    trouble of figuring it out.

2. The indices of this view must define a "nice" sub memory of the parent array.
    This means no reverse indexing, diagonal slicing, non-uniform slicing.
    The view must basically define a continuous, rectilinear sub region of the original
    memory.

Furthermore, we cannot take a view directly of a `Memory`.
Instead, we must `OneDNN.materialize` the memory, take a view of that, and then create a new `Memory` from that `SubArray`.
This is a little more expensive, but keeps way down on the headaches.
