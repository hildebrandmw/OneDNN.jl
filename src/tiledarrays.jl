module TiledArrays

# Ceiling division
cdiv(a::Integer, b::Integer) = cdiv(promote(a, b)...)
cdiv(a::T, b::T) where {T<:Integer} = one(T) + div(a - one(T), b)

# Opposite of `Base.tail`.
# Keep all but the last element in a tuple.
head(x, y...) = (x, head(y...)...)
head(x) = ()

# # Type traits for describing if there's something scary going on with the strides.
# # For now, assume memory strides are monotonically increasing such that the underlying
# # array describes a contiguous block of memory.
# abstract type AbstractStrides end
# struct DenseStrides <: AbstractStrides end

struct Tile
    dim::Int
    size::Int
end
Base.show(io::IO, tile::Tile) = print(io, "Tile($(tile.dim), $(tile.size))")

# Note that `NonBlockedLayout <: MemoryLayout`, but without any tiling, allowing
# for specific handling of the easy case.
const MemoryLayout{N} = Tuple{Vararg{Union{Int,Tile},N}} where {N}
const NonBlockedLayout{N} = Tuple{Vararg{Int,N}} where {N}

# Utility function for removing all "Tile"s from the front of an array.
function strip_leading_tiles(x::Tuple{Vararg{Union{Int,Tile},N}}) where {N}
    return strip_leading_tiles(x...)
end

function strip_leading_tiles(x::Tile, y::Vararg{Union{Int,Tile},N}) where {N}
    return strip_leading_tiles(y...)
end

strip_leading_tiles(x::Int, y::Vararg{Union{Int,Tile},N}) where {N} = (x, y...)

#####
##### Layout Validation
#####

# Validation
# Invariants to maintain:
#
# 1. We must have exactly one occurance of an Integer index.
# 2. All `Tile` indexes must occur BEFORE the corresponding Integer index.
# 3. Each `Tile` index must have a corresponding Integer index
function validate(layout::MemoryLayout{N}) where {N}
    @nospecialize

    # Pass 1 - Make sure there are no nonpositive indices.
    # Also, find the maximum index.
    maximum_index = 0
    for desc in layout
        if isa(desc, Int)
            desc <= 0 && error("Found non-positive index: $desc")
            maximum_index = max(maximum_index, desc)
        elseif isa(desc, Tile)
            desc.dim <= 0 && error("Found non-positive tile index: $desc")
            maximum_index = max(maximum_index, desc.dim)
        else
            # Should never happen due to type restrictions.
            error("Unreachable reached")
        end
    end

    # Pass 2 - Check invariants.
    for index in Base.OneTo(maximum_index)
        saw_int = false
        saw_tile = false
        for desc in layout
            if isa(desc, Int) && desc == index
                saw_int && error("Integer index $index occurs more than once!")
                saw_int = true
            elseif isa(desc, Tile) && desc.dim == index
                saw_int &&
                    error("Integer index $index occurs before its corresponding tile index!")
                saw_tile = true
            end
        end
        # We saw a tile but not an int, throw error
        saw_tile &&
            !saw_int &&
            error("Found tile with index $index with no corresponding Int")

        # We didn't find anything
        !saw_tile && !saw_int && error("Found no entry with index $index")
    end
    return true
end

#####
##### Adjust indexes for padding.
#####

unchecked_divrem(x::Int, y::Int) = (Base.sdiv_int(x, y), Base.srem_int(x, y))

# Note: not necessarily safe since we are using the unchecked integer division functions.
function adjust_for_padding(
    size::NTuple{N,Int}, padded_size::NTuple{N,Int}, index::NTuple{N,Int}
) where {N}
    return ntuple(i -> _adjust_for_padding(size[i], padded_size[i], index[i]), Val(N))
end

# Path where no padding is used.
function adjust_for_padding(size::NTuple{N,Int}, ::Nothing, index::NTuple{N,Int}) where {N}
    return index
end

function _adjust_for_padding(size::Int, padded_size::Int, index::Int)
    size == padded_size && return index
    a, b = unchecked_divrem(index, size)
    return (a * padded_size) + b
end

#####
##### Index Conversion Logic
#####

# Expand the index according to the given layout.
# NOTE: `I` should be post padding adjusting if necessary.
function splitindex(::Val{T}, I::Tuple{Vararg{Int,N}}) where {T,N}
    return _splitindex(T, I)
end

# Work from the first index to the last.
# At each step, strip off the index we processed from `layout` and recurse.
#
# When `layout[1]` is an Integer, we just return the corresponding index from `I`.
#
# When `layout[1]` is a Tile, then we grab the corresponding index from `I`,
# `divrem` that value, then reconstruct `I` with the reduced index.
#
# NB: This depends on Julia's super agressive constant propagation heuristics when dealing
# with simplifying tuple recursive functions to generate great assembly code.
@inline function _splitindex(layout::MemoryLayout, I::Tuple{Vararg{Int,N}}) where {N}
    v, _I = process(layout[1], I)
    return (v, _splitindex(Base.tail(layout), _I)...)
end

### Base case - last item should never be a `Tile`.
@inline function _splitindex(layout::Tuple{Int}, I::Tuple{Vararg{Int,N}}) where {N}
    return (I[layout[1]],)
end

function _splitindex(::Tuple{Tile}, I::Tuple{Vararg{Int,N}}) where {N}
    return error("Malformed memory layout with a `Tile` in the tail position!")
end

@inline process(x::Int, I::Tuple{Vararg{Int,N}}) where {N} = (I[x], I)
@inline function process(x::Tile, I::Tuple{Vararg{Int,N}}) where {N}
    outer, inner = unchecked_divrem(I[x.dim], x.size)
    return (inner, ntuple(i -> i == x.dim ? outer : I[i], Val(N)))
end

#####
##### Obtaining full dimensions
#####

# Move layout from type domain to value domain, providing "const" information to the
# compiler.
@inline dims(valT::Val{T}, size::Tuple{Vararg{Int,N}}) where {T,N} = _dims(T, size)

@inline function _dims(layout::MemoryLayout, size::Tuple{Vararg{Int,N}}) where {N}
    v, _size = _dimcheck(layout[1], size)
    return (v, _dims(Base.tail(layout), _size)...)
end

# Bottom of recursion.
@inline _dims(layout::Tuple{Int}, size::Tuple{Vararg{Int,N}}) where {N} = (size[layout[1]],)

@inline _dimcheck(x::Int, size::Tuple{Vararg{Int,N}}) where {N} = (size[x], size)
@inline function _dimcheck(x::Tile, size::Tuple{Vararg{Int,N}}) where {N}
    newdim = cdiv(size[x.dim], x.size)
    return (x.size, ntuple(i -> i == x.dim ? newdim : size[i], Val(N)))
end

#####
##### Users of "dims"
#####

# NB: Cannot extend `Base.strides` because a `MemoryLayout` can potentially just be a tuple
# of Ints, so we'd be committing type piracy.
function _strides(valT::Val{T}, size::Tuple{Vararg{Int,N}}) where {T,N}
    # Drop the last value from `dims`.
    return cumprod((one(Int), head(dims(valT, size)...)...))
end

# function dimstrides(valT::Val{T}, size::Tuple{Vararg{Int,N}}) where {T,N}
#     return applyperm(_strides(valT, size), T)
# end
#
# @inline function applyperm(size::Tuple{Vararg{Int,N}}, perm::Tuple{Vararg{Int,N}}) where {N}
#     return ntuple(i -> size[perm[i]], Val(N))
# end

function fullsize(valT::Val{T}, size::Tuple{Vararg{Int,N}}) where {T,N}
    # Constant propagation to the rescue.
    # If the layout is non-blocked, just standard or permuted, then there is no
    # over-allocation necessary.
    if isa(T, NonBlockedLayout)
        return prod(size)
        # Otherwise, we need to compute the full dimensions, which could include zero padding
        # to keep the inner blocksizes constant.
    else
        return prod(dims(valT, size))
    end
end

#####
##### Linear Offset
#####

function getoffset(
    valT::Val{T},
    size::Tuple{Vararg{Int,N}},     # Padded Size
    I::Tuple{Vararg{Int,N}},        # Post-padded index
) where {T,N}
    Base.@_inline_meta
    strides = _strides(valT, size)
    index = splitindex(valT, I)
    offset = zero(Int)
    for i in eachindex(strides)
        @inbounds offset += strides[i] * index[i]
    end
    return offset
end

# Generates indices into a tiled array.
# Parameters:
#
# `L`: Layout type parameter.
# `N`: Number of Dims.
struct TiledIndexer{L,N}
    size::NTuple{N,Int}
    padded_size::NTuple{N,Int}
end

function TiledIndexer{L}(size::NTuple{N,Int}, padded_size::NTuple{N,Int}) where {L,N}
    return TiledIndexer{L,N}(size, padded_size)
end

function genindex(x::TiledIndexer{L,N}, I::NTuple{N,Int}) where {L,N}
    return 1 + getoffset(Val(L), x.padded_size, adjust_for_padding(x.size, x.padded_size, I .- 1))
end

# function genindex(x::TiledIndexer{L,N,Nothing}, I::NTuple{N,Int}) where {L,N}
#     return getoffset(Val(N), x.size, I)
# end

# #####
# ##### Tiled Array
# #####
#
# struct TiledArray{T,L,N} <: AbstractArray{T,N}
#     parent::Vector{T}
#     size::NTuple{N,Int}
# end
# Base.parent(A::TiledArray) = A.parent
#
# function TiledArray{T,L}(::UndefInitializer, size::Tuple{Vararg{Int,N}}) where {T,L,N}
#     # Allocate enough room for the whole layout.
#     parent = Vector{T}(undef, fullsize(Val(L), size))
#     return TiledArray{T,L,N}(parent, size)
# end
#
# layout(::Type{<:TiledArray{<:Any,L}}) where {L} = L
# layout(x::T) where {T<:TiledArray} = layout(T)
# vlayout(x) = Val(layout(x))
#
# # Array interface
# @inline Base.size(x::TiledArray) = x.size
# Base.IndexStyle(::Type{<:TiledArray}) = Base.IndexCartesian()
# Base.@propagate_inbounds function Base.getindex(
#     A::TiledArray{T,L,N}, I::Vararg{Int,N}; strides = nothing
# ) where {T,L,N}
#     @boundscheck checkbounds(A, I...)
#     offset = getoffset(vlayout(A), size(A), I, strides) + one(Int)
#     return @inbounds(getindex(parent(A), offset))
# end
#
# Base.@propagate_inbounds function Base.setindex!(
#     A::TiledArray{T,L,N}, v, I::Vararg{Int,N}; strides = nothing
# ) where {T,L,N}
#     @boundscheck checkbounds(A, I...)
#     offset = getoffset(vlayout(A), size(A), I, strides) + one(Int)
#     return @inbounds setindex!(parent(A), v, offset)
# end

end # module
