# This script generates the wrapper code for the C API to OneDNN
#
# This is largely based on the wrapper code for "Sundials":
# https://github.com/SciML/Sundials.jl/blob/master/scripts/wrap_sundials.jl

using Clang
using Clang.Generators
using Clang.LibClang.Clang_jll
using MacroTools

# This is where the wrapper gets generated.
#
# If it looks Okay, move into the `src` folder.
const outpath = joinpath(@__DIR__, "dnnl.jl")
#mkpath(outpath)

# Find the relevant headers.
# We're just looking for `dnnl.h` and `dnnl_types.h`.
const include_dir = joinpath(dirname(@__DIR__), "deps", "dnnl", "include")

const onednn_headers = [
    joinpath(include_dir, "dnnl.h"),
    joinpath(include_dir, "dnnl_threadpool.h"),
]

args = get_default_args()
push!(args, "-I$include_dir")

options = Dict("general" => Dict(
    "output_file_path" => outpath,
    "library_name" => "libdnnl",
    "use_julia_native_enum_type" => false,
    "use_deterministic_symbol" => true,
))

ctx = create_context(onednn_headers, args, options)

# Build without printing so we can do custom rewriting.
build!(ctx, BUILDSTAGE_NO_PRINTING)

# For rewriting purposes, we need to intercept all entries for
# `dnnl_dims_t` and replace them with `Ptr{dnnl_dim_t}`.
custom_pointer_types = [
    :dnnl_memory,
    :dnnl_engine,
    :dnnl_primitive_desc_iterator,
    :dnnl_primitive_desc,
    :dnnl_primitive_attr,
    :dnnl_post_ops,
    :dnnl_primitive,
    :dnnl_stream,
]

rewrite(notexpr) = Any[notexpr]
function rewrite(expr::Expr)
    # Replace objects that Clang turns into "Cvoid" into singleton structs.
    # This provides a bit more type-safety when calling the C API.
    # Also need to make "unsafe_convert" an error to avoid pointers swapping types
    # accidentally.
    if MacroTools.@capture(expr, mutable struct name_ end)
        if in(name, custom_pointer_types)
            expr = quote
                $expr

                function Base.cconvert(::Type{Ptr{$name}}, x::Ptr{$name})
                    return x
                end
                function Base.cconvert(::Type{Ptr{$name}}, x::Ptr)
                    return error("Refusing to convert $(typeof(x)) to a Ptr{$($name)}!")
                end

                function Base.cconvert(::Type{Ptr{Ptr{$name}}}, x::Ptr{Ptr{$name}})
                    return x
                end
                function Base.cconvert(::Type{Ptr{Ptr{$name}}}, x::Ptr)
                    return error("Refusing to convert $(typeof(x)) to a Ptr{Ptr{$($name)}}!")
                end
            end |> MacroTools.prettify
        end
        return [expr]

    # Check if this is a ccall
    elseif expr.head == :function
        # use the powerful macrotools to transform `dnnl_dims_t` to `ptr{dnnl_dim_t}`.
        function_body = expr.args[2]
        expr.args[2] = MacroTools.postwalk(function_body) do x
            if x == :dnnl_dims_t
                return :(Ptr{dnnl_dim_t})
            else
                return x
            end
        end
    elseif expr.head == :const
        expr = MacroTools.postwalk(expr) do ex
            if ex == :INT64_MIN
                return :(typemin(Int64))
            elseif ex == :size_t
                return :unsigned
            # Delete this definition since it depends on something that gets skipped.
            elseif in(ex, (:DNNL_RUNTIME_S32_VAL_REP, :NULL))
                return 0
            end
            return ex
        end
    end
    return [expr]
end

function rewrite!(dag::ExprDAG)
    for node in get_nodes(dag)
        new_exprs = []
        for expr in get_exprs(node)
            append!(new_exprs, rewrite(expr))
        end
        node.exprs .= new_exprs
    end
end

# HACK WARNING!!
# For some reason, the topological sorting algorithm in Clang.jl is not catching this Union
# type ...
# Manually hoist it's definition to the beginning.
function replace_JL_Ctag_89!(dag::ExprDAG)
    nodes = get_nodes(dag)
    uniondef = nodes[end-1]
    @assert isa(uniondef, ExprNode{Clang.Generators.UnionAnonymous, Clang.CLUnionDecl})
    deleteat!(nodes, length(nodes) - 1)

    found = false
    for (i, node) in enumerate(nodes)
        for expr in get_exprs(node)
            if MacroTools.@capture(expr, struct T_ args__ end)
                if T == :dnnl_memory_desc_t
                    found = true
                    break
                end
            end
        end
        if found
            insert!(nodes, i - 1, uniondef)
            break
        end
    end
    @assert found
end

replace_JL_Ctag_89!(ctx.dag)
rewrite!(ctx.dag)

# build
build!(ctx, BUILDSTAGE_PRINTING_ONLY)
