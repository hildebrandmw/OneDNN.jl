# This script generates the wrapper code for the C API to OneDNN
#
# This is largely based on the wrapper code for "Sundials":
# https://github.com/SciML/Sundials.jl/blob/master/scripts/wrap_sundials.jl

using Clang
using MacroTools

# This is where the wrapper gets generated.
#
# If it looks Okay, move into the `src` folder.
const outpath = joinpath(normpath(joinpath(@__DIR__, "wrapped_api")))
mkpath(outpath)

# Find the relevant headers.
# We're just looking for `dnnl.h` and `dnnl_types.h`.
const include_dir = joinpath(dirname(@__DIR__), "deps", "dnnl", "include")

const onednn_headers = [
    joinpath(include_dir, "dnnl.h"),
]

# Includes for Clang.jl
#
# Why does `CLANG_INCLUDES` not exist?
#const clang_includes = [CLANG_INCLUDES]
#const clang_includes = String[]

function wrap_header(top_hdr::AbstractString, cursor_header::AbstractString)
    if occursin("dnnl", cursor_header)
        return true
    end
    return false
end

const REMOVE_MACROS = [
    "DNNL_RUNTIME_DIM_VAL",
    "DNNL_RUNTIME_S32_VAL",
    "DNNL_MEMORY_NONE",
]

function wrap_cursor(cursor_name::AbstractString, cursor)
    # There's one macro definition we have to snipe.
    if isa(cursor, Clang.CLMacroDefinition)
        # Remove the macros we don't want.
        if in(cursor_name, REMOVE_MACROS)
            return false
        else
            return true
        end
    end
    return true
end

# # Mapping of header files to Julia files
# function julia_file(header::AbstractString)
#     src_name = basename(dirname(header))
#     #if src_name == "sundials"
#     #    src_name = "libsundials" # avoid having both Sundials.jl and sundials.jl
#     #end
#     return joinpath(outpath, string(src_name, ".jl"))
# end
# function library_file(header::AbstractString)
#     header_name = basename(header)
#     if startswith(header_name, "nvector")
#         return "libsundials_nvecserial"
#     else
#         return string("libsundials_", basename(dirname(header)))
#     end
# end

#####
##### Context Creation
#####

const context = init(;
    common_file = joinpath(outpath, "types.jl"),
    clang_diagnostics = true,
    clang_includes = [include_dir],
    header_wrapped = wrap_header,
    cursor_wrapped = wrap_cursor,
)
context.headers = onednn_headers

# For rewriting purposes, we need to intercept all entries for
# `dnnl_dims_t` and replace them with `Ptr{dnnl_dim_t}`.

wrap_onednn_api(notexpr) = Any[notexpr]

function wrap_onednn_api(expr::Expr)
    # Check if this is aa ccall
    if expr.head == :function
        # Handle special cases.
        #
        # For some reason, the argument to this method is not captured.
        # I have no idea why ...
        if expr.args[1].args[1] == :dnnl_memory_desc_get_size
            expr = :(
                function dnnl_memory_desc_get_size(memory_desc)
                    ccall((:dnnl_memory_desc_get_size, dnnl), Csize_t, (Ptr{dnnl_memory_desc_t},), memory_desc)
                end
            )
            return [expr]
        end

        # use the powerful macrotools to transform `dnnl_dims_t` to `ptr{dnnl_dim_t}`.
        function_body = expr.args[2]
        expr.args[2] = MacroTools.postwalk(function_body) do x
            if x == :dnnl_dims_t
                return :(Ptr{dnnl_dim_t})
            else
                return x
            end
        end
    end
    return [expr]
end

context.rewriter = function(exprs)
    mod_exprs = sizehint!(Vector{Any}(), length(exprs))
    for expr in exprs
        append!(mod_exprs, wrap_onednn_api(expr))
    end
    return mod_exprs
end

@info("Generating .jl wrappers for OneDNN in $outpath...")
run(context)
@info("Done generating .jl wrappers for OneDNN in $outpath")

