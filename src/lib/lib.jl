module Lib

using Libdl: Libdl

using CEnum

const PKGDIR = dirname(dirname(@__DIR__))
const USRDIR = joinpath(PKGDIR, "deps", "dnnl")
const LIBDIR = joinpath(USRDIR, "lib")

const _flags = Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL

const libdnnl = joinpath(LIBDIR, "libdnnl.so")
Libdl.dlopen(libdnnl, _flags)

#include("types.jl")
include("dnnl.jl")
#include("dnnl_threadpool.jl")

end # module
