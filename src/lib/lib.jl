module Lib

using Libdl: Libdl

using CEnum

const PKGDIR = dirname(dirname(@__DIR__))
const USRDIR = joinpath(PKGDIR, "deps", "dnnl")
const LIBDIR = joinpath(USRDIR, "lib")

const _flags = Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL

const dnnl = joinpath(LIBDIR, "libdnnl.so")
Libdl.dlopen(dnnl, _flags)

include("types.jl")
include("dnnl.jl")

end # module
