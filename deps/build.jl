# url = """
# https://github.com/oneapi-src/oneDNN/releases/download/v1.5/dnnl_lnx_1.5.0_cpu_gomp.tgz
# """
#
# local_tarball = "dnnl.tgz"
# local_dir = "dnnl"
#
# if !ispath(local_tarball)
#     download(url, local_tarball)
#     run(`tar -xvf $localpath`)
# end
#
# if !ispath(local_dir)
#     mv("dnnl_lnx_1.5.0_cpu_gomp", "dnnl")
# end

#####
##### threadpool
#####

import CxxWrap

# Paths for linking
cxxhome = CxxWrap.prefix_path()
juliahome = dirname(Base.Sys.BINDIR)
dnnlhome = joinpath(@__DIR__, "dnnl")

# Use Clang since it seems to get along better with Julia
cxx = "clang++"

cxxflags = [
    "-g",
    "-O3",
    "-Wall",
    "-fPIC",
    "-std=c++17",
    "-DPCM_SILENT",
    "-DJULIA_ENABLE_THREADING",
    "-Dexcept_EXPORTS",
    # Surpress some warnings from Cxx
    "-Wno-unused-variable",
    "-Wno-unused-lambda-capture",
]

includes = [
    "-I$(joinpath(cxxhome, "include"))",
    "-I$(joinpath(juliahome, "include", "julia"))",
    "-I$(joinpath(dnnlhome, "include"))",
]

loadflags = [
    # Linking flags for Julia
    "-L$(joinpath(juliahome, "lib"))",
    "-Wl,--export-dynamic",
    "-Wl,-rpath,$(joinpath(juliahome, "lib"))",
    "-ljulia",
    # Linking Flags for CxxWrap
    "-L$(joinpath(cxxhome, "lib"))",
    "-Wl,-rpath,$(joinpath(cxxhome, "lib"))",
    "-lcxxwrap_julia",
    # # Linking Flags for nGraph
    # "-L$(joinpath(ngraphhome, "lib"))",
    # "-Wl,-rpath,$(joinpath(ngraphhome, "lib"))",
    # "-lngraph",
]

src = joinpath(@__DIR__, "threadpool.cpp")
so = joinpath(@__DIR__, "libthreadpool.so")

cmd = `$cxx $cxxflags $includes -shared $src -lpthread -o $so $loadflags`
@show cmd
run(cmd)

# #####
# ##### Debug Build
# #####
#
# using LibGit2
#
# #####
# ##### Fetch Repo
# #####
#
# url = "https://github.com/oneapi-src/oneDNN"
# branch = "master"
# tag = "v1.5"
#
# localdir = joinpath(@__DIR__, "oneDNN")
# if !ispath(localdir)
#     LibGit2.clone(url, localdir; branch = branch)
#     repo = LibGit2.GitRepo(localdir)
#     commit = LibGit2.GitCommit(repo, tag)
#     LibGit2.checkout!(repo, string(LibGit2.GitHash(commit)))
# end
#
# builddir = joinpath(localdir, "build")
# mkpath(builddir)
# currentdir = pwd()
#
# CC = "clang"
# CXX = "clang++"
# nproc = parse(Int, read(`nproc`, String))
#
# cd(builddir)
#
# buildtype = "DEBUG"
#
# cmake_args = [
#     # configure cmake and installation dir.
#     "-DCMAKE_C_COMPILER=$CC",
#     "-DCMAKE_CXX_COMPILER=$CXX",
#     "-DCMAKE_INSTALL_PREFIX=$(joinpath(@__DIR__, "usr"))",
#     "-DCMAKE_BUILD_TYPE=$buildtype",
#     # oneDNN options
#     "-DDNNL_LIBRARY_TYPE=SHARED",
# ]
#
# run(`cmake .. $cmake_args`)
# run(`make -j $nproc`)
# run(`make install`)
#
