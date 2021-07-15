onednn_url = "https://github.com/oneapi-src/OneDNN"
onednn_tag = "v2.3"
install_dir = joinpath(@__DIR__, "dnnl")

onednn_src = joinpath(@__DIR__, "oneDNN")
onednn_build = joinpath(onednn_src, "build")

# Clone and checkout.
if !isdir(onednn_src)
    run(`git clone $onednn_url $onednn_src`)
    cd(onednn_src)
    run(`git checkout $onednn_tag`)
end

# Apply performance patch.
patchdir = joinpath(@__DIR__, "patches")
patchfile = joinpath(patchdir, "onednn.patch")
cd(onednn_src)
try
    run(`git apply $patchfile`)
catch e
    isa(e, ProcessFailedException) || rethrow(e)
end

# Run cmake
cmake_args = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$(install_dir)",
    "-DDNNL_LIBRARY_TYPE=SHARED",
    "-DCMAKE_CXX_COMPILER=icc",
    "-DCMAKE_C_COMPILER=icc",
    "-DDNNL_BUILD_TESTS=OFF",
    "-DDNNL_BUILD_EXAMPLES=OFF",
    "-DDNNL_CPU_RUNTIME=THREADPOOL",
]

isdir(onednn_build) || mkdir(onednn_build)
cd(onednn_build)
run(`cmake .. $cmake_args`)
nproc = parse(Int, read(`nproc`, String))
run(`make -j$nproc`)
run(`make install`)

# Cleanup
cd(@__DIR__)

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
]

src = joinpath(@__DIR__, "threadpool.cpp")
so = joinpath(@__DIR__, "libthreadpool.so")

cmd = `$cxx $cxxflags $includes -shared $src -lpthread -o $so $loadflags`
@show cmd
run(cmd)

