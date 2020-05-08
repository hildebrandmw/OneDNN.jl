using LibGit2

#####
##### Fetch Repo
#####

url = "https://github.com/oneapi-src/oneDNN"
branch = "master"
tag = "v1.4"

localdir = joinpath(@__DIR__, "oneDNN")
if !ispath(localdir)
    LibGit2.clone(url, localdir; branch = branch)
    repo = LibGit2.GitRepo(localdir)
    commit = LibGit2.GitCommit(repo, tag)
    LibGit2.checkout!(repo, string(LibGit2.GitHash(commit)))
end

#####
##### building
#####

builddir = joinpath(localdir, "build")
mkpath(builddir)
currentdir = pwd()

CC = "clang"
CXX = "clang++"
nproc = parse(Int, read(`nproc`, String))

cd(builddir)

cmake_args = [
    # configure cmake and installation dir.
    "-DCMAKE_C_COMPILER=$CC",
    "-DCMAKE_CXX_COMPILER=$CXX",
    "-DCMAKE_INSTALL_PREFIX=$(joinpath(@__DIR__, "usr"))",
    "-DCMAKE_BUILD_TYPE=DEBUG",
    # oneDNN options
    "-DDNNL_LIBRARY_TYPE=SHARED",
]

run(`cmake .. $cmake_args`)
run(`make -j $nproc`)
run(`make install`)

