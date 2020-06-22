module OneDNN

#####
##### deps
#####

# Include auto-generated wrapper for OneDNN
include("lib/lib.jl")
include("utils.jl")
include("memory.jl")
include("ops/matmul.jl")

#include("dispatch.jl")
#include("primitive.jl")
#include("initializer.jl")

#include("compiler/compiler.jl")

# Just create a global engine and stream for everything to use for now.
const GLOBAL_ENGINE = Ref{Engine}()
const GLOBAL_STREAM = Ref{Stream}()

# Initialize the default Engine and Stream.
function __init__()
    GLOBAL_ENGINE[] = Engine(Lib.dnnl_cpu)
    GLOBAL_STREAM[] = Stream(GLOBAL_ENGINE[])
end

end # module
