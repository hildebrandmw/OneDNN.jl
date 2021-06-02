# Julia wrapper for header: dnnl_threadpool.h
# Automatically generated using Clang.jl


function dnnl_threadpool_interop_stream_create(stream, engine, threadpool)
    ccall((:dnnl_threadpool_interop_stream_create, dnnl), dnnl_status_t, (Ptr{dnnl_stream_t}, dnnl_engine_t, Ptr{Cvoid}), stream, engine, threadpool)
end

function dnnl_threadpool_interop_stream_get_threadpool(astream, threadpool)
    ccall((:dnnl_threadpool_interop_stream_get_threadpool, dnnl), dnnl_status_t, (dnnl_stream_t, Ptr{Ptr{Cvoid}}), astream, threadpool)
end

function dnnl_threadpool_interop_sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, threadpool)
    ccall((:dnnl_threadpool_interop_sgemm, dnnl), dnnl_status_t, (UInt8, UInt8, dnnl_dim_t, dnnl_dim_t, dnnl_dim_t, Cfloat, Ptr{Cfloat}, dnnl_dim_t, Ptr{Cfloat}, dnnl_dim_t, Cfloat, Ptr{Cfloat}, dnnl_dim_t, Ptr{Cvoid}), transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, threadpool)
end

function dnnl_threadpool_interop_gemm_u8s8s32(transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co, threadpool)
    ccall((:dnnl_threadpool_interop_gemm_u8s8s32, dnnl), dnnl_status_t, (UInt8, UInt8, UInt8, dnnl_dim_t, dnnl_dim_t, dnnl_dim_t, Cfloat, Ptr{UInt8}, dnnl_dim_t, UInt8, Ptr{Int8}, dnnl_dim_t, Int8, Cfloat, Ptr{Int32}, dnnl_dim_t, Ptr{Int32}, Ptr{Cvoid}), transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co, threadpool)
end

function dnnl_threadpool_interop_gemm_s8s8s32(transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co, threadpool)
    ccall((:dnnl_threadpool_interop_gemm_s8s8s32, dnnl), dnnl_status_t, (UInt8, UInt8, UInt8, dnnl_dim_t, dnnl_dim_t, dnnl_dim_t, Cfloat, Ptr{Int8}, dnnl_dim_t, Int8, Ptr{Int8}, dnnl_dim_t, Int8, Cfloat, Ptr{Int32}, dnnl_dim_t, Ptr{Int32}, Ptr{Cvoid}), transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co, threadpool)
end
