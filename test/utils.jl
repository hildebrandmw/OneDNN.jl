# Util types for testing
mutable struct CalledConvert
    called::Bool
end
CalledConvert() = CalledConvert(false)
function OneDNN.dnnl_convert(x::CalledConvert)
    x.called = true
    return x
end
reset!(x::CalledConvert) = (x.called = false)
wascalled(x::CalledConvert) = x.called

partial_expand(expr) = macroexpand(OneDNN, OneDNN._apicall_partial_impl(expr))

# Testing `dnnl_arg` pipeline
mutable struct MemoryWrapper
    ptr::OneDNN.Lib.dnnl_memory_t
    context::OneDNN.AccessContext
end
MemoryWrapper() = MemoryWrapper(OneDNN.Lib.dnnl_memory_t(), OneDNN.Unknown())
function OneDNN.dnnl_exec_arg(y::MemoryWrapper, context)
    y.context = context
    return y.ptr
end

@testset "Testing Utils" begin
    @testset "Testing @apicall" begin
        lib = :Lib
        convert = :dnnl_convert
        _convert = :_dnnl_convert
        esca = Expr(:escape, :a)
        escb = Expr(:escape, :b)
        escf = Expr(:escape, :f)

        # Standard Transformation
        expr = partial_expand(:(dnnl_abc(a, b)))
        expected = :($lib.dnnl_abc($convert($esca), $convert($escb)))
        @test expr == expected

        # Does it ignore function calls that don't start with "dnnl"?
        expr = partial_expand(:(f(a, b)))
        expected = :($escf($convert($esca), $convert($escb)))
        @test expr == expected

        # Does it correctly handle varargs
        expr = partial_expand(:(f(a, b...)))
        expected = :($escf($convert($esca), $_convert($escb...)...))
        @test expr == expected

        expr = partial_expand(:(f(a...)))
        expected = :($escf($_convert($esca...)...))
        @test expr == expected

        # Finally, prepend "lib"
        expr = partial_expand(:(dnnl_abc(a, b...)))
        expected = :($lib.dnnl_abc($convert($esca), $_convert($escb...)...))
        @test expr == expected
    end

    @testset "Testing Ancestor" begin
        w = collect(1:10)
        x = view(w, 1:4)
        y = reshape(x, 2, 2)
        z = view(y, :, 1)

        @test OneDNN.ancestor(w) === w
        @test OneDNN.ancestor(x) === w
        @test OneDNN.ancestor(y) === w
        @test OneDNN.ancestor(z) === w
    end

    @testset "Testing Arguments" begin
        @test OneDNN.getsym(:hello) == "hello"
        @test OneDNN.getsym(:(A.B.hello)) == "hello"
        @test OneDNN.getsym(QuoteNode(:hello)) == "hello"

        wrapper = MemoryWrapper()
        @test wrapper.context == OneDNN.Unknown()
        x = OneDNN.dnnl_arg(OneDNN.Lib.DNNL_ARG_SRC, wrapper)
        @test isa(x, OneDNN.Lib.dnnl_exec_arg_t)
        @test wrapper.context == OneDNN.Reading()
        @test length(x) == 1
        @test collect(x) == [x]

        # Construction utilities.
        tuple_no_array = (1,2,3,4)
        @test OneDNN._hasarray(tuple_no_array...) == false
        @test OneDNN._flatcat(tuple_no_array...) == (1,2,3,4)

        tuple_no_array = ((1,3), 2, 3, 4)
        @test OneDNN._hasarray(tuple_no_array...) == false
        @test OneDNN._flatcat(tuple_no_array...) == (1,3,2,3,4)

        tuple_no_array = ((1, 2), 3, (4, (5, 6), 7), 8)
        @test OneDNN._hasarray(tuple_no_array...) == false
        @test OneDNN._flatcat(tuple_no_array...) == (1,2,3,4,5,6,7,8)

        tuple_with_array = ([1,2], 2, 3, 4)
        @test OneDNN._hasarray(tuple_with_array...) == true
        @test OneDNN._vcat(tuple_with_array; init = Int[]) == [1,2,2,3,4]

        tuple_with_array = (1, 2, [3], 4)
        @test OneDNN._hasarray(tuple_with_array...) == true
        @test OneDNN._vcat(tuple_with_array; init = Int[]) == [1,2,3,4]

        tuple_with_array = (1, 2, 3, [4])
        @test OneDNN._hasarray(tuple_with_array...) == true
        @test OneDNN._vcat(tuple_with_array; init = Int[]) == [1,2,3,4]

        tuple_with_array = (1, [2], 3, [4,2])
        @test OneDNN._hasarray(tuple_with_array...) == true
        @test OneDNN._vcat(tuple_with_array; init = Int[]) == [1,2,3,4,2]
    end

    @testset "Testing Conversions" begin
        a = CalledConvert()
        OneDNN.dnnl_convert(a)
        @test wascalled(a)

        x = (CalledConvert(), CalledConvert(), CalledConvert())
        OneDNN._dnnl_convert(x...)
        @test all(wascalled, x)

        # Wrapping and unwrapping.
        x = Ref(10)
        y = 10

        # Equality for "Ref"s doesn't check contents.
        # Define our own notiong of "equality" here to express the semantics that "wrap_ref"
        # should create a "Ref" is the argument is not already a "Ref".
        ref_equal(x::Ref, y::Ref) = x[] === y[]

        @test OneDNN.unwrap_ref(x) === 10
        @test OneDNN.unwrap_ref(y) === 10
        @test ref_equal(OneDNN.wrap_ref(x), Ref(10))
        @test ref_equal(OneDNN.wrap_ref(y), Ref(10))
    end

    @testset "Testing SIMD Compress" begin
        layouts = [
            OneDNN.Lib.dnnl_ab,
            OneDNN.Lib.dnnl_ba,
            OneDNN.Lib.dnnl_AB16b16a,
            OneDNN.Lib.dnnl_AB16b32a,
            OneDNN.Lib.dnnl_AB16b64a,
            OneDNN.Lib.dnnl_AB8b16a2b,
        ]

        src_base = randn(Float32, 2048, 2048)
        dst_base = randn(Float32, 2048, 2048)
        eta = Float32(2.0)
        result_reference = dst_base .- (eta .* src_base)
        for src_layout in layouts, dst_layout in layouts
            @show src_layout, dst_layout
            src = OneDNN.reorder(src_layout, OneDNN.Memory(copy(src_base)))
            dst = OneDNN.reorder(dst_layout, OneDNN.Memory(copy(dst_base)))

            isrc = OneDNN.generate_linear_indices(src)
            idst = OneDNN.generate_linear_indices(dst)
            map = OneDNN.simdcompress(Float32, isrc, idst)
            OneDNN.sgd!(parent(dst), parent(src), map, eta)
            @test OneDNN.materialize(dst) == result_reference

            # Now - try again using the `Flux.update!` API
            # Run it once to trigger caching of the translation layer,
            # then run it again to make sure we actually hit in the cache.
            src = OneDNN.reorder(src_layout, OneDNN.Memory(copy(src_base)))
            dst = OneDNN.reorder(dst_layout, OneDNN.Memory(copy(dst_base)))
            Flux.Optimise.update!(Flux.Descent(eta), dst, src)
            @test OneDNN.materialize(dst) == result_reference
            if src_layout != dst_layout
                key = (OneDNN.memorydesc(dst), OneDNN.memorydesc(src))
                @test haskey(OneDNN.TRANSLATION_DICT, key)

                src = OneDNN.reorder(src_layout, OneDNN.Memory(copy(src_base)))
                dst = OneDNN.reorder(dst_layout, OneDNN.Memory(copy(dst_base)))
                Flux.Optimise.update!(Flux.Descent(eta), dst, src)
                @test OneDNN.materialize(dst) == result_reference
            end
        end
    end
end # @testset
