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

@testset "Testing Utils" begin
    # @testset "Testing @apicall" begin
    #     # Create the correct GlobalRefs.
    #     lib = GlobalRef(OneDNN, :Lib)
    #     convert = GlobalRef(OneDNN, :dnnl_convert)
    #     _convert = GlobalRef(OneDNN, :_dnnl_convert)

    #     # Standard transformation
    #     expr = @macroexpand OneDNN.@apicall dnnl_abc(a, b)
    #     expected = :($lib.dnnl_abc($convert(a), $convert(b)))
    #     @test expr == expected

    #     # Does it ignore function calls that don't start with "dnnl"?
    #     expr = @macroexpand OneDNN.@apicall f(a, b)
    #     expected = :(f($convert(a), $convert(b)))
    #     @test expr == expected

    #     # Does it correctly handle varargs
    #     expr = @macroexpand OneDNN.@apicall f(a, b...)
    #     expected = :(f($convert(a), $_convert(b...)...))
    #     @test expr == expected

    #     expr = @macroexpand OneDNN.@apicall f(a...)
    #     expected = :(f($_convert(a...)...))
    #     @test expr == expected

    #     # Finally, prepend "lib"
    #     expr = @macroexpand OneDNN.@apicall dnnl_abc(a, b...)
    #     expected = :($lib.dnnl_abc($convert(a), $_convert(b...)...))
    #     @test expr == expected
    # end

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
end # @testset
