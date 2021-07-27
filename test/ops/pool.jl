# @testset "Testing Pooling" begin
#     @testset "Testing Dimension Utilities" begin
#         # expand
#         @test OneDNN.expand(Val(2), 1) == (1,1)
#         @test OneDNN.expand(Val(2), 2) == (2,2)
#         @test OneDNN.expand(Val(3), 1) == (1,1,1)
#         @test OneDNN.expand(Val(2), (5,6)) == (5,6)
#
#         x = (1,2)
#         y = (10, 20, 30, 40)
#         @test OneDNN._paddims(x, y) == (1, 2, 30, 40)
#         @test OneDNN._append(Val(4), (2, 3), 1) == (2, 3, 1, 1)
#     end
# end
