# Julia wrapper for header: dnnl.h
# Automatically generated using Clang.jl


function dnnl_primitive_desc_iterator_create(iterator, op_desc, attr, engine, hint_forward_primitive_desc)
    ccall((:dnnl_primitive_desc_iterator_create, dnnl), dnnl_status_t, (Ptr{dnnl_primitive_desc_iterator_t}, const_dnnl_op_desc_t, const_dnnl_primitive_attr_t, dnnl_engine_t, const_dnnl_primitive_desc_t), iterator, op_desc, attr, engine, hint_forward_primitive_desc)
end

function dnnl_primitive_desc_iterator_next(iterator)
    ccall((:dnnl_primitive_desc_iterator_next, dnnl), dnnl_status_t, (dnnl_primitive_desc_iterator_t,), iterator)
end

function dnnl_primitive_desc_iterator_fetch(iterator)
    ccall((:dnnl_primitive_desc_iterator_fetch, dnnl), dnnl_primitive_desc_t, (const_dnnl_primitive_desc_iterator_t,), iterator)
end

function dnnl_primitive_desc_iterator_destroy(iterator)
    ccall((:dnnl_primitive_desc_iterator_destroy, dnnl), dnnl_status_t, (dnnl_primitive_desc_iterator_t,), iterator)
end

function dnnl_primitive_desc_create(primitive_desc, op_desc, attr, engine, hint_forward_primitive_desc)
    ccall((:dnnl_primitive_desc_create, dnnl), dnnl_status_t, (Ptr{dnnl_primitive_desc_t}, const_dnnl_op_desc_t, const_dnnl_primitive_attr_t, dnnl_engine_t, const_dnnl_primitive_desc_t), primitive_desc, op_desc, attr, engine, hint_forward_primitive_desc)
end

function dnnl_primitive_desc_clone(primitive_desc, existing_primitive_desc)
    ccall((:dnnl_primitive_desc_clone, dnnl), dnnl_status_t, (Ptr{dnnl_primitive_desc_t}, const_dnnl_primitive_desc_t), primitive_desc, existing_primitive_desc)
end

function dnnl_primitive_desc_get_attr(primitive_desc, attr)
    ccall((:dnnl_primitive_desc_get_attr, dnnl), dnnl_status_t, (const_dnnl_primitive_desc_t, Ptr{const_dnnl_primitive_attr_t}), primitive_desc, attr)
end

function dnnl_primitive_desc_destroy(primitive_desc)
    ccall((:dnnl_primitive_desc_destroy, dnnl), dnnl_status_t, (dnnl_primitive_desc_t,), primitive_desc)
end

function dnnl_primitive_desc_query(primitive_desc, what, index, result)
    ccall((:dnnl_primitive_desc_query, dnnl), dnnl_status_t, (const_dnnl_primitive_desc_t, dnnl_query_t, Cint, Ptr{Cvoid}), primitive_desc, what, index, result)
end

function dnnl_primitive_desc_query_md(primitive_desc, what, index)
    ccall((:dnnl_primitive_desc_query_md, dnnl), Ptr{dnnl_memory_desc_t}, (const_dnnl_primitive_desc_t, dnnl_query_t, Cint), primitive_desc, what, index)
end

function dnnl_primitive_desc_query_s32(primitive_desc, what, index)
    ccall((:dnnl_primitive_desc_query_s32, dnnl), Cint, (const_dnnl_primitive_desc_t, dnnl_query_t, Cint), primitive_desc, what, index)
end

function dnnl_primitive_create(primitive, primitive_desc)
    ccall((:dnnl_primitive_create, dnnl), dnnl_status_t, (Ptr{dnnl_primitive_t}, const_dnnl_primitive_desc_t), primitive, primitive_desc)
end

function dnnl_primitive_execute(primitive, stream, nargs, args)
    ccall((:dnnl_primitive_execute, dnnl), dnnl_status_t, (const_dnnl_primitive_t, dnnl_stream_t, Cint, Ptr{dnnl_exec_arg_t}), primitive, stream, nargs, args)
end

function dnnl_primitive_get_primitive_desc(primitive, primitive_desc)
    ccall((:dnnl_primitive_get_primitive_desc, dnnl), dnnl_status_t, (const_dnnl_primitive_t, Ptr{const_dnnl_primitive_desc_t}), primitive, primitive_desc)
end

function dnnl_primitive_destroy(primitive)
    ccall((:dnnl_primitive_destroy, dnnl), dnnl_status_t, (dnnl_primitive_t,), primitive)
end

function dnnl_primitive_attr_create(attr)
    ccall((:dnnl_primitive_attr_create, dnnl), dnnl_status_t, (Ptr{dnnl_primitive_attr_t},), attr)
end

function dnnl_primitive_attr_clone(attr, existing_attr)
    ccall((:dnnl_primitive_attr_clone, dnnl), dnnl_status_t, (Ptr{dnnl_primitive_attr_t}, const_dnnl_primitive_attr_t), attr, existing_attr)
end

function dnnl_primitive_attr_destroy(attr)
    ccall((:dnnl_primitive_attr_destroy, dnnl), dnnl_status_t, (dnnl_primitive_attr_t,), attr)
end

function dnnl_primitive_attr_get_scratchpad_mode(attr, mode)
    ccall((:dnnl_primitive_attr_get_scratchpad_mode, dnnl), dnnl_status_t, (const_dnnl_primitive_attr_t, Ptr{dnnl_scratchpad_mode_t}), attr, mode)
end

function dnnl_primitive_attr_set_scratchpad_mode(attr, mode)
    ccall((:dnnl_primitive_attr_set_scratchpad_mode, dnnl), dnnl_status_t, (dnnl_primitive_attr_t, dnnl_scratchpad_mode_t), attr, mode)
end

function dnnl_primitive_attr_get_output_scales(attr, count, mask, scales)
    ccall((:dnnl_primitive_attr_get_output_scales, dnnl), dnnl_status_t, (const_dnnl_primitive_attr_t, Ptr{dnnl_dim_t}, Ptr{Cint}, Ptr{Ptr{Cfloat}}), attr, count, mask, scales)
end

function dnnl_primitive_attr_set_output_scales(attr, count, mask, scales)
    ccall((:dnnl_primitive_attr_set_output_scales, dnnl), dnnl_status_t, (dnnl_primitive_attr_t, dnnl_dim_t, Cint, Ptr{Cfloat}), attr, count, mask, scales)
end

function dnnl_primitive_attr_get_scales(attr, arg, count, mask, scales)
    ccall((:dnnl_primitive_attr_get_scales, dnnl), dnnl_status_t, (dnnl_primitive_attr_t, Cint, Ptr{dnnl_dim_t}, Ptr{Cint}, Ptr{Ptr{Cfloat}}), attr, arg, count, mask, scales)
end

function dnnl_primitive_attr_set_scales(attr, arg, count, mask, scales)
    ccall((:dnnl_primitive_attr_set_scales, dnnl), dnnl_status_t, (dnnl_primitive_attr_t, Cint, dnnl_dim_t, Cint, Ptr{Cfloat}), attr, arg, count, mask, scales)
end

function dnnl_primitive_attr_get_zero_points(attr, arg, count, mask, zero_points)
    ccall((:dnnl_primitive_attr_get_zero_points, dnnl), dnnl_status_t, (const_dnnl_primitive_attr_t, Cint, Ptr{dnnl_dim_t}, Ptr{Cint}, Ptr{Ptr{Int32}}), attr, arg, count, mask, zero_points)
end

function dnnl_primitive_attr_set_zero_points(attr, arg, count, mask, zero_points)
    ccall((:dnnl_primitive_attr_set_zero_points, dnnl), dnnl_status_t, (dnnl_primitive_attr_t, Cint, dnnl_dim_t, Cint, Ptr{Int32}), attr, arg, count, mask, zero_points)
end

function dnnl_primitive_attr_get_post_ops(attr, post_ops)
    ccall((:dnnl_primitive_attr_get_post_ops, dnnl), dnnl_status_t, (const_dnnl_primitive_attr_t, Ptr{const_dnnl_post_ops_t}), attr, post_ops)
end

function dnnl_primitive_attr_set_post_ops(attr, post_ops)
    ccall((:dnnl_primitive_attr_set_post_ops, dnnl), dnnl_status_t, (dnnl_primitive_attr_t, const_dnnl_post_ops_t), attr, post_ops)
end

function dnnl_post_ops_create(post_ops)
    ccall((:dnnl_post_ops_create, dnnl), dnnl_status_t, (Ptr{dnnl_post_ops_t},), post_ops)
end

function dnnl_post_ops_destroy(post_ops)
    ccall((:dnnl_post_ops_destroy, dnnl), dnnl_status_t, (dnnl_post_ops_t,), post_ops)
end

function dnnl_post_ops_len(post_ops)
    ccall((:dnnl_post_ops_len, dnnl), Cint, (const_dnnl_post_ops_t,), post_ops)
end

function dnnl_post_ops_get_kind(post_ops, index)
    ccall((:dnnl_post_ops_get_kind, dnnl), dnnl_primitive_kind_t, (const_dnnl_post_ops_t, Cint), post_ops, index)
end

function dnnl_post_ops_append_sum(post_ops, scale)
    ccall((:dnnl_post_ops_append_sum, dnnl), dnnl_status_t, (dnnl_post_ops_t, Cfloat), post_ops, scale)
end

function dnnl_post_ops_get_params_sum(post_ops, index, scale)
    ccall((:dnnl_post_ops_get_params_sum, dnnl), dnnl_status_t, (const_dnnl_post_ops_t, Cint, Ptr{Cfloat}), post_ops, index, scale)
end

function dnnl_post_ops_append_eltwise(post_ops, scale, alg_kind, alpha, beta)
    ccall((:dnnl_post_ops_append_eltwise, dnnl), dnnl_status_t, (dnnl_post_ops_t, Cfloat, dnnl_alg_kind_t, Cfloat, Cfloat), post_ops, scale, alg_kind, alpha, beta)
end

function dnnl_post_ops_get_params_eltwise(post_ops, index, scale, alg_kind, alpha, beta)
    ccall((:dnnl_post_ops_get_params_eltwise, dnnl), dnnl_status_t, (const_dnnl_post_ops_t, Cint, Ptr{Cfloat}, Ptr{dnnl_alg_kind_t}, Ptr{Cfloat}, Ptr{Cfloat}), post_ops, index, scale, alg_kind, alpha, beta)
end

function dnnl_post_ops_append_dw_k3s1p1(post_ops, weights_data_type, bias_data_type, dst_data_type, count, mask, scales)
    ccall((:dnnl_post_ops_append_dw_k3s1p1, dnnl), dnnl_status_t, (dnnl_post_ops_t, dnnl_data_type_t, dnnl_data_type_t, dnnl_data_type_t, dnnl_dim_t, Cint, Ptr{Cfloat}), post_ops, weights_data_type, bias_data_type, dst_data_type, count, mask, scales)
end

function dnnl_post_ops_get_params_dw_k3s1p1(post_ops, index, weights_data_type, bias_data_type, dst_data_type, count, mask, scales)
    ccall((:dnnl_post_ops_get_params_dw_k3s1p1, dnnl), dnnl_status_t, (const_dnnl_post_ops_t, Cint, Ptr{dnnl_data_type_t}, Ptr{dnnl_data_type_t}, Ptr{dnnl_data_type_t}, Ptr{dnnl_dim_t}, Ptr{Cint}, Ptr{Ptr{Cfloat}}), post_ops, index, weights_data_type, bias_data_type, dst_data_type, count, mask, scales)
end

function dnnl_post_ops_append_dw_k3s2p1(post_ops, weights_data_type, bias_data_type, dst_data_type, count, mask, scales)
    ccall((:dnnl_post_ops_append_dw_k3s2p1, dnnl), dnnl_status_t, (dnnl_post_ops_t, dnnl_data_type_t, dnnl_data_type_t, dnnl_data_type_t, dnnl_dim_t, Cint, Ptr{Cfloat}), post_ops, weights_data_type, bias_data_type, dst_data_type, count, mask, scales)
end

function dnnl_post_ops_get_params_dw_k3s2p1(post_ops, index, weights_data_type, bias_data_type, dst_data_type, count, mask, scales)
    ccall((:dnnl_post_ops_get_params_dw_k3s2p1, dnnl), dnnl_status_t, (const_dnnl_post_ops_t, Cint, Ptr{dnnl_data_type_t}, Ptr{dnnl_data_type_t}, Ptr{dnnl_data_type_t}, Ptr{dnnl_dim_t}, Ptr{Cint}, Ptr{Ptr{Cfloat}}), post_ops, index, weights_data_type, bias_data_type, dst_data_type, count, mask, scales)
end

function dnnl_memory_desc_init_by_strides(memory_desc, ndims, dims, data_type, strides)
    ccall((:dnnl_memory_desc_init_by_strides, dnnl), dnnl_status_t, (Ptr{dnnl_memory_desc_t}, Cint, Ptr{dnnl_dim_t}, dnnl_data_type_t, Ptr{dnnl_dim_t}), memory_desc, ndims, dims, data_type, strides)
end

function dnnl_memory_desc_init_by_tag(memory_desc, ndims, dims, data_type, tag)
    ccall((:dnnl_memory_desc_init_by_tag, dnnl), dnnl_status_t, (Ptr{dnnl_memory_desc_t}, Cint, Ptr{dnnl_dim_t}, dnnl_data_type_t, dnnl_format_tag_t), memory_desc, ndims, dims, data_type, tag)
end

function dnnl_memory_desc_init_submemory(memory_desc, parent_memory_desc, dims, offsets)
    ccall((:dnnl_memory_desc_init_submemory, dnnl), dnnl_status_t, (Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), memory_desc, parent_memory_desc, dims, offsets)
end

function dnnl_memory_desc_reshape(out_memory_desc, in_memory_desc, ndims, dims)
    ccall((:dnnl_memory_desc_reshape, dnnl), dnnl_status_t, (Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Cint, Ptr{dnnl_dim_t}), out_memory_desc, in_memory_desc, ndims, dims)
end

function dnnl_memory_desc_permute_axes(out_memory_desc, in_memory_desc, permutation)
    ccall((:dnnl_memory_desc_permute_axes, dnnl), dnnl_status_t, (Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{Cint}), out_memory_desc, in_memory_desc, permutation)
end

function dnnl_memory_desc_equal(lhs, rhs)
    ccall((:dnnl_memory_desc_equal, dnnl), Cint, (Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}), lhs, rhs)
end

function dnnl_memory_desc_get_size(memory_desc)
    #= /home/mark/projects/OneDNN/scripts/wrap.jl:100 =#
    ccall((:dnnl_memory_desc_get_size, dnnl), Csize_t, (Ptr{dnnl_memory_desc_t},), memory_desc)
end

function dnnl_memory_create(memory, memory_desc, engine, handle)
    ccall((:dnnl_memory_create, dnnl), dnnl_status_t, (Ptr{dnnl_memory_t}, Ptr{dnnl_memory_desc_t}, dnnl_engine_t, Ptr{Cvoid}), memory, memory_desc, engine, handle)
end

function dnnl_memory_get_memory_desc(memory, memory_desc)
    ccall((:dnnl_memory_get_memory_desc, dnnl), dnnl_status_t, (const_dnnl_memory_t, Ptr{Ptr{dnnl_memory_desc_t}}), memory, memory_desc)
end

function dnnl_memory_get_engine(memory, engine)
    ccall((:dnnl_memory_get_engine, dnnl), dnnl_status_t, (const_dnnl_memory_t, Ptr{dnnl_engine_t}), memory, engine)
end

function dnnl_memory_map_data(memory, mapped_ptr)
    ccall((:dnnl_memory_map_data, dnnl), dnnl_status_t, (const_dnnl_memory_t, Ptr{Ptr{Cvoid}}), memory, mapped_ptr)
end

function dnnl_memory_unmap_data(memory, mapped_ptr)
    ccall((:dnnl_memory_unmap_data, dnnl), dnnl_status_t, (const_dnnl_memory_t, Ptr{Cvoid}), memory, mapped_ptr)
end

function dnnl_memory_get_data_handle(memory, handle)
    ccall((:dnnl_memory_get_data_handle, dnnl), dnnl_status_t, (const_dnnl_memory_t, Ptr{Ptr{Cvoid}}), memory, handle)
end

function dnnl_memory_set_data_handle(memory, handle)
    ccall((:dnnl_memory_set_data_handle, dnnl), dnnl_status_t, (dnnl_memory_t, Ptr{Cvoid}), memory, handle)
end

function dnnl_memory_set_data_handle_v2(memory, handle, stream)
    ccall((:dnnl_memory_set_data_handle_v2, dnnl), dnnl_status_t, (dnnl_memory_t, Ptr{Cvoid}, dnnl_stream_t), memory, handle, stream)
end

function dnnl_memory_destroy(memory)
    ccall((:dnnl_memory_destroy, dnnl), dnnl_status_t, (dnnl_memory_t,), memory)
end

function dnnl_reorder_primitive_desc_create(reorder_primitive_desc, src_desc, src_engine, dst_desc, dst_engine, attr)
    ccall((:dnnl_reorder_primitive_desc_create, dnnl), dnnl_status_t, (Ptr{dnnl_primitive_desc_t}, Ptr{dnnl_memory_desc_t}, dnnl_engine_t, Ptr{dnnl_memory_desc_t}, dnnl_engine_t, const_dnnl_primitive_attr_t), reorder_primitive_desc, src_desc, src_engine, dst_desc, dst_engine, attr)
end

function dnnl_concat_primitive_desc_create(concat_primitive_desc, dst_desc, n, concat_dimension, src_descs, attr, engine)
    ccall((:dnnl_concat_primitive_desc_create, dnnl), dnnl_status_t, (Ptr{dnnl_primitive_desc_t}, Ptr{dnnl_memory_desc_t}, Cint, Cint, Ptr{dnnl_memory_desc_t}, const_dnnl_primitive_attr_t, dnnl_engine_t), concat_primitive_desc, dst_desc, n, concat_dimension, src_descs, attr, engine)
end

function dnnl_sum_primitive_desc_create(sum_primitive_desc, dst_desc, n, scales, src_descs, attr, engine)
    ccall((:dnnl_sum_primitive_desc_create, dnnl), dnnl_status_t, (Ptr{dnnl_primitive_desc_t}, Ptr{dnnl_memory_desc_t}, Cint, Ptr{Cfloat}, Ptr{dnnl_memory_desc_t}, const_dnnl_primitive_attr_t, dnnl_engine_t), sum_primitive_desc, dst_desc, n, scales, src_descs, attr, engine)
end

function dnnl_binary_desc_init(binary_desc, alg_kind, src0_desc, src1_desc, dst_desc)
    ccall((:dnnl_binary_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_binary_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}), binary_desc, alg_kind, src0_desc, src1_desc, dst_desc)
end

function dnnl_convolution_forward_desc_init(conv_desc, prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc, strides, padding_l, padding_r)
    ccall((:dnnl_convolution_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_convolution_desc_t}, dnnl_prop_kind_t, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), conv_desc, prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc, strides, padding_l, padding_r)
end

function dnnl_dilated_convolution_forward_desc_init(conv_desc, prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc, strides, dilates, padding_l, padding_r)
    ccall((:dnnl_dilated_convolution_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_convolution_desc_t}, dnnl_prop_kind_t, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), conv_desc, prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc, strides, dilates, padding_l, padding_r)
end

function dnnl_convolution_backward_data_desc_init(conv_desc, alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides, padding_l, padding_r)
    ccall((:dnnl_convolution_backward_data_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_convolution_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), conv_desc, alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides, padding_l, padding_r)
end

function dnnl_dilated_convolution_backward_data_desc_init(conv_desc, alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides, dilates, padding_l, padding_r)
    ccall((:dnnl_dilated_convolution_backward_data_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_convolution_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), conv_desc, alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides, dilates, padding_l, padding_r)
end

function dnnl_convolution_backward_weights_desc_init(conv_desc, alg_kind, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, padding_l, padding_r)
    ccall((:dnnl_convolution_backward_weights_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_convolution_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), conv_desc, alg_kind, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, padding_l, padding_r)
end

function dnnl_dilated_convolution_backward_weights_desc_init(conv_desc, alg_kind, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, dilates, padding_l, padding_r)
    ccall((:dnnl_dilated_convolution_backward_weights_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_convolution_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), conv_desc, alg_kind, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, dilates, padding_l, padding_r)
end

function dnnl_deconvolution_forward_desc_init(deconv_desc, prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc, strides, padding_l, padding_r)
    ccall((:dnnl_deconvolution_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_deconvolution_desc_t}, dnnl_prop_kind_t, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), deconv_desc, prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc, strides, padding_l, padding_r)
end

function dnnl_dilated_deconvolution_forward_desc_init(deconv_desc, prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc, strides, dilates, padding_l, padding_r)
    ccall((:dnnl_dilated_deconvolution_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_deconvolution_desc_t}, dnnl_prop_kind_t, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), deconv_desc, prop_kind, alg_kind, src_desc, weights_desc, bias_desc, dst_desc, strides, dilates, padding_l, padding_r)
end

function dnnl_deconvolution_backward_data_desc_init(deconv_desc, alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides, padding_l, padding_r)
    ccall((:dnnl_deconvolution_backward_data_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_deconvolution_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), deconv_desc, alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides, padding_l, padding_r)
end

function dnnl_dilated_deconvolution_backward_data_desc_init(deconv_desc, alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides, dilates, padding_l, padding_r)
    ccall((:dnnl_dilated_deconvolution_backward_data_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_deconvolution_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), deconv_desc, alg_kind, diff_src_desc, weights_desc, diff_dst_desc, strides, dilates, padding_l, padding_r)
end

function dnnl_deconvolution_backward_weights_desc_init(deconv_desc, alg_kind, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, padding_l, padding_r)
    ccall((:dnnl_deconvolution_backward_weights_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_deconvolution_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), deconv_desc, alg_kind, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, padding_l, padding_r)
end

function dnnl_dilated_deconvolution_backward_weights_desc_init(deconv_desc, alg_kind, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, dilates, padding_l, padding_r)
    ccall((:dnnl_dilated_deconvolution_backward_weights_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_deconvolution_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), deconv_desc, alg_kind, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, dilates, padding_l, padding_r)
end

function dnnl_shuffle_forward_desc_init(shuffle_desc, prop_kind, data_desc, axis, group_size)
    ccall((:dnnl_shuffle_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_shuffle_desc_t}, dnnl_prop_kind_t, Ptr{dnnl_memory_desc_t}, Cint, dnnl_dim_t), shuffle_desc, prop_kind, data_desc, axis, group_size)
end

function dnnl_shuffle_backward_desc_init(shuffle_desc, diff_data_desc, axis, group_size)
    ccall((:dnnl_shuffle_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_shuffle_desc_t}, Ptr{dnnl_memory_desc_t}, Cint, dnnl_dim_t), shuffle_desc, diff_data_desc, axis, group_size)
end

function dnnl_eltwise_forward_desc_init(eltwise_desc, prop_kind, alg_kind, data_desc, alpha, beta)
    ccall((:dnnl_eltwise_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_eltwise_desc_t}, dnnl_prop_kind_t, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Cfloat, Cfloat), eltwise_desc, prop_kind, alg_kind, data_desc, alpha, beta)
end

function dnnl_eltwise_backward_desc_init(eltwise_desc, alg_kind, diff_data_desc, data_desc, alpha, beta)
    ccall((:dnnl_eltwise_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_eltwise_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Cfloat, Cfloat), eltwise_desc, alg_kind, diff_data_desc, data_desc, alpha, beta)
end

function dnnl_softmax_forward_desc_init(softmax_desc, prop_kind, data_desc, softmax_axis)
    ccall((:dnnl_softmax_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_softmax_desc_t}, dnnl_prop_kind_t, Ptr{dnnl_memory_desc_t}, Cint), softmax_desc, prop_kind, data_desc, softmax_axis)
end

function dnnl_softmax_backward_desc_init(softmax_desc, diff_data_desc, data_desc, softmax_axis)
    ccall((:dnnl_softmax_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_softmax_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Cint), softmax_desc, diff_data_desc, data_desc, softmax_axis)
end

function dnnl_logsoftmax_forward_desc_init(logsoftmax_desc, prop_kind, data_desc, logsoftmax_axis)
    ccall((:dnnl_logsoftmax_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_logsoftmax_desc_t}, dnnl_prop_kind_t, Ptr{dnnl_memory_desc_t}, Cint), logsoftmax_desc, prop_kind, data_desc, logsoftmax_axis)
end

function dnnl_logsoftmax_backward_desc_init(logsoftmax_desc, diff_data_desc, data_desc, logsoftmax_axis)
    ccall((:dnnl_logsoftmax_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_logsoftmax_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Cint), logsoftmax_desc, diff_data_desc, data_desc, logsoftmax_axis)
end

function dnnl_pooling_forward_desc_init(pool_desc, prop_kind, alg_kind, src_desc, dst_desc, strides, kernel, padding_l, padding_r)
    ccall((:dnnl_pooling_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_pooling_desc_t}, dnnl_prop_kind_t, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), pool_desc, prop_kind, alg_kind, src_desc, dst_desc, strides, kernel, padding_l, padding_r)
end

function dnnl_pooling_backward_desc_init(pool_desc, alg_kind, diff_src_desc, diff_dst_desc, strides, kernel, padding_l, padding_r)
    ccall((:dnnl_pooling_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_pooling_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}, Ptr{dnnl_dim_t}), pool_desc, alg_kind, diff_src_desc, diff_dst_desc, strides, kernel, padding_l, padding_r)
end

function dnnl_lrn_forward_desc_init(lrn_desc, prop_kind, alg_kind, data_desc, local_size, alpha, beta, k)
    ccall((:dnnl_lrn_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_lrn_desc_t}, dnnl_prop_kind_t, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, dnnl_dim_t, Cfloat, Cfloat, Cfloat), lrn_desc, prop_kind, alg_kind, data_desc, local_size, alpha, beta, k)
end

function dnnl_lrn_backward_desc_init(lrn_desc, alg_kind, diff_data_desc, data_desc, local_size, alpha, beta, k)
    ccall((:dnnl_lrn_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_lrn_desc_t}, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, dnnl_dim_t, Cfloat, Cfloat, Cfloat), lrn_desc, alg_kind, diff_data_desc, data_desc, local_size, alpha, beta, k)
end

function dnnl_batch_normalization_forward_desc_init(bnrm_desc, prop_kind, data_desc, epsilon, flags)
    ccall((:dnnl_batch_normalization_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_batch_normalization_desc_t}, dnnl_prop_kind_t, Ptr{dnnl_memory_desc_t}, Cfloat, UInt32), bnrm_desc, prop_kind, data_desc, epsilon, flags)
end

function dnnl_batch_normalization_backward_desc_init(bnrm_desc, prop_kind, diff_data_desc, data_desc, epsilon, flags)
    ccall((:dnnl_batch_normalization_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_batch_normalization_desc_t}, dnnl_prop_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Cfloat, UInt32), bnrm_desc, prop_kind, diff_data_desc, data_desc, epsilon, flags)
end

function dnnl_layer_normalization_forward_desc_init(lnrm_desc, prop_kind, data_desc, stat_desc, epsilon, flags)
    ccall((:dnnl_layer_normalization_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_layer_normalization_desc_t}, dnnl_prop_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Cfloat, UInt32), lnrm_desc, prop_kind, data_desc, stat_desc, epsilon, flags)
end

function dnnl_layer_normalization_backward_desc_init(lnrm_desc, prop_kind, diff_data_desc, data_desc, stat_desc, epsilon, flags)
    ccall((:dnnl_layer_normalization_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_layer_normalization_desc_t}, dnnl_prop_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Cfloat, UInt32), lnrm_desc, prop_kind, diff_data_desc, data_desc, stat_desc, epsilon, flags)
end

function dnnl_inner_product_forward_desc_init(ip_desc, prop_kind, src_desc, weights_desc, bias_desc, dst_desc)
    ccall((:dnnl_inner_product_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_inner_product_desc_t}, dnnl_prop_kind_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}), ip_desc, prop_kind, src_desc, weights_desc, bias_desc, dst_desc)
end

function dnnl_inner_product_backward_data_desc_init(ip_desc, diff_src_desc, weights_desc, diff_dst_desc)
    ccall((:dnnl_inner_product_backward_data_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_inner_product_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}), ip_desc, diff_src_desc, weights_desc, diff_dst_desc)
end

function dnnl_inner_product_backward_weights_desc_init(ip_desc, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc)
    ccall((:dnnl_inner_product_backward_weights_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_inner_product_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}), ip_desc, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc)
end

function dnnl_primitive_attr_set_rnn_data_qparams(attr, scale, shift)
    ccall((:dnnl_primitive_attr_set_rnn_data_qparams, dnnl), dnnl_status_t, (dnnl_primitive_attr_t, Cfloat, Cfloat), attr, scale, shift)
end

function dnnl_primitive_attr_set_rnn_weights_qparams(attr, count, mask, scales)
    ccall((:dnnl_primitive_attr_set_rnn_weights_qparams, dnnl), dnnl_status_t, (dnnl_primitive_attr_t, dnnl_dim_t, Cint, Ptr{Cfloat}), attr, count, mask, scales)
end

function dnnl_vanilla_rnn_forward_desc_init(rnn_desc, prop_kind, activation, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, flags, alpha, beta)
    ccall((:dnnl_vanilla_rnn_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_alg_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32, Cfloat, Cfloat), rnn_desc, prop_kind, activation, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, flags, alpha, beta)
end

function dnnl_vanilla_rnn_backward_desc_init(rnn_desc, prop_kind, activation, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, diff_src_layer_desc, diff_src_iter_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, flags, alpha, beta)
    ccall((:dnnl_vanilla_rnn_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_alg_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32, Cfloat, Cfloat), rnn_desc, prop_kind, activation, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, diff_src_layer_desc, diff_src_iter_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, flags, alpha, beta)
end

function dnnl_lstm_forward_desc_init(rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, flags)
    ccall((:dnnl_lstm_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32), rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, flags)
end

function dnnl_lstm_forward_desc_init_v2(rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, weights_peephole_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, flags)
    ccall((:dnnl_lstm_forward_desc_init_v2, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32), rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, weights_peephole_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, flags)
end

function dnnl_lstm_forward_desc_init_v3(rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, weights_peephole_desc, weights_projection_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, flags)
    ccall((:dnnl_lstm_forward_desc_init_v3, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32), rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, weights_peephole_desc, weights_projection_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, flags)
end

function dnnl_lstm_backward_desc_init(rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, diff_src_layer_desc, diff_src_iter_desc, diff_src_iter_c_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, diff_dst_iter_c_desc, flags)
    ccall((:dnnl_lstm_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32), rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, diff_src_layer_desc, diff_src_iter_desc, diff_src_iter_c_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, diff_dst_iter_c_desc, flags)
end

function dnnl_lstm_backward_desc_init_v2(rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, weights_peephole_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, diff_src_layer_desc, diff_src_iter_desc, diff_src_iter_c_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_weights_peephole_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, diff_dst_iter_c_desc, flags)
    ccall((:dnnl_lstm_backward_desc_init_v2, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32), rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, weights_peephole_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, diff_src_layer_desc, diff_src_iter_desc, diff_src_iter_c_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_weights_peephole_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, diff_dst_iter_c_desc, flags)
end

function dnnl_lstm_backward_desc_init_v3(rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, weights_peephole_desc, weights_projection_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, diff_src_layer_desc, diff_src_iter_desc, diff_src_iter_c_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_weights_peephole_desc, diff_weights_projection_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, diff_dst_iter_c_desc, flags)
    ccall((:dnnl_lstm_backward_desc_init_v3, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32), rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, src_iter_c_desc, weights_layer_desc, weights_iter_desc, weights_peephole_desc, weights_projection_desc, bias_desc, dst_layer_desc, dst_iter_desc, dst_iter_c_desc, diff_src_layer_desc, diff_src_iter_desc, diff_src_iter_c_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_weights_peephole_desc, diff_weights_projection_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, diff_dst_iter_c_desc, flags)
end

function dnnl_gru_forward_desc_init(rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, flags)
    ccall((:dnnl_gru_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32), rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, flags)
end

function dnnl_gru_backward_desc_init(rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, diff_src_layer_desc, diff_src_iter_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, flags)
    ccall((:dnnl_gru_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32), rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, diff_src_layer_desc, diff_src_iter_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, flags)
end

function dnnl_lbr_gru_forward_desc_init(rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, flags)
    ccall((:dnnl_lbr_gru_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32), rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, flags)
end

function dnnl_lbr_gru_backward_desc_init(rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, diff_src_layer_desc, diff_src_iter_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, flags)
    ccall((:dnnl_lbr_gru_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_rnn_desc_t}, dnnl_prop_kind_t, dnnl_rnn_direction_t, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, UInt32), rnn_desc, prop_kind, direction, src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc, dst_iter_desc, diff_src_layer_desc, diff_src_iter_desc, diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc, flags)
end

function dnnl_matmul_desc_init(matmul_desc, src_desc, weights_desc, bias_desc, dst_desc)
    ccall((:dnnl_matmul_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_matmul_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}), matmul_desc, src_desc, weights_desc, bias_desc, dst_desc)
end

function dnnl_resampling_forward_desc_init(resampling_desc, prop_kind, alg_kind, factors, src_desc, dst_desc)
    ccall((:dnnl_resampling_forward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_resampling_desc_t}, dnnl_prop_kind_t, dnnl_alg_kind_t, Ptr{Cfloat}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}), resampling_desc, prop_kind, alg_kind, factors, src_desc, dst_desc)
end

function dnnl_resampling_backward_desc_init(resampling_desc, alg_kind, factors, diff_src_desc, diff_dst_desc)
    ccall((:dnnl_resampling_backward_desc_init, dnnl), dnnl_status_t, (Ptr{dnnl_resampling_desc_t}, dnnl_alg_kind_t, Ptr{Cfloat}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}), resampling_desc, alg_kind, factors, diff_src_desc, diff_dst_desc)
end

function dnnl_engine_get_count()
    ccall((:dnnl_engine_get_count, dnnl), Cint, ())
end

function dnnl_engine_create(engine, kind, index)
    ccall((:dnnl_engine_create, dnnl), dnnl_status_t, (Ptr{dnnl_engine_t}, dnnl_engine_kind_t, Cint), engine, kind, index)
end

function dnnl_engine_get_kind(engine, kind)
    ccall((:dnnl_engine_get_kind, dnnl), dnnl_status_t, (dnnl_engine_t, Ptr{dnnl_engine_kind_t}), engine, kind)
end

function dnnl_engine_destroy(engine)
    ccall((:dnnl_engine_destroy, dnnl), dnnl_status_t, (dnnl_engine_t,), engine)
end

function dnnl_stream_attr_create(attr, kind)
    ccall((:dnnl_stream_attr_create, dnnl), dnnl_status_t, (Ptr{dnnl_stream_attr_t}, dnnl_engine_kind_t), attr, kind)
end

function dnnl_stream_attr_destroy(attr)
    ccall((:dnnl_stream_attr_destroy, dnnl), dnnl_status_t, (dnnl_stream_attr_t,), attr)
end

function dnnl_stream_create(stream, engine, flags)
    ccall((:dnnl_stream_create, dnnl), dnnl_status_t, (Ptr{dnnl_stream_t}, dnnl_engine_t, UInt32), stream, engine, flags)
end

function dnnl_stream_create_v2(stream, engine, flags, attr)
    ccall((:dnnl_stream_create_v2, dnnl), dnnl_status_t, (Ptr{dnnl_stream_t}, dnnl_engine_t, UInt32, const_dnnl_stream_attr_t), stream, engine, flags, attr)
end

function dnnl_stream_wait(stream)
    ccall((:dnnl_stream_wait, dnnl), dnnl_status_t, (dnnl_stream_t,), stream)
end

function dnnl_stream_destroy(stream)
    ccall((:dnnl_stream_destroy, dnnl), dnnl_status_t, (dnnl_stream_t,), stream)
end

function dnnl_set_verbose(level)
    ccall((:dnnl_set_verbose, dnnl), dnnl_status_t, (Cint,), level)
end

function dnnl_set_jit_dump(enable)
    ccall((:dnnl_set_jit_dump, dnnl), dnnl_status_t, (Cint,), enable)
end

function dnnl_version()
    ccall((:dnnl_version, dnnl), Ptr{dnnl_version_t}, ())
end

function dnnl_set_jit_profiling_flags(flags)
    ccall((:dnnl_set_jit_profiling_flags, dnnl), dnnl_status_t, (UInt32,), flags)
end

function dnnl_set_jit_profiling_jitdumpdir(dir)
    ccall((:dnnl_set_jit_profiling_jitdumpdir, dnnl), dnnl_status_t, (Cstring,), dir)
end

function dnnl_set_max_cpu_isa(isa)
    ccall((:dnnl_set_max_cpu_isa, dnnl), dnnl_status_t, (dnnl_cpu_isa_t,), isa)
end

function dnnl_sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    ccall((:dnnl_sgemm, dnnl), dnnl_status_t, (UInt8, UInt8, dnnl_dim_t, dnnl_dim_t, dnnl_dim_t, Cfloat, Ptr{Cfloat}, dnnl_dim_t, Ptr{Cfloat}, dnnl_dim_t, Cfloat, Ptr{Cfloat}, dnnl_dim_t), transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
end

function dnnl_gemm_u8s8s32(transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co)
    ccall((:dnnl_gemm_u8s8s32, dnnl), dnnl_status_t, (UInt8, UInt8, UInt8, dnnl_dim_t, dnnl_dim_t, dnnl_dim_t, Cfloat, Ptr{UInt8}, dnnl_dim_t, UInt8, Ptr{Int8}, dnnl_dim_t, Int8, Cfloat, Ptr{Int32}, dnnl_dim_t, Ptr{Int32}), transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co)
end

function dnnl_gemm_s8s8s32(transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co)
    ccall((:dnnl_gemm_s8s8s32, dnnl), dnnl_status_t, (UInt8, UInt8, UInt8, dnnl_dim_t, dnnl_dim_t, dnnl_dim_t, Cfloat, Ptr{Int8}, dnnl_dim_t, Int8, Ptr{Int8}, dnnl_dim_t, Int8, Cfloat, Ptr{Int32}, dnnl_dim_t, Ptr{Int32}), transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co)
end
