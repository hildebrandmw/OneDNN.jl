# Automatically generated using Clang.jl


const DNNL_MAX_NDIMS = 12

# Skipping MacroDefinition: DNNL_RUNTIME_SIZE_VAL ( ( size_t ) DNNL_RUNTIME_DIM_VAL )
# Skipping MacroDefinition: DNNL_RUNTIME_F32_VAL ( DNNL_RUNTIME_F32_VAL_REP . f )

const DNNL_RNN_MAX_N_PARTS = 4

# Skipping MacroDefinition: DNNL_MEMORY_ALLOCATE ( ( void * ) ( size_t ) - 1 )

const DNNL_ARG_SRC_0 = 1
const DNNL_ARG_SRC = DNNL_ARG_SRC_0
const DNNL_ARG_SRC_LAYER = DNNL_ARG_SRC_0
const DNNL_ARG_FROM = DNNL_ARG_SRC_0
const DNNL_ARG_SRC_1 = 2
const DNNL_ARG_SRC_ITER = DNNL_ARG_SRC_1
const DNNL_ARG_SRC_2 = 3
const DNNL_ARG_SRC_ITER_C = DNNL_ARG_SRC_2
const DNNL_ARG_DST_0 = 17
const DNNL_ARG_DST = DNNL_ARG_DST_0
const DNNL_ARG_TO = DNNL_ARG_DST_0
const DNNL_ARG_DST_LAYER = DNNL_ARG_DST_0
const DNNL_ARG_DST_1 = 18
const DNNL_ARG_DST_ITER = DNNL_ARG_DST_1
const DNNL_ARG_DST_2 = 19
const DNNL_ARG_DST_ITER_C = DNNL_ARG_DST_2
const DNNL_ARG_WEIGHTS_0 = 33
const DNNL_ARG_WEIGHTS = DNNL_ARG_WEIGHTS_0
const DNNL_ARG_SCALE_SHIFT = DNNL_ARG_WEIGHTS_0
const DNNL_ARG_WEIGHTS_LAYER = DNNL_ARG_WEIGHTS_0
const DNNL_ARG_WEIGHTS_1 = 34
const DNNL_ARG_WEIGHTS_ITER = DNNL_ARG_WEIGHTS_1
const DNNL_ARG_WEIGHTS_2 = 35
const DNNL_ARG_WEIGHTS_PEEPHOLE = DNNL_ARG_WEIGHTS_2
const DNNL_ARG_WEIGHTS_3 = 36
const DNNL_ARG_WEIGHTS_PROJECTION = DNNL_ARG_WEIGHTS_3
const DNNL_ARG_BIAS = 41
const DNNL_ARG_MEAN = 49
const DNNL_ARG_VARIANCE = 50
const DNNL_ARG_WORKSPACE = 64
const DNNL_ARG_SCRATCHPAD = 80
const DNNL_ARG_DIFF_SRC_0 = 129
const DNNL_ARG_DIFF_SRC = DNNL_ARG_DIFF_SRC_0
const DNNL_ARG_DIFF_SRC_LAYER = DNNL_ARG_DIFF_SRC_0
const DNNL_ARG_DIFF_SRC_1 = 130
const DNNL_ARG_DIFF_SRC_ITER = DNNL_ARG_DIFF_SRC_1
const DNNL_ARG_DIFF_SRC_2 = 131
const DNNL_ARG_DIFF_SRC_ITER_C = DNNL_ARG_DIFF_SRC_2
const DNNL_ARG_DIFF_DST_0 = 145
const DNNL_ARG_DIFF_DST = DNNL_ARG_DIFF_DST_0
const DNNL_ARG_DIFF_DST_LAYER = DNNL_ARG_DIFF_DST_0
const DNNL_ARG_DIFF_DST_1 = 146
const DNNL_ARG_DIFF_DST_ITER = DNNL_ARG_DIFF_DST_1
const DNNL_ARG_DIFF_DST_2 = 147
const DNNL_ARG_DIFF_DST_ITER_C = DNNL_ARG_DIFF_DST_2
const DNNL_ARG_DIFF_WEIGHTS_0 = 161
const DNNL_ARG_DIFF_WEIGHTS = DNNL_ARG_DIFF_WEIGHTS_0
const DNNL_ARG_DIFF_SCALE_SHIFT = DNNL_ARG_DIFF_WEIGHTS_0
const DNNL_ARG_DIFF_WEIGHTS_LAYER = DNNL_ARG_DIFF_WEIGHTS_0
const DNNL_ARG_DIFF_WEIGHTS_1 = 162
const DNNL_ARG_DIFF_WEIGHTS_ITER = DNNL_ARG_DIFF_WEIGHTS_1
const DNNL_ARG_DIFF_WEIGHTS_2 = 163
const DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE = DNNL_ARG_DIFF_WEIGHTS_2
const DNNL_ARG_DIFF_WEIGHTS_3 = 164
const DNNL_ARG_DIFF_WEIGHTS_PROJECTION = DNNL_ARG_DIFF_WEIGHTS_3
const DNNL_ARG_DIFF_BIAS = 169
const DNNL_ARG_ATTR_OUTPUT_SCALES = 513
const DNNL_ARG_MULTIPLE_SRC = 1024
const DNNL_ARG_MULTIPLE_DST = 2048
const DNNL_ARG_ATTR_ZERO_POINTS = 4096
const DNNL_ARG_ATTR_POST_OP_DW = 8192
const DNNL_RUNTIME_NONE = UInt32(0)
const DNNL_RUNTIME_SEQ = UInt32(1)
const DNNL_RUNTIME_OMP = UInt32(2)
const DNNL_RUNTIME_TBB = UInt32(4)
const DNNL_RUNTIME_THREADPOOL = UInt32(8)
const DNNL_RUNTIME_OCL = UInt32(256)
const DNNL_JIT_PROFILE_NONE = UInt32(0)
const DNNL_JIT_PROFILE_VTUNE = UInt32(1)
const DNNL_JIT_PROFILE_LINUX_PERFMAP = UInt32(2)
const DNNL_JIT_PROFILE_LINUX_JITDUMP = UInt32(4)
const DNNL_JIT_PROFILE_LINUX_JITDUMP_USE_TSC = UInt32(8)
const DNNL_JIT_PROFILE_LINUX_PERF = DNNL_JIT_PROFILE_LINUX_JITDUMP | DNNL_JIT_PROFILE_LINUX_PERFMAP

# Skipping MacroDefinition: DNNL_HELPER_DLL_IMPORT __attribute__ ( ( visibility ( "default" ) ) )
# Skipping MacroDefinition: DNNL_HELPER_DLL_EXPORT __attribute__ ( ( visibility ( "default" ) ) )
# Skipping MacroDefinition: DNNL_DEPRECATED __attribute__ ( ( deprecated ) )

const DNNL_CPU_THREADING_RUNTIME = DNNL_RUNTIME_OMP
const DNNL_CPU_RUNTIME = DNNL_RUNTIME_OMP
const DNNL_GPU_RUNTIME = DNNL_RUNTIME_NONE
const DNNL_VERSION_MAJOR = 1
const DNNL_VERSION_MINOR = 4
const DNNL_VERSION_PATCH = 0
const DNNL_VERSION_HASH = "f7c41dc7b5471ad8bf7905e459bbed27f9094caa"

@cenum dnnl_status_t::UInt32 begin
    dnnl_success = 0
    dnnl_out_of_memory = 1
    dnnl_invalid_arguments = 2
    dnnl_unimplemented = 3
    dnnl_iterator_ends = 4
    dnnl_runtime_error = 5
    dnnl_not_required = 6
end

@cenum dnnl_data_type_t::UInt32 begin
    dnnl_data_type_undef = 0
    dnnl_f16 = 1
    dnnl_bf16 = 2
    dnnl_f32 = 3
    dnnl_s32 = 4
    dnnl_s8 = 5
    dnnl_u8 = 6
end

@cenum dnnl_format_kind_t::UInt32 begin
    dnnl_format_kind_undef = 0
    dnnl_format_kind_any = 1
    dnnl_blocked = 2
    dnnl_format_kind_wino = 3
    dnnl_format_kind_rnn_packed = 4
end

@cenum dnnl_format_tag_t::UInt32 begin
    dnnl_format_tag_undef = 0
    dnnl_format_tag_any = 1
    dnnl_a = 2
    dnnl_ab = 3
    dnnl_abc = 4
    dnnl_abcd = 5
    dnnl_abcde = 6
    dnnl_abcdef = 7
    dnnl_abdc = 8
    dnnl_abdec = 9
    dnnl_acb = 10
    dnnl_acbde = 11
    dnnl_acbdef = 12
    dnnl_acdb = 13
    dnnl_acdeb = 14
    dnnl_ba = 15
    dnnl_bac = 16
    dnnl_bacd = 17
    dnnl_bca = 18
    dnnl_bcda = 19
    dnnl_bcdea = 20
    dnnl_cba = 21
    dnnl_cdba = 22
    dnnl_dcab = 23
    dnnl_cdeba = 24
    dnnl_decab = 25
    dnnl_defcab = 26
    dnnl_Abc16a = 27
    dnnl_ABc16a16b = 28
    dnnl_ABc4a4b = 29
    dnnl_aBc16b = 30
    dnnl_ABc16b16a = 31
    dnnl_Abc4a = 32
    dnnl_aBc4b = 33
    dnnl_ABc4b16a4b = 34
    dnnl_ABc2b8a4b = 35
    dnnl_ABc4b4a = 36
    dnnl_ABc8a16b2a = 37
    dnnl_ABc8a8b = 38
    dnnl_aBc8b = 39
    dnnl_ABc8b16a2b = 40
    dnnl_BAc8a16b2a = 41
    dnnl_ABc8b8a = 42
    dnnl_Abcd16a = 43
    dnnl_Abcd8a = 44
    dnnl_ABcd16a16b = 45
    dnnl_ABcd32a32b = 46
    dnnl_aBcd16b = 47
    dnnl_ABcd16b16a = 48
    dnnl_aBCd16b16c = 49
    dnnl_aBCd16c16b = 50
    dnnl_Abcd4a = 51
    dnnl_aBcd4b = 52
    dnnl_ABcd4b16a4b = 53
    dnnl_ABcd4b4a = 54
    dnnl_ABcd4a4b = 55
    dnnl_aBCd2c4b2c = 56
    dnnl_aBCd4b8c2b = 57
    dnnl_aBCd4c16b4c = 58
    dnnl_aBCd2c8b4c = 59
    dnnl_aBCd4c4b = 60
    dnnl_aBCd4b4c = 61
    dnnl_ABcd8a16b2a = 62
    dnnl_ABcd2b8a4b = 63
    dnnl_ABcd8a8b = 64
    dnnl_aBcd8b = 65
    dnnl_aBCd4c8b2c = 66
    dnnl_ABcd8b16a2b = 67
    dnnl_aBCd8b16c2b = 68
    dnnl_BAcd8a16b2a = 69
    dnnl_ABcd8b8a = 70
    dnnl_aBCd8b8c = 71
    dnnl_aBCd8c16b2c = 72
    dnnl_ABcde8a16b2a = 73
    dnnl_aCBd8b16c2b = 74
    dnnl_aBCd8c8b = 75
    dnnl_Abcde16a = 76
    dnnl_ABcde16a16b = 77
    dnnl_BAcde8a16b2a = 78
    dnnl_aBCd2b4c2b = 79
    dnnl_ABcde4b16a4b = 80
    dnnl_ABcde2b8a4b = 81
    dnnl_aBcde16b = 82
    dnnl_ABcde16b16a = 83
    dnnl_aBCde16b16c = 84
    dnnl_aBCde16c16b = 85
    dnnl_aBCde2c8b4c = 86
    dnnl_Abcde4a = 87
    dnnl_aBcde4b = 88
    dnnl_ABcde4b4a = 89
    dnnl_ABcde4a4b = 90
    dnnl_aBCde4b4c = 91
    dnnl_aBCde2c4b2c = 92
    dnnl_aBCde4b8c2b = 93
    dnnl_aBCde4c16b4c = 94
    dnnl_aBCde4c4b = 95
    dnnl_Abcde8a = 96
    dnnl_ABcde8a8b = 97
    dnnl_BAcde16b16a = 98
    dnnl_aBcde8b = 99
    dnnl_ABcde8b16a2b = 100
    dnnl_aBCde8b16c2b = 101
    dnnl_aBCde4c8b2c = 102
    dnnl_aCBde8b16c2b = 103
    dnnl_ABcde8b8a = 104
    dnnl_aBCde8b8c = 105
    dnnl_ABcd4a8b8a4b = 106
    dnnl_ABcd2a8b8a2b = 107
    dnnl_aBCde4b8c8b4c = 108
    dnnl_aBCde2b8c8b2c = 109
    dnnl_aBCde8c16b2c = 110
    dnnl_aBCde8c8b = 111
    dnnl_aBCde2b4c2b = 112
    dnnl_aBcdef16b = 113
    dnnl_aBCdef16b16c = 114
    dnnl_aBCdef16c16b = 115
    dnnl_aBCdef4c16b4c = 116
    dnnl_aBCdef3c8b4c = 117
    dnnl_aBCdef4c8b2c = 118
    dnnl_aBCdef2b4c2b = 119
    dnnl_aBcdef4b = 120
    dnnl_aBCdef4c4b = 121
    dnnl_aBCdef4b4c = 122
    dnnl_aBCdef2c4b2c = 123
    dnnl_aBCdef4b8c2b = 124
    dnnl_aBCdef8b8c = 125
    dnnl_aBCdef8c16b2c = 126
    dnnl_aBCdef8b16c2b = 127
    dnnl_aCBdef8b16c2b = 128
    dnnl_aBCdef8c8b = 129
    dnnl_aBdc16b = 130
    dnnl_aBdC16b2c = 131
    dnnl_aBdc4b = 132
    dnnl_aBdc8b = 133
    dnnl_aBdec16b = 134
    dnnl_aBdeC16b2c = 135
    dnnl_aBdec32b = 136
    dnnl_aBdec4b = 137
    dnnl_aBdec8b = 138
    dnnl_aBdefc16b = 139
    dnnl_aBdefC16b2c = 140
    dnnl_aCBdef16c16b = 141
    dnnl_aBdefc4b = 142
    dnnl_aBdefc8b = 143
    dnnl_Abcdef16a = 144
    dnnl_Acb16a = 145
    dnnl_AcB16a2b = 146
    dnnl_Acb4a = 147
    dnnl_Acb8a = 148
    dnnl_aCBd16b16c = 149
    dnnl_aCBd16c16b = 150
    dnnl_aCBde16b16c = 151
    dnnl_aCBde16c16b = 152
    dnnl_Acdb16a = 153
    dnnl_AcdB16a2b = 154
    dnnl_Acdb32a = 155
    dnnl_Acdb4a = 156
    dnnl_Acdb8a = 157
    dnnl_Acdeb16a = 158
    dnnl_AcdeB16a2b = 159
    dnnl_Acdeb4a = 160
    dnnl_Acdeb8a = 161
    dnnl_BAc16a16b = 162
    dnnl_BAc16b16a = 163
    dnnl_BAcd16a16b = 164
    dnnl_BAcd16b16a = 165
    dnnl_BAcde16a16b = 166
    dnnl_aCBdef16b16c = 167
    dnnl_format_tag_last = 168
    dnnl_x = 2
    dnnl_nc = 3
    dnnl_cn = 15
    dnnl_tn = 3
    dnnl_nt = 15
    dnnl_ncw = 4
    dnnl_nwc = 10
    dnnl_nchw = 5
    dnnl_nhwc = 13
    dnnl_chwn = 19
    dnnl_ncdhw = 6
    dnnl_ndhwc = 14
    dnnl_oi = 3
    dnnl_io = 15
    dnnl_oiw = 4
    dnnl_owi = 10
    dnnl_wio = 21
    dnnl_iwo = 18
    dnnl_oihw = 5
    dnnl_hwio = 22
    dnnl_ohwi = 13
    dnnl_ihwo = 19
    dnnl_iohw = 17
    dnnl_oidhw = 6
    dnnl_dhwio = 24
    dnnl_odhwi = 14
    dnnl_idhwo = 20
    dnnl_goiw = 5
    dnnl_wigo = 23
    dnnl_goihw = 6
    dnnl_hwigo = 25
    dnnl_giohw = 11
    dnnl_goidhw = 7
    dnnl_giodhw = 12
    dnnl_dhwigo = 26
    dnnl_tnc = 4
    dnnl_ntc = 16
    dnnl_ldnc = 5
    dnnl_ldigo = 6
    dnnl_ldgoi = 9
    dnnl_ldio = 5
    dnnl_ldoi = 8
    dnnl_ldgo = 5
    dnnl_nCdhw16c = 82
    dnnl_nCdhw4c = 88
    dnnl_nCdhw8c = 99
    dnnl_nChw16c = 47
    dnnl_nChw4c = 52
    dnnl_nChw8c = 65
    dnnl_nCw16c = 30
    dnnl_nCw4c = 33
    dnnl_nCw8c = 39
    dnnl_NCw16n16c = 28
    dnnl_NCdhw16n16c = 77
    dnnl_NChw16n16c = 45
    dnnl_NChw32n32c = 46
    dnnl_IOw16o16i = 162
    dnnl_IOw16i16o = 163
    dnnl_OIw16i16o = 31
    dnnl_OIw16o16i = 28
    dnnl_Oiw16o = 27
    dnnl_OIw4i16o4i = 34
    dnnl_OIw2i8o4i = 35
    dnnl_OIw4i4o = 36
    dnnl_OIw4o4i = 29
    dnnl_Oiw4o = 32
    dnnl_OIw8i16o2i = 40
    dnnl_OIw8i8o = 42
    dnnl_OIw8o16i2o = 37
    dnnl_IOw8o16i2o = 41
    dnnl_OIw8o8i = 38
    dnnl_Owi16o = 145
    dnnl_OwI16o2i = 146
    dnnl_Owi4o = 147
    dnnl_Owi8o = 148
    dnnl_IOhw16i16o = 165
    dnnl_IOhw16o16i = 164
    dnnl_Ohwi16o = 153
    dnnl_OhwI16o2i = 154
    dnnl_Ohwi32o = 155
    dnnl_Ohwi4o = 156
    dnnl_Ohwi8o = 157
    dnnl_OIhw16i16o = 48
    dnnl_OIhw16o16i = 45
    dnnl_Oihw16o = 43
    dnnl_OIhw4i16o4i = 53
    dnnl_OIhw4i4o = 54
    dnnl_OIhw4o4i = 55
    dnnl_Oihw4o = 51
    dnnl_OIhw8i16o2i = 67
    dnnl_OIhw8i8o = 70
    dnnl_OIhw8o16i2o = 62
    dnnl_OIhw2i8o4i = 63
    dnnl_IOhw8o16i2o = 69
    dnnl_OIhw8o8i = 64
    dnnl_Odhwi16o = 158
    dnnl_OdhwI16o2i = 159
    dnnl_Odhwi4o = 160
    dnnl_Odhwi8o = 161
    dnnl_OIdhw16i16o = 83
    dnnl_OIdhw16o16i = 77
    dnnl_Oidhw16o = 76
    dnnl_OIdhw4i4o = 89
    dnnl_OIdhw4o4i = 90
    dnnl_Oidhw4o = 87
    dnnl_OIdhw8i16o2i = 100
    dnnl_OIdhw8i8o = 104
    dnnl_OIdhw8o16i2o = 73
    dnnl_IOdhw8o16i2o = 78
    dnnl_OIdhw4i16o4i = 80
    dnnl_OIdhw2i8o4i = 81
    dnnl_OIdhw8o8i = 97
    dnnl_IOdhw16i16o = 98
    dnnl_IOdhw16o16i = 166
    dnnl_Goiw16g = 43
    dnnl_Goiw8g = 44
    dnnl_gIOw16o16i = 149
    dnnl_gIOw16i16o = 150
    dnnl_gOIw16i16o = 50
    dnnl_gOIw16o16i = 49
    dnnl_gOiw16o = 47
    dnnl_gOIw4i16o4i = 58
    dnnl_gOIw2i8o4i = 59
    dnnl_gOIw4i4o = 60
    dnnl_gOIw4o4i = 61
    dnnl_gOiw4o = 52
    dnnl_gOIw8i16o2i = 72
    dnnl_gOIw8i8o = 75
    dnnl_gOIw8o16i2o = 68
    dnnl_gIOw8o16i2o = 74
    dnnl_gOIw8o8i = 71
    dnnl_gOwi16o = 130
    dnnl_gOwI16o2i = 131
    dnnl_gOwi4o = 132
    dnnl_gOwi8o = 133
    dnnl_gOIw2i4o2i = 56
    dnnl_gOIw2o4i2o = 79
    dnnl_gOIw4i8o2i = 66
    dnnl_gOIw4o8i2o = 57
    dnnl_gIOhw16i16o = 152
    dnnl_gIOhw16o16i = 151
    dnnl_gOhwi16o = 134
    dnnl_gOhwI16o2i = 135
    dnnl_gOhwi32o = 136
    dnnl_gOhwi4o = 137
    dnnl_gOhwi8o = 138
    dnnl_Goihw16g = 76
    dnnl_gOIhw16i16o = 85
    dnnl_gOIhw16o16i = 84
    dnnl_gOihw16o = 82
    dnnl_gOIhw2i8o4i = 86
    dnnl_gOIhw4i16o4i = 94
    dnnl_gOIhw4i4o = 95
    dnnl_gOIhw4o4i = 91
    dnnl_gOihw4o = 88
    dnnl_Goihw8g = 96
    dnnl_gOIhw8i16o2i = 110
    dnnl_gOIhw8i8o = 111
    dnnl_gOIhw8o16i2o = 101
    dnnl_gIOhw8o16i2o = 103
    dnnl_gOIhw8o8i = 105
    dnnl_OIhw4o8i8o4i = 106
    dnnl_OIhw2o8i8o2i = 107
    dnnl_gOIhw4o8i8o4i = 108
    dnnl_gOIhw2o8i8o2i = 109
    dnnl_gOIhw2i4o2i = 92
    dnnl_gOIhw2o4i2o = 112
    dnnl_gOIhw4i8o2i = 102
    dnnl_gOIhw4o8i2o = 93
    dnnl_gIOdhw16i16o = 141
    dnnl_gIOdhw16o16i = 167
    dnnl_gOdhwi16o = 139
    dnnl_gOdhwI16o2i = 140
    dnnl_gOdhwi4o = 142
    dnnl_gOdhwi8o = 143
    dnnl_gOIdhw16i16o = 115
    dnnl_gOIdhw4i16o4i = 116
    dnnl_gOIdhw2i8o4i = 117
    dnnl_gOIdhw16o16i = 114
    dnnl_gOidhw16o = 113
    dnnl_gOIdhw4i4o = 121
    dnnl_gOIdhw4o4i = 122
    dnnl_gOidhw4o = 120
    dnnl_gOIdhw8i16o2i = 126
    dnnl_gOIdhw8i8o = 129
    dnnl_gOIdhw8o16i2o = 127
    dnnl_gIOdhw8o16i2o = 128
    dnnl_gOIdhw8o8i = 125
    dnnl_Goidhw16g = 144
    dnnl_gOIdhw2i4o2i = 123
    dnnl_gOIdhw4i8o2i = 118
    dnnl_gOIdhw2o4i2o = 119
    dnnl_gOIdhw4o8i2o = 124
end

@cenum dnnl_prop_kind_t::UInt32 begin
    dnnl_prop_kind_undef = 0
    dnnl_forward_training = 64
    dnnl_forward_inference = 96
    dnnl_forward_scoring = 96
    dnnl_forward = 64
    dnnl_backward = 128
    dnnl_backward_data = 160
    dnnl_backward_weights = 192
    dnnl_backward_bias = 193
end

@cenum dnnl_primitive_kind_t::UInt32 begin
    dnnl_undefined_primitive = 0
    dnnl_reorder = 1
    dnnl_shuffle = 2
    dnnl_concat = 3
    dnnl_sum = 4
    dnnl_convolution = 5
    dnnl_deconvolution = 6
    dnnl_eltwise = 7
    dnnl_softmax = 8
    dnnl_pooling = 9
    dnnl_lrn = 10
    dnnl_batch_normalization = 11
    dnnl_layer_normalization = 12
    dnnl_inner_product = 13
    dnnl_rnn = 14
    dnnl_gemm = 15
    dnnl_binary = 16
    dnnl_logsoftmax = 17
    dnnl_matmul = 18
    dnnl_resampling = 19
end

@cenum dnnl_alg_kind_t::UInt32 begin
    dnnl_alg_kind_undef = 0
    dnnl_convolution_direct = 1
    dnnl_convolution_winograd = 2
    dnnl_convolution_auto = 3
    dnnl_deconvolution_direct = 10
    dnnl_deconvolution_winograd = 11
    dnnl_eltwise_relu = 31
    dnnl_eltwise_tanh = 47
    dnnl_eltwise_elu = 63
    dnnl_eltwise_square = 79
    dnnl_eltwise_abs = 95
    dnnl_eltwise_sqrt = 111
    dnnl_eltwise_linear = 127
    dnnl_eltwise_bounded_relu = 143
    dnnl_eltwise_soft_relu = 159
    dnnl_eltwise_logistic = 175
    dnnl_eltwise_exp = 191
    dnnl_eltwise_gelu_tanh = 207
    dnnl_eltwise_gelu = 207
    dnnl_eltwise_swish = 223
    dnnl_eltwise_log = 239
    dnnl_eltwise_clip = 255
    dnnl_eltwise_pow = 32
    dnnl_eltwise_gelu_erf = 48
    dnnl_eltwise_relu_use_dst_for_bwd = 256
    dnnl_eltwise_tanh_use_dst_for_bwd = 257
    dnnl_eltwise_elu_use_dst_for_bwd = 258
    dnnl_eltwise_sqrt_use_dst_for_bwd = 259
    dnnl_eltwise_logistic_use_dst_for_bwd = 260
    dnnl_eltwise_exp_use_dst_for_bwd = 261
    dnnl_pooling_max = 511
    dnnl_pooling_avg_include_padding = 767
    dnnl_pooling_avg_exclude_padding = 1023
    dnnl_pooling_avg = 1023
    dnnl_lrn_across_channels = 2815
    dnnl_lrn_within_channel = 3071
    dnnl_vanilla_rnn = 8191
    dnnl_vanilla_lstm = 12287
    dnnl_vanilla_gru = 16383
    dnnl_lbr_gru = 20479
    dnnl_binary_add = 131056
    dnnl_binary_mul = 131057
    dnnl_binary_max = 131058
    dnnl_binary_min = 131059
    dnnl_resampling_nearest = 196592
    dnnl_resampling_linear = 196593
end

@cenum dnnl_normalization_flags_t::UInt32 begin
    dnnl_normalization_flags_none = 0
    dnnl_use_global_stats = 1
    dnnl_use_scaleshift = 2
    dnnl_fuse_norm_relu = 4
end


const dnnl_dim_t = Int64
const dnnl_dims_t = NTuple{12, dnnl_dim_t}

struct dnnl_blocking_desc_t
    strides::dnnl_dims_t
    inner_nblks::Cint
    inner_blks::dnnl_dims_t
    inner_idxs::dnnl_dims_t
end

@cenum dnnl_wino_memory_format_t::UInt32 begin
    dnnl_wino_undef = 0
    dnnl_wino_wei_aaOIoi = 1
    dnnl_wino_wei_aaOio = 2
    dnnl_wino_wei_aaOBiOo = 3
    dnnl_wino_wei_OBaaIBOIio = 4
end


struct dnnl_wino_desc_t
    wino_format::dnnl_wino_memory_format_t
    r::Cint
    alpha::Cint
    ic::Cint
    oc::Cint
    ic_block::Cint
    oc_block::Cint
    ic2_block::Cint
    oc2_block::Cint
    adj_scale::Cfloat
    size::Cint
end

@cenum dnnl_rnn_packed_memory_format_t::UInt32 begin
    dnnl_packed_format_undef = 0
    dnnl_ldigo_p = 1
    dnnl_ldgoi_p = 2
end


struct dnnl_rnn_packed_desc_t
    format::dnnl_rnn_packed_memory_format_t
    n_parts::Cint
    n::Cint
    ldb::Cint
    parts::NTuple{4, Cint}
    part_pack_size::NTuple{4, Cint}
    pack_part::NTuple{4, UInt32}
    offset_compensation::Cint
    size::Cint
    reserved::NTuple{200, UInt8}
end

@cenum dnnl_memory_extra_flags_t::UInt32 begin
    dnnl_memory_extra_flag_none = 0
    dnnl_memory_extra_flag_compensation_conv_s8s8 = 1
    dnnl_memory_extra_flag_scale_adjust = 2
    dnnl_memory_extra_flag_gpu_rnn_u8s8_compensation = 4
end


struct dnnl_memory_extra_desc_t
    flags::UInt64
    compensation_mask::Cint
    scale_adjust::Cfloat
    reserved::NTuple{64, UInt8}
end

struct ANONYMOUS1_format_desc
    blocking::dnnl_blocking_desc_t
end

struct dnnl_memory_desc_t
    ndims::Cint
    dims::dnnl_dims_t
    data_type::dnnl_data_type_t
    padded_dims::dnnl_dims_t
    padded_offsets::dnnl_dims_t
    offset0::dnnl_dim_t
    format_kind::dnnl_format_kind_t
    format_desc::ANONYMOUS1_format_desc
    extra::dnnl_memory_extra_desc_t
end

const dnnl_memory = Cvoid
const dnnl_memory_t = Ptr{dnnl_memory}
const const_dnnl_memory_t = Ptr{dnnl_memory}
const dnnl_op_desc_t = Ptr{Cvoid}
const const_dnnl_op_desc_t = Ptr{Cvoid}

struct dnnl_convolution_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    alg_kind::dnnl_alg_kind_t
    src_desc::dnnl_memory_desc_t
    diff_src_desc::dnnl_memory_desc_t
    weights_desc::dnnl_memory_desc_t
    diff_weights_desc::dnnl_memory_desc_t
    bias_desc::dnnl_memory_desc_t
    diff_bias_desc::dnnl_memory_desc_t
    dst_desc::dnnl_memory_desc_t
    diff_dst_desc::dnnl_memory_desc_t
    strides::dnnl_dims_t
    dilates::dnnl_dims_t
    padding::NTuple{2, dnnl_dims_t}
    accum_data_type::dnnl_data_type_t
end

const dnnl_deconvolution_desc_t = dnnl_convolution_desc_t

struct dnnl_shuffle_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    data_desc::dnnl_memory_desc_t
    axis::Cint
    group_size::dnnl_dim_t
end

struct dnnl_eltwise_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    alg_kind::dnnl_alg_kind_t
    data_desc::dnnl_memory_desc_t
    diff_data_desc::dnnl_memory_desc_t
    alpha::Cfloat
    beta::Cfloat
end

struct dnnl_softmax_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    data_desc::dnnl_memory_desc_t
    diff_desc::dnnl_memory_desc_t
    softmax_axis::Cint
end

const dnnl_logsoftmax_desc_t = dnnl_softmax_desc_t

struct dnnl_pooling_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    alg_kind::dnnl_alg_kind_t
    src_desc::dnnl_memory_desc_t
    diff_src_desc::dnnl_memory_desc_t
    dst_desc::dnnl_memory_desc_t
    diff_dst_desc::dnnl_memory_desc_t
    strides::dnnl_dims_t
    kernel::dnnl_dims_t
    padding::NTuple{2, dnnl_dims_t}
    accum_data_type::dnnl_data_type_t
end

struct dnnl_lrn_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    alg_kind::dnnl_alg_kind_t
    data_desc::dnnl_memory_desc_t
    diff_data_desc::dnnl_memory_desc_t
    local_size::dnnl_dim_t
    lrn_alpha::Cfloat
    lrn_beta::Cfloat
    lrn_k::Cfloat
end

struct dnnl_batch_normalization_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    data_desc::dnnl_memory_desc_t
    diff_data_desc::dnnl_memory_desc_t
    data_scaleshift_desc::dnnl_memory_desc_t
    diff_data_scaleshift_desc::dnnl_memory_desc_t
    stat_desc::dnnl_memory_desc_t
    batch_norm_epsilon::Cfloat
    flags::UInt32
end

struct dnnl_layer_normalization_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    data_desc::dnnl_memory_desc_t
    diff_data_desc::dnnl_memory_desc_t
    data_scaleshift_desc::dnnl_memory_desc_t
    diff_data_scaleshift_desc::dnnl_memory_desc_t
    stat_desc::dnnl_memory_desc_t
    layer_norm_epsilon::Cfloat
    flags::UInt32
end

struct dnnl_inner_product_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    src_desc::dnnl_memory_desc_t
    diff_src_desc::dnnl_memory_desc_t
    weights_desc::dnnl_memory_desc_t
    diff_weights_desc::dnnl_memory_desc_t
    bias_desc::dnnl_memory_desc_t
    diff_bias_desc::dnnl_memory_desc_t
    dst_desc::dnnl_memory_desc_t
    diff_dst_desc::dnnl_memory_desc_t
    accum_data_type::dnnl_data_type_t
end

@cenum dnnl_rnn_flags_t::UInt32 begin
    dnnl_rnn_flags_undef = 0
end

@cenum dnnl_rnn_direction_t::UInt32 begin
    dnnl_unidirectional_left2right = 0
    dnnl_unidirectional_right2left = 1
    dnnl_bidirectional_concat = 2
    dnnl_bidirectional_sum = 3
    dnnl_unidirectional = 0
end


struct dnnl_rnn_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    cell_kind::dnnl_alg_kind_t
    direction::dnnl_rnn_direction_t
    src_layer_desc::dnnl_memory_desc_t
    src_iter_desc::dnnl_memory_desc_t
    src_iter_c_desc::dnnl_memory_desc_t
    weights_layer_desc::dnnl_memory_desc_t
    weights_iter_desc::dnnl_memory_desc_t
    bias_desc::dnnl_memory_desc_t
    dst_layer_desc::dnnl_memory_desc_t
    dst_iter_desc::dnnl_memory_desc_t
    dst_iter_c_desc::dnnl_memory_desc_t
    weights_peephole_desc::dnnl_memory_desc_t
    weights_projection_desc::dnnl_memory_desc_t
    diff_src_layer_desc::dnnl_memory_desc_t
    diff_src_iter_desc::dnnl_memory_desc_t
    diff_src_iter_c_desc::dnnl_memory_desc_t
    diff_weights_layer_desc::dnnl_memory_desc_t
    diff_weights_iter_desc::dnnl_memory_desc_t
    diff_bias_desc::dnnl_memory_desc_t
    diff_dst_layer_desc::dnnl_memory_desc_t
    diff_dst_iter_desc::dnnl_memory_desc_t
    diff_dst_iter_c_desc::dnnl_memory_desc_t
    diff_weights_peephole_desc::dnnl_memory_desc_t
    diff_weights_projection_desc::dnnl_memory_desc_t
    flags::UInt32
    activation_kind::dnnl_alg_kind_t
    alpha::Cfloat
    beta::Cfloat
end

struct dnnl_binary_desc_t
    primitive_kind::dnnl_primitive_kind_t
    alg_kind::dnnl_alg_kind_t
    src_desc::NTuple{2, dnnl_memory_desc_t}
    dst_desc::dnnl_memory_desc_t
end

struct dnnl_matmul_desc_t
    primitive_kind::dnnl_primitive_kind_t
    src_desc::dnnl_memory_desc_t
    weights_desc::dnnl_memory_desc_t
    bias_desc::dnnl_memory_desc_t
    dst_desc::dnnl_memory_desc_t
    accum_data_type::dnnl_data_type_t
end

struct dnnl_resampling_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    alg_kind::dnnl_alg_kind_t
    src_desc::dnnl_memory_desc_t
    diff_src_desc::dnnl_memory_desc_t
    dst_desc::dnnl_memory_desc_t
    diff_dst_desc::dnnl_memory_desc_t
    factors::NTuple{12, Cfloat}
end

@cenum dnnl_engine_kind_t::UInt32 begin
    dnnl_any_engine = 0
    dnnl_cpu = 1
    dnnl_gpu = 2
end


const dnnl_engine = Cvoid
const dnnl_engine_t = Ptr{dnnl_engine}
const dnnl_primitive_desc_iterator = Cvoid
const dnnl_primitive_desc_iterator_t = Ptr{dnnl_primitive_desc_iterator}
const const_dnnl_primitive_desc_iterator_t = Ptr{dnnl_primitive_desc_iterator}
const dnnl_primitive_desc = Cvoid
const dnnl_primitive_desc_t = Ptr{dnnl_primitive_desc}
const const_dnnl_primitive_desc_t = Ptr{dnnl_primitive_desc}

@cenum dnnl_scratchpad_mode_t::UInt32 begin
    dnnl_scratchpad_mode_library = 0
    dnnl_scratchpad_mode_user = 1
end


const dnnl_primitive_attr = Cvoid
const dnnl_primitive_attr_t = Ptr{dnnl_primitive_attr}
const const_dnnl_primitive_attr_t = Ptr{dnnl_primitive_attr}
const dnnl_post_ops = Cvoid
const dnnl_post_ops_t = Ptr{dnnl_post_ops}
const const_dnnl_post_ops_t = Ptr{dnnl_post_ops}
const dnnl_primitive = Cvoid
const dnnl_primitive_t = Ptr{dnnl_primitive}
const const_dnnl_primitive_t = Ptr{dnnl_primitive}

struct dnnl_exec_arg_t
    arg::Cint
    memory::dnnl_memory_t
end

@cenum dnnl_query_t::UInt32 begin
    dnnl_query_undef = 0
    dnnl_query_engine = 1
    dnnl_query_primitive_kind = 2
    dnnl_query_num_of_inputs_s32 = 3
    dnnl_query_num_of_outputs_s32 = 4
    dnnl_query_time_estimate_f64 = 5
    dnnl_query_memory_consumption_s64 = 6
    dnnl_query_scratchpad_engine = 7
    dnnl_query_impl_info_str = 8
    dnnl_query_reorder_src_engine = 9
    dnnl_query_reorder_dst_engine = 10
    dnnl_query_prop_kind = 11
    dnnl_query_some_d = 64
    dnnl_query_op_d = 65
    dnnl_query_convolution_d = 66
    dnnl_query_deconvolution_d = 67
    dnnl_query_shuffle_d = 68
    dnnl_query_eltwise_d = 69
    dnnl_query_softmax_d = 70
    dnnl_query_pooling_d = 71
    dnnl_query_lrn_d = 72
    dnnl_query_batch_normalization_d = 73
    dnnl_query_layer_normalization_d = 74
    dnnl_query_inner_product_d = 75
    dnnl_query_rnn_d = 76
    dnnl_query_gemm_d = 77
    dnnl_query_binary_d = 78
    dnnl_query_logsoftmax_d = 79
    dnnl_query_matmul_d = 80
    dnnl_query_resampling_d = 81
    dnnl_query_some_md = 128
    dnnl_query_src_md = 129
    dnnl_query_diff_src_md = 130
    dnnl_query_weights_md = 131
    dnnl_query_diff_weights_md = 132
    dnnl_query_dst_md = 133
    dnnl_query_diff_dst_md = 134
    dnnl_query_workspace_md = 135
    dnnl_query_scratchpad_md = 136
    dnnl_query_exec_arg_md = 255
end

@cenum dnnl_stream_flags_t::UInt32 begin
    dnnl_stream_default_order = 1
    dnnl_stream_in_order = 2
    dnnl_stream_out_of_order = 4
    dnnl_stream_default_flags = 1
end


const dnnl_stream = Cvoid
const dnnl_stream_t = Ptr{dnnl_stream}
const const_dnnl_stream_t = Ptr{dnnl_stream}
const dnnl_stream_attr = Cvoid
const dnnl_stream_attr_t = Ptr{dnnl_stream_attr}
const const_dnnl_stream_attr_t = Ptr{dnnl_stream_attr}

struct dnnl_version_t
    major::Cint
    minor::Cint
    patch::Cint
    hash::Cstring
    cpu_runtime::UInt32
    gpu_runtime::UInt32
end

@cenum dnnl_cpu_isa_t::UInt32 begin
    dnnl_cpu_isa_all = 0
    dnnl_cpu_isa_sse41 = 1
    dnnl_cpu_isa_avx = 3
    dnnl_cpu_isa_avx2 = 7
    dnnl_cpu_isa_avx512_mic = 15
    dnnl_cpu_isa_avx512_mic_4ops = 31
    dnnl_cpu_isa_avx512_core = 39
    dnnl_cpu_isa_avx512_core_vnni = 103
    dnnl_cpu_isa_avx512_core_bf16 = 231
end

