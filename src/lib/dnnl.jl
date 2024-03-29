using CEnum

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
    dnnl_acbd = 6
    dnnl_abcde = 7
    dnnl_abcdef = 8
    dnnl_abcdefg = 9
    dnnl_abcdefgh = 10
    dnnl_abcdefghi = 11
    dnnl_abcdefghij = 12
    dnnl_abcdefghijk = 13
    dnnl_abcdefghijkl = 14
    dnnl_abdc = 15
    dnnl_abdec = 16
    dnnl_acb = 17
    dnnl_acbde = 18
    dnnl_acbdef = 19
    dnnl_acdb = 20
    dnnl_acdeb = 21
    dnnl_ba = 22
    dnnl_bac = 23
    dnnl_bacd = 24
    dnnl_bacde = 25
    dnnl_bca = 26
    dnnl_bcda = 27
    dnnl_bcdea = 28
    dnnl_cba = 29
    dnnl_cdba = 30
    dnnl_dcab = 31
    dnnl_cdeba = 32
    dnnl_decab = 33
    dnnl_defcab = 34
    dnnl_abced = 35
    dnnl_abcdfe = 36
    dnnl_abcdegf = 37
    dnnl_abcdefhg = 38
    dnnl_abcdefgih = 39
    dnnl_abcdefghji = 40
    dnnl_abcdefghikj = 41
    dnnl_abcdefghijlk = 42
    dnnl_Abc16a = 43
    dnnl_ABc16a16b = 44
    dnnl_ABc32a32b = 45
    dnnl_ABc4a4b = 46
    dnnl_aBc16b = 47
    dnnl_ABc16b16a = 48
    dnnl_Abc4a = 49
    dnnl_aBc32b = 50
    dnnl_aBc4b = 51
    dnnl_ABc4b16a4b = 52
    dnnl_ABc2b8a4b = 53
    dnnl_ABc16b16a4b = 54
    dnnl_ABc16b16a2b = 55
    dnnl_ABc4b4a = 56
    dnnl_ABc8a16b2a = 57
    dnnl_ABc8a8b = 58
    dnnl_ABc8a4b = 59
    dnnl_aBc8b = 60
    dnnl_ABc8b16a2b = 61
    dnnl_BAc8a16b2a = 62
    dnnl_ABc8b8a = 63
    dnnl_Abcd16a = 64
    dnnl_Abcd8a = 65
    dnnl_ABcd16a16b = 66
    dnnl_Abcd32a = 67
    dnnl_ABcd32a32b = 68
    dnnl_aBcd16b = 69
    dnnl_ABcd16b16a = 70
    dnnl_aBCd16b16c = 71
    dnnl_aBCd16c16b = 72
    dnnl_Abcd4a = 73
    dnnl_aBcd32b = 74
    dnnl_aBcd4b = 75
    dnnl_ABcd4b16a4b = 76
    dnnl_ABcd16b16a4b = 77
    dnnl_ABcd16b16a2b = 78
    dnnl_ABcd4b4a = 79
    dnnl_ABcd4a4b = 80
    dnnl_aBCd2c4b2c = 81
    dnnl_aBCd4b8c2b = 82
    dnnl_aBCd4c16b4c = 83
    dnnl_aBCd2c8b4c = 84
    dnnl_aBCd16c16b4c = 85
    dnnl_aBCd16c16b2c = 86
    dnnl_aBCd4c4b = 87
    dnnl_aBCd4b4c = 88
    dnnl_ABcd8a16b2a = 89
    dnnl_ABcd2b8a4b = 90
    dnnl_ABcd8a8b = 91
    dnnl_ABcd8a4b = 92
    dnnl_aBcd8b = 93
    dnnl_aBCd4c8b2c = 94
    dnnl_ABcd8b16a2b = 95
    dnnl_aBCd8b16c2b = 96
    dnnl_BAcd8a16b2a = 97
    dnnl_ABcd8b8a = 98
    dnnl_aBCd8b8c = 99
    dnnl_aBCd8b4c = 100
    dnnl_aBCd8c16b2c = 101
    dnnl_ABcde8a16b2a = 102
    dnnl_aCBd8b16c2b = 103
    dnnl_aBCd8c8b = 104
    dnnl_Abcde16a = 105
    dnnl_Abcde32a = 106
    dnnl_ABcde16a16b = 107
    dnnl_BAcde8a16b2a = 108
    dnnl_aBCd2b4c2b = 109
    dnnl_ABcde4b16a4b = 110
    dnnl_ABcde2b8a4b = 111
    dnnl_aBcde16b = 112
    dnnl_ABcde16b16a = 113
    dnnl_aBCde16b16c = 114
    dnnl_aBCde16c16b = 115
    dnnl_aBCde2c8b4c = 116
    dnnl_Abcde4a = 117
    dnnl_aBcde32b = 118
    dnnl_aBcde4b = 119
    dnnl_ABcde4b4a = 120
    dnnl_ABcde4a4b = 121
    dnnl_aBCde4b4c = 122
    dnnl_aBCde2c4b2c = 123
    dnnl_aBCde4b8c2b = 124
    dnnl_aBCde4c16b4c = 125
    dnnl_aBCde16c16b4c = 126
    dnnl_aBCde16c16b2c = 127
    dnnl_aBCde4c4b = 128
    dnnl_Abcde8a = 129
    dnnl_ABcde8a8b = 130
    dnnl_ABcde8a4b = 131
    dnnl_BAcde16b16a = 132
    dnnl_aBcde8b = 133
    dnnl_ABcde8b16a2b = 134
    dnnl_aBCde8b16c2b = 135
    dnnl_aBCde4c8b2c = 136
    dnnl_aCBde8b16c2b = 137
    dnnl_ABcde8b8a = 138
    dnnl_ABcde32a32b = 139
    dnnl_aBCde8b8c = 140
    dnnl_aBCde8b4c = 141
    dnnl_ABc4a8b8a4b = 142
    dnnl_ABcd4a8b8a4b = 143
    dnnl_ABcde4a8b8a4b = 144
    dnnl_BAc4b8a8b4a = 145
    dnnl_BAcd4b8a8b4a = 146
    dnnl_BAcde4b8a8b4a = 147
    dnnl_ABcd2a8b8a2b = 148
    dnnl_aBCd4b8c8b4c = 149
    dnnl_aBCde4b8c8b4c = 150
    dnnl_aBCde2b8c8b2c = 151
    dnnl_aBCde8c16b2c = 152
    dnnl_aBCde8c8b = 153
    dnnl_aBCde2b4c2b = 154
    dnnl_aBcdef16b = 155
    dnnl_aBCdef16b16c = 156
    dnnl_aBCdef16c16b = 157
    dnnl_aBCdef4c16b4c = 158
    dnnl_aBCdef2c8b4c = 159
    dnnl_aBCdef4c8b2c = 160
    dnnl_aBCdef2b4c2b = 161
    dnnl_aBcdef4b = 162
    dnnl_aBCdef4c4b = 163
    dnnl_aBCdef4b4c = 164
    dnnl_aBCdef2c4b2c = 165
    dnnl_aBCdef4b8c2b = 166
    dnnl_aBCdef8b8c = 167
    dnnl_aBCdef8b4c = 168
    dnnl_aBCdef8c16b2c = 169
    dnnl_aBCdef4b8c8b4c = 170
    dnnl_aBCdef8b16c2b = 171
    dnnl_aCBdef8b16c2b = 172
    dnnl_aBCdef8c8b = 173
    dnnl_aBdc16b = 174
    dnnl_aBdC16b2c = 175
    dnnl_aBdC16b4c = 176
    dnnl_aBdc4b = 177
    dnnl_aBdc8b = 178
    dnnl_aBdec16b = 179
    dnnl_aBdeC16b2c = 180
    dnnl_aBdeC16b4c = 181
    dnnl_aBdec32b = 182
    dnnl_aBdec4b = 183
    dnnl_aBdec8b = 184
    dnnl_aBdefc16b = 185
    dnnl_aBdefC16b2c = 186
    dnnl_aCBdef16c16b = 187
    dnnl_aBdefc4b = 188
    dnnl_aBdefc8b = 189
    dnnl_Abcdef16a = 190
    dnnl_Abcdef32a = 191
    dnnl_aBedc16b = 192
    dnnl_Acb16a = 193
    dnnl_AcB16a2b = 194
    dnnl_AcB16a4b = 195
    dnnl_Acb4a = 196
    dnnl_Acb8a = 197
    dnnl_aCBd16b16c = 198
    dnnl_aCBd16c16b = 199
    dnnl_aCBde16b16c = 200
    dnnl_aCBde16c16b = 201
    dnnl_Acdb16a = 202
    dnnl_AcdB16a2b = 203
    dnnl_AcdB16a4b = 204
    dnnl_Acdb32a = 205
    dnnl_Acdb4a = 206
    dnnl_Acdb8a = 207
    dnnl_Acdeb16a = 208
    dnnl_AcdeB16a2b = 209
    dnnl_Acdeb4a = 210
    dnnl_Acdeb8a = 211
    dnnl_Adcb16a = 212
    dnnl_BAc16a16b = 213
    dnnl_BAc16b16a = 214
    dnnl_BAcd16a16b = 215
    dnnl_BAcd16b16a = 216
    dnnl_aCBd4c8b8c4b = 217
    dnnl_aCBde4c8b8c4b = 218
    dnnl_aCBdef4c8b8c4b = 219
    dnnl_BAcde16a16b = 220
    dnnl_aCBdef16b16c = 221
    dnnl_abdfce = 222
    dnnl_abdefc = 223
    dnnl_ABc16b32a = 224
    dnnl_ABc16b64a = 225
    dnnl_ABc4b32a4b = 226
    dnnl_ABc4b64a4b = 227
    dnnl_ABc8b32a2b = 228
    dnnl_ABc8b64a2b = 229
    dnnl_AB16b16a = 230
    dnnl_AB16b32a = 231
    dnnl_AB16b64a = 232
    dnnl_AB8b16a2b = 233
    dnnl_AB8b32a2b = 234
    dnnl_AB8b64a2b = 235
    dnnl_AB4b16a4b = 236
    dnnl_AB4b32a4b = 237
    dnnl_AB4b64a4b = 238
    dnnl_AB16b16a4b = 239
    dnnl_ABcd16b32a = 240
    dnnl_ABcd16b64a = 241
    dnnl_ABcd4b32a4b = 242
    dnnl_ABcd4b64a4b = 243
    dnnl_ABcd8b32a2b = 244
    dnnl_ABcd8b64a2b = 245
    dnnl_ABcde4b32a4b = 246
    dnnl_ABcde4b64a4b = 247
    dnnl_ABcde16b16a4b = 248
    dnnl_ABcde16b16a2b = 249
    dnnl_ABcde16b32a = 250
    dnnl_ABcde16b64a = 251
    dnnl_ABcde8b32a2b = 252
    dnnl_ABcde8b64a2b = 253
    dnnl_aBCdef16c16b4c = 254
    dnnl_aBCdef16c16b2c = 255
    dnnl_AB32a32b8a4b = 256
    dnnl_AB8a4b = 257
    dnnl_AB32a32b8a2b = 258
    dnnl_AB8a2b = 259
    dnnl_abDc32d = 260
    dnnl_abDC32d4c = 261
    dnnl_abdEc32e = 262
    dnnl_abdEC32e2c = 263
    dnnl_abdEC32e4c = 264
    dnnl_aBdefC16b4c = 265
    dnnl_AcdeB16a4b = 266
    dnnl_ABcd16a16b2a = 267
    dnnl_ABc16a16b2a = 268
    dnnl_aBCd16b16c2b = 269
    dnnl_aBCde16b16c2b = 270
    dnnl_Acb32a = 271
    dnnl_AcB32a2b = 272
    dnnl_AcB32a4b = 273
    dnnl_Acb48a = 274
    dnnl_AcB48a2b = 275
    dnnl_AcB48a4b = 276
    dnnl_Acb64a = 277
    dnnl_AcB64a2b = 278
    dnnl_AcB64a4b = 279
    dnnl_cBa2b = 280
    dnnl_cBa4b = 281
    dnnl_aBdc32b = 282
    dnnl_aBdC32b2c = 283
    dnnl_aBdC32b4c = 284
    dnnl_aBdc48b = 285
    dnnl_aBdC48b2c = 286
    dnnl_aBdC48b4c = 287
    dnnl_aBdc64b = 288
    dnnl_aBdC64b2c = 289
    dnnl_aBdC64b4c = 290
    dnnl_adcb = 291
    dnnl_adCb2c = 292
    dnnl_adCb4c = 293
    dnnl_AcdB32a2b = 294
    dnnl_AcdB32a4b = 295
    dnnl_Acdb48a = 296
    dnnl_AcdB48a2b = 297
    dnnl_AcdB48a4b = 298
    dnnl_Acdb64a = 299
    dnnl_AcdB64a2b = 300
    dnnl_AcdB64a4b = 301
    dnnl_cdBa2b = 302
    dnnl_cdBa4b = 303
    dnnl_aBdeC32b2c = 304
    dnnl_aBdeC32b4c = 305
    dnnl_aBdec48b = 306
    dnnl_aBdeC48b2c = 307
    dnnl_aBdeC48b4c = 308
    dnnl_aBdec64b = 309
    dnnl_aBdeC64b2c = 310
    dnnl_aBdeC64b4c = 311
    dnnl_adecb = 312
    dnnl_adeCb2c = 313
    dnnl_adeCb4c = 314
    dnnl_Acdeb32a = 315
    dnnl_AcdeB32a2b = 316
    dnnl_AcdeB32a4b = 317
    dnnl_Acdeb48a = 318
    dnnl_AcdeB48a2b = 319
    dnnl_AcdeB48a4b = 320
    dnnl_Acdeb64a = 321
    dnnl_AcdeB64a2b = 322
    dnnl_AcdeB64a4b = 323
    dnnl_cdeBa2b = 324
    dnnl_cdeBa4b = 325
    dnnl_aBdefc32b = 326
    dnnl_aBdefC32b2c = 327
    dnnl_aBdefC32b4c = 328
    dnnl_aBdefc48b = 329
    dnnl_aBdefC48b2c = 330
    dnnl_aBdefC48b4c = 331
    dnnl_aBdefc64b = 332
    dnnl_aBdefC64b2c = 333
    dnnl_aBdefC64b4c = 334
    dnnl_adefcb = 335
    dnnl_adefCb2c = 336
    dnnl_adefCb4c = 337
    dnnl_AB16b32a4b = 338
    dnnl_AB16b48a4b = 339
    dnnl_AB16b64a4b = 340
    dnnl_AB16b16a2b = 341
    dnnl_AB16b32a2b = 342
    dnnl_AB16b48a2b = 343
    dnnl_AB16b64a2b = 344
    dnnl_ABc16b32a4b = 345
    dnnl_ABc16b48a4b = 346
    dnnl_ABc16b64a4b = 347
    dnnl_ABc16b32a2b = 348
    dnnl_ABc16b48a2b = 349
    dnnl_ABc16b64a2b = 350
    dnnl_ABcd16b32a4b = 351
    dnnl_ABcd16b48a4b = 352
    dnnl_ABcd16b64a4b = 353
    dnnl_ABcd16b32a2b = 354
    dnnl_ABcd16b48a2b = 355
    dnnl_ABcd16b64a2b = 356
    dnnl_ABcde16b32a4b = 357
    dnnl_ABcde16b48a4b = 358
    dnnl_ABcde16b64a4b = 359
    dnnl_ABcde16b32a2b = 360
    dnnl_ABcde16b48a2b = 361
    dnnl_ABcde16b64a2b = 362
    dnnl_ABc32a16b = 363
    dnnl_ABcd32a16b = 364
    dnnl_ABcde32a16b = 365
    dnnl_AB48a16b = 366
    dnnl_AB48a32b = 367
    dnnl_ABc40a16b = 368
    dnnl_ABc40a32b = 369
    dnnl_aBC48b16c = 370
    dnnl_aBC48b32c = 371
    dnnl_ABcd40a16b = 372
    dnnl_ABcd40a32b = 373
    dnnl_abCd32c = 374
    dnnl_abdCe32c = 375
    dnnl_abdCE32c2e = 376
    dnnl_BA16a16b2a = 377
    dnnl_BA16a32b2a = 378
    dnnl_BA16a48b2a = 379
    dnnl_BA16a64b2a = 380
    dnnl_BA16a16b4a = 381
    dnnl_BA16a32b4a = 382
    dnnl_BA16a48b4a = 383
    dnnl_BA16a64b4a = 384
    dnnl_format_tag_last = 385
    dnnl_x = 2
    dnnl_nc = 3
    dnnl_cn = 22
    dnnl_tn = 3
    dnnl_nt = 22
    dnnl_ncw = 4
    dnnl_nwc = 17
    dnnl_nchw = 5
    dnnl_nhwc = 20
    dnnl_chwn = 27
    dnnl_ncdhw = 7
    dnnl_ndhwc = 21
    dnnl_oi = 3
    dnnl_io = 22
    dnnl_oiw = 4
    dnnl_owi = 17
    dnnl_wio = 29
    dnnl_iwo = 26
    dnnl_oihw = 5
    dnnl_hwio = 30
    dnnl_ohwi = 20
    dnnl_ihwo = 27
    dnnl_iohw = 24
    dnnl_oidhw = 7
    dnnl_iodhw = 25
    dnnl_dhwio = 32
    dnnl_odhwi = 21
    dnnl_idhwo = 28
    dnnl_goiw = 5
    dnnl_gowi = 15
    dnnl_wigo = 31
    dnnl_goihw = 7
    dnnl_gohwi = 16
    dnnl_hwigo = 33
    dnnl_giohw = 18
    dnnl_goidhw = 8
    dnnl_godhwi = 223
    dnnl_giodhw = 19
    dnnl_dhwigo = 34
    dnnl_tnc = 4
    dnnl_ntc = 23
    dnnl_ldnc = 5
    dnnl_ldigo = 7
    dnnl_ldgoi = 16
    dnnl_ldio = 5
    dnnl_ldoi = 15
    dnnl_ldgo = 5
    dnnl_ldOi32o = 260
    dnnl_ldOI32o4i = 261
    dnnl_ldIo32i = 374
    dnnl_ldgOi32o = 262
    dnnl_ldgOI32o2i = 263
    dnnl_ldgOI32o4i = 264
    dnnl_ldgIo32i = 375
    dnnl_ldgIO32i2o = 376
    dnnl_nCdhw32c = 118
    dnnl_nCdhw16c = 112
    dnnl_nCdhw4c = 119
    dnnl_nCdhw8c = 133
    dnnl_nChw32c = 74
    dnnl_nChw16c = 69
    dnnl_nChw4c = 75
    dnnl_nChw8c = 93
    dnnl_nCw32c = 50
    dnnl_nCw16c = 47
    dnnl_nCw4c = 51
    dnnl_nCw8c = 60
    dnnl_NCw16n16c = 44
    dnnl_NCdhw16n16c = 107
    dnnl_NChw16n16c = 66
    dnnl_NCw32n16c = 363
    dnnl_NChw32n16c = 364
    dnnl_NCdhw32n16c = 365
    dnnl_NCw32n32c = 45
    dnnl_NChw32n32c = 68
    dnnl_NCdhw32n32c = 139
    dnnl_OI16i16o = 230
    dnnl_OI16i32o = 231
    dnnl_OI16i64o = 232
    dnnl_OI8i16o2i = 233
    dnnl_OI8i32o2i = 234
    dnnl_OI8i64o2i = 235
    dnnl_OI4i16o4i = 236
    dnnl_OI4i32o4i = 237
    dnnl_OI4i64o4i = 238
    dnnl_OI16i16o4i = 239
    dnnl_IOw16o16i = 213
    dnnl_IOw16i16o = 214
    dnnl_OIw16i16o = 48
    dnnl_OIw16i32o = 224
    dnnl_OIw16i64o = 225
    dnnl_OIw16o16i = 44
    dnnl_Oiw16o = 43
    dnnl_OIw4i16o4i = 52
    dnnl_OIw4i32o4i = 226
    dnnl_OIw4i64o4i = 227
    dnnl_OIw2i8o4i = 53
    dnnl_OIw16i16o4i = 54
    dnnl_OIw16i16o2i = 55
    dnnl_OIw16o16i2o = 268
    dnnl_OIw4i4o = 56
    dnnl_OIw4o4i = 46
    dnnl_Oiw4o = 49
    dnnl_OIw8i16o2i = 61
    dnnl_OIw8i32o2i = 228
    dnnl_OIw8i64o2i = 229
    dnnl_OIw8i8o = 63
    dnnl_OIw8o16i2o = 57
    dnnl_IOw8o16i2o = 62
    dnnl_OIw8o8i = 58
    dnnl_OIw8o4i = 59
    dnnl_Owi16o = 193
    dnnl_OwI16o2i = 194
    dnnl_OwI16o4i = 195
    dnnl_Owi4o = 196
    dnnl_Owi8o = 197
    dnnl_IOhw16i16o = 216
    dnnl_IOhw16o16i = 215
    dnnl_Ohwi16o = 202
    dnnl_OhwI16o2i = 203
    dnnl_OhwI16o4i = 204
    dnnl_Ohwi32o = 205
    dnnl_Ohwi4o = 206
    dnnl_Ohwi8o = 207
    dnnl_OIhw16i16o = 70
    dnnl_OIhw16i32o = 240
    dnnl_OIhw16i64o = 241
    dnnl_OIhw16o16i = 66
    dnnl_Oihw16o = 64
    dnnl_OIhw4i16o4i = 76
    dnnl_OIhw4i32o4i = 242
    dnnl_OIhw4i64o4i = 243
    dnnl_OIhw16i16o4i = 77
    dnnl_OIhw16i16o2i = 78
    dnnl_OIhw16o16i2o = 267
    dnnl_OIhw4i4o = 79
    dnnl_OIhw4o4i = 80
    dnnl_Oihw4o = 73
    dnnl_OIhw8i16o2i = 95
    dnnl_OIhw8i32o2i = 244
    dnnl_OIhw8i64o2i = 245
    dnnl_OIhw8i8o = 98
    dnnl_OIhw8o16i2o = 89
    dnnl_OIhw2i8o4i = 90
    dnnl_IOhw8o16i2o = 97
    dnnl_OIhw8o8i = 91
    dnnl_OIhw8o4i = 92
    dnnl_Owhi16o = 212
    dnnl_Odhwi16o = 208
    dnnl_OdhwI16o2i = 209
    dnnl_OdhwI16o4i = 266
    dnnl_Odhwi4o = 210
    dnnl_Odhwi8o = 211
    dnnl_OIdhw16i16o = 113
    dnnl_OIdhw16i32o = 250
    dnnl_OIdhw16i64o = 251
    dnnl_OIdhw16o16i = 107
    dnnl_Oidhw16o = 105
    dnnl_OIdhw4i4o = 120
    dnnl_OIdhw4o4i = 121
    dnnl_Oidhw4o = 117
    dnnl_OIdhw8i16o2i = 134
    dnnl_OIdhw8i32o2i = 252
    dnnl_OIdhw8i64o2i = 253
    dnnl_OIdhw8i8o = 138
    dnnl_OIdhw8o16i2o = 102
    dnnl_IOdhw8o16i2o = 108
    dnnl_OIdhw4i16o4i = 110
    dnnl_OIdhw4i32o4i = 246
    dnnl_OIdhw4i64o4i = 247
    dnnl_OIdhw16i16o4i = 248
    dnnl_OIdhw16i16o2i = 249
    dnnl_OIdhw2i8o4i = 111
    dnnl_OIdhw8o8i = 130
    dnnl_OIdhw8o4i = 131
    dnnl_IOdhw16i16o = 132
    dnnl_OIdhw4o8i8o4i = 144
    dnnl_IOdhw16o16i = 220
    dnnl_Goiw16g = 64
    dnnl_Goiw8g = 65
    dnnl_Goiw4g = 73
    dnnl_gIOw16o16i = 198
    dnnl_gIOw16i16o = 199
    dnnl_gOIw16i16o = 72
    dnnl_gOIw16o16i = 71
    dnnl_gOiw16o = 69
    dnnl_gOIw4i16o4i = 83
    dnnl_gOIw2i8o4i = 84
    dnnl_gOIw16i16o4i = 85
    dnnl_gOIw16i16o2i = 86
    dnnl_gOIw16o16i2o = 269
    dnnl_gOIw4i4o = 87
    dnnl_gOIw4o4i = 88
    dnnl_gOiw4o = 75
    dnnl_gOIw8i16o2i = 101
    dnnl_gOIw8i8o = 104
    dnnl_gOIw8o16i2o = 96
    dnnl_gIOw8o16i2o = 103
    dnnl_gOIw8o8i = 99
    dnnl_gOIw8o4i = 100
    dnnl_gOwi16o = 174
    dnnl_gOwI16o2i = 175
    dnnl_gOwI16o4i = 176
    dnnl_gOwi4o = 177
    dnnl_gOwi8o = 178
    dnnl_Goiw32g = 67
    dnnl_gOIw2i4o2i = 81
    dnnl_gOIw2o4i2o = 109
    dnnl_gOIw4i8o2i = 94
    dnnl_gOIw4o8i2o = 82
    dnnl_gIOhw16i16o = 201
    dnnl_gIOhw16o16i = 200
    dnnl_gOhwi16o = 179
    dnnl_gOhwI16o2i = 180
    dnnl_gOhwI16o4i = 181
    dnnl_gOhwi32o = 182
    dnnl_gOhwi4o = 183
    dnnl_gOhwi8o = 184
    dnnl_Goihw16g = 105
    dnnl_gOIhw16i16o = 115
    dnnl_gOIhw16o16i = 114
    dnnl_gOihw16o = 112
    dnnl_gOIhw2i8o4i = 116
    dnnl_gOIhw4i16o4i = 125
    dnnl_gOIhw16i16o4i = 126
    dnnl_gOIhw16i16o2i = 127
    dnnl_gOIhw16o16i2o = 270
    dnnl_gOIhw4i4o = 128
    dnnl_gOIhw4o4i = 122
    dnnl_gOihw4o = 119
    dnnl_Goihw8g = 129
    dnnl_Goihw4g = 117
    dnnl_gOIhw8i16o2i = 152
    dnnl_gOIhw8i8o = 153
    dnnl_gOIhw8o16i2o = 135
    dnnl_gIOhw8o16i2o = 137
    dnnl_gOIhw8o8i = 140
    dnnl_gOIhw8o4i = 141
    dnnl_Goihw32g = 106
    dnnl_gOwhi16o = 192
    dnnl_OIw4o8i8o4i = 142
    dnnl_OIhw4o8i8o4i = 143
    dnnl_IOw4i8o8i4o = 145
    dnnl_IOhw4i8o8i4o = 146
    dnnl_IOdhw4i8o8i4o = 147
    dnnl_OIhw2o8i8o2i = 148
    dnnl_gOIw4o8i8o4i = 149
    dnnl_gOIhw4o8i8o4i = 150
    dnnl_gOIdhw4o8i8o4i = 170
    dnnl_gIOw4i8o8i4o = 217
    dnnl_gIOhw4i8o8i4o = 218
    dnnl_gIOdhw4i8o8i4o = 219
    dnnl_gOIhw2o8i8o2i = 151
    dnnl_gOIhw2i4o2i = 123
    dnnl_gOIhw2o4i2o = 154
    dnnl_gOIhw4i8o2i = 136
    dnnl_gOIhw4o8i2o = 124
    dnnl_gIOdhw16i16o = 187
    dnnl_gIOdhw16o16i = 221
    dnnl_gOdhwi16o = 185
    dnnl_gOdhwI16o2i = 186
    dnnl_gOdhwI16o4i = 265
    dnnl_gOdhwi4o = 188
    dnnl_gOdhwi8o = 189
    dnnl_gOIdhw16i16o = 157
    dnnl_gOIdhw4i16o4i = 158
    dnnl_gOIdhw16i16o4i = 254
    dnnl_gOIdhw2i8o4i = 159
    dnnl_gOIdhw16i16o2i = 255
    dnnl_gOIdhw16o16i = 156
    dnnl_gOidhw16o = 155
    dnnl_gOIdhw4i4o = 163
    dnnl_gOIdhw4o4i = 164
    dnnl_gOidhw4o = 162
    dnnl_gOIdhw8i16o2i = 169
    dnnl_gOIdhw8i8o = 173
    dnnl_gOIdhw8o16i2o = 171
    dnnl_gIOdhw8o16i2o = 172
    dnnl_gOIdhw8o8i = 167
    dnnl_gOIdhw8o4i = 168
    dnnl_Goidhw16g = 190
    dnnl_Goidhw32g = 191
    dnnl_gOIdhw2i4o2i = 165
    dnnl_gOIdhw4i8o2i = 160
    dnnl_gOIdhw2o4i2o = 161
    dnnl_gOIdhw4o8i2o = 166
    dnnl_Owi32o = 271
    dnnl_OwI32o2i = 272
    dnnl_OwI32o4i = 273
    dnnl_Owi48o = 274
    dnnl_OwI48o2i = 275
    dnnl_OwI48o4i = 276
    dnnl_Owi64o = 277
    dnnl_OwI64o2i = 278
    dnnl_OwI64o4i = 279
    dnnl_wIo2i = 280
    dnnl_wIo4i = 281
    dnnl_gOwi32o = 282
    dnnl_gOwI32o2i = 283
    dnnl_gOwI32o4i = 284
    dnnl_gOwi48o = 285
    dnnl_gOwI48o2i = 286
    dnnl_gOwI48o4i = 287
    dnnl_gOwi64o = 288
    dnnl_gOwI64o2i = 289
    dnnl_gOwI64o4i = 290
    dnnl_gwio = 291
    dnnl_gwIo2i = 292
    dnnl_gwIo4i = 293
    dnnl_OhwI32o = 205
    dnnl_OhwI32o2i = 294
    dnnl_OhwI32o4i = 295
    dnnl_Ohwi48o = 296
    dnnl_OhwI48o2i = 297
    dnnl_OhwI48o4i = 298
    dnnl_Ohwi64o = 299
    dnnl_OhwI64o2i = 300
    dnnl_OhwI64o4i = 301
    dnnl_hwIo2i = 302
    dnnl_hwIo4i = 303
    dnnl_gOhwI32o = 182
    dnnl_gOhwI32o2i = 304
    dnnl_gOhwI32o4i = 305
    dnnl_gOhwi48o = 306
    dnnl_gOhwI48o2i = 307
    dnnl_gOhwI48o4i = 308
    dnnl_gOhwi64o = 309
    dnnl_gOhwI64o2i = 310
    dnnl_gOhwI64o4i = 311
    dnnl_ghwio = 312
    dnnl_ghwIo2i = 313
    dnnl_ghwIo4i = 314
    dnnl_Odhwi32o = 315
    dnnl_OdhwI32o2i = 316
    dnnl_OdhwI32o4i = 317
    dnnl_Odhwi48o = 318
    dnnl_OdhwI48o2i = 319
    dnnl_OdhwI48o4i = 320
    dnnl_Odhwi64o = 321
    dnnl_OdhwI64o2i = 322
    dnnl_OdhwI64o4i = 323
    dnnl_dhwIo2i = 324
    dnnl_dhwIo4i = 325
    dnnl_gOdhwi32o = 326
    dnnl_gOdhwI32o2i = 327
    dnnl_gOdhwI32o4i = 328
    dnnl_gOdhwi48o = 329
    dnnl_gOdhwI48o2i = 330
    dnnl_gOdhwI48o4i = 331
    dnnl_gOdhwi64o = 332
    dnnl_gOdhwI64o2i = 333
    dnnl_gOdhwI64o4i = 334
    dnnl_gdhwio = 335
    dnnl_gdhwIo2i = 336
    dnnl_gdhwIo4i = 337
    dnnl_OI16i32o4i = 338
    dnnl_OI16i48o4i = 339
    dnnl_OI16i64o4i = 340
    dnnl_OI16i16o2i = 341
    dnnl_OI16i32o2i = 342
    dnnl_OI16i48o2i = 343
    dnnl_OI16i64o2i = 344
    dnnl_OIw16i32o4i = 345
    dnnl_OIw16i48o4i = 346
    dnnl_OIw16i64o4i = 347
    dnnl_OIw16i32o2i = 348
    dnnl_OIw16i48o2i = 349
    dnnl_OIw16i64o2i = 350
    dnnl_OIhw16i32o4i = 351
    dnnl_OIhw16i48o4i = 352
    dnnl_OIhw16i64o4i = 353
    dnnl_OIhw16i32o2i = 354
    dnnl_OIhw16i48o2i = 355
    dnnl_OIhw16i64o2i = 356
    dnnl_OIdhw16i32o4i = 357
    dnnl_OIdhw16i48o4i = 358
    dnnl_OIdhw16i64o4i = 359
    dnnl_OIdhw16i32o2i = 360
    dnnl_OIdhw16i48o2i = 361
    dnnl_OIdhw16i64o2i = 362
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
    dnnl_pooling_v2 = 20
    dnnl_reduction = 21
    dnnl_prelu = 22
    dnnl_primitive_kind_max = 32767
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
    dnnl_eltwise_clip_v2 = 16
    dnnl_eltwise_pow = 32
    dnnl_eltwise_gelu_erf = 48
    dnnl_eltwise_round = 64
    dnnl_eltwise_logsigmoid = 80
    dnnl_eltwise_mish = 96
    dnnl_eltwise_hardswish = 112
    dnnl_eltwise_relu_use_dst_for_bwd = 256
    dnnl_eltwise_tanh_use_dst_for_bwd = 257
    dnnl_eltwise_elu_use_dst_for_bwd = 258
    dnnl_eltwise_sqrt_use_dst_for_bwd = 259
    dnnl_eltwise_logistic_use_dst_for_bwd = 260
    dnnl_eltwise_exp_use_dst_for_bwd = 261
    dnnl_eltwise_clip_v2_use_dst_for_bwd = 262
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
    dnnl_binary_div = 131060
    dnnl_binary_sub = 131061
    dnnl_binary_ge = 131062
    dnnl_binary_gt = 131063
    dnnl_binary_le = 131064
    dnnl_binary_lt = 131065
    dnnl_binary_eq = 131066
    dnnl_binary_ne = 131067
    dnnl_resampling_nearest = 196592
    dnnl_resampling_linear = 196593
    dnnl_reduction_max = 196594
    dnnl_reduction_min = 196595
    dnnl_reduction_sum = 196596
    dnnl_reduction_mul = 196597
    dnnl_reduction_mean = 196598
    dnnl_reduction_norm_lp_max = 196599
    dnnl_reduction_norm_lp_sum = 196600
    dnnl_reduction_norm_lp_power_p_max = 196601
    dnnl_reduction_norm_lp_power_p_sum = 196602
end

@cenum dnnl_normalization_flags_t::UInt32 begin
    dnnl_normalization_flags_none = 0
    dnnl_use_global_stats = 1
    dnnl_use_scaleshift = 2
    dnnl_fuse_norm_relu = 4
    dnnl_use_scale = 8
    dnnl_use_shift = 16
end

struct __JL_Ctag_10
    data::NTuple{4,UInt8}
end

function Base.getproperty(x::Ptr{__JL_Ctag_10}, f::Symbol)
    f === :u && return Ptr{Cuint}(x + 0)
    f === :f && return Ptr{Cfloat}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_10, f::Symbol)
    r = Ref{__JL_Ctag_10}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_10}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_10}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

const dnnl_dim_t = Int64

const dnnl_dims_t = NTuple{12,dnnl_dim_t}

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
    size::Csize_t
end

@cenum dnnl_rnn_packed_memory_format_t::UInt32 begin
    dnnl_packed_format_undef = 0
    dnnl_ldigo_p = 1
    dnnl_ldgoi_p = 2
    dnnl_ldio_p = 3
end

struct dnnl_rnn_packed_desc_t
    format::dnnl_rnn_packed_memory_format_t
    n_parts::Cint
    n::Cint
    ldb::Cint
    parts::NTuple{4,Cint}
    part_pack_size::NTuple{4,Csize_t}
    pack_part::NTuple{4,Cuint}
    offset_compensation::Csize_t
    size::Csize_t
    reserved::NTuple{200,Cchar}
end

@cenum dnnl_memory_extra_flags_t::UInt32 begin
    dnnl_memory_extra_flag_none = 0
    dnnl_memory_extra_flag_compensation_conv_s8s8 = 1
    dnnl_memory_extra_flag_scale_adjust = 2
    dnnl_memory_extra_flag_rnn_u8s8_compensation = 4
    dnnl_memory_extra_flag_gpu_rnn_u8s8_compensation = 4
    dnnl_memory_extra_flag_compensation_conv_asymmetric_src = 8
    dnnl_memory_extra_flag_rnn_s8s8_compensation = 22
end

struct dnnl_memory_extra_desc_t
    flags::UInt64
    compensation_mask::Cint
    scale_adjust::Cfloat
    asymm_compensation_mask::Cint
    reserved::NTuple{60,Cchar}
end

struct __JL_Ctag_89
    data::NTuple{296,UInt8}
end

function Base.getproperty(x::Ptr{__JL_Ctag_89}, f::Symbol)
    f === :blocking && return Ptr{dnnl_blocking_desc_t}(x + 0)
    f === :wino_desc && return Ptr{dnnl_wino_desc_t}(x + 0)
    f === :rnn_packed_desc && return Ptr{dnnl_rnn_packed_desc_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_89, f::Symbol)
    r = Ref{__JL_Ctag_89}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_89}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_89}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct dnnl_memory_desc_t
    ndims::Cint
    dims::dnnl_dims_t
    data_type::dnnl_data_type_t
    padded_dims::dnnl_dims_t
    padded_offsets::dnnl_dims_t
    offset0::dnnl_dim_t
    format_kind::dnnl_format_kind_t
    #format_desc::__JL_Ctag_89
    format_desc::dnnl_blocking_desc_t
    extra::dnnl_memory_extra_desc_t
end

begin
    struct dnnl_memory end
    function Base.cconvert(::Type{Ptr{dnnl_memory}}, x::Ptr{dnnl_memory})
        return x
    end
    function Base.cconvert(::Type{Ptr{dnnl_memory}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{$(dnnl_memory)}!")
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_memory}}}, x::Ptr{Ptr{dnnl_memory}})
        return x
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_memory}}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{Ptr{$(dnnl_memory)}}!")
    end
end

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
    padding::NTuple{2,dnnl_dims_t}
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
    padding::NTuple{2,dnnl_dims_t}
    accum_data_type::dnnl_data_type_t
end

struct dnnl_pooling_v2_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    alg_kind::dnnl_alg_kind_t
    src_desc::dnnl_memory_desc_t
    diff_src_desc::dnnl_memory_desc_t
    dst_desc::dnnl_memory_desc_t
    diff_dst_desc::dnnl_memory_desc_t
    strides::dnnl_dims_t
    kernel::dnnl_dims_t
    padding::NTuple{2,dnnl_dims_t}
    accum_data_type::dnnl_data_type_t
    dilation::dnnl_dims_t
end

struct dnnl_prelu_desc_t
    primitive_kind::dnnl_primitive_kind_t
    prop_kind::dnnl_prop_kind_t
    data_desc::dnnl_memory_desc_t
    weights_desc::dnnl_memory_desc_t
    diff_data_desc::dnnl_memory_desc_t
    diff_weights_desc::dnnl_memory_desc_t
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
    flags::Cuint
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
    flags::Cuint
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
    flags::Cuint
    activation_kind::dnnl_alg_kind_t
    alpha::Cfloat
    beta::Cfloat
end

struct dnnl_binary_desc_t
    primitive_kind::dnnl_primitive_kind_t
    alg_kind::dnnl_alg_kind_t
    src_desc::NTuple{2,dnnl_memory_desc_t}
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
    factors::NTuple{12,Cfloat}
end

struct dnnl_reduction_desc_t
    primitive_kind::dnnl_primitive_kind_t
    alg_kind::dnnl_alg_kind_t
    src_desc::dnnl_memory_desc_t
    dst_desc::dnnl_memory_desc_t
    p::Cfloat
    eps::Cfloat
end

@cenum dnnl_engine_kind_t::UInt32 begin
    dnnl_any_engine = 0
    dnnl_cpu = 1
    dnnl_gpu = 2
end

begin
    struct dnnl_engine end
    function Base.cconvert(::Type{Ptr{dnnl_engine}}, x::Ptr{dnnl_engine})
        return x
    end
    function Base.cconvert(::Type{Ptr{dnnl_engine}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{$(dnnl_engine)}!")
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_engine}}}, x::Ptr{Ptr{dnnl_engine}})
        return x
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_engine}}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{Ptr{$(dnnl_engine)}}!")
    end
end

const dnnl_engine_t = Ptr{dnnl_engine}

begin
    struct dnnl_primitive_desc_iterator end
    function Base.cconvert(
        ::Type{Ptr{dnnl_primitive_desc_iterator}}, x::Ptr{dnnl_primitive_desc_iterator}
    )
        return x
    end
    function Base.cconvert(::Type{Ptr{dnnl_primitive_desc_iterator}}, x::Ptr)
        return error(
            "Refusing to convert $(typeof(x)) to a Ptr{$(dnnl_primitive_desc_iterator)}!"
        )
    end
    function Base.cconvert(
        ::Type{Ptr{Ptr{dnnl_primitive_desc_iterator}}},
        x::Ptr{Ptr{dnnl_primitive_desc_iterator}},
    )
        return x
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_primitive_desc_iterator}}}, x::Ptr)
        return error(
            "Refusing to convert $(typeof(x)) to a Ptr{Ptr{$(dnnl_primitive_desc_iterator)}}!",
        )
    end
end

const dnnl_primitive_desc_iterator_t = Ptr{dnnl_primitive_desc_iterator}

const const_dnnl_primitive_desc_iterator_t = Ptr{dnnl_primitive_desc_iterator}

begin
    struct dnnl_primitive_desc end
    function Base.cconvert(::Type{Ptr{dnnl_primitive_desc}}, x::Ptr{dnnl_primitive_desc})
        return x
    end
    function Base.cconvert(::Type{Ptr{dnnl_primitive_desc}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{$(dnnl_primitive_desc)}!")
    end
    function Base.cconvert(
        ::Type{Ptr{Ptr{dnnl_primitive_desc}}}, x::Ptr{Ptr{dnnl_primitive_desc}}
    )
        return x
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_primitive_desc}}}, x::Ptr)
        return error(
            "Refusing to convert $(typeof(x)) to a Ptr{Ptr{$(dnnl_primitive_desc)}}!"
        )
    end
end

const dnnl_primitive_desc_t = Ptr{dnnl_primitive_desc}

const const_dnnl_primitive_desc_t = Ptr{dnnl_primitive_desc}

@cenum dnnl_scratchpad_mode_t::UInt32 begin
    dnnl_scratchpad_mode_library = 0
    dnnl_scratchpad_mode_user = 1
end

begin
    struct dnnl_primitive_attr end
    function Base.cconvert(::Type{Ptr{dnnl_primitive_attr}}, x::Ptr{dnnl_primitive_attr})
        return x
    end
    function Base.cconvert(::Type{Ptr{dnnl_primitive_attr}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{$(dnnl_primitive_attr)}!")
    end
    function Base.cconvert(
        ::Type{Ptr{Ptr{dnnl_primitive_attr}}}, x::Ptr{Ptr{dnnl_primitive_attr}}
    )
        return x
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_primitive_attr}}}, x::Ptr)
        return error(
            "Refusing to convert $(typeof(x)) to a Ptr{Ptr{$(dnnl_primitive_attr)}}!"
        )
    end
end

const dnnl_primitive_attr_t = Ptr{dnnl_primitive_attr}

const const_dnnl_primitive_attr_t = Ptr{dnnl_primitive_attr}

begin
    struct dnnl_post_ops end
    function Base.cconvert(::Type{Ptr{dnnl_post_ops}}, x::Ptr{dnnl_post_ops})
        return x
    end
    function Base.cconvert(::Type{Ptr{dnnl_post_ops}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{$(dnnl_post_ops)}!")
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_post_ops}}}, x::Ptr{Ptr{dnnl_post_ops}})
        return x
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_post_ops}}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{Ptr{$(dnnl_post_ops)}}!")
    end
end

const dnnl_post_ops_t = Ptr{dnnl_post_ops}

const const_dnnl_post_ops_t = Ptr{dnnl_post_ops}

begin
    struct dnnl_primitive end
    function Base.cconvert(::Type{Ptr{dnnl_primitive}}, x::Ptr{dnnl_primitive})
        return x
    end
    function Base.cconvert(::Type{Ptr{dnnl_primitive}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{$(dnnl_primitive)}!")
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_primitive}}}, x::Ptr{Ptr{dnnl_primitive}})
        return x
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_primitive}}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{Ptr{$(dnnl_primitive)}}!")
    end
end

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
    dnnl_query_pooling_v2_d = 82
    dnnl_query_reduction_d = 83
    dnnl_query_prelu_d = 84
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
    dnnl_query_max = 32767
end

@cenum dnnl_stream_flags_t::UInt32 begin
    dnnl_stream_in_order = 1
    dnnl_stream_out_of_order = 2
    dnnl_stream_default_flags = 1
end

begin
    struct dnnl_stream end
    function Base.cconvert(::Type{Ptr{dnnl_stream}}, x::Ptr{dnnl_stream})
        return x
    end
    function Base.cconvert(::Type{Ptr{dnnl_stream}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{$(dnnl_stream)}!")
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_stream}}}, x::Ptr{Ptr{dnnl_stream}})
        return x
    end
    function Base.cconvert(::Type{Ptr{Ptr{dnnl_stream}}}, x::Ptr)
        return error("Refusing to convert $(typeof(x)) to a Ptr{Ptr{$(dnnl_stream)}}!")
    end
end

const dnnl_stream_t = Ptr{dnnl_stream}

const const_dnnl_stream_t = Ptr{dnnl_stream}

struct dnnl_version_t
    major::Cint
    minor::Cint
    patch::Cint
    hash::Ptr{Cchar}
    cpu_runtime::Cuint
    gpu_runtime::Cuint
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
    dnnl_cpu_isa_avx512_core_amx = 999
    dnnl_cpu_isa_avx2_vnni = 1031
end

@cenum dnnl_cpu_isa_hints_t::UInt32 begin
    dnnl_cpu_isa_no_hints = 0
    dnnl_cpu_isa_prefer_ymm = 1
end

function dnnl_primitive_desc_iterator_create(
    iterator, op_desc, attr, engine, hint_forward_primitive_desc
)
    return ccall(
        (:dnnl_primitive_desc_iterator_create, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_primitive_desc_iterator_t},
            const_dnnl_op_desc_t,
            const_dnnl_primitive_attr_t,
            dnnl_engine_t,
            const_dnnl_primitive_desc_t,
        ),
        iterator,
        op_desc,
        attr,
        engine,
        hint_forward_primitive_desc,
    )
end

function dnnl_primitive_desc_iterator_next(iterator)
    return ccall(
        (:dnnl_primitive_desc_iterator_next, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_desc_iterator_t,),
        iterator,
    )
end

function dnnl_primitive_desc_iterator_fetch(iterator)
    return ccall(
        (:dnnl_primitive_desc_iterator_fetch, libdnnl),
        dnnl_primitive_desc_t,
        (const_dnnl_primitive_desc_iterator_t,),
        iterator,
    )
end

function dnnl_primitive_desc_iterator_destroy(iterator)
    return ccall(
        (:dnnl_primitive_desc_iterator_destroy, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_desc_iterator_t,),
        iterator,
    )
end

function dnnl_primitive_desc_create(
    primitive_desc, op_desc, attr, engine, hint_forward_primitive_desc
)
    return ccall(
        (:dnnl_primitive_desc_create, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_primitive_desc_t},
            const_dnnl_op_desc_t,
            const_dnnl_primitive_attr_t,
            dnnl_engine_t,
            const_dnnl_primitive_desc_t,
        ),
        primitive_desc,
        op_desc,
        attr,
        engine,
        hint_forward_primitive_desc,
    )
end

function dnnl_primitive_desc_clone(primitive_desc, existing_primitive_desc)
    return ccall(
        (:dnnl_primitive_desc_clone, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_primitive_desc_t}, const_dnnl_primitive_desc_t),
        primitive_desc,
        existing_primitive_desc,
    )
end

function dnnl_primitive_desc_get_attr(primitive_desc, attr)
    return ccall(
        (:dnnl_primitive_desc_get_attr, libdnnl),
        dnnl_status_t,
        (const_dnnl_primitive_desc_t, Ptr{const_dnnl_primitive_attr_t}),
        primitive_desc,
        attr,
    )
end

function dnnl_primitive_desc_destroy(primitive_desc)
    return ccall(
        (:dnnl_primitive_desc_destroy, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_desc_t,),
        primitive_desc,
    )
end

function dnnl_primitive_desc_query(primitive_desc, what, index, result)
    return ccall(
        (:dnnl_primitive_desc_query, libdnnl),
        dnnl_status_t,
        (const_dnnl_primitive_desc_t, dnnl_query_t, Cint, Ptr{Cvoid}),
        primitive_desc,
        what,
        index,
        result,
    )
end

function dnnl_primitive_desc_query_md(primitive_desc, what, index)
    return ccall(
        (:dnnl_primitive_desc_query_md, libdnnl),
        Ptr{dnnl_memory_desc_t},
        (const_dnnl_primitive_desc_t, dnnl_query_t, Cint),
        primitive_desc,
        what,
        index,
    )
end

function dnnl_primitive_desc_query_s32(primitive_desc, what, index)
    return ccall(
        (:dnnl_primitive_desc_query_s32, libdnnl),
        Cint,
        (const_dnnl_primitive_desc_t, dnnl_query_t, Cint),
        primitive_desc,
        what,
        index,
    )
end

function dnnl_primitive_create(primitive, primitive_desc)
    return ccall(
        (:dnnl_primitive_create, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_primitive_t}, const_dnnl_primitive_desc_t),
        primitive,
        primitive_desc,
    )
end

function dnnl_primitive_execute(primitive, stream, nargs, args)
    return ccall(
        (:dnnl_primitive_execute, libdnnl),
        dnnl_status_t,
        (const_dnnl_primitive_t, dnnl_stream_t, Cint, Ptr{dnnl_exec_arg_t}),
        primitive,
        stream,
        nargs,
        args,
    )
end

function dnnl_primitive_get_primitive_desc(primitive, primitive_desc)
    return ccall(
        (:dnnl_primitive_get_primitive_desc, libdnnl),
        dnnl_status_t,
        (const_dnnl_primitive_t, Ptr{const_dnnl_primitive_desc_t}),
        primitive,
        primitive_desc,
    )
end

function dnnl_primitive_destroy(primitive)
    return ccall(
        (:dnnl_primitive_destroy, libdnnl), dnnl_status_t, (dnnl_primitive_t,), primitive
    )
end

function dnnl_primitive_attr_create(attr)
    return ccall(
        (:dnnl_primitive_attr_create, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_primitive_attr_t},),
        attr,
    )
end

function dnnl_primitive_attr_clone(attr, existing_attr)
    return ccall(
        (:dnnl_primitive_attr_clone, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_primitive_attr_t}, const_dnnl_primitive_attr_t),
        attr,
        existing_attr,
    )
end

function dnnl_primitive_attr_destroy(attr)
    return ccall(
        (:dnnl_primitive_attr_destroy, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_attr_t,),
        attr,
    )
end

function dnnl_primitive_attr_get_scratchpad_mode(attr, mode)
    return ccall(
        (:dnnl_primitive_attr_get_scratchpad_mode, libdnnl),
        dnnl_status_t,
        (const_dnnl_primitive_attr_t, Ptr{dnnl_scratchpad_mode_t}),
        attr,
        mode,
    )
end

function dnnl_primitive_attr_set_scratchpad_mode(attr, mode)
    return ccall(
        (:dnnl_primitive_attr_set_scratchpad_mode, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_attr_t, dnnl_scratchpad_mode_t),
        attr,
        mode,
    )
end

function dnnl_primitive_attr_get_output_scales(attr, count, mask, scales)
    return ccall(
        (:dnnl_primitive_attr_get_output_scales, libdnnl),
        dnnl_status_t,
        (const_dnnl_primitive_attr_t, Ptr{dnnl_dim_t}, Ptr{Cint}, Ptr{Ptr{Cfloat}}),
        attr,
        count,
        mask,
        scales,
    )
end

function dnnl_primitive_attr_set_output_scales(attr, count, mask, scales)
    return ccall(
        (:dnnl_primitive_attr_set_output_scales, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_attr_t, dnnl_dim_t, Cint, Ptr{Cfloat}),
        attr,
        count,
        mask,
        scales,
    )
end

function dnnl_primitive_attr_get_scales(attr, arg, count, mask, scales)
    return ccall(
        (:dnnl_primitive_attr_get_scales, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_attr_t, Cint, Ptr{dnnl_dim_t}, Ptr{Cint}, Ptr{Ptr{Cfloat}}),
        attr,
        arg,
        count,
        mask,
        scales,
    )
end

function dnnl_primitive_attr_set_scales(attr, arg, count, mask, scales)
    return ccall(
        (:dnnl_primitive_attr_set_scales, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_attr_t, Cint, dnnl_dim_t, Cint, Ptr{Cfloat}),
        attr,
        arg,
        count,
        mask,
        scales,
    )
end

function dnnl_primitive_attr_get_zero_points(attr, arg, count, mask, zero_points)
    return ccall(
        (:dnnl_primitive_attr_get_zero_points, libdnnl),
        dnnl_status_t,
        (const_dnnl_primitive_attr_t, Cint, Ptr{dnnl_dim_t}, Ptr{Cint}, Ptr{Ptr{Int32}}),
        attr,
        arg,
        count,
        mask,
        zero_points,
    )
end

function dnnl_primitive_attr_set_zero_points(attr, arg, count, mask, zero_points)
    return ccall(
        (:dnnl_primitive_attr_set_zero_points, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_attr_t, Cint, dnnl_dim_t, Cint, Ptr{Int32}),
        attr,
        arg,
        count,
        mask,
        zero_points,
    )
end

function dnnl_primitive_attr_get_post_ops(attr, post_ops)
    return ccall(
        (:dnnl_primitive_attr_get_post_ops, libdnnl),
        dnnl_status_t,
        (const_dnnl_primitive_attr_t, Ptr{const_dnnl_post_ops_t}),
        attr,
        post_ops,
    )
end

function dnnl_primitive_attr_set_post_ops(attr, post_ops)
    return ccall(
        (:dnnl_primitive_attr_set_post_ops, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_attr_t, const_dnnl_post_ops_t),
        attr,
        post_ops,
    )
end

function dnnl_post_ops_create(post_ops)
    return ccall(
        (:dnnl_post_ops_create, libdnnl), dnnl_status_t, (Ptr{dnnl_post_ops_t},), post_ops
    )
end

function dnnl_post_ops_destroy(post_ops)
    return ccall(
        (:dnnl_post_ops_destroy, libdnnl), dnnl_status_t, (dnnl_post_ops_t,), post_ops
    )
end

function dnnl_post_ops_len(post_ops)
    return ccall((:dnnl_post_ops_len, libdnnl), Cint, (const_dnnl_post_ops_t,), post_ops)
end

function dnnl_post_ops_get_kind(post_ops, index)
    return ccall(
        (:dnnl_post_ops_get_kind, libdnnl),
        dnnl_primitive_kind_t,
        (const_dnnl_post_ops_t, Cint),
        post_ops,
        index,
    )
end

function dnnl_post_ops_append_sum(post_ops, scale)
    return ccall(
        (:dnnl_post_ops_append_sum, libdnnl),
        dnnl_status_t,
        (dnnl_post_ops_t, Cfloat),
        post_ops,
        scale,
    )
end

function dnnl_post_ops_append_sum_v2(post_ops, scale, data_type)
    return ccall(
        (:dnnl_post_ops_append_sum_v2, libdnnl),
        dnnl_status_t,
        (dnnl_post_ops_t, Cfloat, dnnl_data_type_t),
        post_ops,
        scale,
        data_type,
    )
end

function dnnl_post_ops_get_params_sum(post_ops, index, scale)
    return ccall(
        (:dnnl_post_ops_get_params_sum, libdnnl),
        dnnl_status_t,
        (const_dnnl_post_ops_t, Cint, Ptr{Cfloat}),
        post_ops,
        index,
        scale,
    )
end

function dnnl_post_ops_get_params_sum_v2(post_ops, index, scale, data_type)
    return ccall(
        (:dnnl_post_ops_get_params_sum_v2, libdnnl),
        dnnl_status_t,
        (const_dnnl_post_ops_t, Cint, Ptr{Cfloat}, Ptr{dnnl_data_type_t}),
        post_ops,
        index,
        scale,
        data_type,
    )
end

function dnnl_post_ops_append_eltwise(post_ops, scale, alg_kind, alpha, beta)
    return ccall(
        (:dnnl_post_ops_append_eltwise, libdnnl),
        dnnl_status_t,
        (dnnl_post_ops_t, Cfloat, dnnl_alg_kind_t, Cfloat, Cfloat),
        post_ops,
        scale,
        alg_kind,
        alpha,
        beta,
    )
end

function dnnl_post_ops_get_params_eltwise(post_ops, index, scale, alg_kind, alpha, beta)
    return ccall(
        (:dnnl_post_ops_get_params_eltwise, libdnnl),
        dnnl_status_t,
        (
            const_dnnl_post_ops_t,
            Cint,
            Ptr{Cfloat},
            Ptr{dnnl_alg_kind_t},
            Ptr{Cfloat},
            Ptr{Cfloat},
        ),
        post_ops,
        index,
        scale,
        alg_kind,
        alpha,
        beta,
    )
end

function dnnl_post_ops_append_dw_k3s1p1(
    post_ops, weights_data_type, bias_data_type, dst_data_type, count, mask, scales
)
    return ccall(
        (:dnnl_post_ops_append_dw_k3s1p1, libdnnl),
        dnnl_status_t,
        (
            dnnl_post_ops_t,
            dnnl_data_type_t,
            dnnl_data_type_t,
            dnnl_data_type_t,
            dnnl_dim_t,
            Cint,
            Ptr{Cfloat},
        ),
        post_ops,
        weights_data_type,
        bias_data_type,
        dst_data_type,
        count,
        mask,
        scales,
    )
end

function dnnl_post_ops_get_params_dw_k3s1p1(
    post_ops, index, weights_data_type, bias_data_type, dst_data_type, count, mask, scales
)
    return ccall(
        (:dnnl_post_ops_get_params_dw_k3s1p1, libdnnl),
        dnnl_status_t,
        (
            const_dnnl_post_ops_t,
            Cint,
            Ptr{dnnl_data_type_t},
            Ptr{dnnl_data_type_t},
            Ptr{dnnl_data_type_t},
            Ptr{dnnl_dim_t},
            Ptr{Cint},
            Ptr{Ptr{Cfloat}},
        ),
        post_ops,
        index,
        weights_data_type,
        bias_data_type,
        dst_data_type,
        count,
        mask,
        scales,
    )
end

function dnnl_post_ops_append_dw_k3s2p1(
    post_ops, weights_data_type, bias_data_type, dst_data_type, count, mask, scales
)
    return ccall(
        (:dnnl_post_ops_append_dw_k3s2p1, libdnnl),
        dnnl_status_t,
        (
            dnnl_post_ops_t,
            dnnl_data_type_t,
            dnnl_data_type_t,
            dnnl_data_type_t,
            dnnl_dim_t,
            Cint,
            Ptr{Cfloat},
        ),
        post_ops,
        weights_data_type,
        bias_data_type,
        dst_data_type,
        count,
        mask,
        scales,
    )
end

function dnnl_post_ops_get_params_dw_k3s2p1(
    post_ops, index, weights_data_type, bias_data_type, dst_data_type, count, mask, scales
)
    return ccall(
        (:dnnl_post_ops_get_params_dw_k3s2p1, libdnnl),
        dnnl_status_t,
        (
            const_dnnl_post_ops_t,
            Cint,
            Ptr{dnnl_data_type_t},
            Ptr{dnnl_data_type_t},
            Ptr{dnnl_data_type_t},
            Ptr{dnnl_dim_t},
            Ptr{Cint},
            Ptr{Ptr{Cfloat}},
        ),
        post_ops,
        index,
        weights_data_type,
        bias_data_type,
        dst_data_type,
        count,
        mask,
        scales,
    )
end

function dnnl_post_ops_append_binary(post_ops, alg_kind, src1_desc)
    return ccall(
        (:dnnl_post_ops_append_binary, libdnnl),
        dnnl_status_t,
        (dnnl_post_ops_t, dnnl_alg_kind_t, Ptr{dnnl_memory_desc_t}),
        post_ops,
        alg_kind,
        src1_desc,
    )
end

function dnnl_post_ops_get_params_binary(post_ops, index, alg_kind, src1_desc)
    return ccall(
        (:dnnl_post_ops_get_params_binary, libdnnl),
        dnnl_status_t,
        (const_dnnl_post_ops_t, Cint, Ptr{dnnl_alg_kind_t}, Ptr{Ptr{dnnl_memory_desc_t}}),
        post_ops,
        index,
        alg_kind,
        src1_desc,
    )
end

function dnnl_memory_desc_init_by_strides(memory_desc, ndims, dims, data_type, strides)
    return ccall(
        (:dnnl_memory_desc_init_by_strides, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_memory_desc_t}, Cint, Ptr{Clong}, dnnl_data_type_t, Ptr{Clong}),
        memory_desc,
        ndims,
        dims,
        data_type,
        strides,
    )
end

function dnnl_memory_desc_init_by_tag(memory_desc, ndims, dims, data_type, tag)
    return ccall(
        (:dnnl_memory_desc_init_by_tag, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_memory_desc_t}, Cint, Ptr{Clong}, dnnl_data_type_t, dnnl_format_tag_t),
        memory_desc,
        ndims,
        dims,
        data_type,
        tag,
    )
end

function dnnl_memory_desc_init_submemory(memory_desc, parent_memory_desc, dims, offsets)
    return ccall(
        (:dnnl_memory_desc_init_submemory, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{Clong}, Ptr{Clong}),
        memory_desc,
        parent_memory_desc,
        dims,
        offsets,
    )
end

function dnnl_memory_desc_reshape(out_memory_desc, in_memory_desc, ndims, dims)
    return ccall(
        (:dnnl_memory_desc_reshape, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Cint, Ptr{Clong}),
        out_memory_desc,
        in_memory_desc,
        ndims,
        dims,
    )
end

function dnnl_memory_desc_permute_axes(out_memory_desc, in_memory_desc, permutation)
    return ccall(
        (:dnnl_memory_desc_permute_axes, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{Cint}),
        out_memory_desc,
        in_memory_desc,
        permutation,
    )
end

function dnnl_memory_desc_equal(lhs, rhs)
    return ccall(
        (:dnnl_memory_desc_equal, libdnnl),
        Cint,
        (Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}),
        lhs,
        rhs,
    )
end

function dnnl_memory_desc_get_size(memory_desc)
    return ccall(
        (:dnnl_memory_desc_get_size, libdnnl),
        Csize_t,
        (Ptr{dnnl_memory_desc_t},),
        memory_desc,
    )
end

function dnnl_data_type_size(data_type)
    return ccall((:dnnl_data_type_size, libdnnl), Csize_t, (dnnl_data_type_t,), data_type)
end

function dnnl_memory_create(memory, memory_desc, engine, handle)
    return ccall(
        (:dnnl_memory_create, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_memory_t}, Ptr{dnnl_memory_desc_t}, dnnl_engine_t, Ptr{Cvoid}),
        memory,
        memory_desc,
        engine,
        handle,
    )
end

function dnnl_memory_get_memory_desc(memory, memory_desc)
    return ccall(
        (:dnnl_memory_get_memory_desc, libdnnl),
        dnnl_status_t,
        (const_dnnl_memory_t, Ptr{Ptr{dnnl_memory_desc_t}}),
        memory,
        memory_desc,
    )
end

function dnnl_memory_get_engine(memory, engine)
    return ccall(
        (:dnnl_memory_get_engine, libdnnl),
        dnnl_status_t,
        (const_dnnl_memory_t, Ptr{dnnl_engine_t}),
        memory,
        engine,
    )
end

function dnnl_memory_map_data(memory, mapped_ptr)
    return ccall(
        (:dnnl_memory_map_data, libdnnl),
        dnnl_status_t,
        (const_dnnl_memory_t, Ptr{Ptr{Cvoid}}),
        memory,
        mapped_ptr,
    )
end

function dnnl_memory_unmap_data(memory, mapped_ptr)
    return ccall(
        (:dnnl_memory_unmap_data, libdnnl),
        dnnl_status_t,
        (const_dnnl_memory_t, Ptr{Cvoid}),
        memory,
        mapped_ptr,
    )
end

function dnnl_memory_get_data_handle(memory, handle)
    return ccall(
        (:dnnl_memory_get_data_handle, libdnnl),
        dnnl_status_t,
        (const_dnnl_memory_t, Ptr{Ptr{Cvoid}}),
        memory,
        handle,
    )
end

function dnnl_memory_set_data_handle(memory, handle)
    return ccall(
        (:dnnl_memory_set_data_handle, libdnnl),
        dnnl_status_t,
        (dnnl_memory_t, Ptr{Cvoid}),
        memory,
        handle,
    )
end

function dnnl_memory_set_data_handle_v2(memory, handle, stream)
    return ccall(
        (:dnnl_memory_set_data_handle_v2, libdnnl),
        dnnl_status_t,
        (dnnl_memory_t, Ptr{Cvoid}, dnnl_stream_t),
        memory,
        handle,
        stream,
    )
end

function dnnl_memory_destroy(memory)
    return ccall((:dnnl_memory_destroy, libdnnl), dnnl_status_t, (dnnl_memory_t,), memory)
end

function dnnl_reorder_primitive_desc_create(
    reorder_primitive_desc, src_desc, src_engine, dst_desc, dst_engine, attr
)
    return ccall(
        (:dnnl_reorder_primitive_desc_create, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_primitive_desc_t},
            Ptr{dnnl_memory_desc_t},
            dnnl_engine_t,
            Ptr{dnnl_memory_desc_t},
            dnnl_engine_t,
            const_dnnl_primitive_attr_t,
        ),
        reorder_primitive_desc,
        src_desc,
        src_engine,
        dst_desc,
        dst_engine,
        attr,
    )
end

function dnnl_concat_primitive_desc_create(
    concat_primitive_desc, dst_desc, n, concat_dimension, src_descs, attr, engine
)
    return ccall(
        (:dnnl_concat_primitive_desc_create, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_primitive_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cint,
            Cint,
            Ptr{dnnl_memory_desc_t},
            const_dnnl_primitive_attr_t,
            dnnl_engine_t,
        ),
        concat_primitive_desc,
        dst_desc,
        n,
        concat_dimension,
        src_descs,
        attr,
        engine,
    )
end

function dnnl_sum_primitive_desc_create(
    sum_primitive_desc, dst_desc, n, scales, src_descs, attr, engine
)
    return ccall(
        (:dnnl_sum_primitive_desc_create, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_primitive_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cint,
            Ptr{Cfloat},
            Ptr{dnnl_memory_desc_t},
            const_dnnl_primitive_attr_t,
            dnnl_engine_t,
        ),
        sum_primitive_desc,
        dst_desc,
        n,
        scales,
        src_descs,
        attr,
        engine,
    )
end

function dnnl_binary_desc_init(binary_desc, alg_kind, src0_desc, src1_desc, dst_desc)
    return ccall(
        (:dnnl_binary_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_binary_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
        ),
        binary_desc,
        alg_kind,
        src0_desc,
        src1_desc,
        dst_desc,
    )
end

function dnnl_convolution_forward_desc_init(
    conv_desc,
    prop_kind,
    alg_kind,
    src_desc,
    weights_desc,
    bias_desc,
    dst_desc,
    strides,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_convolution_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_convolution_desc_t},
            dnnl_prop_kind_t,
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        conv_desc,
        prop_kind,
        alg_kind,
        src_desc,
        weights_desc,
        bias_desc,
        dst_desc,
        strides,
        padding_l,
        padding_r,
    )
end

function dnnl_dilated_convolution_forward_desc_init(
    conv_desc,
    prop_kind,
    alg_kind,
    src_desc,
    weights_desc,
    bias_desc,
    dst_desc,
    strides,
    dilates,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_dilated_convolution_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_convolution_desc_t},
            dnnl_prop_kind_t,
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        conv_desc,
        prop_kind,
        alg_kind,
        src_desc,
        weights_desc,
        bias_desc,
        dst_desc,
        strides,
        dilates,
        padding_l,
        padding_r,
    )
end

function dnnl_convolution_backward_data_desc_init(
    conv_desc,
    alg_kind,
    diff_src_desc,
    weights_desc,
    diff_dst_desc,
    strides,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_convolution_backward_data_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_convolution_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        conv_desc,
        alg_kind,
        diff_src_desc,
        weights_desc,
        diff_dst_desc,
        strides,
        padding_l,
        padding_r,
    )
end

function dnnl_dilated_convolution_backward_data_desc_init(
    conv_desc,
    alg_kind,
    diff_src_desc,
    weights_desc,
    diff_dst_desc,
    strides,
    dilates,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_dilated_convolution_backward_data_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_convolution_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        conv_desc,
        alg_kind,
        diff_src_desc,
        weights_desc,
        diff_dst_desc,
        strides,
        dilates,
        padding_l,
        padding_r,
    )
end

function dnnl_convolution_backward_weights_desc_init(
    conv_desc,
    alg_kind,
    src_desc,
    diff_weights_desc,
    diff_bias_desc,
    diff_dst_desc,
    strides,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_convolution_backward_weights_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_convolution_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        conv_desc,
        alg_kind,
        src_desc,
        diff_weights_desc,
        diff_bias_desc,
        diff_dst_desc,
        strides,
        padding_l,
        padding_r,
    )
end

function dnnl_dilated_convolution_backward_weights_desc_init(
    conv_desc,
    alg_kind,
    src_desc,
    diff_weights_desc,
    diff_bias_desc,
    diff_dst_desc,
    strides,
    dilates,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_dilated_convolution_backward_weights_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_convolution_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        conv_desc,
        alg_kind,
        src_desc,
        diff_weights_desc,
        diff_bias_desc,
        diff_dst_desc,
        strides,
        dilates,
        padding_l,
        padding_r,
    )
end

function dnnl_deconvolution_forward_desc_init(
    deconv_desc,
    prop_kind,
    alg_kind,
    src_desc,
    weights_desc,
    bias_desc,
    dst_desc,
    strides,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_deconvolution_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_deconvolution_desc_t},
            dnnl_prop_kind_t,
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        deconv_desc,
        prop_kind,
        alg_kind,
        src_desc,
        weights_desc,
        bias_desc,
        dst_desc,
        strides,
        padding_l,
        padding_r,
    )
end

function dnnl_dilated_deconvolution_forward_desc_init(
    deconv_desc,
    prop_kind,
    alg_kind,
    src_desc,
    weights_desc,
    bias_desc,
    dst_desc,
    strides,
    dilates,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_dilated_deconvolution_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_deconvolution_desc_t},
            dnnl_prop_kind_t,
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        deconv_desc,
        prop_kind,
        alg_kind,
        src_desc,
        weights_desc,
        bias_desc,
        dst_desc,
        strides,
        dilates,
        padding_l,
        padding_r,
    )
end

function dnnl_deconvolution_backward_data_desc_init(
    deconv_desc,
    alg_kind,
    diff_src_desc,
    weights_desc,
    diff_dst_desc,
    strides,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_deconvolution_backward_data_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_deconvolution_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        deconv_desc,
        alg_kind,
        diff_src_desc,
        weights_desc,
        diff_dst_desc,
        strides,
        padding_l,
        padding_r,
    )
end

function dnnl_dilated_deconvolution_backward_data_desc_init(
    deconv_desc,
    alg_kind,
    diff_src_desc,
    weights_desc,
    diff_dst_desc,
    strides,
    dilates,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_dilated_deconvolution_backward_data_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_deconvolution_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        deconv_desc,
        alg_kind,
        diff_src_desc,
        weights_desc,
        diff_dst_desc,
        strides,
        dilates,
        padding_l,
        padding_r,
    )
end

function dnnl_deconvolution_backward_weights_desc_init(
    deconv_desc,
    alg_kind,
    src_desc,
    diff_weights_desc,
    diff_bias_desc,
    diff_dst_desc,
    strides,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_deconvolution_backward_weights_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_deconvolution_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        deconv_desc,
        alg_kind,
        src_desc,
        diff_weights_desc,
        diff_bias_desc,
        diff_dst_desc,
        strides,
        padding_l,
        padding_r,
    )
end

function dnnl_dilated_deconvolution_backward_weights_desc_init(
    deconv_desc,
    alg_kind,
    src_desc,
    diff_weights_desc,
    diff_bias_desc,
    diff_dst_desc,
    strides,
    dilates,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_dilated_deconvolution_backward_weights_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_deconvolution_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        deconv_desc,
        alg_kind,
        src_desc,
        diff_weights_desc,
        diff_bias_desc,
        diff_dst_desc,
        strides,
        dilates,
        padding_l,
        padding_r,
    )
end

function dnnl_shuffle_forward_desc_init(
    shuffle_desc, prop_kind, data_desc, axis, group_size
)
    return ccall(
        (:dnnl_shuffle_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_shuffle_desc_t},
            dnnl_prop_kind_t,
            Ptr{dnnl_memory_desc_t},
            Cint,
            dnnl_dim_t,
        ),
        shuffle_desc,
        prop_kind,
        data_desc,
        axis,
        group_size,
    )
end

function dnnl_shuffle_backward_desc_init(shuffle_desc, diff_data_desc, axis, group_size)
    return ccall(
        (:dnnl_shuffle_backward_desc_init, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_shuffle_desc_t}, Ptr{dnnl_memory_desc_t}, Cint, dnnl_dim_t),
        shuffle_desc,
        diff_data_desc,
        axis,
        group_size,
    )
end

function dnnl_eltwise_forward_desc_init(
    eltwise_desc, prop_kind, alg_kind, data_desc, alpha, beta
)
    return ccall(
        (:dnnl_eltwise_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_eltwise_desc_t},
            dnnl_prop_kind_t,
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Cfloat,
            Cfloat,
        ),
        eltwise_desc,
        prop_kind,
        alg_kind,
        data_desc,
        alpha,
        beta,
    )
end

function dnnl_eltwise_backward_desc_init(
    eltwise_desc, alg_kind, diff_data_desc, data_desc, alpha, beta
)
    return ccall(
        (:dnnl_eltwise_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_eltwise_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cfloat,
            Cfloat,
        ),
        eltwise_desc,
        alg_kind,
        diff_data_desc,
        data_desc,
        alpha,
        beta,
    )
end

function dnnl_softmax_forward_desc_init(softmax_desc, prop_kind, data_desc, softmax_axis)
    return ccall(
        (:dnnl_softmax_forward_desc_init, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_softmax_desc_t}, dnnl_prop_kind_t, Ptr{dnnl_memory_desc_t}, Cint),
        softmax_desc,
        prop_kind,
        data_desc,
        softmax_axis,
    )
end

function dnnl_softmax_backward_desc_init(
    softmax_desc, diff_data_desc, data_desc, softmax_axis
)
    return ccall(
        (:dnnl_softmax_backward_desc_init, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_softmax_desc_t}, Ptr{dnnl_memory_desc_t}, Ptr{dnnl_memory_desc_t}, Cint),
        softmax_desc,
        diff_data_desc,
        data_desc,
        softmax_axis,
    )
end

function dnnl_logsoftmax_forward_desc_init(
    logsoftmax_desc, prop_kind, data_desc, logsoftmax_axis
)
    return ccall(
        (:dnnl_logsoftmax_forward_desc_init, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_logsoftmax_desc_t}, dnnl_prop_kind_t, Ptr{dnnl_memory_desc_t}, Cint),
        logsoftmax_desc,
        prop_kind,
        data_desc,
        logsoftmax_axis,
    )
end

function dnnl_logsoftmax_backward_desc_init(
    logsoftmax_desc, diff_data_desc, data_desc, logsoftmax_axis
)
    return ccall(
        (:dnnl_logsoftmax_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_logsoftmax_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cint,
        ),
        logsoftmax_desc,
        diff_data_desc,
        data_desc,
        logsoftmax_axis,
    )
end

function dnnl_pooling_forward_desc_init(
    pool_desc,
    prop_kind,
    alg_kind,
    src_desc,
    dst_desc,
    strides,
    kernel,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_pooling_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_pooling_desc_t},
            dnnl_prop_kind_t,
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        pool_desc,
        prop_kind,
        alg_kind,
        src_desc,
        dst_desc,
        strides,
        kernel,
        padding_l,
        padding_r,
    )
end

function dnnl_pooling_backward_desc_init(
    pool_desc, alg_kind, diff_src_desc, diff_dst_desc, strides, kernel, padding_l, padding_r
)
    return ccall(
        (:dnnl_pooling_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_pooling_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        pool_desc,
        alg_kind,
        diff_src_desc,
        diff_dst_desc,
        strides,
        kernel,
        padding_l,
        padding_r,
    )
end

function dnnl_pooling_v2_forward_desc_init(
    pool_desc,
    prop_kind,
    alg_kind,
    src_desc,
    dst_desc,
    strides,
    kernel,
    dilation,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_pooling_v2_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_pooling_v2_desc_t},
            dnnl_prop_kind_t,
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        pool_desc,
        prop_kind,
        alg_kind,
        src_desc,
        dst_desc,
        strides,
        kernel,
        dilation,
        padding_l,
        padding_r,
    )
end

function dnnl_pooling_v2_backward_desc_init(
    pool_desc,
    alg_kind,
    diff_src_desc,
    diff_dst_desc,
    strides,
    kernel,
    dilation,
    padding_l,
    padding_r,
)
    return ccall(
        (:dnnl_pooling_v2_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_pooling_v2_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
            Ptr{Clong},
        ),
        pool_desc,
        alg_kind,
        diff_src_desc,
        diff_dst_desc,
        strides,
        kernel,
        dilation,
        padding_l,
        padding_r,
    )
end

function dnnl_prelu_forward_desc_init(prelu_desc, prop_kind, data_desc, weights_desc)
    return ccall(
        (:dnnl_prelu_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_prelu_desc_t},
            dnnl_prop_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
        ),
        prelu_desc,
        prop_kind,
        data_desc,
        weights_desc,
    )
end

function dnnl_prelu_backward_desc_init(
    prelu_desc, data_desc, weights_desc, diff_data_desc, diff_weights_desc
)
    return ccall(
        (:dnnl_prelu_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_prelu_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
        ),
        prelu_desc,
        data_desc,
        weights_desc,
        diff_data_desc,
        diff_weights_desc,
    )
end

function dnnl_lrn_forward_desc_init(
    lrn_desc, prop_kind, alg_kind, data_desc, local_size, alpha, beta, k
)
    return ccall(
        (:dnnl_lrn_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_lrn_desc_t},
            dnnl_prop_kind_t,
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            dnnl_dim_t,
            Cfloat,
            Cfloat,
            Cfloat,
        ),
        lrn_desc,
        prop_kind,
        alg_kind,
        data_desc,
        local_size,
        alpha,
        beta,
        k,
    )
end

function dnnl_lrn_backward_desc_init(
    lrn_desc, alg_kind, diff_data_desc, data_desc, local_size, alpha, beta, k
)
    return ccall(
        (:dnnl_lrn_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_lrn_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            dnnl_dim_t,
            Cfloat,
            Cfloat,
            Cfloat,
        ),
        lrn_desc,
        alg_kind,
        diff_data_desc,
        data_desc,
        local_size,
        alpha,
        beta,
        k,
    )
end

function dnnl_batch_normalization_forward_desc_init(
    bnrm_desc, prop_kind, data_desc, epsilon, flags
)
    return ccall(
        (:dnnl_batch_normalization_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_batch_normalization_desc_t},
            dnnl_prop_kind_t,
            Ptr{dnnl_memory_desc_t},
            Cfloat,
            Cuint,
        ),
        bnrm_desc,
        prop_kind,
        data_desc,
        epsilon,
        flags,
    )
end

function dnnl_batch_normalization_backward_desc_init(
    bnrm_desc, prop_kind, diff_data_desc, data_desc, epsilon, flags
)
    return ccall(
        (:dnnl_batch_normalization_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_batch_normalization_desc_t},
            dnnl_prop_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cfloat,
            Cuint,
        ),
        bnrm_desc,
        prop_kind,
        diff_data_desc,
        data_desc,
        epsilon,
        flags,
    )
end

function dnnl_layer_normalization_forward_desc_init(
    lnrm_desc, prop_kind, data_desc, stat_desc, epsilon, flags
)
    return ccall(
        (:dnnl_layer_normalization_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_layer_normalization_desc_t},
            dnnl_prop_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cfloat,
            Cuint,
        ),
        lnrm_desc,
        prop_kind,
        data_desc,
        stat_desc,
        epsilon,
        flags,
    )
end

function dnnl_layer_normalization_backward_desc_init(
    lnrm_desc, prop_kind, diff_data_desc, data_desc, stat_desc, epsilon, flags
)
    return ccall(
        (:dnnl_layer_normalization_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_layer_normalization_desc_t},
            dnnl_prop_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cfloat,
            Cuint,
        ),
        lnrm_desc,
        prop_kind,
        diff_data_desc,
        data_desc,
        stat_desc,
        epsilon,
        flags,
    )
end

function dnnl_inner_product_forward_desc_init(
    ip_desc, prop_kind, src_desc, weights_desc, bias_desc, dst_desc
)
    return ccall(
        (:dnnl_inner_product_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_inner_product_desc_t},
            dnnl_prop_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
        ),
        ip_desc,
        prop_kind,
        src_desc,
        weights_desc,
        bias_desc,
        dst_desc,
    )
end

function dnnl_inner_product_backward_data_desc_init(
    ip_desc, diff_src_desc, weights_desc, diff_dst_desc
)
    return ccall(
        (:dnnl_inner_product_backward_data_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_inner_product_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
        ),
        ip_desc,
        diff_src_desc,
        weights_desc,
        diff_dst_desc,
    )
end

function dnnl_inner_product_backward_weights_desc_init(
    ip_desc, src_desc, diff_weights_desc, diff_bias_desc, diff_dst_desc
)
    return ccall(
        (:dnnl_inner_product_backward_weights_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_inner_product_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
        ),
        ip_desc,
        src_desc,
        diff_weights_desc,
        diff_bias_desc,
        diff_dst_desc,
    )
end

function dnnl_primitive_attr_set_rnn_data_qparams(attr, scale, shift)
    return ccall(
        (:dnnl_primitive_attr_set_rnn_data_qparams, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_attr_t, Cfloat, Cfloat),
        attr,
        scale,
        shift,
    )
end

function dnnl_primitive_attr_get_rnn_data_qparams(attr, scale, shift)
    return ccall(
        (:dnnl_primitive_attr_get_rnn_data_qparams, libdnnl),
        dnnl_status_t,
        (const_dnnl_primitive_attr_t, Ptr{Cfloat}, Ptr{Cfloat}),
        attr,
        scale,
        shift,
    )
end

function dnnl_primitive_attr_set_rnn_weights_qparams(attr, count, mask, scales)
    return ccall(
        (:dnnl_primitive_attr_set_rnn_weights_qparams, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_attr_t, dnnl_dim_t, Cint, Ptr{Cfloat}),
        attr,
        count,
        mask,
        scales,
    )
end

function dnnl_primitive_attr_get_rnn_weights_qparams(attr, count, mask, scales)
    return ccall(
        (:dnnl_primitive_attr_get_rnn_weights_qparams, libdnnl),
        dnnl_status_t,
        (const_dnnl_primitive_attr_t, Ptr{dnnl_dim_t}, Ptr{Cint}, Ptr{Ptr{Cfloat}}),
        attr,
        count,
        mask,
        scales,
    )
end

function dnnl_primitive_attr_set_rnn_weights_projection_qparams(attr, count, mask, scales)
    return ccall(
        (:dnnl_primitive_attr_set_rnn_weights_projection_qparams, libdnnl),
        dnnl_status_t,
        (dnnl_primitive_attr_t, dnnl_dim_t, Cint, Ptr{Cfloat}),
        attr,
        count,
        mask,
        scales,
    )
end

function dnnl_primitive_attr_get_rnn_weights_projection_qparams(attr, count, mask, scales)
    return ccall(
        (:dnnl_primitive_attr_get_rnn_weights_projection_qparams, libdnnl),
        dnnl_status_t,
        (const_dnnl_primitive_attr_t, Ptr{dnnl_dim_t}, Ptr{Cint}, Ptr{Ptr{Cfloat}}),
        attr,
        count,
        mask,
        scales,
    )
end

function dnnl_vanilla_rnn_forward_desc_init(
    rnn_desc,
    prop_kind,
    activation,
    direction,
    src_layer_desc,
    src_iter_desc,
    weights_layer_desc,
    weights_iter_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    flags,
    alpha,
    beta,
)
    return ccall(
        (:dnnl_vanilla_rnn_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_alg_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
            Cfloat,
            Cfloat,
        ),
        rnn_desc,
        prop_kind,
        activation,
        direction,
        src_layer_desc,
        src_iter_desc,
        weights_layer_desc,
        weights_iter_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        flags,
        alpha,
        beta,
    )
end

function dnnl_vanilla_rnn_backward_desc_init(
    rnn_desc,
    prop_kind,
    activation,
    direction,
    src_layer_desc,
    src_iter_desc,
    weights_layer_desc,
    weights_iter_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    diff_src_layer_desc,
    diff_src_iter_desc,
    diff_weights_layer_desc,
    diff_weights_iter_desc,
    diff_bias_desc,
    diff_dst_layer_desc,
    diff_dst_iter_desc,
    flags,
    alpha,
    beta,
)
    return ccall(
        (:dnnl_vanilla_rnn_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_alg_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
            Cfloat,
            Cfloat,
        ),
        rnn_desc,
        prop_kind,
        activation,
        direction,
        src_layer_desc,
        src_iter_desc,
        weights_layer_desc,
        weights_iter_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        diff_src_layer_desc,
        diff_src_iter_desc,
        diff_weights_layer_desc,
        diff_weights_iter_desc,
        diff_bias_desc,
        diff_dst_layer_desc,
        diff_dst_iter_desc,
        flags,
        alpha,
        beta,
    )
end

function dnnl_lstm_forward_desc_init(
    rnn_desc,
    prop_kind,
    direction,
    src_layer_desc,
    src_iter_desc,
    src_iter_c_desc,
    weights_layer_desc,
    weights_iter_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    dst_iter_c_desc,
    flags,
)
    return ccall(
        (:dnnl_lstm_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
        ),
        rnn_desc,
        prop_kind,
        direction,
        src_layer_desc,
        src_iter_desc,
        src_iter_c_desc,
        weights_layer_desc,
        weights_iter_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        dst_iter_c_desc,
        flags,
    )
end

function dnnl_lstm_forward_desc_init_v2(
    rnn_desc,
    prop_kind,
    direction,
    src_layer_desc,
    src_iter_desc,
    src_iter_c_desc,
    weights_layer_desc,
    weights_iter_desc,
    weights_peephole_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    dst_iter_c_desc,
    flags,
)
    return ccall(
        (:dnnl_lstm_forward_desc_init_v2, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
        ),
        rnn_desc,
        prop_kind,
        direction,
        src_layer_desc,
        src_iter_desc,
        src_iter_c_desc,
        weights_layer_desc,
        weights_iter_desc,
        weights_peephole_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        dst_iter_c_desc,
        flags,
    )
end

function dnnl_lstm_forward_desc_init_v3(
    rnn_desc,
    prop_kind,
    direction,
    src_layer_desc,
    src_iter_desc,
    src_iter_c_desc,
    weights_layer_desc,
    weights_iter_desc,
    weights_peephole_desc,
    weights_projection_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    dst_iter_c_desc,
    flags,
)
    return ccall(
        (:dnnl_lstm_forward_desc_init_v3, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
        ),
        rnn_desc,
        prop_kind,
        direction,
        src_layer_desc,
        src_iter_desc,
        src_iter_c_desc,
        weights_layer_desc,
        weights_iter_desc,
        weights_peephole_desc,
        weights_projection_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        dst_iter_c_desc,
        flags,
    )
end

function dnnl_lstm_backward_desc_init(
    rnn_desc,
    prop_kind,
    direction,
    src_layer_desc,
    src_iter_desc,
    src_iter_c_desc,
    weights_layer_desc,
    weights_iter_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    dst_iter_c_desc,
    diff_src_layer_desc,
    diff_src_iter_desc,
    diff_src_iter_c_desc,
    diff_weights_layer_desc,
    diff_weights_iter_desc,
    diff_bias_desc,
    diff_dst_layer_desc,
    diff_dst_iter_desc,
    diff_dst_iter_c_desc,
    flags,
)
    return ccall(
        (:dnnl_lstm_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
        ),
        rnn_desc,
        prop_kind,
        direction,
        src_layer_desc,
        src_iter_desc,
        src_iter_c_desc,
        weights_layer_desc,
        weights_iter_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        dst_iter_c_desc,
        diff_src_layer_desc,
        diff_src_iter_desc,
        diff_src_iter_c_desc,
        diff_weights_layer_desc,
        diff_weights_iter_desc,
        diff_bias_desc,
        diff_dst_layer_desc,
        diff_dst_iter_desc,
        diff_dst_iter_c_desc,
        flags,
    )
end

function dnnl_lstm_backward_desc_init_v2(
    rnn_desc,
    prop_kind,
    direction,
    src_layer_desc,
    src_iter_desc,
    src_iter_c_desc,
    weights_layer_desc,
    weights_iter_desc,
    weights_peephole_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    dst_iter_c_desc,
    diff_src_layer_desc,
    diff_src_iter_desc,
    diff_src_iter_c_desc,
    diff_weights_layer_desc,
    diff_weights_iter_desc,
    diff_weights_peephole_desc,
    diff_bias_desc,
    diff_dst_layer_desc,
    diff_dst_iter_desc,
    diff_dst_iter_c_desc,
    flags,
)
    return ccall(
        (:dnnl_lstm_backward_desc_init_v2, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
        ),
        rnn_desc,
        prop_kind,
        direction,
        src_layer_desc,
        src_iter_desc,
        src_iter_c_desc,
        weights_layer_desc,
        weights_iter_desc,
        weights_peephole_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        dst_iter_c_desc,
        diff_src_layer_desc,
        diff_src_iter_desc,
        diff_src_iter_c_desc,
        diff_weights_layer_desc,
        diff_weights_iter_desc,
        diff_weights_peephole_desc,
        diff_bias_desc,
        diff_dst_layer_desc,
        diff_dst_iter_desc,
        diff_dst_iter_c_desc,
        flags,
    )
end

function dnnl_lstm_backward_desc_init_v3(
    rnn_desc,
    prop_kind,
    direction,
    src_layer_desc,
    src_iter_desc,
    src_iter_c_desc,
    weights_layer_desc,
    weights_iter_desc,
    weights_peephole_desc,
    weights_projection_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    dst_iter_c_desc,
    diff_src_layer_desc,
    diff_src_iter_desc,
    diff_src_iter_c_desc,
    diff_weights_layer_desc,
    diff_weights_iter_desc,
    diff_weights_peephole_desc,
    diff_weights_projection_desc,
    diff_bias_desc,
    diff_dst_layer_desc,
    diff_dst_iter_desc,
    diff_dst_iter_c_desc,
    flags,
)
    return ccall(
        (:dnnl_lstm_backward_desc_init_v3, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
        ),
        rnn_desc,
        prop_kind,
        direction,
        src_layer_desc,
        src_iter_desc,
        src_iter_c_desc,
        weights_layer_desc,
        weights_iter_desc,
        weights_peephole_desc,
        weights_projection_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        dst_iter_c_desc,
        diff_src_layer_desc,
        diff_src_iter_desc,
        diff_src_iter_c_desc,
        diff_weights_layer_desc,
        diff_weights_iter_desc,
        diff_weights_peephole_desc,
        diff_weights_projection_desc,
        diff_bias_desc,
        diff_dst_layer_desc,
        diff_dst_iter_desc,
        diff_dst_iter_c_desc,
        flags,
    )
end

function dnnl_gru_forward_desc_init(
    rnn_desc,
    prop_kind,
    direction,
    src_layer_desc,
    src_iter_desc,
    weights_layer_desc,
    weights_iter_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    flags,
)
    return ccall(
        (:dnnl_gru_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
        ),
        rnn_desc,
        prop_kind,
        direction,
        src_layer_desc,
        src_iter_desc,
        weights_layer_desc,
        weights_iter_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        flags,
    )
end

function dnnl_gru_backward_desc_init(
    rnn_desc,
    prop_kind,
    direction,
    src_layer_desc,
    src_iter_desc,
    weights_layer_desc,
    weights_iter_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    diff_src_layer_desc,
    diff_src_iter_desc,
    diff_weights_layer_desc,
    diff_weights_iter_desc,
    diff_bias_desc,
    diff_dst_layer_desc,
    diff_dst_iter_desc,
    flags,
)
    return ccall(
        (:dnnl_gru_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
        ),
        rnn_desc,
        prop_kind,
        direction,
        src_layer_desc,
        src_iter_desc,
        weights_layer_desc,
        weights_iter_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        diff_src_layer_desc,
        diff_src_iter_desc,
        diff_weights_layer_desc,
        diff_weights_iter_desc,
        diff_bias_desc,
        diff_dst_layer_desc,
        diff_dst_iter_desc,
        flags,
    )
end

function dnnl_lbr_gru_forward_desc_init(
    rnn_desc,
    prop_kind,
    direction,
    src_layer_desc,
    src_iter_desc,
    weights_layer_desc,
    weights_iter_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    flags,
)
    return ccall(
        (:dnnl_lbr_gru_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
        ),
        rnn_desc,
        prop_kind,
        direction,
        src_layer_desc,
        src_iter_desc,
        weights_layer_desc,
        weights_iter_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        flags,
    )
end

function dnnl_lbr_gru_backward_desc_init(
    rnn_desc,
    prop_kind,
    direction,
    src_layer_desc,
    src_iter_desc,
    weights_layer_desc,
    weights_iter_desc,
    bias_desc,
    dst_layer_desc,
    dst_iter_desc,
    diff_src_layer_desc,
    diff_src_iter_desc,
    diff_weights_layer_desc,
    diff_weights_iter_desc,
    diff_bias_desc,
    diff_dst_layer_desc,
    diff_dst_iter_desc,
    flags,
)
    return ccall(
        (:dnnl_lbr_gru_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_rnn_desc_t},
            dnnl_prop_kind_t,
            dnnl_rnn_direction_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cuint,
        ),
        rnn_desc,
        prop_kind,
        direction,
        src_layer_desc,
        src_iter_desc,
        weights_layer_desc,
        weights_iter_desc,
        bias_desc,
        dst_layer_desc,
        dst_iter_desc,
        diff_src_layer_desc,
        diff_src_iter_desc,
        diff_weights_layer_desc,
        diff_weights_iter_desc,
        diff_bias_desc,
        diff_dst_layer_desc,
        diff_dst_iter_desc,
        flags,
    )
end

function dnnl_matmul_desc_init(matmul_desc, src_desc, weights_desc, bias_desc, dst_desc)
    return ccall(
        (:dnnl_matmul_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_matmul_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
        ),
        matmul_desc,
        src_desc,
        weights_desc,
        bias_desc,
        dst_desc,
    )
end

function dnnl_resampling_forward_desc_init(
    resampling_desc, prop_kind, alg_kind, factors, src_desc, dst_desc
)
    return ccall(
        (:dnnl_resampling_forward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_resampling_desc_t},
            dnnl_prop_kind_t,
            dnnl_alg_kind_t,
            Ptr{Cfloat},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
        ),
        resampling_desc,
        prop_kind,
        alg_kind,
        factors,
        src_desc,
        dst_desc,
    )
end

function dnnl_resampling_backward_desc_init(
    resampling_desc, alg_kind, factors, diff_src_desc, diff_dst_desc
)
    return ccall(
        (:dnnl_resampling_backward_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_resampling_desc_t},
            dnnl_alg_kind_t,
            Ptr{Cfloat},
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
        ),
        resampling_desc,
        alg_kind,
        factors,
        diff_src_desc,
        diff_dst_desc,
    )
end

function dnnl_reduction_desc_init(desc, alg_kind, src_desc, dst_desc, p, eps)
    return ccall(
        (:dnnl_reduction_desc_init, libdnnl),
        dnnl_status_t,
        (
            Ptr{dnnl_reduction_desc_t},
            dnnl_alg_kind_t,
            Ptr{dnnl_memory_desc_t},
            Ptr{dnnl_memory_desc_t},
            Cfloat,
            Cfloat,
        ),
        desc,
        alg_kind,
        src_desc,
        dst_desc,
        p,
        eps,
    )
end

function dnnl_engine_get_count(kind)
    return ccall((:dnnl_engine_get_count, libdnnl), Csize_t, (dnnl_engine_kind_t,), kind)
end

function dnnl_engine_create(engine, kind, index)
    return ccall(
        (:dnnl_engine_create, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_engine_t}, dnnl_engine_kind_t, Csize_t),
        engine,
        kind,
        index,
    )
end

function dnnl_engine_get_kind(engine, kind)
    return ccall(
        (:dnnl_engine_get_kind, libdnnl),
        dnnl_status_t,
        (dnnl_engine_t, Ptr{dnnl_engine_kind_t}),
        engine,
        kind,
    )
end

function dnnl_engine_destroy(engine)
    return ccall((:dnnl_engine_destroy, libdnnl), dnnl_status_t, (dnnl_engine_t,), engine)
end

function dnnl_stream_create(stream, engine, flags)
    return ccall(
        (:dnnl_stream_create, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_stream_t}, dnnl_engine_t, Cuint),
        stream,
        engine,
        flags,
    )
end

function dnnl_stream_get_engine(stream, engine)
    return ccall(
        (:dnnl_stream_get_engine, libdnnl),
        dnnl_status_t,
        (const_dnnl_stream_t, Ptr{dnnl_engine_t}),
        stream,
        engine,
    )
end

function dnnl_stream_wait(stream)
    return ccall((:dnnl_stream_wait, libdnnl), dnnl_status_t, (dnnl_stream_t,), stream)
end

function dnnl_stream_destroy(stream)
    return ccall((:dnnl_stream_destroy, libdnnl), dnnl_status_t, (dnnl_stream_t,), stream)
end

function dnnl_get_primitive_cache_capacity(capacity)
    return ccall(
        (:dnnl_get_primitive_cache_capacity, libdnnl), dnnl_status_t, (Ptr{Cint},), capacity
    )
end

function dnnl_set_primitive_cache_capacity(capacity)
    return ccall(
        (:dnnl_set_primitive_cache_capacity, libdnnl), dnnl_status_t, (Cint,), capacity
    )
end

function dnnl_set_verbose(level)
    return ccall((:dnnl_set_verbose, libdnnl), dnnl_status_t, (Cint,), level)
end

function dnnl_set_jit_dump(enable)
    return ccall((:dnnl_set_jit_dump, libdnnl), dnnl_status_t, (Cint,), enable)
end

function dnnl_version()
    return ccall((:dnnl_version, libdnnl), Ptr{dnnl_version_t}, ())
end

function dnnl_set_jit_profiling_flags(flags)
    return ccall((:dnnl_set_jit_profiling_flags, libdnnl), dnnl_status_t, (Cuint,), flags)
end

function dnnl_set_jit_profiling_jitdumpdir(dir)
    return ccall(
        (:dnnl_set_jit_profiling_jitdumpdir, libdnnl), dnnl_status_t, (Ptr{Cchar},), dir
    )
end

function dnnl_set_max_cpu_isa(isa)
    return ccall((:dnnl_set_max_cpu_isa, libdnnl), dnnl_status_t, (dnnl_cpu_isa_t,), isa)
end

function dnnl_get_effective_cpu_isa()
    return ccall((:dnnl_get_effective_cpu_isa, libdnnl), dnnl_cpu_isa_t, ())
end

function dnnl_set_cpu_isa_hints(isa_hints)
    return ccall(
        (:dnnl_set_cpu_isa_hints, libdnnl),
        dnnl_status_t,
        (dnnl_cpu_isa_hints_t,),
        isa_hints,
    )
end

function dnnl_get_cpu_isa_hints()
    return ccall((:dnnl_get_cpu_isa_hints, libdnnl), dnnl_cpu_isa_hints_t, ())
end

function dnnl_sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    return ccall(
        (:dnnl_sgemm, libdnnl),
        dnnl_status_t,
        (
            Cchar,
            Cchar,
            dnnl_dim_t,
            dnnl_dim_t,
            dnnl_dim_t,
            Cfloat,
            Ptr{Cfloat},
            dnnl_dim_t,
            Ptr{Cfloat},
            dnnl_dim_t,
            Cfloat,
            Ptr{Cfloat},
            dnnl_dim_t,
        ),
        transa,
        transb,
        M,
        N,
        K,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc,
    )
end

function dnnl_gemm_u8s8s32(
    transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co
)
    return ccall(
        (:dnnl_gemm_u8s8s32, libdnnl),
        dnnl_status_t,
        (
            Cchar,
            Cchar,
            Cchar,
            dnnl_dim_t,
            dnnl_dim_t,
            dnnl_dim_t,
            Cfloat,
            Ptr{UInt8},
            dnnl_dim_t,
            UInt8,
            Ptr{Int8},
            dnnl_dim_t,
            Int8,
            Cfloat,
            Ptr{Int32},
            dnnl_dim_t,
            Ptr{Int32},
        ),
        transa,
        transb,
        offsetc,
        M,
        N,
        K,
        alpha,
        A,
        lda,
        ao,
        B,
        ldb,
        bo,
        beta,
        C,
        ldc,
        co,
    )
end

function dnnl_gemm_s8s8s32(
    transa, transb, offsetc, M, N, K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co
)
    return ccall(
        (:dnnl_gemm_s8s8s32, libdnnl),
        dnnl_status_t,
        (
            Cchar,
            Cchar,
            Cchar,
            dnnl_dim_t,
            dnnl_dim_t,
            dnnl_dim_t,
            Cfloat,
            Ptr{Int8},
            dnnl_dim_t,
            Int8,
            Ptr{Int8},
            dnnl_dim_t,
            Int8,
            Cfloat,
            Ptr{Int32},
            dnnl_dim_t,
            Ptr{Int32},
        ),
        transa,
        transb,
        offsetc,
        M,
        N,
        K,
        alpha,
        A,
        lda,
        ao,
        B,
        ldb,
        bo,
        beta,
        C,
        ldc,
        co,
    )
end

function dnnl_threadpool_interop_stream_create(stream, engine, threadpool)
    return ccall(
        (:dnnl_threadpool_interop_stream_create, libdnnl),
        dnnl_status_t,
        (Ptr{dnnl_stream_t}, dnnl_engine_t, Ptr{Cvoid}),
        stream,
        engine,
        threadpool,
    )
end

function dnnl_threadpool_interop_stream_get_threadpool(astream, threadpool)
    return ccall(
        (:dnnl_threadpool_interop_stream_get_threadpool, libdnnl),
        dnnl_status_t,
        (dnnl_stream_t, Ptr{Ptr{Cvoid}}),
        astream,
        threadpool,
    )
end

function dnnl_threadpool_interop_sgemm(
    transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, threadpool
)
    return ccall(
        (:dnnl_threadpool_interop_sgemm, libdnnl),
        dnnl_status_t,
        (
            Cchar,
            Cchar,
            dnnl_dim_t,
            dnnl_dim_t,
            dnnl_dim_t,
            Cfloat,
            Ptr{Cfloat},
            dnnl_dim_t,
            Ptr{Cfloat},
            dnnl_dim_t,
            Cfloat,
            Ptr{Cfloat},
            dnnl_dim_t,
            Ptr{Cvoid},
        ),
        transa,
        transb,
        M,
        N,
        K,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc,
        threadpool,
    )
end

function dnnl_threadpool_interop_gemm_u8s8s32(
    transa,
    transb,
    offsetc,
    M,
    N,
    K,
    alpha,
    A,
    lda,
    ao,
    B,
    ldb,
    bo,
    beta,
    C,
    ldc,
    co,
    threadpool,
)
    return ccall(
        (:dnnl_threadpool_interop_gemm_u8s8s32, libdnnl),
        dnnl_status_t,
        (
            Cchar,
            Cchar,
            Cchar,
            dnnl_dim_t,
            dnnl_dim_t,
            dnnl_dim_t,
            Cfloat,
            Ptr{UInt8},
            dnnl_dim_t,
            UInt8,
            Ptr{Int8},
            dnnl_dim_t,
            Int8,
            Cfloat,
            Ptr{Int32},
            dnnl_dim_t,
            Ptr{Int32},
            Ptr{Cvoid},
        ),
        transa,
        transb,
        offsetc,
        M,
        N,
        K,
        alpha,
        A,
        lda,
        ao,
        B,
        ldb,
        bo,
        beta,
        C,
        ldc,
        co,
        threadpool,
    )
end

function dnnl_threadpool_interop_gemm_s8s8s32(
    transa,
    transb,
    offsetc,
    M,
    N,
    K,
    alpha,
    A,
    lda,
    ao,
    B,
    ldb,
    bo,
    beta,
    C,
    ldc,
    co,
    threadpool,
)
    return ccall(
        (:dnnl_threadpool_interop_gemm_s8s8s32, libdnnl),
        dnnl_status_t,
        (
            Cchar,
            Cchar,
            Cchar,
            dnnl_dim_t,
            dnnl_dim_t,
            dnnl_dim_t,
            Cfloat,
            Ptr{Int8},
            dnnl_dim_t,
            Int8,
            Ptr{Int8},
            dnnl_dim_t,
            Int8,
            Cfloat,
            Ptr{Int32},
            dnnl_dim_t,
            Ptr{Int32},
            Ptr{Cvoid},
        ),
        transa,
        transb,
        offsetc,
        M,
        N,
        K,
        alpha,
        A,
        lda,
        ao,
        B,
        ldb,
        bo,
        beta,
        C,
        ldc,
        co,
        threadpool,
    )
end

const DNNL_MAX_NDIMS = 12

const DNNL_RUNTIME_DIM_VAL = typemin(Int64)

const DNNL_RUNTIME_SIZE_VAL = unsigned(DNNL_RUNTIME_DIM_VAL)

# Skipping MacroDefinition: DNNL_RUNTIME_F32_VAL ( DNNL_RUNTIME_F32_VAL_REP . f )

const DNNL_RUNTIME_S32_VAL = 0

const DNNL_RNN_MAX_N_PARTS = 4

const DNNL_MEMORY_NONE = 0

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

const DNNL_ARG_SCALE = 51

const DNNL_ARG_SHIFT = 52

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

const DNNL_ARG_DIFF_SCALE = 255

const DNNL_ARG_DIFF_SHIFT = 256

const DNNL_ARG_ATTR_OUTPUT_SCALES = 513

const DNNL_ARG_MULTIPLE_SRC = 1024

const DNNL_ARG_MULTIPLE_DST = 2048

const DNNL_ARG_ATTR_ZERO_POINTS = 4096

const DNNL_ARG_ATTR_POST_OP_DW = 8192

const DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE = 16384

const DNNL_ARG_ATTR_INPUT_SCALES = 1048576

const DNNL_RUNTIME_NONE = Cuint(0)

const DNNL_RUNTIME_SEQ = Cuint(1)

const DNNL_RUNTIME_OMP = Cuint(2)

const DNNL_RUNTIME_TBB = Cuint(4)

const DNNL_RUNTIME_THREADPOOL = Cuint(8)

const DNNL_RUNTIME_OCL = Cuint(256)

const DNNL_RUNTIME_SYCL = Cuint(512)

const DNNL_RUNTIME_DPCPP = DNNL_RUNTIME_SYCL

const DNNL_JIT_PROFILE_NONE = Cuint(0)

const DNNL_JIT_PROFILE_VTUNE = Cuint(1)

const DNNL_JIT_PROFILE_LINUX_PERFMAP = Cuint(2)

const DNNL_JIT_PROFILE_LINUX_JITDUMP = Cuint(4)

const DNNL_JIT_PROFILE_LINUX_JITDUMP_USE_TSC = Cuint(8)

const DNNL_JIT_PROFILE_LINUX_PERF =
    DNNL_JIT_PROFILE_LINUX_JITDUMP | DNNL_JIT_PROFILE_LINUX_PERFMAP

# Skipping MacroDefinition: DNNL_HELPER_DLL_IMPORT __attribute__ ( ( visibility ( "default" ) ) )

# Skipping MacroDefinition: DNNL_HELPER_DLL_EXPORT __attribute__ ( ( visibility ( "default" ) ) )

# Skipping MacroDefinition: DNNL_DEPRECATED __attribute__ ( ( deprecated ) )

const DNNL_CPU_THREADING_RUNTIME = DNNL_RUNTIME_THREADPOOL

const DNNL_CPU_RUNTIME = DNNL_RUNTIME_THREADPOOL

const DNNL_GPU_RUNTIME = DNNL_RUNTIME_NONE

const DNNL_VERSION_MAJOR = 2

const DNNL_VERSION_MINOR = 3

const DNNL_VERSION_PATCH = 0

const DNNL_VERSION_HASH = "81a1d98844a2687779c050c2460dc353bd9b15e9"
