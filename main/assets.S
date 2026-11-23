/* Embeds the .tflite models and test set into flash (.rodata).
 * 16-byte alignment is required by TFLite Micro for the flatbuffers. */
    .section .rodata.assets, "a"

.macro ASSET name, file
    .global \name
    .global \name\()_end
    .balign 16
\name:
    .incbin "\file"
\name\()_end:
    .byte 0
.endm

    ASSET model_cnn_i8,       "models/CNN_int8.tflite"
    ASSET model_lstm_i8,      "models/LSTM_int8.tflite"
    ASSET model_cnnlstm_i8,   "models/CNN-LSTM_int8.tflite"
    ASSET test_x,             "test_x.bin"
    ASSET test_y,             "test_y.bin"
