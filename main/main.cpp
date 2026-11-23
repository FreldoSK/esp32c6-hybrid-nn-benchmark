// Benchmark of CNN, LSTM and CNN-LSTM door-sensor classifiers on the ESP32-C6.
//
// Every int8 model is run over the embedded test set.
// For each one we print per-sample predictions, accuracy, mean inference time
// and the size of the tensor arena it needed, followed by a CSV summary.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"


// ============================================================================
//  Embedded assets (see assets.S)
// ============================================================================

#define DECLARE_ASSET(name) extern "C" const uint8_t name[], name##_end[]

DECLARE_ASSET(model_cnn_i8);
DECLARE_ASSET(model_lstm_i8);
DECLARE_ASSET(model_cnnlstm_i8);
DECLARE_ASSET(test_x);  // int16 [N][kSteps], value = x * kInputScale
DECLARE_ASSET(test_y);  // uint8 [N], index into kClassNames


// ============================================================================
//  Configuration
// ============================================================================

// How many test samples to run per model. 0 = the whole test set.
constexpr int kMaxSamples = 20;

// Length of one input window (sensor readings per sample).
constexpr int kSteps = 400;

// Fixed-point scale of test_x. Must match SCALE in tools/make_test_bin.py.
constexpr float kInputScale = 8192.0f;

// Generous on purpose. Shrink to the "arena used" value printed at runtime.
constexpr size_t kArenaSize = 320 * 1024;

constexpr const char* kClassNames[] = {"No passage", "Entry", "Exit", "Walk-by"};
constexpr int kNumClasses = sizeof(kClassNames) / sizeof(kClassNames[0]);

struct ModelInfo {
    const char*    name;
    const uint8_t* data;
};

constexpr ModelInfo kModels[] = {
    {"CNN_int8",      model_cnn_i8},
    {"LSTM_int8",     model_lstm_i8},
    {"CNN-LSTM_int8", model_cnnlstm_i8},
};
constexpr int kNumModels = sizeof(kModels) / sizeof(kModels[0]);

alignas(16) static uint8_t tensor_arena[kArenaSize];


// ============================================================================
//  Results
// ============================================================================

struct BenchmarkResult {
    bool     ok              = false;
    int      num_samples     = 0;
    int      num_correct     = 0;
    int64_t  total_invoke_us = 0;
    unsigned arena_used      = 0;

    float accuracy_pct()   const { return 100.0f * num_correct / num_samples; }
    float mean_invoke_ms() const { return total_invoke_us / 1000.0f / num_samples; }
};


// ============================================================================
//  Helpers
// ============================================================================

// Registers every operator used by the CNN, LSTM and CNN-LSTM models.
static const tflite::MicroOpResolver& get_op_resolver() {
    static tflite::MicroMutableOpResolver<11> resolver;
    static bool initialized = false;

    if (!initialized) {
        resolver.AddConv2D();
        resolver.AddMaxPool2D();
        resolver.AddReshape();
        resolver.AddExpandDims();
        resolver.AddSqueeze();
        resolver.AddUnidirectionalSequenceLSTM();
        resolver.AddStridedSlice();
        resolver.AddFullyConnected();
        resolver.AddMean();
        resolver.AddQuantize();
        resolver.AddDequantize();
        initialized = true;
    }
    return resolver;
}

static int count_test_samples() {
    const int available = static_cast<int>(test_y_end - test_y);
    if (kMaxSamples > 0 && available > kMaxSamples) {
        return kMaxSamples;
    }
    return available;
}

// Reads test sample `index` as floats.
// test_x is not guaranteed to be aligned, so it is copied with memcpy.
static void load_test_sample(int index, float* out) {
    int16_t raw[kSteps];
    memcpy(raw, test_x + static_cast<size_t>(index) * sizeof(raw), sizeof(raw));

    for (int i = 0; i < kSteps; i++) {
        out[i] = raw[i] / kInputScale;
    }
}

static int8_t quantize_to_int8(float value, float scale, int zero_point) {
    int q = static_cast<int>(lroundf(value / scale)) + zero_point;
    if (q < -128) q = -128;
    if (q > 127)  q = 127;
    return static_cast<int8_t>(q);
}

// Writes one sample into the model input, quantizing it if the model is int8.
static void fill_input(TfLiteTensor* input, const float* x) {
    if (input->type == kTfLiteInt8) {
        const float scale      = input->params.scale;
        const int   zero_point = input->params.zero_point;
        for (int i = 0; i < kSteps; i++) {
            input->data.int8[i] = quantize_to_int8(x[i], scale, zero_point);
        }
    } else {
        memcpy(input->data.f, x, kSteps * sizeof(float));
    }
}

// Index of the highest score. Works on raw int8 values too, because
// quantization preserves ordering (the scale is always positive).
static int argmax(const TfLiteTensor* output) {
    const bool is_int8 = output->type == kTfLiteInt8;
    int best = 0;

    for (int c = 1; c < kNumClasses; c++) {
        const bool better = is_int8 ? output->data.int8[c] > output->data.int8[best]
                                    : output->data.f[c]    > output->data.f[best];
        if (better) best = c;
    }
    return best;
}


// ============================================================================
//  Benchmark
// ============================================================================

static BenchmarkResult run_benchmark(const ModelInfo& info, int num_samples) {
    BenchmarkResult result;

    const tflite::Model* model = tflite::GetModel(info.data);
    tflite::MicroInterpreter interpreter(model, get_op_resolver(), tensor_arena, kArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed\n");
        return result;
    }
    result.arena_used = static_cast<unsigned>(interpreter.arena_used_bytes());
    printf("arena used: %u B\n", result.arena_used);

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    float x[kSteps];

    for (int s = 0; s < num_samples; s++) {
        load_test_sample(s, x);
        fill_input(input, x);

        const int64_t start_us = esp_timer_get_time();
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("Invoke failed\n");
            return result;
        }
        const int64_t invoke_us = esp_timer_get_time() - start_us;

        const int expected  = test_y[s];
        const int predicted = argmax(output);

        result.total_invoke_us += invoke_us;
        result.num_correct     += (predicted == expected);
        result.num_samples++;

        printf("%4d  true %-10s  pred %-10s  %lld ms\n",
               s, kClassNames[expected], kClassNames[predicted], invoke_us / 1000);

        vTaskDelay(1);
    }

    result.ok = result.num_samples > 0;
    return result;
}

static void print_summary(const BenchmarkResult* results) {
    printf("\n%-18s %10s %13s %9s %12s\n",
           "Model", "Arena [KB]", "Accuracy [%]", "Correct", "Invoke [ms]");
    for (int i = 0; i < 66; i++) putchar('-');
    putchar('\n');

    for (int i = 0; i < kNumModels; i++) {
        const BenchmarkResult& r = results[i];
        if (!r.ok) {
            printf("%-18s %10s\n", kModels[i].name, "FAILED");
            continue;
        }
        printf("%-18s %10.1f %13.2f %4d/%-4d %12.1f\n",
               kModels[i].name, r.arena_used / 1024.0, r.accuracy_pct(),
               r.num_correct, r.num_samples, r.mean_invoke_ms());
    }
}


// ============================================================================
//  Entry point
// ============================================================================

extern "C" void app_main(void) {
    const int num_samples = count_test_samples();
    static BenchmarkResult results[kNumModels];

    for (int i = 0; i < kNumModels; i++) {
        const char* name = kModels[i].name;
        printf("\n=== %s (%d samples) ===\n", name, num_samples);

        results[i] = run_benchmark(kModels[i], num_samples);

        const BenchmarkResult& r = results[i];
        if (r.ok) {
            printf("\n%s accuracy: %.1f %% (%d/%d), mean invoke: %.0f ms\n",
                   name, r.accuracy_pct(), r.num_correct, r.num_samples, r.mean_invoke_ms());
        }
    }

    print_summary(results);
}
