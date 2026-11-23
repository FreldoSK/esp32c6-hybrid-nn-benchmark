# ESP32-C6 Hybrid Neural Network Benchmark

Benchmarks quantized **CNN**, **LSTM** and hybrid **CNN-LSTM** classifiers on an
ESP32-C6 using TensorFlow Lite Micro. The models recognize door-sensor events
from a 400-sample window: *No passage*, *Entry*, *Exit* and *Walk-by*.

For every model the firmware runs the embedded test set and reports per-sample
predictions, accuracy, mean inference time and tensor-arena usage, followed by
a CSV summary.

## Project layout

| Path | Purpose |
|---|---|
| `main/main.cpp` | Benchmark firmware |
| `main/models/` | int8 `.tflite` models |
| `main/assets.S` | Embeds the models and test set into flash |
| `main/test_x.bin`, `main/test_y.bin` | Test set in compact binary form |
| `main/door_sensor_dataset_full.csv` | Source dataset |
| `tools/make_test_bin.py` | Converts the CSV into the `.bin` test set |

## Build and run

Requires ESP-IDF v6.0 or newer.

```sh
idf.py build
idf.py -p <PORT> flash monitor
```

The number of test samples per model is set by `kMaxSamples` in `main/main.cpp`
(`0` = whole test set).

## Regenerating the test set

```sh
python tools/make_test_bin.py [max_samples]
```

## Example output

```
===== SUMMARY =====
model,arena_B,accuracy_pct,correct,n,mean_invoke_ms
CNN_int8,...
```
