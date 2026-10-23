// Bridging header — открывает C API whisper.cpp для Swift.
// Реальная reference: vendor/whisper.cpp/include/whisper.h
#ifndef KEYBOOP_WHISPER_BRIDGING_H
#define KEYBOOP_WHISPER_BRIDGING_H

#include "whisper.h"

// sherpa-onnx (движок GigaAM). Подключается, только если vendor/sherpa-onnx на месте:
// build-app.sh тогда же определяет KEYBOOP_GIGAAM для Swift. Без него сборка как у апстрима.
#if __has_include("sherpa-onnx/c-api/c-api.h")
#include "sherpa-onnx/c-api/c-api.h"
#endif

#endif
