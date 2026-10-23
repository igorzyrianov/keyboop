import Foundation

/// Распознавание русской речи моделью GigaAM v3 E2E-RNN-T (Сбер, salute-developers/GigaAM) через
/// sherpa-onnx (ONNX Runtime, CPU). Модель сама ставит пунктуацию и регистр («e2e»).
///
/// ФОРМАТ МОДЕЛИ — папка в формате sherpa-onnx:
///   encoder*.onnx, decoder*.onnx, joiner*.onnx (или joint*.onnx) и tokens.txt.
/// Ровно так выглядит стоковый экспорт sherpa-onnx
/// (sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-*), его же делает скрипт
/// scripts/nemo/GigaAM/export-onnx-rnnt-v3-punct.py из sherpa-onnx — им конвертируется и
/// дообученный чекпоинт. Свою папку человек подкладывает кнопкой «Из папки…», мы копируем её
/// к себе (Application Support/Keyboop/models/gigaam), чтобы переезд исходника её не ломал.
///
/// ДЛИННЫЕ ДИКТОВКИ. GigaAM обучен на фразах до ~25 с, на длинном куске целиком он начинает
/// терять слова и пунктуацию (проверено: 47 с одним куском — пропуск точек и обрезанный хвост).
/// Поэтому запись длиннее `maxChunkSeconds` режем по паузам: Silero VAD находит речь, соседние
/// куски склеиваем обратно, пока общий отрезок не превысит лимит, и режем оригинальный буфер
/// с запасом по краям (иначе VAD отъедает первый слог: «Ничьих» → «чьих»). Без VAD-модели в
/// бандле запасной резак — по самому тихому месту в окне.
final class GigaAMEngine {
    static let shared = GigaAMEngine()

    /// Кусок, который отдаём модели целиком. Короче этого — без нарезки.
    private static let maxChunkSeconds: Double = 20
    private static let sampleRate: Int32 = 16000

    /// Папка модели внутри наших моделей (рядом с ggml-*.bin whisper).
    static var modelDir: URL {
        VoiceController.modelsDir.appendingPathComponent("gigaam", isDirectory: true)
    }

    struct ModelFiles { let encoder: URL; let decoder: URL; let joiner: URL; let tokens: URL }

    /// Найти файлы модели в папке. Имена допускаем гибкие: encoder.int8.onnx / encoder.onnx /
    /// v3_e2e_rnnt_encoder.onnx и т.п. При нескольких энкодерах берём int8 (он и меньше, и быстрее).
    static func findFiles(in dir: URL) -> ModelFiles? {
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return nil }
        let onnx = names.filter { $0.lowercased().hasSuffix(".onnx") }
        func pick(_ keys: [String]) -> String? {
            let c = onnx.filter { n in keys.contains { n.lowercased().contains($0) } }.sorted()
            return c.first { $0.lowercased().contains("int8") } ?? c.first
        }
        guard let e = pick(["encoder"]), let d = pick(["decoder"]), let j = pick(["joiner", "joint"]),
              names.contains("tokens.txt") else { return nil }
        return ModelFiles(encoder: dir.appendingPathComponent(e), decoder: dir.appendingPathComponent(d),
                          joiner: dir.appendingPathComponent(j), tokens: dir.appendingPathComponent("tokens.txt"))
    }

    static var modelInstalled: Bool {
        #if KEYBOOP_GIGAAM
        return findFiles(in: modelDir) != nil
        #else
        return false
        #endif
    }

    /// Стоковая модель (sherpa-onnx, int8, с пунктуацией) — для кнопки «Скачать».
    static let stockURL = URL(string: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16.tar.bz2")!
    static let stockSHA256 = "f9620a0099019c6afcee26525ef9ed3297fa50dd5691c1902af0c948fc1a470b"

    private(set) var ready = false
    /// Все обращения к sherpa — строго по одному, на этой очереди: распознаватель не потокобезопасен.
    private let queue = DispatchQueue(label: "keyboop.gigaam", qos: .userInitiated)
    private var loading = false
    private var waiters: [(Bool) -> Void] = []
    private let gate = NSLock()

    #if KEYBOOP_GIGAAM
    private var recognizer: OpaquePointer?
    #endif

    // MARK: - Загрузка

    /// Идемпотентно, параллельные заявки ждут одну загрузку (как у Parakeet).
    func loadIfNeeded() async -> Bool {
        if ready { return true }
        return await withCheckedContinuation { (c: CheckedContinuation<Bool, Never>) in
            gate.lock()
            waiters.append { c.resume(returning: $0) }
            let start = !loading
            loading = true
            gate.unlock()
            guard start else { return }
            queue.async { [weak self] in
                guard let self else { return }
                let ok = self.loadNow()
                self.gate.lock()
                let ws = self.waiters; self.waiters = []; self.loading = false
                self.gate.unlock()
                ws.forEach { $0(ok) }
            }
        }
    }

    func waitForPendingLoad() async {
        gate.lock(); let busy = loading; gate.unlock()
        if busy { _ = await loadIfNeeded() }
    }

    /// Только на `queue`.
    private func loadNow() -> Bool {
        #if KEYBOOP_GIGAAM
        if recognizer != nil { ready = true; return true }
        guard let f = Self.findFiles(in: Self.modelDir) else { kbLog("gigaam: модель не установлена"); return false }
        let t0 = ProcessInfo.processInfo.systemUptime
        var cfg = SherpaOnnxOfflineRecognizerConfig()
        cfg.feat_config.sample_rate = Self.sampleRate
        cfg.feat_config.feature_dim = 64            // GigaAM: 64 мела (sherpa всё равно выставит сам по метаданным)
        let enc = strdup(f.encoder.path), dec = strdup(f.decoder.path), joi = strdup(f.joiner.path)
        let tok = strdup(f.tokens.path), cpu = strdup("cpu"), mtype = strdup("nemo_transducer")
        let method = strdup("greedy_search")
        defer { [enc, dec, joi, tok, cpu, mtype, method].forEach { free($0) } }
        cfg.model_config.transducer.encoder = UnsafePointer(enc)
        cfg.model_config.transducer.decoder = UnsafePointer(dec)
        cfg.model_config.transducer.joiner = UnsafePointer(joi)
        cfg.model_config.tokens = UnsafePointer(tok)
        cfg.model_config.provider = UnsafePointer(cpu)
        cfg.model_config.model_type = UnsafePointer(mtype)
        // M2: 4 производительных ядра. Больше потоков ONNX Runtime на E-ядрах только мешает.
        cfg.model_config.num_threads = Int32(max(1, min(4, ProcessInfo.processInfo.activeProcessorCount / 2)))
        cfg.decoding_method = UnsafePointer(method)
        cfg.max_active_paths = 4
        guard let r = SherpaOnnxCreateOfflineRecognizer(&cfg) else {
            kbLog("gigaam: sherpa-onnx не принял модель (\(f.encoder.lastPathComponent)) — формат не sherpa-onnx?")
            return false
        }
        recognizer = r
        ready = true
        kbLog("gigaam: модель загружена за \(Int((ProcessInfo.processInfo.systemUptime - t0) * 1000))мс (\(f.encoder.lastPathComponent))")
        return true
        #else
        return false
        #endif
    }

    /// Выгрузить из памяти (настройка «выгружать после диктовки»). false — идёт загрузка.
    @discardableResult
    func unload() -> Bool {
        gate.lock(); let busy = loading; gate.unlock()
        if busy { return false }
        guard ready else { return true }
        ready = false
        queue.async { [weak self] in
            #if KEYBOOP_GIGAAM
            if let r = self?.recognizer { SherpaOnnxDestroyOfflineRecognizer(r); self?.recognizer = nil }
            #endif
            kbLog("gigaam: модель выгружена")
        }
        return true
    }

    // MARK: - Распознавание

    /// 16 кГц mono Float32. nil — отказ движка, "" — тишина. `language` игнорируем: модель только русская.
    func transcribe(samples: [Float], language: String) async -> String? {
        await withCheckedContinuation { (c: CheckedContinuation<String?, Never>) in
            queue.async { [weak self] in
                guard let self else { return c.resume(returning: nil) }
                c.resume(returning: self.transcribeNow(samples))
            }
        }
    }

    private func transcribeNow(_ samples: [Float]) -> String? {
        #if KEYBOOP_GIGAAM
        guard let r = recognizer, ready else { return nil }
        let t0 = ProcessInfo.processInfo.systemUptime
        let chunks = Self.split(samples)
        var parts: [String] = []
        let tail = [Float](repeating: 0, count: 4800)   // 0.3 с тишины в конец: модель охотнее ставит финальный знак
        for ch in chunks {
            guard let s = SherpaOnnxCreateOfflineStream(r) else { return nil }
            (ch + tail).withUnsafeBufferPointer { SherpaOnnxAcceptWaveformOffline(s, Self.sampleRate, $0.baseAddress, Int32($0.count)) }
            SherpaOnnxDecodeOfflineStream(r, s)
            if let res = SherpaOnnxGetOfflineStreamResult(s) {
                if let t = res.pointee.text {
                    let txt = String(cString: t).trimmingCharacters(in: .whitespacesAndNewlines)
                    if !txt.isEmpty { parts.append(txt) }
                }
                SherpaOnnxDestroyOfflineRecognizerResult(res)
            }
            SherpaOnnxDestroyOfflineStream(s)
        }
        let text = Self.join(parts)
        kbLog("gigaam: \(String(format: "%.1f", Double(samples.count) / 16000))с, кусков=\(chunks.count) → \(text.count) симв. за \(Int((ProcessInfo.processInfo.systemUptime - t0) * 1000))мс")
        return text
        #else
        return nil
        #endif
    }

    /// Склейка кусков: модель ставит заглавную в начале КАЖДОГО куска, а кусок мы резали по паузе,
    /// не по концу фразы. Если предыдущий кусок не кончился на . ! ? …, гасим заглавную у следующего
    /// (кроме слов целиком из заглавных — аббревиатур вроде «МФЦ»).
    static func join(_ parts: [String]) -> String {
        var out = ""
        for p in parts {
            guard !out.isEmpty else { out = p; continue }
            var next = p
            if let last = out.last, !".!?…".contains(last), let first = next.first, first.isUppercase {
                let word = next.prefix { $0.isLetter }
                let isAbbr = word.count > 1 && word.allSatisfy { $0.isUppercase }
                if !isAbbr { next = first.lowercased() + next.dropFirst() }
            }
            out += " " + next
        }
        return out
    }

    // MARK: - Нарезка длинной записи

    /// Короткую запись отдаём целиком, длинную — кусками ≤ maxChunkSeconds по паузам.
    static func split(_ samples: [Float]) -> [[Float]] {
        let maxN = Int(maxChunkSeconds * 16000)
        guard samples.count > maxN else { return [samples] }
        #if KEYBOOP_GIGAAM
        if let spans = vadSpans(samples), !spans.isEmpty {
            return mergeSpans(spans, total: samples.count, maxN: maxN).map { Array(samples[$0.lowerBound..<$0.upperBound]) }
        }
        #endif
        return energySplit(samples, maxN: maxN)
    }

    /// Склеиваем соседние участки речи в отрезки ≤ maxN (с паузами между ними — модели нужен
    /// контекст для пунктуации) и расширяем каждый на 0.3 с, но не дальше середины паузы до соседа.
    static func mergeSpans(_ spans: [Range<Int>], total: Int, maxN: Int) -> [Range<Int>] {
        var groups: [Range<Int>] = []
        for s in spans {
            if let g = groups.last, s.upperBound - g.lowerBound <= maxN {
                groups[groups.count - 1] = g.lowerBound..<s.upperBound
            } else if s.count > maxN {
                // Сплошная речь длиннее лимита (VAD уже режет по max_speech_duration, это страховка).
                var a = s.lowerBound
                while a < s.upperBound { let b = min(a + maxN, s.upperBound); groups.append(a..<b); a = b }
            } else {
                groups.append(s)
            }
        }
        // Запас несимметричный — подобран замером 23.10 на тестовой записи sherpa-onnx: перед речью
        // 0.2 с (с 0.3+ модель на этом же звуке отдаёт «Ничь» вместо «Ничьих»), после — 0.4 с
        // (без хвостовой тишины теряется финальная точка).
        let padLead = Int(0.2 * 16000), padTrail = Int(0.4 * 16000)
        var out: [Range<Int>] = []
        for (i, g) in groups.enumerated() {
            let prevEnd = i > 0 ? groups[i - 1].upperBound : 0
            let nextStart = i + 1 < groups.count ? groups[i + 1].lowerBound : total
            let lo = max(i > 0 ? (prevEnd + g.lowerBound) / 2 : 0, g.lowerBound - padLead)
            let hi = min(i + 1 < groups.count ? (g.upperBound + nextStart) / 2 : total, g.upperBound + padTrail)
            if hi > lo { out.append(lo..<hi) }
        }
        return out
    }

    #if KEYBOOP_GIGAAM
    /// Участки речи от Silero VAD (модель silero_vad.onnx лежит в Resources бандла).
    private static func vadSpans(_ samples: [Float]) -> [Range<Int>]? {
        guard let path = Bundle.main.path(forResource: "silero_vad", ofType: "onnx") else {
            kbLog("gigaam: silero_vad.onnx нет в бандле — режу по тишине")
            return nil
        }
        var cfg = SherpaOnnxVadModelConfig()
        let p = strdup(path), cpu = strdup("cpu")
        defer { free(p); free(cpu) }
        cfg.silero_vad.model = UnsafePointer(p)
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.3
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.max_speech_duration = Float(maxChunkSeconds)
        cfg.silero_vad.window_size = 512
        cfg.sample_rate = sampleRate
        cfg.num_threads = 1
        cfg.provider = UnsafePointer(cpu)
        let seconds = Float(samples.count) / 16000 + 5
        guard let vad = SherpaOnnxCreateVoiceActivityDetector(&cfg, seconds) else { return nil }
        defer { SherpaOnnxDestroyVoiceActivityDetector(vad) }
        var spans: [Range<Int>] = []
        func drain() {
            while SherpaOnnxVoiceActivityDetectorEmpty(vad) == 0 {
                if let seg = SherpaOnnxVoiceActivityDetectorFront(vad) {
                    let a = Int(seg.pointee.start), n = Int(seg.pointee.n)
                    if n > 0 { spans.append(a..<min(a + n, samples.count)) }
                    SherpaOnnxDestroySpeechSegment(seg)
                }
                SherpaOnnxVoiceActivityDetectorPop(vad)
            }
        }
        samples.withUnsafeBufferPointer { buf in
            var i = 0
            while i < buf.count {
                let n = min(512, buf.count - i)
                SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad, buf.baseAddress! + i, Int32(n))
                i += n
                drain()
            }
        }
        SherpaOnnxVoiceActivityDetectorFlush(vad)
        drain()
        return spans.sorted { $0.lowerBound < $1.lowerBound }
    }
    #endif

    /// Запасной резак без VAD: в окне [maxN·0.6, maxN] ищем самые тихие 100 мс и режем там.
    static func energySplit(_ samples: [Float], maxN: Int) -> [[Float]] {
        var out: [[Float]] = []
        var start = 0
        let win = 1600
        while samples.count - start > maxN {
            let lo = start + Int(Double(maxN) * 0.6), hi = start + maxN - win
            var best = hi, bestE = Float.greatestFiniteMagnitude
            var i = lo
            while i <= hi {
                var e: Float = 0
                for k in i..<(i + win) { e += samples[k] * samples[k] }
                if e < bestE { bestE = e; best = i }
                i += 400
            }
            let cut = best + win / 2
            out.append(Array(samples[start..<cut])); start = cut
        }
        out.append(Array(samples[start...]))
        return out
    }

    // MARK: - Установка модели

    /// Скопировать папку модели человека к нам. Проверяем состав ДО копирования.
    func importModel(from src: URL, completion: @escaping (Bool, String?) -> Void) {
        guard Self.findFiles(in: src) != nil else {
            completion(false, L10n.t("voice.gigaBadFolder")); return
        }
        let dst = Self.modelDir
        unloadForReplace()
        queue.async {
            let fm = FileManager.default
            let tmp = VoiceController.modelsDir.appendingPathComponent("gigaam.importing", isDirectory: true)
            try? fm.removeItem(at: tmp)
            do {
                try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
                // Копируем только нужное: в папке экспорта может лежать и чекпоинт на гигабайт.
                for name in try fm.contentsOfDirectory(atPath: src.path) {
                    let l = name.lowercased()
                    guard l.hasSuffix(".onnx") || l == "tokens.txt" || l.hasSuffix(".onnx.data") else { continue }
                    try fm.copyItem(at: src.appendingPathComponent(name), to: tmp.appendingPathComponent(name))
                }
                try? fm.removeItem(at: dst)
                try fm.moveItem(at: tmp, to: dst)
                kbLog("gigaam: модель импортирована из \(src.lastPathComponent)")
                DispatchQueue.main.async { completion(true, nil) }
            } catch {
                try? fm.removeItem(at: tmp)
                kbLog("gigaam: импорт не удался: \(error)")
                DispatchQueue.main.async { completion(false, error.localizedDescription) }
            }
        }
    }

    /// Скачать стоковую модель (~170 МБ архив, ~230 МБ на диске) и распаковать.
    func download(progress: @escaping (Double) -> Void) async -> Bool {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("gigaam-\(UUID().uuidString).tar.bz2")
        guard await TarballDownloader.download(Self.stockURL, to: tmp, progress: progress) else { return false }
        defer { try? FileManager.default.removeItem(at: tmp) }
        // Целостность, как у остальных моделей проекта: архив на релизе sherpa-onnx неизменяемый,
        // хеш снят с него же 23.10.2026.
        guard ModelDownloader.sha256(ofFileAt: tmp) == Self.stockSHA256 else {
            kbLog("gigaam: SHA-256 архива не сошёлся — отклонён"); return false
        }
        let unpack = FileManager.default.temporaryDirectory.appendingPathComponent("gigaam-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: unpack) }
        do {
            try FileManager.default.createDirectory(at: unpack, withIntermediateDirectories: true)
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
            p.arguments = ["--no-same-owner", "-xjf", tmp.path, "-C", unpack.path]
            try p.run(); p.waitUntilExit()
            guard p.terminationStatus == 0 else { kbLog("gigaam: tar=\(p.terminationStatus)"); return false }
        } catch { kbLog("gigaam: распаковка: \(error)"); return false }
        // Внутри архива одна папка sherpa-onnx-nemo-transducer-…; ищем её (или файлы в корне).
        let fm = FileManager.default
        var src = unpack
        if Self.findFiles(in: src) == nil,
           let sub = (try? fm.contentsOfDirectory(at: unpack, includingPropertiesForKeys: nil))?
               .first(where: { Self.findFiles(in: $0) != nil }) { src = sub }
        let ok: Bool = await withCheckedContinuation { c in
            importModel(from: src) { ok, _ in c.resume(returning: ok) }
        }
        return ok
    }

    func deleteModel(completion: @escaping (Bool) -> Void) {
        unloadForReplace()
        queue.async {
            var ok = true
            if FileManager.default.fileExists(atPath: Self.modelDir.path) {
                do { try FileManager.default.removeItem(at: Self.modelDir); kbLog("gigaam: модель удалена") }
                catch { ok = false; kbLog("gigaam: удаление не удалось: \(error)") }
            }
            DispatchQueue.main.async { completion(ok) }
        }
    }

    /// Сбросить распознаватель перед заменой/удалением файлов (на очереди, после текущей задачи).
    private func unloadForReplace() {
        ready = false
        queue.async { [weak self] in
            #if KEYBOOP_GIGAAM
            if let r = self?.recognizer { SherpaOnnxDestroyOfflineRecognizer(r); self?.recognizer = nil }
            #endif
        }
    }
}
