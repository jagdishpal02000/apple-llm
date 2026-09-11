//
// apple-llm helper — the whole on-device path.
//
// Extracted from api-scribe (MIT, https://github.com/…/api-scribe), where it
// lived as a TypeScript string literal in `src/llm/apple-helper.ts`. This file
// is the single source of truth: `scripts/embed-helper.mjs` copies it verbatim
// into both the npm and the pip package, and a test in each asserts the copies
// still hash equal to this file.
//
// It needs no entitlements and no license agreement — unlike `/usr/bin/fm`,
// which ships with macOS 27 but is gated behind a machine-wide `sudo fm
// license`. Do not depend on `fm`.
//
// Protocol:
//   helper --probe   -> stdout: one JSON object describing this machine.
//   helper --serve   -> one request JSON per stdin line,
//                       one response JSON per stdout line, in order.
//                       EXCEPT op "stream": N delta lines + one final line,
//                       all belonging to that single request. The next request
//                       is not read until the stream completes, so ordering is
//                       preserved: the client must keep reading lines until
//                       "done":true before sending the next request.
//   helper           -> one request on stdin, one response on stdout
//                       ("stream" collapses to a single final response here,
//                       since one-shot stdout is not incremental).
//
//   request:  {"op"?:"generate"|"stream"|"countTokens"|"prewarm"|"history"|"reset",
//                       // default generate
//              "instructions":string,"prompt":string,"schema"?:object|null,
//              "temperature"?:number,"maxTokens"?:number,
//              "includeSchemaInPrompt"?:bool,"reuseSession"?:bool,
//              "sessionId"?:string,                  // multi-turn conversation
//              "tools"?:["ocr"|"barcode"|"spotlight"], // built-in FM tools, 27+
//              "images"?:[string|{"path":string,"label"?:string}],
//                                                    // file paths, macOS 27+
//              "useCase"?:"general"|"contentTagging",
//              "guardrails"?:"default"|"permissive",
//              "sampling"?:{"mode":"greedy"}
//                        | {"mode":"topK","k":int,"seed"?:int}
//                        | {"mode":"threshold","p":number,"seed"?:int}}
//   response: {"ok":true,"content":string}                // generate
//           | {"ok":true,"delta":string,"done":false}     // stream partial
//           | {"ok":true,"content":string,"done":true}    // stream final
//           | {"ok":true,"tokens":int}                    // countTokens
//           | {"ok":true,"prewarmed":true}                // prewarm
//           | {"ok":true,"history":[{role,content}],      // history
//              "instructions":string,"sessionId":string}
//           | {"ok":true,"reset":true}                    // reset
//           | {"ok":false,"error":string,"kind"?:string}
//
//   `kind` is one of availability | schema | context | quota | guardrail |
//   timeout | unsupported | generation, so callers can raise typed errors
//   instead of matching on strings. A quota error may carry `resetDate`.
//
//   --probe reports availability, contextSize, variant, the model's
//   capabilities (vision / guidedGeneration / reasoning / toolCalling) and the
//   Private Cloud Compute quota status. That last one is readable without the
//   `com.apple.developer.private-cloud-compute` entitlement that blocks PCC
//   *inference*, so it is the one piece of first-party cloud state available
//   to an unentitled process. It also reports `features` (streaming, sessions,
//   labelledAttachments, builtInTools) so callers can fail fast with a message
//   instead of a stale binary's silent misbehaviour.
//

import Foundation
import FoundationModels
#if canImport(_Vision_FoundationModels)
import _Vision_FoundationModels
#endif
#if canImport(_CoreSpotlight_FoundationModels)
import _CoreSpotlight_FoundationModels
#endif

private func emit(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object),
          let text = String(data: data, encoding: .utf8) else {
        print("{\"ok\":false,\"error\":\"helper could not encode its response\"}")
        fflush(stdout)
        return
    }
    // stdout is a pipe here, so it is fully buffered; the parent is waiting on
    // this line and would otherwise see nothing until the process exits.
    print(text)
    fflush(stdout)
}

/// Optional session reuse, off by default.
///
/// api-scribe kept one session alive while the instructions were unchanged, on
/// the finding that allocating a fresh LanguageModelSession per request made
/// every other call stall ~16s while assets cycled — a strict 17s/1.5s
/// alternation, uncorrelated with prompt size.
///
/// That did not reproduce here. Measured on macOS 27 / M-series inside one
/// long-lived process, 6 calls per arm: session reused -> median 0.63s; a fresh
/// session every call -> median 0.64s, max 0.68s, no variance. The alternation
/// would have hit three times in six calls. The load-bearing part of the fix
/// appears to be the long-lived *process* (api-scribe measured ~17s a call when
/// it spawned a helper per request, against ~1.5s once resident), not the
/// shared session.
///
/// So the default here is a fresh session per request, because a library cannot
/// let two unrelated calls share a transcript — api-scribe could, since every
/// one of its prompts carried its own whole context. The old behaviour is kept
/// behind `reuseSession` rather than deleted: the original finding was measured
/// too, possibly on macOS 26, and the doubt is worth preserving. If per-call
/// latency ever regresses to ~17s, try `reuseSession: true` first.
///
/// Named sessions (`sessionId`) are the multi-turn counterpart: same process,
/// but the session is keyed by id so a conversation accumulates a native
/// transcript across calls. History is also mirrored to
/// `~/Library/Caches/apple-llm/sessions/<id>.json` so `history` survives a
/// helper restart; the native transcript does not — after a restart the next
/// call recreates the native session and continues from the stored history
/// length, which callers should treat as a context break, not a loss.
@available(macOS 26.0, *)
private final class SessionHolder {
    static var session: LanguageModelSession?
    static var instructions: String?
    static var turns = 0
}

@available(macOS 26.0, *)
private struct NamedSession {
    var session: LanguageModelSession
    var instructions: String
    var fingerprint: String
    var history: [[String: String]]
}

@available(macOS 26.0, *)
private final class SessionStore {
    static var named: [String: NamedSession] = [:]

    static func sessionsDir() -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home
            .appendingPathComponent("Library/Caches/apple-llm/sessions", isDirectory: true)
    }

    /// Filename-safe: Siri conversation titles can contain anything.
    static func safeId(_ id: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let mapped = id.unicodeScalars.map { allowed.contains($0) ? String($0) : "_" }.joined()
        let trimmed = String(mapped.prefix(64))
        return trimmed.isEmpty ? "session" : trimmed
    }

    static func fileFor(_ id: String) -> URL {
        sessionsDir().appendingPathComponent("\(safeId(id)).json")
    }

    static func loadHistory(_ id: String) -> (instructions: String?, history: [[String: String]]) {
        let url = fileFor(id)
        guard let data = try? Data(contentsOf: url),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return (nil, []) }
        let history = obj["history"] as? [[String: String]] ?? []
        let instructions = obj["instructions"] as? String
        return (instructions, history)
    }

    static func saveHistory(id: String, instructions: String, history: [[String: String]]) {
        let url = fileFor(id)
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let obj: [String: Any] = [
            "version": 1, "sessionId": id,
            "instructions": instructions, "history": history,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: obj) {
            try? data.write(to: url, options: .atomic)
        }
    }

    static func removeFile(_ id: String) {
        try? FileManager.default.removeItem(at: fileFor(id))
    }
}

@available(macOS 26.0, *)
private func sessionFingerprint(useCase: String, guardrails: String, tools: [String],
                                instructions: String) -> String {
    "\(useCase)|\(guardrails)|\(tools.sorted().joined(separator: ","))|\(instructions)"
}

@available(macOS 26.0, *)
private func sessionFor(
    model: SystemLanguageModel, instructions: String, reuse: Bool
) -> LanguageModelSession {
    // Note a prewarmed session is deliberately *not* reused here. Doing so was
    // tried and broke seeded reproducibility: the first call after a prewarm ran
    // against the warmed session while later calls got fresh ones, so identical
    // seeds produced different text. What prewarming actually buys is loading
    // the model assets, and that is process-wide rather than session-bound, so
    // discarding the session costs nothing and keeps every request identical.
    guard reuse else { return LanguageModelSession(model: model, instructions: instructions) }
    if let existing = SessionHolder.session,
       SessionHolder.instructions == instructions,
       SessionHolder.turns < 16 {
        SessionHolder.turns += 1
        return existing
    }
    let session = LanguageModelSession(model: model, instructions: instructions)
    SessionHolder.session = session
    SessionHolder.instructions = instructions
    SessionHolder.turns = 1
    return session
}

/// Named multi-turn session. Tools are fixed at session construction, so a
/// changed tool set recreates the native session; the mirrored history is kept
/// so nothing the caller said is silently dropped from `history`.
@available(macOS 26.0, *)
private func namedSessionFor(
    id: String, model: SystemLanguageModel, instructions: String,
    useCase: String, guardrails: String, tools: [any Tool]
) -> LanguageModelSession {
    let toolNames = currentToolNames()
    let want = sessionFingerprint(useCase: useCase, guardrails: guardrails,
                                  tools: toolNames, instructions: instructions)
    if let existing = SessionStore.named[id], existing.fingerprint == want {
        return existing.session
    }
    // Recreate (first use, param change, or restart). Restore mirrored history
    // so `history` is continuous even though the native transcript restarts.
    let stored = SessionStore.loadHistory(id)
    let history = SessionStore.named[id]?.history ?? stored.history
    let session: LanguageModelSession
    if tools.isEmpty {
        session = LanguageModelSession(model: model, instructions: instructions)
    } else {
        session = LanguageModelSession(model: model, tools: tools, instructions: instructions)
    }
    SessionStore.named[id] = NamedSession(
        session: session, instructions: instructions,
        fingerprint: want, history: history)
    // Persist immediately so a fresh id is visible to `history` even before
    // its first turn completes.
    SessionStore.saveHistory(id: id, instructions: instructions, history: history)
    return session
}

// The tool names for the in-flight request, stashed so namedSessionFor can
// fingerprint on them without threading another parameter through sessionFor.
@available(macOS 26.0, *)
private final class RequestContext {
    static var toolNames: [String] = []
}

@available(macOS 26.0, *)
private func currentToolNames() -> [String] { RequestContext.toolNames }

/// Classify a generation failure so callers can raise a typed error rather than
/// matching on a message Apple may reword in any OS release.
///
/// macOS 27 replaced `LanguageModelSession.GenerationError` with
/// `LanguageModelError`; both are consulted so one helper serves 26 and 27.
/// `rateLimited` carries a `resetDate`, which is what makes an on-device quota
/// error actionable rather than just a failure.
@available(macOS 26.0, *)
private func classify(_ error: Error) -> (kind: String, resetDate: String?) {
    if #available(macOS 27.0, *) {
        if let modern = error as? LanguageModelError {
            switch modern {
            case .contextSizeExceeded: return ("context", nil)
            case .rateLimited(let info):
                return ("quota", info.resetDate.map(ISO8601DateFormatter().string(from:)))
            case .guardrailViolation, .refusal: return ("guardrail", nil)
            case .timeout: return ("timeout", nil)
            case .unsupportedCapability, .unsupportedGenerationGuide,
                 .unsupportedLanguageOrLocale, .unsupportedTranscriptContent:
                return ("unsupported", nil)
            @unknown default: return ("generation", nil)
            }
        }
    }
    return (legacyKind(error), nil)
}

/// macOS 26's error type, kept so one helper source serves both OS versions.
///
/// Building on macOS 27 emits one deprecation warning at the call site below.
/// That is benign and self-correcting: the deployment target is derived from the
/// host SDK, so on a macOS 26 machine — the only place this branch is reachable —
/// the target is macos26.0 and nothing is deprecated yet. On macOS 27 the
/// warning points at code `#available` has already made unreachable.
@available(macOS 26.0, *)
@available(macOS, deprecated: 27.0)
private func legacyKind(_ error: Error) -> String {
    if let legacy = error as? LanguageModelSession.GenerationError {
        switch legacy {
        case .exceededContextWindowSize: return "context"
        case .guardrailViolation, .refusal: return "guardrail"
        case .rateLimited: return "quota"
        default: return "generation"
        }
    }
    return "generation"
}

/// Models are keyed by use case and guardrails, because both are fixed at
/// construction. Building one is cheap; keeping them avoids re-resolving assets
/// when a caller alternates between, say, general and contentTagging.
@available(macOS 26.0, *)
private final class ModelCache {
    static var models: [String: SystemLanguageModel] = [:]
}

/// `contentTagging` is a use case Apple ships specifically for tagging and
/// topic extraction; `permissive` relaxes the guardrails for content
/// *transformation* tasks (rewriting, summarising text that the default
/// guardrails would refuse). Both are macOS 26+.
@available(macOS 26.0, *)
private func modelFor(useCase: String, guardrails: String) -> SystemLanguageModel {
    let key = "\(useCase)|\(guardrails)"
    if let cached = ModelCache.models[key] { return cached }
    let resolvedUseCase: SystemLanguageModel.UseCase =
        useCase == "contentTagging" ? .contentTagging : .general
    let resolvedGuardrails: SystemLanguageModel.Guardrails =
        guardrails == "permissive" ? .permissiveContentTransformations : .default
    let model = SystemLanguageModel(useCase: resolvedUseCase, guardrails: resolvedGuardrails)
    ModelCache.models[key] = model
    return model
}

/// Apple's built-in tools: on-device OCR and barcode reading (Vision) and the
/// Spotlight semantic index (local RAG). All macOS 27+. Unknown names are
/// rejected loudly — a silently ignored tool would mislead the caller into
/// thinking the model had a capability it did not.
@available(macOS 26.0, *)
private func toolsFor(_ names: [String]) throws -> [any Tool] {
    var out: [any Tool] = []
    for name in names {
        switch name {
        case "ocr":
            if #available(macOS 27.0, *) {
                #if canImport(_Vision_FoundationModels)
                out.append(OCRTool())
                #else
                throw ToolError.unsupported("ocr needs macOS 27 with Vision tools")
                #endif
            } else {
                throw ToolError.unsupported("ocr needs macOS 27 or later")
            }
        case "barcode":
            if #available(macOS 27.0, *) {
                #if canImport(_Vision_FoundationModels)
                out.append(BarcodeReaderTool())
                #else
                throw ToolError.unsupported("barcode needs macOS 27 with Vision tools")
                #endif
            } else {
                throw ToolError.unsupported("barcode needs macOS 27 or later")
            }
        case "spotlight":
            if #available(macOS 27.0, *) {
                #if canImport(_CoreSpotlight_FoundationModels)
                out.append(SpotlightSearchTool())
                #else
                throw ToolError.unsupported("spotlight needs macOS 27 with Spotlight tools")
                #endif
            } else {
                throw ToolError.unsupported("spotlight needs macOS 27 or later")
            }
        default:
            throw ToolError.unknown("unknown tool \"\(name)\" (want ocr, barcode, spotlight)")
        }
    }
    return out
}

private enum ToolError: Error {
    case unknown(String)
    case unsupported(String)
}

/// Coerce a JSON number regardless of int/double mismatch.
///
/// `JSONSerialization` produces `NSNumber`, but `as? Double` fails for an
/// integer-valued `NSNumber` (and vice versa), so a caller passing
/// `temperature: 1` or `maxTokens: 100.0` would silently get `nil`.
private func num(_ v: Any?) -> Double? { (v as? NSNumber)?.doubleValue }

/// Coerce a JSON number to `Int`, returning `nil` for missing or out-of-range
/// values rather than trapping or wrapping.
private func intNum(_ v: Any?) -> Int? {
    guard let n = v as? NSNumber else { return nil }
    let d = n.doubleValue
    guard d.isFinite, d >= Double(Int.min), d <= Double(Int.max) else { return nil }
    return n.intValue
}

/// Coerce a JSON number to `UInt64` for seeds, which may exceed `Int.max`.
private func u64Num(_ v: Any?) -> UInt64? { (v as? NSNumber)?.uint64Value }

/// Sampling mode.
///
/// `greedy` is deterministic but degenerates under guided generation — it padded
/// an unbounded array forever, then ran away inside a single string. The seeded
/// modes give the same determinism *without* that failure: `topK` with a seed
/// returned byte-identical output across three fresh sessions here. Note the
/// determinism depends on a fresh session per request, which is what this helper
/// does by default; reusing a session changes the transcript and with it the
/// output.
@available(macOS 26.0, *)
private func samplingFrom(_ value: Any?) -> GenerationOptions.SamplingMode? {
    guard let spec = value as? [String: Any], let mode = spec["mode"] as? String else { return nil }
    let seed = u64Num(spec["seed"])
    switch mode {
    case "greedy":
        return .greedy
    case "topK":
        return .random(top: intNum(spec["k"]) ?? 50, seed: seed)
    case "threshold":
        return .random(probabilityThreshold: num(spec["p"]) ?? 0.9, seed: seed)
    default:
        return nil
    }
}

@available(macOS 26.0, *)
private struct ImageSpec {
    var path: String
    var label: String
}

/// `images` accepts a plain path or `{"path":..,"label":..}`. Labels mirror
/// `fm --label`: they let the caller name attachments ("receipt", "chart")
/// so follow-up turns can refer to them. A missing file is an error, never a
/// silent drop.
@available(macOS 26.0, *)
private func imageSpecsFrom(_ value: Any?) -> [ImageSpec] {
    guard let raw = value as? [Any] else { return [] }
    var out: [ImageSpec] = []
    for (index, item) in raw.enumerated() {
        if let path = item as? String {
            out.append(ImageSpec(path: path, label: "image\(index + 1)"))
        } else if let dict = item as? [String: Any],
                  let path = dict["path"] as? String {
            let label = (dict["label"] as? String)?.isEmpty == false
                ? dict["label"] as! String : "image\(index + 1)"
            out.append(ImageSpec(path: path, label: label))
        }
    }
    return out
}

/// Build the prompt, attaching any images. Vision is macOS 27+; on 26 the paths
/// are reported as unsupported rather than silently dropped, because a caller
/// who asked about an image and got an answer that ignored it has been misled.
@available(macOS 26.0, *)
private func promptWith(text: String, imageSpecs: [ImageSpec]) -> Prompt {
    guard !imageSpecs.isEmpty else { return Prompt(text) }
    if #available(macOS 27.0, *) {
        let attachments = imageSpecs.map { spec in
            Attachment(imageURL: URL(fileURLWithPath: spec.path)).label(spec.label)
        }
        return Prompt {
            for attachment in attachments { attachment }
            text
        }
    }
    return Prompt(text)
}

@available(macOS 26.0, *)
private func stringArray(_ value: Any?) -> [String] {
    (value as? [Any] ?? []).compactMap { $0 as? String }
}

/// One request/response cycle. Kept separate from the transport so that
/// one-shot and serve modes cannot drift apart.
@available(macOS 26.0, *)
private func handle(
    envelope: [String: Any],
    schemaCache: inout [String: GenerationSchema]
) async {
    await handleEnvelope(envelope: envelope, schemaCache: &schemaCache, streaming: false)
}

@available(macOS 26.0, *)
private func handleEnvelope(
    envelope: [String: Any],
    schemaCache: inout [String: GenerationSchema],
    streaming: Bool
) async {
    let useCase = envelope["useCase"] as? String ?? "general"
    let guardrails = envelope["guardrails"] as? String ?? "default"
    let model = modelFor(useCase: useCase, guardrails: guardrails)

    switch model.availability {
    case .available:
        break
    case .unavailable(let reason):
        emit(["ok": false, "kind": "availability",
              "error": "the on-device model is unavailable: \(describe(reason))"])
        return
    }

    let instructions = envelope["instructions"] as? String ?? ""
    let promptText = envelope["prompt"] as? String ?? ""
    let imageSpecs = imageSpecsFrom(envelope["images"])
    let toolNames = stringArray(envelope["tools"])
    RequestContext.toolNames = toolNames
    let sessionId = envelope["sessionId"] as? String

    if !imageSpecs.isEmpty {
        if #available(macOS 27.0, *) {
            guard model.capabilities.contains(.vision) else {
                emit(["ok": false, "kind": "unsupported",
                      "error": "this model does not support vision"])
                return
            }
            for spec in imageSpecs where !FileManager.default.fileExists(atPath: spec.path) {
                emit(["ok": false, "kind": "unsupported",
                      "error": "image not found: \(spec.path)"])
                return
            }
        } else {
            emit(["ok": false, "kind": "unsupported",
                  "error": "images need macOS 27 or later"])
            return
        }
    }

    let tools: [any Tool]
    do {
        tools = try toolsFor(toolNames)
    } catch let err as ToolError {
        switch err {
        case .unknown(let m), .unsupported(let m):
            emit(["ok": false, "kind": "unsupported", "error": m])
            return
        }
    } catch {
        emit(["ok": false, "kind": "unsupported", "error": "\(error)"])
        return
    }
    if !tools.isEmpty {
        guard model.capabilities.contains(.toolCalling) else {
            emit(["ok": false, "kind": "unsupported",
                  "error": "this model does not support tool calling"])
            return
        }
    }

    // Token counting and prewarming share the model but not the generation path.
    // An empty op defaults to generate; an unknown op is an error rather than
    // silently running generate (which would hide a caller typo like
    // "countToken").
    let rawOp = envelope["op"] as? String
    let op = (rawOp == nil || rawOp == "") ? "generate" : rawOp!
    switch op {
    case "generate", "stream":
        break
    case "countTokens":
        guard #available(macOS 27.0, *) else {
            emit(["ok": false, "kind": "unsupported",
                  "error": "counting tokens needs macOS 27 or later"])
            return
        }
        do {
            // Instructions and schema ride in the same window as the prompt, so
            // a caller budgeting against contextSize needs all three counted.
            var total = try await model.tokenCount(for: promptWith(text: promptText,
                                                                   imageSpecs: imageSpecs))
            if !instructions.isEmpty {
                total += try await model.tokenCount(for: Instructions(instructions))
            }
            if !tools.isEmpty {
                total += (try? await model.tokenCount(for: tools)) ?? 0
            }
            emit(["ok": true, "tokens": total, "contextSize": model.contextSize])
        } catch {
            emit(["ok": false, "kind": classify(error).kind, "error": "\(error)"])
        }
        return

    case "prewarm":
        // Loads model assets now so the first real call does not pay for it.
        // Measured benefit is small once the assets are resident system-wide
        // (0.31s vs 0.36s for an unprewarmed first call here); the win is on a
        // genuinely cold system, where the first framework call took 7.8s.
        // The session is discarded on purpose; see sessionFor above.
        if tools.isEmpty {
            LanguageModelSession(model: model, instructions: instructions).prewarm()
        } else {
            LanguageModelSession(model: model, tools: tools, instructions: instructions).prewarm()
        }
        emit(["ok": true, "prewarmed": true])
        return

    case "history":
        guard let sid = sessionId, !sid.isEmpty else {
            emit(["ok": false, "kind": "unsupported",
                  "error": "history needs a sessionId"])
            return
        }
        if let entry = SessionStore.named[sid] {
            emit(["ok": true, "sessionId": sid, "instructions": entry.instructions,
                  "history": entry.history])
        } else {
            let stored = SessionStore.loadHistory(sid)
            emit(["ok": true, "sessionId": sid,
                  "instructions": stored.instructions ?? instructions,
                  "history": stored.history])
        }
        return

    case "reset":
        if let sid = sessionId, !sid.isEmpty {
            SessionStore.named.removeValue(forKey: sid)
            SessionStore.removeFile(sid)
        } else {
            SessionStore.named.removeAll()
        }
        emit(["ok": true, "reset": true])
        return

    default:
        emit(["ok": false, "kind": "unsupported",
              "error": "unsupported op \"\(op)\""])
        return
    }

    // Streaming + guided generation do not mix in v1: the schema stream yields
    // GeneratedContent snapshots whose partials are not plain-text deltas.
    // Reject loudly rather than emitting misleading partial JSON.
    let wantsStream = (op == "stream") || streaming
    if wantsStream,
       let schemaObject = envelope["schema"], !(schemaObject is NSNull) {
        emit(["ok": false, "kind": "unsupported",
              "error": "streaming with a schema is not supported; use generate for JSON"])
        return
    }

    // No schema means a free-text call. api-scribe only ever asked for JSON and
    // so required a schema on every request; `text()` needs the unconstrained
    // `respond(to:options:)` overload instead.
    var schema: GenerationSchema? = nil
    if let schemaObject = envelope["schema"], !(schemaObject is NSNull) {
        guard let schemaData = try? JSONSerialization.data(withJSONObject: schemaObject),
              let schemaKey = String(data: schemaData, encoding: .utf8) else {
            emit(["ok": false, "kind": "schema",
                  "error": "helper payload is missing a usable schema"])
            return
        }
        // A changing GenerationSchema costs only ~0.15s per call, so per-request
        // schemas are fine; this cache just makes a repeated one free.
        if let cached = schemaCache[schemaKey] {
            schema = cached
        } else {
            do {
                let decoded = try JSONDecoder().decode(GenerationSchema.self, from: schemaData)
                // Bound the cache: a long-lived server may see many schemas.
                if schemaCache.count >= 64 { schemaCache.removeAll() }
                schemaCache[schemaKey] = decoded
                schema = decoded
            } catch {
                emit(["ok": false, "kind": "schema",
                      "error": "Apple rejected the response schema: \(error)"])
                return
            }
        }
    }

    let options = GenerationOptions(
        samplingMode: samplingFrom(envelope["sampling"]),
        temperature: num(envelope["temperature"]),
        maximumResponseTokens: intNum(envelope["maxTokens"])
    )
    let reuse = envelope["reuseSession"] as? Bool ?? false
    // api-scribe passed false because it spelled the schema out in its own
    // system prompt. A library cannot assume that, and a schema's `description`
    // fields are how the decoder gets its generation guidance, so default true.
    let includeSchema = envelope["includeSchemaInPrompt"] as? Bool ?? true

    let session: LanguageModelSession
    // Cross-process continuity: a fresh native session (new process, or param
    // change) has no transcript, but the mirrored history survived on disk.
    // Reprise the recent turns as prompt context so `run --session` remembers
    // across CLI invocations too. Bounded to the last 10 turns; in-process
    // calls skip this and use the native transcript alone.
    var effectivePromptText = promptText
    if let sid = sessionId, !sid.isEmpty {
        let hadMemory = SessionStore.named[sid] != nil
        session = namedSessionFor(id: sid, model: model, instructions: instructions,
                                  useCase: useCase, guardrails: guardrails, tools: tools)
        if !hadMemory, let entry = SessionStore.named[sid], !entry.history.isEmpty {
            let recent = entry.history.suffix(10).map { turn -> String in
                let who = (turn["role"] == "user") ? "User" : "Assistant"
                return "\(who): \(turn["content"] ?? "")"
            }.joined(separator: "\n")
            effectivePromptText =
                "Previous conversation:\n\(recent)\n\nCurrent request:\n\(promptText)"
        }
    } else if tools.isEmpty {
        session = sessionFor(model: model, instructions: instructions, reuse: reuse)
    } else {
        session = LanguageModelSession(model: model, tools: tools, instructions: instructions)
    }
    let prompt = promptWith(text: effectivePromptText, imageSpecs: imageSpecs)

    if wantsStream {
        await handleStream(session: session, prompt: prompt, options: options,
                           sessionId: sessionId, promptText: promptText)
        return
    }

    do {
        let content: String
        if let schema {
            let response = try await session.respond(
                to: prompt, schema: schema, includeSchemaInPrompt: includeSchema, options: options
            )
            content = response.content.jsonString
        } else {
            // Non-streaming. `session.streamResponse` is the streaming path;
            // see handleStream below.
            let response = try await session.respond(to: prompt, options: options)
            content = response.content
        }
        recordTurn(sessionId: sessionId, prompt: promptText, content: content,
                   instructions: instructions)
        emit(["ok": true, "content": content])
    } catch {
        let (kind, resetDate) = classify(error)
        var payload: [String: Any] = ["ok": false, "kind": kind, "error": "\(error)"]
        if let resetDate { payload["resetDate"] = resetDate }
        emit(payload)
    }
}

@available(macOS 26.0, *)
private func recordTurn(sessionId: String?, prompt: String, content: String,
                        instructions: String) {
    guard let sid = sessionId, !sid.isEmpty else { return }
    if var entry = SessionStore.named[sid] {
        entry.history.append(["role": "user", "content": prompt])
        entry.history.append(["role": "assistant", "content": content])
        // Bound mirrored history: native transcript still holds the full
        // context for this process lifetime; the mirror is for `history` and
        // restart continuity, not inference.
        if entry.history.count > 200 { entry.history.removeFirst(entry.history.count - 200) }
        SessionStore.named[sid] = entry
        SessionStore.saveHistory(id: sid, instructions: entry.instructions,
                                 history: entry.history)
    }
}

/// Streaming generation. Each snapshot's `content` is the cumulative partial,
/// so deltas are computed by stripping the previous prefix; when the model
/// revises earlier text (rare for plain prose) the whole new partial is sent
/// so the client never silently drops a correction.
@available(macOS 26.0, *)
private func handleStream(session: LanguageModelSession, prompt: Prompt,
                          options: GenerationOptions, sessionId: String?,
                          promptText: String) async {
    do {
        let stream = session.streamResponse(to: prompt, options: options)
        var previous = ""
        var full = ""
        for try await snapshot in stream {
            let current: String = snapshot.content
            full = current
            let delta: String
            if current.hasPrefix(previous) {
                delta = String(current.dropFirst(previous.count))
            } else {
                delta = current
            }
            previous = current
            if !delta.isEmpty {
                emit(["ok": true, "delta": delta, "done": false])
            }
        }
        recordTurn(sessionId: sessionId, prompt: promptText, content: full,
                   instructions: "")
        // Refresh persisted instructions for named sessions without clobbering.
        emit(["ok": true, "content": full, "done": true])
    } catch {
        let (kind, resetDate) = classify(error)
        var payload: [String: Any] = ["ok": false, "kind": kind, "error": "\(error)"]
        if let resetDate { payload["resetDate"] = resetDate }
        emit(payload)
    }
}

private func describe(_ reason: SystemLanguageModel.Availability.UnavailableReason) -> String {
    switch reason {
    case .deviceNotEligible: return "deviceNotEligible"
    case .appleIntelligenceNotEnabled: return "appleIntelligenceNotEnabled"
    case .modelNotReady: return "modelNotReady"
    @unknown default: return "unknown"
    }
}

/// Private Cloud Compute quota, read without calling it.
///
/// PCC *inference* needs `com.apple.developer.private-cloud-compute`, which is
/// AMFI-restricted and unavailable to any installable package — that is why the
/// cloud tier goes through Shortcuts. But `quotaUsage` is readable from an
/// unentitled process, so a caller can see whether the cloud tier is worth
/// trying before spending a Shortcuts round trip on it.
@available(macOS 27.0, *)
private func cloudQuota() -> [String: Any] {
    let pcc = PrivateCloudComputeLanguageModel()
    var out: [String: Any] = ["isAvailable": pcc.isAvailable]
    let usage = pcc.quotaUsage
    switch usage.status {
    case .belowLimit(let below):
        out["status"] = "belowLimit"
        out["approachingLimit"] = below.isApproachingLimit
    case .limitReached:
        out["status"] = "limitReached"
    @unknown default:
        out["status"] = "unknown"
    }
    if let reset = usage.resetDate {
        out["resetDate"] = ISO8601DateFormatter().string(from: reset)
    }
    return out
}

@main
struct AppleLLMHelper {
    static func main() async {
        let model = SystemLanguageModel.default

        if CommandLine.arguments.contains("--probe") {
            var payload: [String: Any] = ["contextSize": model.contextSize]
            switch model.availability {
            case .available:
                payload["available"] = true
            case .unavailable(let reason):
                payload["available"] = false
                payload["reason"] = describe(reason)
            }
            if #available(macOS 27.0, *) {
                payload["variant"] = model.variant.displayName
                // What the model can actually do, rather than what the docs
                // imply. On this machine the on-device model reports vision and
                // tool calling but *not* reasoning, while PCC reports all four.
                let capabilities = model.capabilities
                payload["capabilities"] = [
                    "vision": capabilities.contains(.vision),
                    "guidedGeneration": capabilities.contains(.guidedGeneration),
                    "reasoning": capabilities.contains(.reasoning),
                    "toolCalling": capabilities.contains(.toolCalling),
                ]
                payload["useCases"] = ["general", "contentTagging"]
                payload["cloud"] = cloudQuota()
                payload["features"] = [
                    "streaming": true,
                    "sessions": true,
                    "history": true,
                    "labelledAttachments": true,
                    "builtInTools": ["ocr", "barcode", "spotlight"],
                ]
            }
            emit(payload)
            return
        }

        var schemaCache: [String: GenerationSchema] = [:]

        // Serve mode: one request per stdin line, one response per stdout line,
        // for as long as the parent keeps the pipe open. Spawning a process per
        // request instead makes the model reload between calls, which measured
        // at ~17s per request against ~1.5s once it is resident.
        //
        // Streaming is the one exception to one-line-per-request: op "stream"
        // emits N {"delta","done":false} lines plus a final {"content",
        // "done":true}. The parent must keep reading until done:true before
        // sending the next request; the next stdin line is not consumed until
        // the stream completes, so replies cannot interleave.
        if CommandLine.arguments.contains("--serve") {
            while let line = readLine(strippingNewline: true) {
                if line.isEmpty { continue }
                guard let data = line.data(using: .utf8),
                      let envelope = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
                else {
                    emit(["ok": false, "error": "helper could not parse a request line as JSON"])
                    continue
                }
                await handle(envelope: envelope, schemaCache: &schemaCache)
            }
            return
        }

        let input = FileHandle.standardInput.readDataToEndOfFile()
        guard let envelope = (try? JSONSerialization.jsonObject(with: input)) as? [String: Any] else {
            emit(["ok": false, "error": "helper could not parse its stdin payload as JSON"])
            return
        }

        await handle(envelope: envelope, schemaCache: &schemaCache)
    }
}
