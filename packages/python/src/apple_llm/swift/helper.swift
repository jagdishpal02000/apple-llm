//
// apple-llm helper — the whole on-device path.
//
// This file is the single source of truth: `scripts/embed-helper.mjs` copies it verbatim
// into both the npm and the pip package, and a test in each asserts the copies
// still hash equal to this file.
//
// It needs no entitlements and no license agreement — unlike `/usr/bin/fm`,
// which ships with macOS 27 but is gated behind a machine-wide `sudo fm
// license`. Do not depend on `fm`.
//
// Protocol (version 2; every version-1 request still works unchanged):
//   helper --probe   -> stdout: one JSON object describing this machine.
//   helper --serve   -> one request JSON per stdin line; for each request, zero
//                       or more *event* lines ("done":false) and then exactly one
//                       *final* line (anything without "done":false), in order.
//                       Plain generate emits no events, so a version-1 client
//                       that reads one line per request keeps working.
//                       The client must not send the next request until the
//                       final line arrives. It MAY send control lines (cancel,
//                       toolResult) while a request is in flight: stdin is read
//                       on its own thread, so those are seen mid-generation.
//   helper           -> one request on stdin, one response on stdout
//                       ("stream" collapses to a single final response here,
//                       since one-shot stdout is not incremental; function
//                       tools need --serve, since stdin is already consumed).
//
//   request:  {"op"?:"generate"|"stream"|"countTokens"|"prewarm"|"history"|"reset",
//                       // default generate
//              "id"?:string,                         // target for "cancel"
//              "instructions":string,"prompt":string,"schema"?:object|null,
//              "temperature"?:number,"maxTokens"?:number,
//              "includeSchemaInPrompt"?:bool,"reuseSession"?:bool,
//              "sessionId"?:string,                  // multi-turn conversation
//              "history"?:[{"role":"user"|"assistant","content":string,
//                           "toolCalls"?:[{id,name,arguments}]}
//                         |{"role":"tool","toolCallId":string,"name":string,
//                           "content":string}],     // stateless multi-turn
//              "trimHistory"?:bool,                  // drop oldest turns to fit
//              "tools"?:["ocr"|"barcode"|"spotlight"], // built-in FM tools, 27+
//              "functions"?:[{"name":string,"description":string,
//                             "parameters":object}], // tools run by the client
//              "images"?:[string|{"path":string,"label"?:string}],
//                                                    // file paths, macOS 27+
//              "useCase"?:"general"|"contentTagging",
//              "guardrails"?:"default"|"permissive",
//              "sampling"?:{"mode":"greedy"}
//                        | {"mode":"topK","k":int,"seed"?:int}
//                        | {"mode":"threshold","p":number,"seed"?:int}}
//   control:  {"op":"cancel","id"?:string}           // no reply line of its own
//           | {"op":"toolResult","callId":string,"output"?:string,
//              "isError"?:bool,"stop"?:bool}         // answers a toolCall event
//   event:    {"ok":true,"delta":string,"done":false}       // stream, text
//           | {"ok":true,"partial":string,"done":false}     // stream, schema:
//                                                           //  JSON so far
//           | {"ok":true,"toolCall":{"id","name","arguments"},"done":false}
//   final:    {"ok":true,"content":string,               // generate / stream
//              "done"?:true,"finishReason"?:"stop"|"length"|"toolCalls",
//              "usage"?:{inputTokens,outputTokens,cachedInputTokens},
//              "toolCalls"?:[{id,name,arguments,output?}],
//              "trimmedTurns"?:int}
//           | {"ok":true,"tokens":int,"contextSize":int} // countTokens
//           | {"ok":true,"prewarmed":true}                // prewarm
//           | {"ok":true,"history":[{role,content}],      // history
//              "instructions":string,"sessionId":string}
//           | {"ok":true,"reset":true}                    // reset
//           | {"ok":false,"error":string,"kind"?:string}
//
//   `kind` is one of availability | schema | context | quota | guardrail |
//   timeout | unsupported | tool | cancelled | busy | generation, so callers can raise
//   typed errors instead of matching on strings. A quota error may carry
//   `resetDate`; a context error may carry `contextSize` and `tokenCount`.
//
//   A function tool call is answered by the client with a toolResult line. A
//   reply with "stop":true ends generation there with finishReason "toolCalls"
//   and empty content — the client executes the call itself and continues the
//   conversation later through `history`, which is how an OpenAI-style caller
//   (or the Vercel AI SDK) drives tools.
//
//   --probe reports availability, contextSize, variant, the model's
//   capabilities (vision / guidedGeneration / reasoning / toolCalling) and,
//   under "cloud", the same for Private Cloud Compute plus its quota. Those are
//   readable without the `com.apple.developer.private-cloud-compute`
//   entitlement that blocks PCC *inference*, so they are the first-party cloud
//   state available to an unentitled process. It also reports `features` so callers can fail
//   fast with a message instead of a stale binary's silent misbehaviour.
//

import Foundation
import FoundationModels
#if canImport(_Vision_FoundationModels)
import _Vision_FoundationModels
#endif
#if canImport(_CoreSpotlight_FoundationModels)
import _CoreSpotlight_FoundationModels
#endif

/// Serialises writes. Function tools can run concurrently when the model asks
/// for several at once, and two unsynchronised `print`s can interleave bytes
/// inside a line, which would corrupt the framing for every later request.
private let emitLock = NSLock()

private func emit(_ object: [String: Any]) {
    var text = "{\"ok\":false,\"error\":\"helper could not encode its response\"}"
    if let data = try? JSONSerialization.data(withJSONObject: object),
       let encoded = String(data: data, encoding: .utf8) {
        text = encoded
    }
    emitLock.lock()
    // stdout is a pipe here, so it is fully buffered; the parent is waiting on
    // this line and would otherwise see nothing until the process exits.
    print(text)
    fflush(stdout)
    emitLock.unlock()
}

private func jsonValue(_ text: String) -> Any? {
    guard let data = text.data(using: .utf8) else { return nil }
    return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
}

private func jsonText(_ value: Any) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed]),
          let text = String(data: data, encoding: .utf8) else { return "{}" }
    return text
}

/// A reply to a function tool call, sent by the parent as a toolResult line
/// while the request is still in flight.
private enum ToolReply {
    case output(String)
    case failure(String)
    case stop
    case cancelled
}

/// Routes the control lines that have to be seen *while* a request runs —
/// cancellation and tool results. Everything else is a request and waits its
/// turn in the serial queue.
///
/// Why a reader thread rather than reading stdin between requests: a tool call
/// suspends generation until the parent answers, and the answer arrives on
/// stdin. Reading stdin only between requests would deadlock the first tool
/// call; it would also make cancellation impossible, since the cancel line
/// would sit unread until the generation it was meant to stop had finished.
private final class Control: @unchecked Sendable {
    static let shared = Control()

    private let lock = NSLock()
    private var currentId: String?
    private var currentTask: Task<Void, Never>?
    /// Cancels that arrived before their request started. Bounded below.
    private var cancelledEarly: Set<String> = []
    private var waiting: [String: CheckedContinuation<ToolReply, Never>] = [:]
    private var closed = false

    /// True when `line` was a control message and has been handled here.
    func intercept(_ line: String) -> Bool {
        // Cheap pre-check: a prompt can be tens of KB, and most lines are
        // requests, which never carry these two ops.
        guard line.contains("\"cancel\"") || line.contains("\"toolResult\""),
              let object = jsonValue(line) as? [String: Any],
              let op = object["op"] as? String else { return false }
        switch op {
        case "cancel":
            cancel(id: object["id"] as? String)
            return true
        case "toolResult":
            guard let callId = object["callId"] as? String else { return true }
            let reply: ToolReply
            if object["stop"] as? Bool == true {
                reply = .stop
            } else if object["isError"] as? Bool == true {
                reply = .failure(object["output"] as? String ?? "the tool failed")
            } else {
                reply = .output(object["output"] as? String ?? "")
            }
            deliver(callId: callId, reply: reply)
            return true
        default:
            return false
        }
    }

    private func cancel(id: String?) {
        lock.lock()
        defer { lock.unlock() }
        if id == nil || id == currentId {
            currentTask?.cancel()
            // A tool call parked on the parent would otherwise never notice.
            for continuation in waiting.values { continuation.resume(returning: .cancelled) }
            waiting.removeAll()
        } else if let id {
            // The request line may already be queued but not yet started.
            if cancelledEarly.count > 256 { cancelledEarly.removeAll() }
            cancelledEarly.insert(id)
        }
    }

    private func deliver(callId: String, reply: ToolReply) {
        lock.lock()
        let continuation = waiting.removeValue(forKey: callId)
        lock.unlock()
        continuation?.resume(returning: reply)
    }

    /// Park a tool call until the parent answers it. `announce` emits the
    /// toolCall event, and runs only once the call is registered, so an answer
    /// cannot arrive before anything is waiting for it.
    func awaitToolResult(callId: String, announce: () -> Void) async -> ToolReply {
        await withCheckedContinuation { (continuation: CheckedContinuation<ToolReply, Never>) in
            lock.lock()
            if closed || currentTask?.isCancelled == true {
                lock.unlock()
                continuation.resume(returning: .cancelled)
                return
            }
            waiting[callId] = continuation
            lock.unlock()
            announce()
        }
    }

    /// Run one request as a cancellable task, recording it as the target of
    /// any cancel line that arrives meanwhile.
    func run(id: String?, _ body: @escaping () async -> Void) async {
        let task: Task<Void, Never>? = lock.withLock {
            if let id, cancelledEarly.remove(id) != nil { return nil }
            let task = Task { await body() }
            currentId = id
            currentTask = task
            return task
        }
        guard let task else {
            emit(["ok": false, "kind": "cancelled", "error": "the request was cancelled"])
            return
        }
        await task.value
        lock.withLock {
            currentId = nil
            currentTask = nil
        }
    }

    /// The parent closed stdin: nothing will ever answer a parked tool call.
    func stdinClosed() {
        lock.lock()
        closed = true
        currentTask?.cancel()
        for continuation in waiting.values { continuation.resume(returning: .cancelled) }
        waiting.removeAll()
        lock.unlock()
    }
}

/// Why a function tool call did not produce output.
private enum ClientToolFailure: Error, CustomStringConvertible {
    case failed(name: String, message: String)
    /// The client will run the call itself; end generation here.
    case stop

    var description: String {
        switch self {
        case .failed(let name, let message): return "tool \"\(name)\" failed: \(message)"
        case .stop: return "stopped for a client tool call"
        }
    }
}

/// A tool defined by the client and executed there. The model's arguments are
/// sent up as a toolCall event and the call suspends until a toolResult line
/// answers it. `parameters` is the client's JSON Schema, already rewritten into
/// Apple's dialect, so the arguments are decoded under the same constrained
/// decoding guarantee as `json()`.
@available(macOS 26.0, *)
private struct ClientTool: Tool {
    typealias Arguments = GeneratedContent
    typealias Output = String

    let name: String
    let description: String
    let parameters: GenerationSchema

    func call(arguments: GeneratedContent) async throws -> String {
        let callId = "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24)
        let args = jsonValue(arguments.jsonString) ?? [String: Any]()
        let reply = await Control.shared.awaitToolResult(callId: callId) {
            emit(["ok": true, "done": false,
                  "toolCall": ["id": callId, "name": name, "arguments": args]])
        }
        switch reply {
        case .output(let text): return text
        case .failure(let message): throw ClientToolFailure.failed(name: name, message: message)
        case .stop: throw ClientToolFailure.stop
        case .cancelled: throw CancellationError()
        }
    }
}

/// Optional session reuse, off by default.
///
/// An earlier prototype kept one session alive while the instructions were
/// unchanged, on the finding that allocating a fresh LanguageModelSession per
/// request made every other call stall ~16s while assets cycled — a strict
/// 17s/1.5s alternation, uncorrelated with prompt size.
///
/// That did not reproduce here. Measured on macOS 27 / M-series inside one
/// long-lived process, 6 calls per arm: session reused -> median 0.63s; a fresh
/// session every call -> median 0.64s, max 0.68s, no variance. The alternation
/// would have hit three times in six calls. The load-bearing part of the fix
/// appears to be the long-lived *process* (~17s a call was measured when a
/// helper was spawned per request, against ~1.5s once resident), not the
/// shared session.
///
/// So the default here is a fresh session per request, because a library cannot
/// let two unrelated calls share a transcript — the prototype could, since every
/// one of its prompts carried its own whole context. The old behaviour is kept
/// behind `reuseSession` rather than deleted: the original finding was measured
/// too, possibly on macOS 26, and the doubt is worth preserving. If per-call
/// latency ever regresses to ~17s, try `reuseSession: true` first.
///
/// Named sessions (`sessionId`) are the multi-turn counterpart: same process,
/// but the session is keyed by id so a conversation accumulates a native
/// transcript across calls. History is also mirrored to
/// `~/Library/Caches/apple-llm/sessions/<id>.json`. When the native session
/// has to be rebuilt — a helper restart, changed instructions or tools, or a
/// context overflow — it is rebuilt as a native transcript from that mirror,
/// oldest turns dropped until it fits, so a conversation survives all three.
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

/// Decoded schemas, keyed by their JSON text. A changing GenerationSchema costs
/// only ~0.15s per call, so per-request schemas are fine; this just makes a
/// repeated one free. Bounded, since a long-lived server may see many.
@available(macOS 26.0, *)
private final class SchemaCache {
    static var decoded: [String: GenerationSchema] = [:]

    static func decode(_ object: Any) throws -> GenerationSchema {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        let key = String(data: data, encoding: .utf8) ?? ""
        if let cached = decoded[key] { return cached }
        let schema = try JSONDecoder().decode(GenerationSchema.self, from: data)
        if decoded.count >= 64 { decoded.removeAll() }
        decoded[key] = schema
        return schema
    }
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

/// Turn history turns into native transcript entries.
///
/// A native transcript rather than a "Previous conversation:" preamble: the
/// model was trained on its own turn structure, and a preamble also costs the
/// history twice when a tool call echoes the prompt back.
@available(macOS 26.0, *)
private func transcriptEntries(from history: [[String: Any]]) -> [Transcript.Entry] {
    var entries: [Transcript.Entry] = []
    for turn in history {
        let role = turn["role"] as? String ?? ""
        let content = turn["content"] as? String ?? ""
        switch role {
        case "user":
            entries.append(.prompt(Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: content))])))
        case "assistant":
            if !content.isEmpty {
                entries.append(.response(Transcript.Response(
                    assetIDs: [], segments: [.text(Transcript.TextSegment(content: content))])))
            }
            let calls = (turn["toolCalls"] as? [[String: Any]] ?? []).compactMap {
                call -> Transcript.ToolCall? in
                guard let name = call["name"] as? String,
                      let arguments = try? GeneratedContent(json: jsonText(call["arguments"] ?? [String: Any]()))
                else { return nil }
                return Transcript.ToolCall(id: call["id"] as? String ?? UUID().uuidString,
                                           toolName: name, arguments: arguments)
            }
            if !calls.isEmpty { entries.append(.toolCalls(Transcript.ToolCalls(calls))) }
        case "tool":
            entries.append(.toolOutput(Transcript.ToolOutput(
                id: turn["toolCallId"] as? String ?? UUID().uuidString,
                toolName: turn["name"] as? String ?? "tool",
                segments: [.text(Transcript.TextSegment(content: content))])))
        default:
            continue
        }
    }
    return entries
}

/// Drop the oldest turns until instructions, tools, history, prompt and the
/// response budget all fit in the context window. A turn is dropped whole —
/// from one user prompt up to the next — so a tool call is never left without
/// its output. Returns how many user turns were dropped, which the caller
/// reports: trimming is never silent.
///
/// Runs only after a context overflow (or when rebuilding a named session),
/// never ahead of every call, and binary-searches the cut: log2(turns) token
/// counts rather than one per dropped turn.
///
/// macOS 27 only, since it needs `tokenCount`; on 26 the history is left as is
/// and an overflow surfaces as a ContextLengthError.
@available(macOS 26.0, *)
private func fitted(
    _ entries: [Transcript.Entry], model: SystemLanguageModel, instructions: String,
    tools: [any Tool], prompt: Prompt, reserve: Int
) async -> (entries: [Transcript.Entry], dropped: Int) {
    guard #available(macOS 27.0, *), !entries.isEmpty else { return (entries, 0) }
    var fixed = (try? await model.tokenCount(for: prompt)) ?? 0
    if !instructions.isEmpty {
        fixed += (try? await model.tokenCount(for: Instructions(instructions))) ?? 0
    }
    if !tools.isEmpty { fixed += (try? await model.tokenCount(for: tools)) ?? 0 }
    let budget = model.contextSize - reserve - fixed
    func fits(_ slice: ArraySlice<Transcript.Entry>) async -> Bool {
        if slice.isEmpty { return true }
        let used = (try? await model.tokenCount(for: Array(slice))) ?? Int.max
        return used <= budget
    }
    if await fits(entries[...]) { return (entries, 0) }
    // Cut points: the start of each user turn, plus "drop everything".
    var cuts = entries.indices.filter { $0 > 0 && isPrompt(entries[$0]) }
    cuts.append(entries.count)
    var low = 0
    var high = cuts.count - 1
    while low < high {
        let mid = (low + high) / 2
        if await fits(entries[cuts[mid]...]) { high = mid } else { low = mid + 1 }
    }
    return (Array(entries[cuts[low]...]), low + 1)
}

@available(macOS 26.0, *)
private func isPrompt(_ entry: Transcript.Entry) -> Bool {
    if case .prompt = entry { return true }
    return false
}

@available(macOS 26.0, *)
private func transcriptSession(
    model: SystemLanguageModel, instructions: String, tools: [any Tool],
    entries: [Transcript.Entry]
) -> LanguageModelSession {
    if entries.isEmpty {
        return tools.isEmpty
            ? LanguageModelSession(model: model, instructions: instructions)
            : LanguageModelSession(model: model, tools: tools, instructions: instructions)
    }
    var all: [Transcript.Entry] = []
    if !instructions.isEmpty || !tools.isEmpty {
        let segments: [Transcript.Segment] =
            instructions.isEmpty ? [] : [.text(Transcript.TextSegment(content: instructions))]
        all.append(.instructions(Transcript.Instructions(
            segments: segments,
            toolDefinitions: tools.map { Transcript.ToolDefinition(tool: $0) })))
    }
    all.append(contentsOf: entries)
    return LanguageModelSession(model: model, tools: tools, transcript: Transcript(entries: all))
}

/// Named multi-turn session. Tools are fixed at session construction, so a
/// changed tool set recreates the native session. Recreation (first use in
/// this process, changed parameters, or `rebuild` after a context overflow)
/// replays the mirrored history as a native transcript, trimmed to fit, so
/// nothing the caller said is silently dropped from `history` and the model
/// keeps as much of the conversation as the window allows.
@available(macOS 26.0, *)
private func namedSessionFor(
    id: String, model: SystemLanguageModel, instructions: String, fingerprint want: String,
    tools: [any Tool], prompt: Prompt, reserve: Int, rebuild: Bool
) async -> (session: LanguageModelSession, dropped: Int) {
    if !rebuild, let existing = SessionStore.named[id], existing.fingerprint == want {
        return (existing.session, 0)
    }
    let history = SessionStore.named[id]?.history ?? SessionStore.loadHistory(id).history
    let replay = transcriptEntries(from: history.suffix(40).map { $0 as [String: Any] })
    let (kept, dropped) = await fitted(replay, model: model, instructions: instructions,
                                       tools: tools, prompt: prompt, reserve: reserve)
    let session = transcriptSession(model: model, instructions: instructions, tools: tools, entries: kept)
    SessionStore.named[id] = NamedSession(
        session: session, instructions: instructions,
        fingerprint: want, history: history)
    // Persist immediately so a fresh id is visible to `history` even before
    // its first turn completes.
    SessionStore.saveHistory(id: id, instructions: instructions, history: history)
    return (session, dropped)
}

/// Classify a generation failure so callers can raise a typed error rather than
/// matching on a message Apple may reword in any OS release.
///
/// macOS 27 replaced `LanguageModelSession.GenerationError` with
/// `LanguageModelError`; both are consulted so one helper serves 26 and 27.
/// `rateLimited` carries a `resetDate`, which is what makes an on-device quota
/// error actionable rather than just a failure.
@available(macOS 26.0, *)
private func classify(_ error: Error) -> (kind: String, extra: [String: Any]) {
    if error is CancellationError { return ("cancelled", [:]) }
    // Several processes using the model at once: the model manager turns a
    // request away with ModelManagerError 1042, wrapped in an otherwise
    // uninformative LanguageModelError -1. Nothing was generated, so the
    // client can retry it; naming it lets the client do that safely.
    let described = String(describing: error)
    if described.contains("ModelManagerError"), described.contains("1042") { return ("busy", [:]) }
    if let toolError = error as? LanguageModelSession.ToolCallError {
        if toolError.underlyingError is CancellationError { return ("cancelled", [:]) }
        if let client = toolError.underlyingError as? ClientToolFailure, case .stop = client {
            return ("stopped", [:])
        }
        return ("tool", ["tool": toolError.tool.name])
    }
    if #available(macOS 27.0, *) {
        if let modern = error as? LanguageModelError {
            switch modern {
            case .contextSizeExceeded(let info):
                return ("context", ["contextSize": info.contextSize, "tokenCount": info.tokenCount])
            case .rateLimited(let info):
                if let reset = info.resetDate {
                    return ("quota", ["resetDate": ISO8601DateFormatter().string(from: reset)])
                }
                return ("quota", [:])
            case .guardrailViolation, .refusal: return ("guardrail", [:])
            case .timeout: return ("timeout", [:])
            case .unsupportedCapability, .unsupportedGenerationGuide,
                 .unsupportedLanguageOrLocale, .unsupportedTranscriptContent:
                return ("unsupported", [:])
            @unknown default: return ("generation", [:])
            }
        }
    }
    return (legacyKind(error), [:])
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

@available(macOS 26.0, *)
private func failure(_ error: Error) -> [String: Any] {
    let (kind, extra) = classify(error)
    var payload: [String: Any] = ["ok": false, "kind": kind, "error": "\(error)"]
    if kind == "cancelled" { payload["error"] = "the request was cancelled" }
    for (key, value) in extra { payload[key] = value }
    return payload
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

/// Function tools from the request. Their parameter schemas go through Apple's
/// decoder like any response schema, and a rejected one is a schema error
/// naming the tool rather than a generic failure.
@available(macOS 26.0, *)
private func functionToolsFrom(_ value: Any?) throws -> [any Tool] {
    guard let specs = value as? [[String: Any]] else { return [] }
    var out: [any Tool] = []
    for spec in specs {
        guard let name = spec["name"] as? String, !name.isEmpty else {
            throw ToolError.unknown("every function tool needs a name")
        }
        let parameters = spec["parameters"] ?? [
            "type": "object", "title": "\(name)Arguments", "properties": [String: Any](),
            "x-order": [String](), "required": [String](), "additionalProperties": false,
        ]
        let schema: GenerationSchema
        do {
            schema = try SchemaCache.decode(parameters)
        } catch {
            throw ToolError.schema("Apple rejected the parameters schema of tool \"\(name)\": \(error)")
        }
        out.append(ClientTool(name: name, description: spec["description"] as? String ?? "",
                              parameters: schema))
    }
    return out
}

private enum ToolError: Error {
    case unknown(String)
    case unsupported(String)
    case schema(String)
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

/// Every tool call made while producing one response, with its output where
/// the transcript recorded one — built-in and function tools alike, so a
/// caller can see *why* an answer says what it says.
@available(macOS 26.0, *)
private func toolActivity(_ entries: some Sequence<Transcript.Entry>) -> [[String: Any]] {
    var calls: [[String: Any]] = []
    var outputs: [String: String] = [:]
    for entry in entries {
        switch entry {
        case .toolCalls(let batch):
            for call in batch {
                calls.append(["id": call.id, "name": call.toolName,
                              "arguments": jsonValue(call.arguments.jsonString) ?? NSNull()])
            }
        case .toolOutput(let output):
            outputs[output.id] = output.segments.compactMap { segment -> String? in
                if case .text(let text) = segment { return text.content }
                return nil
            }.joined()
        default:
            continue
        }
    }
    return calls.map { call in
        var out = call
        if let id = call["id"] as? String, let text = outputs[id] { out["output"] = text }
        return out
    }
}

/// What one successful generation produced, before it is framed as a line.
private struct Outcome {
    var content: String
    var usage: [String: Any]?
    var toolCalls: [[String: Any]] = []
    var finishReason = "stop"
}

@available(macOS 27.0, *)
private func usageOf(_ usage: LanguageModelSession.Usage) -> [String: Any] {
    ["inputTokens": usage.input.totalTokenCount,
     "cachedInputTokens": usage.input.cachedTokenCount,
     "outputTokens": usage.output.totalTokenCount]
}

/// A response that used its whole token budget was cut off, not finished.
/// Reported so a caller can tell a short answer from a truncated one.
private func finishReason(usage: [String: Any]?, maxTokens: Int?) -> String {
    if let maxTokens, let out = usage?["outputTokens"] as? Int, out >= maxTokens { return "length" }
    return "stop"
}

/// One request/response cycle. Kept separate from the transport so that
/// one-shot and serve modes cannot drift apart.
@available(macOS 26.0, *)
private func handle(envelope: [String: Any]) async {
    await handleEnvelope(envelope: envelope, streaming: false)
}

@available(macOS 26.0, *)
private func handleEnvelope(envelope: [String: Any], streaming: Bool) async {
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
    let sessionId = envelope["sessionId"] as? String
    let historyTurns = envelope["history"] as? [[String: Any]] ?? []

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
        tools = try toolsFor(toolNames) + functionToolsFrom(envelope["functions"])
    } catch let err as ToolError {
        switch err {
        case .unknown(let m), .unsupported(let m):
            emit(["ok": false, "kind": "unsupported", "error": m])
        case .schema(let m):
            emit(["ok": false, "kind": "schema", "error": m])
        }
        return
    } catch {
        emit(["ok": false, "kind": "unsupported", "error": "\(error)"])
        return
    }
    if !tools.isEmpty {
        if #available(macOS 27.0, *) {
            guard model.capabilities.contains(.toolCalling) else {
                emit(["ok": false, "kind": "unsupported",
                      "error": "this model does not support tool calling"])
                return
            }
        }
    }

    // A response schema, decoded once for every op that needs it.
    var schema: GenerationSchema? = nil
    if let schemaObject = envelope["schema"], !(schemaObject is NSNull) {
        do {
            schema = try SchemaCache.decode(schemaObject)
        } catch {
            emit(["ok": false, "kind": "schema",
                  "error": "Apple rejected the response schema: \(error)"])
            return
        }
    }
    let prompt = promptWith(text: promptText, imageSpecs: imageSpecs)
    let history = transcriptEntries(from: historyTurns)

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
            // Instructions, tools, schema and history all ride in the same
            // window as the prompt, so a caller budgeting against contextSize
            // needs every one of them counted.
            var total = try await model.tokenCount(for: prompt)
            if !instructions.isEmpty {
                total += try await model.tokenCount(for: Instructions(instructions))
            }
            if !tools.isEmpty {
                total += (try? await model.tokenCount(for: tools)) ?? 0
            }
            if let schema {
                total += (try? await model.tokenCount(for: schema)) ?? 0
            }
            if !history.isEmpty {
                total += (try? await model.tokenCount(for: history)) ?? 0
            }
            emit(["ok": true, "tokens": total, "contextSize": model.contextSize])
        } catch {
            emit(failure(error))
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

    let wantsStream = (op == "stream") || streaming
    let maxTokens = intNum(envelope["maxTokens"])
    let options = GenerationOptions(
        samplingMode: samplingFrom(envelope["sampling"]),
        temperature: num(envelope["temperature"]),
        maximumResponseTokens: maxTokens
    )
    let reuse = envelope["reuseSession"] as? Bool ?? false
    // A caller that spells the schema out in its own system prompt could pass
    // false. A library cannot assume that, and a schema's `description`
    // fields are how the decoder gets its generation guidance, so default true.
    let includeSchema = envelope["includeSchemaInPrompt"] as? Bool ?? true
    // Room left for the response when trimming history to fit.
    let reserve = min(maxTokens ?? 1024, model.contextSize / 2)
    let fingerprint = [useCase, guardrails, toolNames.sorted().joined(separator: ","),
                       jsonText(envelope["functions"] ?? [Any]()), instructions].joined(separator: "|")

    var trimmed = 0
    func makeSession(rebuild: Bool) async -> LanguageModelSession {
        if let sid = sessionId, !sid.isEmpty {
            let (named, dropped) = await namedSessionFor(
                id: sid, model: model, instructions: instructions, fingerprint: fingerprint,
                tools: tools, prompt: prompt, reserve: reserve, rebuild: rebuild)
            trimmed += dropped
            return named
        }
        if !history.isEmpty {
            // Trimmed only on the retry after an overflow: counting tokens up
            // front would tax every call to learn, almost always, that it fits.
            var entries = history
            if rebuild {
                let (kept, dropped) = await fitted(history, model: model, instructions: instructions,
                                                   tools: tools, prompt: prompt, reserve: reserve)
                entries = kept
                trimmed += dropped
            }
            return transcriptSession(model: model, instructions: instructions, tools: tools, entries: entries)
        }
        if tools.isEmpty { return sessionFor(model: model, instructions: instructions, reuse: reuse) }
        return LanguageModelSession(model: model, tools: tools, instructions: instructions)
    }

    var emittedEvents = false
    func generate(on session: LanguageModelSession) async throws -> Outcome {
        if wantsStream {
            return try await streamed(session: session, prompt: prompt, schema: schema,
                                      includeSchema: includeSchema, options: options,
                                      maxTokens: maxTokens, onEvent: { emittedEvents = true })
        }
        if let schema {
            let response = try await session.respond(
                to: prompt, schema: schema, includeSchemaInPrompt: includeSchema, options: options)
            try Task.checkCancellation()
            var outcome = Outcome(content: response.content.jsonString)
            outcome.toolCalls = toolActivity(response.transcriptEntries)
            if #available(macOS 27.0, *) { outcome.usage = usageOf(response.usage) }
            outcome.finishReason = finishReason(usage: outcome.usage, maxTokens: maxTokens)
            return outcome
        }
        let response = try await session.respond(to: prompt, options: options)
        try Task.checkCancellation()
        var outcome = Outcome(content: response.content)
        outcome.toolCalls = toolActivity(response.transcriptEntries)
        if #available(macOS 27.0, *) { outcome.usage = usageOf(response.usage) }
        outcome.finishReason = finishReason(usage: outcome.usage, maxTokens: maxTokens)
        return outcome
    }

    // A named session recovers from overflow on its own: it is stateful, so
    // without this one long turn would leave it permanently unusable.
    // Stateless history is trimmed only when the caller opted in.
    let canTrim = (sessionId?.isEmpty == false)
        || (!history.isEmpty && envelope["trimHistory"] as? Bool == true)
    var outcome: Outcome
    do {
        do {
            outcome = try await generate(on: await makeSession(rebuild: false))
        } catch let error where classify(error).kind == "context" && canTrim && !emittedEvents {
            // The conversation outgrew the window. Rebuild it from the newest
            // turns that still fit and try once more. Only when nothing has
            // been streamed yet: a retry after deltas would duplicate output.
            outcome = try await generate(on: await makeSession(rebuild: true))
        }
    } catch {
        if classify(error).kind == "stopped" {
            // The client is running a tool call itself; it saw the call as an
            // event and will continue through `history`.
            var payload: [String: Any] = ["ok": true, "content": "", "finishReason": "toolCalls"]
            if wantsStream { payload["done"] = true }
            if trimmed > 0 { payload["trimmedTurns"] = trimmed }
            emit(payload)
            return
        }
        var payload = failure(error)
        if trimmed > 0 { payload["trimmedTurns"] = trimmed }
        emit(payload)
        return
    }

    recordTurn(sessionId: sessionId, prompt: promptText, content: outcome.content)
    var payload: [String: Any] = ["ok": true, "content": outcome.content,
                                  "finishReason": outcome.finishReason]
    if wantsStream { payload["done"] = true }
    if let usage = outcome.usage { payload["usage"] = usage }
    if !outcome.toolCalls.isEmpty { payload["toolCalls"] = outcome.toolCalls }
    if trimmed > 0 { payload["trimmedTurns"] = trimmed }
    emit(payload)
}

@available(macOS 26.0, *)
private func recordTurn(sessionId: String?, prompt: String, content: String) {
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

/// Streaming generation. Text snapshots carry the cumulative partial, so deltas
/// are computed by stripping the previous prefix; when the model revises
/// earlier text (rare for plain prose) the whole new partial is sent so the
/// client never silently drops a correction.
///
/// With a schema, each snapshot is the JSON generated so far, sent whole as
/// `partial`: a partial object is a usable thing to render, where a raw text
/// delta of JSON is not.
@available(macOS 26.0, *)
private func streamed(
    session: LanguageModelSession, prompt: Prompt, schema: GenerationSchema?,
    includeSchema: Bool, options: GenerationOptions, maxTokens: Int?,
    onEvent: () -> Void
) async throws -> Outcome {
    if let schema {
        let stream = session.streamResponse(to: prompt, schema: schema,
                                            includeSchemaInPrompt: includeSchema, options: options)
        var last = ""
        var outcome = Outcome(content: "")
        for try await snapshot in stream {
            try Task.checkCancellation()
            let json = snapshot.rawContent.jsonString
            if json != last {
                emit(["ok": true, "partial": json, "done": false])
                onEvent()
                last = json
            }
            if #available(macOS 27.0, *) {
                outcome.usage = usageOf(snapshot.usage)
                outcome.toolCalls = toolActivity(snapshot.transcriptEntries)
            }
        }
        // A cancelled stream can end quietly rather than throw; without this the
        // truncated text would be reported as a normal, finished answer.
        try Task.checkCancellation()
        outcome.content = last
        outcome.finishReason = finishReason(usage: outcome.usage, maxTokens: maxTokens)
        return outcome
    }
    let stream = session.streamResponse(to: prompt, options: options)
    var previous = ""
    var outcome = Outcome(content: "")
    for try await snapshot in stream {
        try Task.checkCancellation()
        let current: String = snapshot.content
        let delta = current.hasPrefix(previous) ? String(current.dropFirst(previous.count)) : current
        previous = current
        if !delta.isEmpty {
            emit(["ok": true, "delta": delta, "done": false])
            onEvent()
        }
        if #available(macOS 27.0, *) {
            outcome.usage = usageOf(snapshot.usage)
            outcome.toolCalls = toolActivity(snapshot.transcriptEntries)
        }
    }
    try Task.checkCancellation()
    outcome.content = previous
    outcome.finishReason = finishReason(usage: outcome.usage, maxTokens: maxTokens)
    return outcome
}

private func describe(_ reason: SystemLanguageModel.Availability.UnavailableReason) -> String {
    switch reason {
    case .deviceNotEligible: return "deviceNotEligible"
    case .appleIntelligenceNotEnabled: return "appleIntelligenceNotEnabled"
    case .modelNotReady: return "modelNotReady"
    @unknown default: return "unknown"
    }
}

/// Private Cloud Compute, described without calling it.
///
/// PCC *inference* needs `com.apple.developer.private-cloud-compute`, which is
/// AMFI-restricted and unavailable to any installable package — that is why the
/// cloud tier goes through Shortcuts. But the model's quota, capabilities and
/// context size are all readable from an unentitled process, so a caller can
/// see what the cloud tier offers, and whether it is worth trying, before
/// spending a Shortcuts round trip on it. On macOS 27 Golden Gate this is the
/// next-generation server model Siri AI is built on: it reports reasoning,
/// vision and tool calling, with a 32,768-token window.
@available(macOS 27.0, *)
private func cloudModel() async -> [String: Any] {
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
    let capabilities = pcc.capabilities
    out["capabilities"] = [
        "vision": capabilities.contains(.vision),
        "guidedGeneration": capabilities.contains(.guidedGeneration),
        "reasoning": capabilities.contains(.reasoning),
        "toolCalling": capabilities.contains(.toolCalling),
    ]
    if let contextSize = try? await pcc.contextSize { out["contextSize"] = contextSize }
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
                payload["cloud"] = await cloudModel()
            }
            payload["features"] = [
                "protocol": 2,
                "streaming": true,
                "structuredStreaming": true,
                "sessions": true,
                "history": true,
                "transcripts": true,
                "labelledAttachments": true,
                "functionTools": true,
                "cancellation": true,
                "builtInTools": ["ocr", "barcode", "spotlight"],
            ]
            emit(payload)
            return
        }

        // Serve mode: one request per stdin line, for as long as the parent
        // keeps the pipe open. Spawning a process per request instead makes the
        // model reload between calls, which measured at ~17s per request
        // against ~1.5s once it is resident.
        //
        // stdin is read on its own thread so control lines (cancel, toolResult)
        // reach a request that is still running; see Control. Requests are
        // still handled strictly one at a time, in order.
        if CommandLine.arguments.contains("--serve") {
            let requests = AsyncStream<String> { continuation in
                let reader = Thread {
                    while let line = readLine(strippingNewline: true) {
                        if line.isEmpty || Control.shared.intercept(line) { continue }
                        continuation.yield(line)
                    }
                    Control.shared.stdinClosed()
                    continuation.finish()
                }
                reader.start()
            }
            for await line in requests {
                guard let data = line.data(using: .utf8),
                      let envelope = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
                else {
                    emit(["ok": false, "error": "helper could not parse a request line as JSON"])
                    continue
                }
                await Control.shared.run(id: envelope["id"] as? String) {
                    await handle(envelope: envelope)
                }
            }
            return
        }

        let input = FileHandle.standardInput.readDataToEndOfFile()
        guard let envelope = (try? JSONSerialization.jsonObject(with: input)) as? [String: Any] else {
            emit(["ok": false, "error": "helper could not parse its stdin payload as JSON"])
            return
        }
        Control.shared.stdinClosed()
        await handle(envelope: envelope)
    }
}
