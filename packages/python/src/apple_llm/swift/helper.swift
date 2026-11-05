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
//   Speech (macOS 26+). These ops never touch the language model, so they work
//   with Apple Intelligence turned off. A client should run them on a helper of
//   their own: requests are handled one at a time, and a long transcription
//   must not hold up generation.
//     {"op":"voices"}
//       -> {"ok":true,"voices":[{id,name,language,quality,gender,novelty,
//                                personal,default,engine}],"systemVoice":{..}}
//     {"op":"speak","text":string,"voice"?:string,"language"?:string,"ssml"?:bool,
//      "speed"?:number,"pitch"?:number,"volume"?:number,"words"?:bool}
//       -> {"ok":true,"spoken":true,"duration":number,"voice":{..}}
//     {"op":"synthesize", ...the same..., "out":path,"format"?:string,"sampleRate"?:number}
//       -> {"ok":true,"path","format","duration","sampleRate","bytes","voice","words"?}
//     {"op":"transcribe","path":string,"language"?:string,"words"?:bool,
//      "alternatives"?:bool,"hints"?:[string],"download"?:bool}
//       -> {"ok":true,"text","segments":[{text,start,end,confidence?,words?,
//           alternatives?}],"language","engine","duration"}
//     {"op":"listen","language"?:string,"silence"?:number,"maxSeconds"?:number,
//      "words"?:bool,"hints"?:[string],"download"?:bool}
//       -> the same as transcribe; {"op":"endInput"} finishes it and keeps the
//          text, where "cancel" discards it.
//     {"op":"speechLanguages"}
//       -> {"ok":true,"languages":[{id,name,installed,engine}],"default":string}
//     {"op":"installSpeechLanguage","language":string} -> {"ok":true,"language":string}
//     {"op":"removeSpeechLanguage","language":string}  -> {"ok":true,"removed":bool}
//   events:  {"ok":true,"heartbeat":true,"done":false}       // every 20s, so a
//                                                            //  30 minute read-aloud
//                                                            //  is not taken for a wedge
//          | {"ok":true,"word":{text,start,end,offset,length},"done":false}  // speak
//          | {"ok":true,"started":{language,engine,duration},"done":false}   // transcribe
//          | {"ok":true,"segment":{..},"done":false}                         // transcribe
//          | {"ok":true,"listening":true,"language":string,"done":false}     // listen
//          | {"ok":true,"transcript":{text,final,start,end,..},"done":false} // listen
//          | {"ok":true,"download":{language,fraction},"done":false}
//   extra failure kinds: permission | setup (a language model is not installed)
//                        | speech
//

import AVFoundation
import AppKit
import Foundation
import FoundationModels
import Speech
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
    /// Finishes a running `listen`, keeping what it heard.
    private var endInput: (() -> Void)?

    /// Register what an `endInput` line should do; nil when nothing is listening.
    func onEndInput(_ action: (() -> Void)?) {
        lock.withLock { endInput = action }
    }

    /// True when `line` was a control message and has been handled here.
    func intercept(_ line: String) -> Bool {
        // Cheap pre-check: a prompt can be tens of KB, and most lines are
        // requests, which never carry these ops.
        guard line.contains("\"cancel\"") || line.contains("\"toolResult\"") || line.contains("\"endInput\""),
              let object = jsonValue(line) as? [String: Any],
              let op = object["op"] as? String else { return false }
        switch op {
        case "cancel":
            cancel(id: object["id"] as? String)
            return true
        case "endInput":
            lock.withLock { endInput }?()
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
    // Speech goes first: handleEnvelope turns everything away when the language
    // model is unavailable, and speech does not need it.
    if let op = envelope["op"] as? String, speechOps.contains(op) {
        await handleSpeech(op: op, envelope: envelope)
        return
    }
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

// MARK: - Speech

private let speechOps: Set<String> = [
    "voices", "speak", "synthesize", "transcribe", "listen",
    "speechLanguages", "installSpeechLanguage", "removeSpeechLanguage",
]

/// Why a speech op failed, in the `kind` vocabulary the clients map to typed errors.
private enum SpeechFailure: Error {
    case unsupported(String)
    case permission(String)
    /// The language's model is not installed and the caller said not to download it.
    case notInstalled(language: String, message: String)
    case failed(String)
}

private func speechFailure(_ error: Error) -> [String: Any] {
    if error is CancellationError {
        return ["ok": false, "kind": "cancelled", "error": "the request was cancelled"]
    }
    guard let failure = error as? SpeechFailure else {
        return ["ok": false, "kind": "speech", "error": (error as NSError).localizedDescription]
    }
    switch failure {
    case .unsupported(let message): return ["ok": false, "kind": "unsupported", "error": message]
    case .permission(let message): return ["ok": false, "kind": "permission", "error": message]
    case .notInstalled(let language, let message):
        return ["ok": false, "kind": "setup", "language": language, "error": message]
    case .failed(let message): return ["ok": false, "kind": "speech", "error": message]
    }
}

@available(macOS 26.0, *)
private func handleSpeech(op: String, envelope: [String: Any]) async {
    do {
        let payload: [String: Any] = try await withHeartbeat {
            switch op {
            case "voices": return voicesPayload()
            case "speak": return try await speakAloud(envelope)
            case "synthesize": return try await synthesizeToFile(envelope)
            case "transcribe": return try await transcribeFile(envelope)
            case "listen": return try await listenToMicrophone(envelope)
            case "speechLanguages": return try await languagesPayload()
            case "installSpeechLanguage": return try await installLanguage(envelope)
            default: return await removeLanguage(envelope)
            }
        }
        emit(payload)
    } catch {
        emit(speechFailure(error))
    }
}

/// Keep the client's wedge timer quiet while a long op runs.
///
/// That timer measures silence, and it is the one real limit on long audio:
/// Apple renders an hour of speech without complaint, but a 30 minute read-aloud
/// emits nothing between its first and last word unless something says it is
/// still alive. The beat is awaited on the way out, so no event can follow the
/// final line.
private func withHeartbeat<T>(_ body: () async throws -> T) async throws -> T {
    let beat = Task {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 20_000_000_000)
            if !Task.isCancelled { emit(["ok": true, "heartbeat": true, "done": false]) }
        }
    }
    do {
        let result = try await body()
        beat.cancel()
        _ = await beat.value
        return result
    } catch {
        beat.cancel()
        _ = await beat.value
        throw error
    }
}

/// Runs a closure at most once, whichever of several racing callers gets there.
private final class Once: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false

    func run(_ body: () -> Void) {
        let first = lock.withLock { () -> Bool in
            if fired { return false }
            fired = true
            return true
        }
        if first { body() }
    }
}

/// A cancellation target for code that starts running after the task may
/// already have been cancelled.
private final class Canceller: @unchecked Sendable {
    private let lock = NSLock()
    private var action: (() -> Void)?
    private var cancelled = false

    func set(_ newAction: @escaping () -> Void) {
        let runNow = lock.withLock { () -> Bool in
            if cancelled { return true }
            action = newAction
            return false
        }
        if runNow { newAction() }
    }

    func cancel() {
        let pending = lock.withLock { () -> (() -> Void)? in
            cancelled = true
            defer { action = nil }
            return action
        }
        pending?()
    }
}

private func seconds(_ time: CMTime) -> Double {
    guard time.isNumeric else { return 0 }
    return (time.seconds * 1000).rounded() / 1000
}

/// JSON readable by people: 0.801, not 0.8009999999999999.
private func rounded3(_ value: Double) -> Double {
    (value * 1000).rounded() / 1000
}

// MARK: Voices

/// One voice, however it is reached.
///
/// `AVSpeechSynthesizer` is the better engine: word timings, pitch, volume and
/// SSML. But it lists fewer voices than macOS has — Aman, Tara, Aru and Ona were
/// audibly distinct through `say` and absent from `speechVoices()` on macOS 27 —
/// so those are reached through `say` rather than left out.
private struct SpeechVoice {
    let id: String
    let name: String
    let fullName: String
    let language: String
    let quality: String
    let gender: String
    let novelty: Bool
    let personal: Bool
    let avVoice: AVSpeechSynthesisVoice?

    func json(isDefault: Bool) -> [String: Any] {
        ["id": id, "name": name, "language": language, "quality": quality, "gender": gender,
         "novelty": novelty, "personal": personal, "default": isDefault,
         "engine": avVoice == nil ? "say" : "avspeech"]
    }

    var summary: [String: Any] {
        ["id": id, "name": name, "language": language, "engine": avVoice == nil ? "say" : "avspeech"]
    }
}

/// "Daniel (English (UK))" and "Aru ((null))" are how macOS names voices for
/// `say`; the name a person means is the part before the bracket.
private func baseVoiceName(_ full: String) -> String {
    full.components(separatedBy: " (").first ?? full
}

/// Listing voices costs about 100ms, and that would land on every call. A voice
/// downloaded in System Settings shows up within the minute.
private final class VoiceCatalogCache: @unchecked Sendable {
    static let shared = VoiceCatalogCache()
    private let lock = NSLock()
    private var voices: [SpeechVoice] = []
    private var stamp = Date.distantPast

    func current() -> [SpeechVoice] {
        lock.withLock {
            if Date().timeIntervalSince(stamp) > 60 {
                voices = buildVoiceCatalog()
                stamp = Date()
            }
            return voices
        }
    }
}

private func voiceCatalog() -> [SpeechVoice] {
    VoiceCatalogCache.shared.current()
}

private func buildVoiceCatalog() -> [SpeechVoice] {
    var out: [SpeechVoice] = []
    var known = Set<String>()
    for voice in AVSpeechSynthesisVoice.speechVoices() {
        let quality: String
        switch voice.quality {
        case .premium: quality = "premium"
        case .enhanced: quality = "enhanced"
        default: quality = "default"
        }
        let gender: String
        switch voice.gender {
        case .female: gender = "female"
        case .male: gender = "male"
        default: gender = "unspecified"
        }
        let language = voice.language.replacingOccurrences(of: "_", with: "-")
        known.insert("\(voice.name)|\(language)")
        out.append(SpeechVoice(
            id: voice.identifier, name: voice.name, fullName: voice.name, language: language,
            quality: quality, gender: gender, novelty: voice.voiceTraits.contains(.isNoveltyVoice),
            personal: voice.voiceTraits.contains(.isPersonalVoice), avVoice: voice))
    }
    for raw in NSSpeechSynthesizer.availableVoices {
        // The two APIs can disagree on a voice's id (compact, super-compact) and
        // its language (zh-HK, yue-HK), but AVSpeech resolves either spelling.
        if AVSpeechSynthesisVoice(identifier: raw.rawValue) != nil { continue }
        let attributes = NSSpeechSynthesizer.attributes(forVoice: raw)
        let full = attributes[.name] as? String ?? raw.rawValue
        let name = baseVoiceName(full)
        let language = (attributes[.localeIdentifier] as? String ?? "").replacingOccurrences(of: "_", with: "-")
        // The same voice often appears twice, as a base id and a .premium one,
        // and `say` can only tell them apart by name anyway.
        guard known.insert("\(name)|\(language)").inserted else { continue }
        let gender = attributes[.gender] as? String ?? ""
        out.append(SpeechVoice(
            id: raw.rawValue, name: name, fullName: full, language: language, quality: "default",
            gender: gender.contains("Female") ? "female" : gender.contains("Male") ? "male" : "unspecified",
            novelty: false, personal: false, avVoice: nil))
    }
    return out
}

/// The "System voice" setting under System Settings › Accessibility. Only this
/// deprecated API names it: `AVSpeechSynthesisVoice(language:)` picks a voice
/// for the language, which is not the same thing.
private func systemVoiceId() -> String {
    NSSpeechSynthesizer.defaultVoice.rawValue
}

private func voicesPayload() -> [String: Any] {
    let systemId = systemVoiceId()
    // AVSpeech and NSSpeech can spell one voice two ways (compact, super-compact).
    let matchId = AVSpeechSynthesisVoice(identifier: systemId)?.identifier ?? systemId
    let catalog = voiceCatalog()
    let attributes = NSSpeechSynthesizer.attributes(forVoice: NSSpeechSynthesizer.VoiceName(rawValue: systemId))
    let systemName = baseVoiceName(attributes[.name] as? String ?? systemId)
    return [
        "ok": true,
        "voices": catalog.map { $0.json(isDefault: $0.id == matchId) },
        "systemVoice": [
            "id": systemId,
            "name": systemName,
            "language": (attributes[.localeIdentifier] as? String ?? "").replacingOccurrences(of: "_", with: "-"),
            "siri": systemId.lowercased().contains("siri"),
            "listed": catalog.contains { $0.id == matchId },
        ] as [String: Any],
    ]
}

/// Lower sorts first. System voice, then real voices over novelties, then the
/// best quality, then the speaker's own region, then AVSpeech over `say`.
private func voiceRank(_ voice: SpeechVoice, systemId: String, likelyRegion: String?) -> [Int] {
    let qualityRank = ["premium": 0, "enhanced": 1][voice.quality] ?? 2
    let region = voice.language.split(separator: "-").last.map(String.init)?.uppercased()
    return [
        voice.id == systemId ? 0 : 1,
        voice.novelty ? 1 : 0,
        qualityRank,
        region == Locale.current.region?.identifier ? 0 : 1,
        region == likelyRegion ? 0 : 1,
        voice.avVoice == nil ? 1 : 0,
        voice.id.hasPrefix("com.apple.voice.") ? 0 : 1,
    ]
}

private func bestVoice(_ voices: [SpeechVoice], language: String?) -> SpeechVoice? {
    let systemId = systemVoiceId()
    let matchId = AVSpeechSynthesisVoice(identifier: systemId)?.identifier ?? systemId
    let likely = language.flatMap { Locale.Language(identifier: $0).maximalIdentifier.split(separator: "-").last.map(String.init)?.uppercased() }
    return voices.min { a, b in
        let (ra, rb) = (voiceRank(a, systemId: matchId, likelyRegion: likely),
                        voiceRank(b, systemId: matchId, likelyRegion: likely))
        return ra.lexicographicallyPrecedes(rb)
    }
}

/// Exact language first, then the same language in any region: "fr" and "fr-CH"
/// both find a French voice.
private func voices(in catalog: [SpeechVoice], speaking wanted: String) -> [SpeechVoice] {
    let normal = wanted.replacingOccurrences(of: "_", with: "-").lowercased()
    let exact = catalog.filter { $0.language.lowercased() == normal }
    if !exact.isEmpty { return exact }
    let family = normal.split(separator: "-").first.map(String.init) ?? normal
    return catalog.filter { $0.language.lowercased().split(separator: "-").first.map(String.init) == family }
}

private struct ResolvedVoice {
    enum Engine {
        case avSpeech(AVSpeechSynthesisVoice)
        /// nil means no `-v`: whatever the System Voice is. That is the one
        /// route to a Siri voice, because Apple does not offer them to apps.
        case say(String?)
    }

    let engine: Engine
    let summary: [String: Any]

    init(_ voice: SpeechVoice) {
        engine = voice.avVoice.map { .avSpeech($0) } ?? .say(voice.name)
        summary = voice.summary
    }

    init(systemVoiceId id: String) {
        let attributes = NSSpeechSynthesizer.attributes(forVoice: NSSpeechSynthesizer.VoiceName(rawValue: id))
        engine = .say(nil)
        summary = ["id": "system", "name": baseVoiceName(attributes[.name] as? String ?? id),
                   "language": (attributes[.localeIdentifier] as? String ?? "").replacingOccurrences(of: "_", with: "-"),
                   "engine": "say"]
    }
}

private func resolveVoice(_ envelope: [String: Any]) throws -> ResolvedVoice {
    let spec = (envelope["voice"] as? String)?.trimmingCharacters(in: .whitespaces) ?? ""
    let language = (envelope["language"] as? String).flatMap { $0.isEmpty ? nil : $0 }
    let catalog = voiceCatalog()

    if spec.lowercased() == "system" { return ResolvedVoice(systemVoiceId: systemVoiceId()) }

    if spec.isEmpty {
        if let language {
            guard let best = bestVoice(voices(in: catalog, speaking: language), language: language) else {
                throw SpeechFailure.unsupported(
                    "No voice speaks \"\(language)\". Run `apple-llm voices --all` to see what is installed.")
            }
            return ResolvedVoice(best)
        }
        let systemId = systemVoiceId()
        let matchId = AVSpeechSynthesisVoice(identifier: systemId)?.identifier ?? systemId
        if let listed = catalog.first(where: { $0.id == matchId }) { return ResolvedVoice(listed) }
        return ResolvedVoice(systemVoiceId: systemId)
    }

    if let byId = catalog.first(where: { $0.id == spec }) { return ResolvedVoice(byId) }
    let needle = spec.lowercased()
    var named = catalog.filter { $0.name.lowercased() == needle || $0.fullName.lowercased() == needle }
    if let language, !named.isEmpty {
        named = voices(in: named, speaking: language)
    } else if named.count > 1, let current = Locale.current.language.languageCode?.identifier {
        // "Eddy" exists in fifteen languages; the speaker's own comes first.
        let own = voices(in: named, speaking: current)
        if !own.isEmpty { named = own }
    }
    if let best = bestVoice(named, language: language ?? Locale.current.language.languageCode?.identifier) {
        return ResolvedVoice(best)
    }
    let close = Set(catalog.map(\.name).filter {
        $0.lowercased().hasPrefix(String(needle.prefix(2))) || needle.hasPrefix($0.lowercased().prefix(3))
    }).sorted().prefix(5)
    let hint = close.isEmpty ? "" : " Did you mean \(close.joined(separator: ", "))?"
    throw SpeechFailure.unsupported(
        "Unknown voice \"\(spec)\".\(hint) Run `apple-llm voices --all` to list every voice.")
}

// MARK: Text to speech

/// How far `AVSpeechUtterance.rate` is from a speed multiplier. The rate is not
/// linear and 0.5 is normal, so "twice as fast" is not a number a person can
/// guess. Measured on macOS 27 against a 49 second passage: rate 0 gives 88.5s,
/// rate 1 gives 12.2s.
private let avRateForSpeed: [(speed: Double, value: Double)] = [
    (0.56, 0), (0.67, 0.1), (0.72, 0.2), (0.83, 0.3), (0.91, 0.4), (1, 0.5),
    (1.29, 0.55), (1.59, 0.6), (2.11, 0.7), (2.80, 0.8), (3.52, 0.9), (4.04, 1),
]

/// The same for `say -r`, in words per minute, on the same passage.
private let sayRateForSpeed: [(speed: Double, value: Double)] = [
    (0.83, 90), (1, 175), (1.12, 200), (1.83, 350), (2.49, 500), (3.52, 700), (4.04, 800),
]

/// Linear between measured points, clamped at both ends.
private func interpolate(_ table: [(speed: Double, value: Double)], at speed: Double) -> Double {
    guard let first = table.first, let last = table.last else { return speed }
    if speed <= first.speed { return first.value }
    if speed >= last.speed { return last.value }
    for (low, high) in zip(table, table.dropFirst()) where speed <= high.speed {
        return low.value + (speed - low.speed) / (high.speed - low.speed) * (high.value - low.value)
    }
    return last.value
}

private struct SpeechSettings {
    let text: String
    let ssml: Bool
    let speed: Double?
    let pitch: Double?
    let volume: Double?
    let wantWords: Bool

    init(_ envelope: [String: Any]) throws {
        text = envelope["text"] as? String ?? ""
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SpeechFailure.unsupported("There is no text to speak.")
        }
        ssml = envelope["ssml"] as? Bool ?? false
        speed = num(envelope["speed"])
        pitch = num(envelope["pitch"])
        volume = num(envelope["volume"])
        wantWords = envelope["words"] as? Bool ?? false
        if let speed, !(0.25...4).contains(speed) {
            throw SpeechFailure.unsupported("speed must be between 0.25 and 4 (got \(speed)).")
        }
        if let pitch, !(0.5...2).contains(pitch) {
            throw SpeechFailure.unsupported("pitch must be between 0.5 and 2 (got \(pitch)).")
        }
        if let volume, !(0...1).contains(volume) {
            throw SpeechFailure.unsupported("volume must be between 0 and 1 (got \(volume)).")
        }
    }

    /// `say` has no pitch, volume or SSML. Refused rather than ignored: a caller
    /// who asked for quiet audio and got loud audio has been misled.
    func assertSayCanDoThis(voiceName: String) throws {
        var missing: [String] = []
        if pitch != nil { missing.append("pitch") }
        if volume != nil { missing.append("volume") }
        if ssml { missing.append("SSML") }
        if !missing.isEmpty {
            throw SpeechFailure.unsupported(
                "The voice \"\(voiceName)\" is only reachable through macOS `say`, which has no "
                + "\(missing.joined(separator: " or ")). Pick a voice whose engine is avspeech.")
        }
    }

    func utterance(voice: AVSpeechSynthesisVoice) throws -> AVSpeechUtterance {
        let made: AVSpeechUtterance
        if ssml {
            guard let parsed = AVSpeechUtterance(ssmlRepresentation: text) else {
                throw SpeechFailure.unsupported("The text is not valid SSML.")
            }
            made = parsed
        } else {
            made = AVSpeechUtterance(string: text)
        }
        made.voice = voice
        if let speed { made.rate = Float(interpolate(avRateForSpeed, at: speed)) }
        if let pitch { made.pitchMultiplier = Float(pitch) }
        if let volume { made.volume = Float(volume) }
        return made
    }
}

/// One synthesizer for the life of the process: a new one costs about 300ms
/// before its first buffer, a reused one about 10ms, which is the difference
/// between a voice assistant that answers and one that hesitates. Requests are
/// handled one at a time, so it is never shared between two at once.
private let sharedSynthesizer: AVSpeechSynthesizer = {
    let synthesizer = AVSpeechSynthesizer()
    synthesizer.delegate = SynthesisRouter.shared
    return synthesizer
}()

/// The shared synthesizer's only delegate, alive as long as the process.
///
/// A delegate per request crashed the helper: after a cancelled `speak`, the
/// next request died with SIGSEGV in `objc_retain` under TextToSpeech, on a block
/// the framework had queued on the main thread for an object already released.
/// So nothing the framework may still be holding gets freed: this object never
/// is, and the last few utterances are kept. Callbacks are matched by utterance,
/// so a cancelled one's late didCancel cannot be taken for the next one finishing.
private final class SynthesisRouter: NSObject, AVSpeechSynthesizerDelegate, @unchecked Sendable {
    static let shared = SynthesisRouter()

    private let lock = NSLock()
    private var current: AVSpeechUtterance?
    private var recent: [AVSpeechUtterance] = []
    private var onDone: ((_ cancelled: Bool) -> Void)?
    private var onRange: ((NSRange) -> Void)?

    func track(
        _ utterance: AVSpeechUtterance, onDone: @escaping (_ cancelled: Bool) -> Void,
        onRange: ((NSRange) -> Void)? = nil
    ) {
        lock.withLock {
            current = utterance
            recent.append(utterance)
            if recent.count > 4 { recent.removeFirst() }
            self.onDone = onDone
            self.onRange = onRange
        }
    }

    private func finished(_ utterance: AVSpeechUtterance, cancelled: Bool) {
        let done = lock.withLock { () -> ((Bool) -> Void)? in
            guard utterance === current else { return nil }
            defer {
                onDone = nil
                onRange = nil
            }
            return onDone
        }
        done?(cancelled)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        finished(utterance, cancelled: false)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        finished(utterance, cancelled: true)
    }

    func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer, willSpeakRangeOfSpeechString characterRange: NSRange,
        utterance: AVSpeechUtterance
    ) {
        let handler = lock.withLock { utterance === current ? onRange : nil }
        handler?(characterRange)
    }
}

/// Stop the shared synthesizer and let its didCancel arrive before moving on.
/// It may never come for a render, so waiting is bounded.
private func stopSynthesizer(then finish: @escaping @Sendable () -> Void) {
    sharedSynthesizer.stopSpeaking(at: .immediate)
    DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) { finish() }
}

/// Speech that macOS cancelled when nobody asked it to.
///
/// Reporting this as success would be the worst outcome: if the helper's file is
/// deleted while it runs, as a disk cleaner might do to a cache, AVSpeech fires
/// didCancel within 0.1s and speaks nothing, and a render comes back cut short.
/// Word callbacks cannot catch this, because some voices never send any.
private let stoppedEarly = SpeechFailure.failed(
    "macOS stopped this speech before it finished, and nothing here asked it to. "
    + "If a disk cleaner removed apple-llm's cache while this program was running, restart it.")

private final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    var isSet: Bool { lock.withLock { value } }

    func set(_ cancelled: Bool) {
        if cancelled { lock.withLock { value = true } }
    }
}

/// Audio formats by name, and the file extension each is written with.
private let audioExtensions: [String: String] = [
    "wav": "wav", "aiff": "aiff", "caf": "caf", "m4a": "m4a", "aac": "aac", "flac": "flac", "pcm": "pcm",
]

/// Named in the error because these are what people ask for first.
private let unencodable = "macOS can read MP3, Ogg and Opus but has no encoder for them. "
    + "Use m4a, aac, flac or wav, and convert afterwards with ffmpeg if you need to."

private func audioFormat(named name: String?, path: String) throws -> String {
    let aliases = ["wave": "wav", "aif": "aiff", "mp4": "m4a", "raw": "pcm"]
    let raw = (name?.isEmpty == false ? name! : (path as NSString).pathExtension).lowercased()
    let format = aliases[raw] ?? raw
    if ["mp3", "ogg", "oga", "opus"].contains(format) { throw SpeechFailure.unsupported(unencodable) }
    guard audioExtensions[format] != nil else {
        throw SpeechFailure.unsupported(
            "Unknown audio format \"\(raw)\" (want wav, aiff, caf, m4a, aac, flac or pcm).")
    }
    return format
}

private func fileSettings(format: String, sampleRate rate: Double) -> [String: Any] {
    var linear: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: rate, AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false,
        AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false,
    ]
    switch format {
    case "aiff":
        linear[AVLinearPCMIsBigEndianKey] = true
        return linear
    case "m4a", "aac":
        var aac: [String: Any] = [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: rate, AVNumberOfChannelsKey: 1]
        // Plenty for one voice; the encoder rejects this rate at very low sample rates.
        if rate >= 16000 { aac[AVEncoderBitRateKey] = 48000 }
        return aac
    case "flac":
        return [AVFormatIDKey: kAudioFormatFLAC, AVSampleRateKey: rate, AVNumberOfChannelsKey: 1,
                AVEncoderBitDepthHintKey: 16]
    default:
        return linear
    }
}

/// Writes audio buffers to a file, converting format and sample rate as needed.
///
/// Written beside the destination and renamed at the end, so a run that fails
/// or is cancelled cannot leave half a file under the name the caller asked for.
/// Takes buffers one at a time, so an hour of audio costs a buffer's worth of
/// memory rather than an hour's.
private final class AudioSink: @unchecked Sendable {
    let format: String
    private let destination: URL
    private let temporary: URL
    private let requestedRate: Double?
    private let lock = NSLock()
    private var file: AVAudioFile?
    private var raw: FileHandle?
    private var target: AVAudioFormat?
    private var converter: AVAudioConverter?
    private var closed = false

    private(set) var failure: Error?
    /// What arrived, in the source's own frames: word markers count in these.
    private(set) var sourceFrames: Int64 = 0
    private(set) var sourceRate: Double = 0
    private(set) var sourceBytesPerFrame: Double = 0
    /// What was written, in the output's own frames.
    private(set) var outputFrames: Int64 = 0
    private(set) var outputRate: Double = 0

    init(destination: URL, format: String, sampleRate: Double?) {
        self.destination = destination
        self.format = format
        requestedRate = sampleRate
        temporary = destination.deletingLastPathComponent()
            .appendingPathComponent(".apple-llm-\(UUID().uuidString).\(audioExtensions[format] ?? format)")
    }

    func write(_ input: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard !closed, failure == nil, input.frameLength > 0 else { return }
        do {
            if target == nil { try open(for: input.format) }
            sourceFrames += Int64(input.frameLength)
            guard let target else { return }
            if input.format == target {
                try store(input)
            } else {
                guard let output = try converted(input, to: target, endOfStream: false) else { return }
                try store(output)
            }
        } catch {
            failure = error
        }
    }

    private func open(for source: AVAudioFormat) throws {
        sourceRate = source.sampleRate
        sourceBytesPerFrame = Double(source.streamDescription.pointee.mBytesPerFrame)
        let rate = requestedRate ?? source.sampleRate
        outputRate = rate
        let made: AVAudioFormat?
        if format == "pcm" {
            guard FileManager.default.createFile(atPath: temporary.path, contents: nil) else {
                throw SpeechFailure.failed("Could not write to \(destination.deletingLastPathComponent().path).")
            }
            raw = try FileHandle(forWritingTo: temporary)
            made = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: rate, channels: 1, interleaved: true)
        } else {
            let opened = try AVAudioFile(
                forWriting: temporary, settings: fileSettings(format: format, sampleRate: rate),
                commonFormat: .pcmFormatFloat32, interleaved: false)
            file = opened
            made = opened.processingFormat
        }
        guard let made else { throw SpeechFailure.failed("Could not set up audio format \(format) at \(rate) Hz.") }
        target = made
        if source != made {
            guard let conversion = AVAudioConverter(from: source, to: made) else {
                throw SpeechFailure.failed("macOS cannot convert \(source) to \(made).")
            }
            converter = conversion
        }
    }

    private func converted(
        _ input: AVAudioPCMBuffer?, to target: AVAudioFormat, endOfStream: Bool
    ) throws -> AVAudioPCMBuffer? {
        guard let converter else { return nil }
        let frames = Double(input?.frameLength ?? 4096)
        let capacity = AVAudioFrameCount((frames * target.sampleRate / max(sourceRate, 1)).rounded(.up)) + 4096
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return nil }
        var supplied = false
        var problem: NSError?
        converter.convert(to: output, error: &problem) { _, status in
            if supplied || input == nil {
                status.pointee = endOfStream ? .endOfStream : .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return input
        }
        if let problem { throw problem }
        return output.frameLength > 0 ? output : nil
    }

    private func store(_ buffer: AVAudioPCMBuffer) throws {
        if let file {
            try file.write(from: buffer)
        } else if let raw, let samples = buffer.int16ChannelData {
            try raw.write(contentsOf: Data(bytes: samples[0], count: Int(buffer.frameLength) * 2))
        }
        outputFrames += Int64(buffer.frameLength)
    }

    /// Flush, close and move into place. Returns the size in bytes.
    func finish() throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        closed = true
        if let failure { throw failure }
        guard outputFrames > 0 || sourceFrames > 0 else { throw SpeechFailure.failed("No audio came back.") }
        if let target, converter != nil {
            while let tail = try converted(nil, to: target, endOfStream: true) { try store(tail) }
        }
        file?.close()
        file = nil
        try raw?.close()
        raw = nil
        guard rename(temporary.path, destination.path) == 0 else {
            throw SpeechFailure.failed("Could not write \(destination.path): \(String(cString: strerror(errno))).")
        }
        let size = try FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? Int
        return size ?? 0
    }

    func abandon() {
        lock.lock()
        defer { lock.unlock() }
        closed = true
        file?.close()
        file = nil
        try? raw?.close()
        raw = nil
        try? FileManager.default.removeItem(at: temporary)
    }
}

/// Word positions from a render, turned into seconds at the end.
///
/// A marker's `byteSampleOffset` counts from the start of the current *segment*,
/// not the utterance: AVSpeech ends a segment with a zero-length buffer, several
/// times in a long text. Taken at face value the last word of a 30 minute render
/// reported 11.9s, so each marker is stored with the frames that preceded its
/// segment.
private final class WordMarks: @unchecked Sendable {
    private let lock = NSLock()
    private let text: NSString
    private var segmentBase: Int64 = 0
    private var marks: [(range: NSRange, base: Int64, offset: Int)] = []

    init(text: String) {
        self.text = text as NSString
    }

    func segmentEnded(framesSoFar: Int64) {
        lock.withLock { segmentBase = framesSoFar }
    }

    func add(_ markers: [AVSpeechSynthesisMarker]) {
        lock.withLock {
            for marker in markers where marker.mark == .word {
                marks.append((marker.textRange, segmentBase, Int(marker.byteSampleOffset)))
            }
        }
    }

    func words(sampleRate: Double, bytesPerFrame: Double, duration: Double) -> [[String: Any]] {
        guard sampleRate > 0, bytesPerFrame > 0 else { return [] }
        let starts = marks.map { (Double($0.base) + Double($0.offset) / bytesPerFrame) / sampleRate }
        return marks.enumerated().map { index, mark in
            let word: String
            if NSMaxRange(mark.range) <= text.length {
                word = text.substring(with: mark.range).trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                word = ""
            }
            return [
                "text": word, "start": rounded3(starts[index]),
                "end": rounded3(index + 1 < starts.count ? starts[index + 1] : duration),
                "offset": mark.range.location, "length": mark.range.length,
            ]
        }
    }
}

private func renderWithAVSpeech(
    _ settings: SpeechSettings, voice: AVSpeechSynthesisVoice, into sink: AudioSink
) async throws -> WordMarks {
    let utterance = try settings.utterance(voice: voice)
    let synthesizer = sharedSynthesizer
    let marks = WordMarks(text: utterance.speechString)
    let canceller = Canceller()
    let macOSCancelled = CancelFlag()
    await withTaskCancellationHandler {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let once = Once()
            let finish: @Sendable () -> Void = { once.run { continuation.resume() } }
            // A zero-length buffer ends a segment of the text, not the text: only
            // this delegate call means the whole utterance is done.
            SynthesisRouter.shared.track(utterance, onDone: { cancelled in
                macOSCancelled.set(cancelled)
                finish()
            })
            canceller.set { stopSynthesizer(then: finish) }
            let onBuffer: AVSpeechSynthesizer.BufferCallback = { buffer in
                guard let pcm = buffer as? AVAudioPCMBuffer else { return }
                if pcm.frameLength == 0 {
                    marks.segmentEnded(framesSoFar: sink.sourceFrames)
                } else {
                    sink.write(pcm)
                }
            }
            if settings.wantWords {
                synthesizer.write(utterance, toBufferCallback: onBuffer, toMarkerCallback: { marks.add($0) })
            } else {
                synthesizer.write(utterance, toBufferCallback: onBuffer)
            }
        }
    } onCancel: {
        canceller.cancel()
    }
    try Task.checkCancellation()
    if macOSCancelled.isSet { throw stoppedEarly }
    return marks
}

/// Run `/usr/bin/say`, reading the text from a file so nothing in it can be
/// taken for an option, and so its length is no concern of argv.
private func runSay(voice: String?, text: String, speed: Double?, output: URL?) async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("apple-llm-say-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let textFile = directory.appendingPathComponent("text.txt")
    try text.write(to: textFile, atomically: false, encoding: .utf8)

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
    var arguments: [String] = []
    if let voice { arguments += ["-v", voice] }
    if let speed { arguments += ["-r", String(Int(interpolate(sayRateForSpeed, at: speed).rounded()))] }
    if let output { arguments += ["-o", output.path] }
    process.arguments = arguments + ["-f", textFile.path]
    process.standardInput = FileHandle.nullDevice
    process.standardOutput = FileHandle.nullDevice
    let errors = Pipe()
    process.standardError = errors

    let canceller = Canceller()
    try await withTaskCancellationHandler {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            process.terminationHandler = { finished in
                if finished.terminationReason == .exit, finished.terminationStatus == 0 {
                    continuation.resume()
                    return
                }
                if finished.terminationReason == .uncaughtSignal {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                let detail = String(data: errors.fileHandleForReading.availableData, encoding: .utf8) ?? ""
                continuation.resume(throwing: SpeechFailure.failed(
                    "macOS `say` failed (exit \(finished.terminationStatus)): \(detail.trimmingCharacters(in: .whitespacesAndNewlines))"))
            }
            do {
                try process.run()
                canceller.set { if process.isRunning { process.terminate() } }
            } catch {
                process.terminationHandler = nil
                continuation.resume(throwing: SpeechFailure.failed("Could not run /usr/bin/say: \(error.localizedDescription)"))
            }
        }
    } onCancel: {
        canceller.cancel()
    }
}

/// Have `say` render to a file, then feed that through the sink, which gives
/// these voices every output format the AVSpeech route has.
///
/// A render is only kept once two in a row agree, because `say` can exit 0 having
/// written something wrong. With no load every voice was deterministic: 15 renders
/// in 15 came out the same length. With the language model busy in other processes,
/// a quarter of renders in Aman and Tara lost words ("The quick over, it was a
/// bright cold day in April"), and a few were silent or 0.02s long. Bad renders
/// vary, so two agreeing means both are good. AVSpeech has none of this: 4,130
/// renders under the same load came out identical.
///
/// Some macOS versions are also reported to write silence for Siri voices every
/// time. That still ends in an error: a silent file with no error is the worst way
/// to find out.
private func renderWithSay(
    _ settings: SpeechSettings, voice: String?, into sink: AudioSink
) async throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("apple-llm-render-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let rendered = directory.appendingPathComponent("speech.aiff")

    var previousFrames: Int64?
    var sawSound = false
    for attempt in 1...6 {
        if attempt > 1 { try await Task.sleep(nanoseconds: 100_000_000) }
        try await runSay(voice: voice, text: settings.text, speed: settings.speed, output: rendered)
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: rendered)
        } catch {
            throw SpeechFailure.failed("macOS `say` produced no audio for this voice.")
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 65536) else {
            throw SpeechFailure.failed("Could not read the audio `say` produced.")
        }
        var peak: Float = 0
        while file.framePosition < file.length {
            try Task.checkCancellation()
            try file.read(into: buffer)
            if buffer.frameLength == 0 { break }
            if let samples = buffer.floatChannelData {
                for index in 0..<Int(buffer.frameLength) { peak = max(peak, abs(samples[0][index])) }
            }
        }
        guard peak >= 0.0005 else {
            previousFrames = nil
            continue
        }
        sawSound = true
        // Ona alone was once 10ms out with nothing else running, so not exact.
        if let previousFrames, abs(previousFrames - file.length) <= max(previousFrames / 200, 512) {
            file.framePosition = 0
            while file.framePosition < file.length {
                try Task.checkCancellation()
                try file.read(into: buffer)
                if buffer.frameLength == 0 { break }
                sink.write(buffer)
            }
            return
        }
        previousFrames = file.length
    }
    if sawSound {
        throw SpeechFailure.failed(
            "macOS `say` could not render this voice cleanly while the Mac was this busy. "
            + "Try again, or pick a voice whose engine is avspeech.")
    }
    throw SpeechFailure.failed(
        "This voice rendered silence to a file, which some macOS versions do for Siri voices. "
        + "It can still speak aloud; to get a file, pick another voice.")
}

@available(macOS 26.0, *)
private func synthesizeToFile(_ envelope: [String: Any]) async throws -> [String: Any] {
    let settings = try SpeechSettings(envelope)
    guard let out = envelope["out"] as? String, !out.isEmpty else {
        throw SpeechFailure.unsupported("synthesize needs an output path.")
    }
    let format = try audioFormat(named: envelope["format"] as? String, path: out)
    let rate = num(envelope["sampleRate"])
    if let rate, !(8000...192000).contains(rate) {
        throw SpeechFailure.unsupported("sampleRate must be between 8000 and 192000 (got \(rate)).")
    }
    let voice = try resolveVoice(envelope)
    let destination = URL(fileURLWithPath: out)
    var isDirectory: ObjCBool = false
    let parent = destination.deletingLastPathComponent().path
    guard FileManager.default.fileExists(atPath: parent, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw SpeechFailure.unsupported("The folder \(parent) does not exist.")
    }

    let sink = AudioSink(destination: destination, format: format, sampleRate: rate)
    do {
        var marks: WordMarks?
        switch voice.engine {
        case .avSpeech(let chosen):
            marks = try await renderWithAVSpeech(settings, voice: chosen, into: sink)
        case .say(let name):
            try settings.assertSayCanDoThis(voiceName: name ?? "system")
            try await renderWithSay(settings, voice: name, into: sink)
        }
        try Task.checkCancellation()
        if let failure = sink.failure { throw failure }
        guard sink.sourceFrames > 0 else {
            throw SpeechFailure.failed(
                "No audio came back from this voice. Try another, or voice \"system\" to go through macOS `say`.")
        }
        let bytes = try sink.finish()
        let duration = sink.outputRate > 0 ? Double(sink.outputFrames) / sink.outputRate : 0
        var payload: [String: Any] = [
            "ok": true, "path": destination.path, "format": format, "duration": rounded3(duration),
            "sampleRate": sink.outputRate, "bytes": bytes, "voice": voice.summary,
        ]
        if settings.wantWords, let marks {
            payload["words"] = marks.words(
                sampleRate: sink.sourceRate, bytesPerFrame: sink.sourceBytesPerFrame, duration: duration)
        }
        return payload
    } catch {
        sink.abandon()
        throw error
    }
}

@available(macOS 26.0, *)
private func speakAloud(_ envelope: [String: Any]) async throws -> [String: Any] {
    let settings = try SpeechSettings(envelope)
    let voice = try resolveVoice(envelope)
    let started = Date()
    switch voice.engine {
    case .say(let name):
        try settings.assertSayCanDoThis(voiceName: name ?? "system")
        try await runSay(voice: name, text: settings.text, speed: settings.speed, output: nil)
    case .avSpeech(let chosen):
        let utterance = try settings.utterance(voice: chosen)
        let spoken = utterance.speechString as NSString
        let onRange: ((NSRange) -> Void)? = !settings.wantWords ? nil : { range in
            guard NSMaxRange(range) <= spoken.length else { return }
            emit(["ok": true, "done": false, "word": [
                "text": spoken.substring(with: range), "start": rounded3(Date().timeIntervalSince(started)),
                "offset": range.location, "length": range.length,
            ] as [String: Any]])
        }
        let canceller = Canceller()
        let macOSCancelled = CancelFlag()
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                let once = Once()
                let finish: @Sendable () -> Void = { once.run { continuation.resume() } }
                SynthesisRouter.shared.track(utterance, onDone: { cancelled in
                    macOSCancelled.set(cancelled)
                    finish()
                }, onRange: onRange)
                canceller.set { stopSynthesizer(then: finish) }
                sharedSynthesizer.speak(utterance)
            }
        } onCancel: {
            canceller.cancel()
        }
        try Task.checkCancellation()
        if macOSCancelled.isSet { throw stoppedEarly }
    }
    try Task.checkCancellation()
    return ["ok": true, "spoken": true, "duration": rounded3(Date().timeIntervalSince(started)),
            "voice": voice.summary]
}

// MARK: Speech to text

/// `text` and `alternatives` live on each transcriber's own Result type rather
/// than on the protocol they share.
@available(macOS 26.0, *)
private protocol SpeechTextResult: SpeechModuleResult {
    var text: AttributedString { get }
    var alternatives: [AttributedString] { get }
}

@available(macOS 26.0, *)
extension SpeechTranscriber.Result: SpeechTextResult {}

@available(macOS 26.0, *)
extension DictationTranscriber.Result: SpeechTextResult {}

/// One recognised phrase, either transcriber's, flattened for the wire.
private struct RecognizedPiece: @unchecked Sendable {
    let text: String
    let isFinal: Bool
    let start: Double
    let end: Double
    let confidence: Double?
    let words: [[String: Any]]
    let alternatives: [String]

    func json(includeWords: Bool) -> [String: Any] {
        var out: [String: Any] = [
            "text": text.trimmingCharacters(in: .whitespacesAndNewlines), "start": start, "end": end,
        ]
        if let confidence { out["confidence"] = confidence }
        if includeWords, !words.isEmpty { out["words"] = words }
        if !alternatives.isEmpty { out["alternatives"] = alternatives }
        return out
    }
}

@available(macOS 26.0, *)
private func recognizedPiece<R: SpeechTextResult>(_ result: R) -> RecognizedPiece {
    var words: [[String: Any]] = []
    var confidences: [Double] = []
    for run in result.text.runs {
        let chunk = String(result.text[run.range].characters).trimmingCharacters(in: .whitespacesAndNewlines)
        if let confidence = run.transcriptionConfidence { confidences.append(confidence) }
        if chunk.isEmpty { continue }
        if let range = run.audioTimeRange {
            var word: [String: Any] = ["text": chunk, "start": seconds(range.start), "end": seconds(range.end)]
            if let confidence = run.transcriptionConfidence { word["confidence"] = rounded3(confidence) }
            words.append(word)
        } else if var last = words.popLast() {
            // Punctuation can arrive untimed; it belongs to the word before it.
            last["text"] = (last["text"] as? String ?? "") + chunk
            words.append(last)
        }
    }
    let alternatives = result.alternatives.dropFirst().map { String($0.characters) }
    return RecognizedPiece(
        text: String(result.text.characters), isFinal: result.isFinal,
        start: seconds(result.range.start), end: seconds(result.range.end),
        confidence: confidences.isEmpty ? nil : rounded3(confidences.reduce(0, +) / Double(confidences.count)),
        words: words, alternatives: Array(alternatives))
}

@available(macOS 26.0, *)
private func recognizedStream<M: SpeechModule>(_ module: M) -> AsyncThrowingStream<RecognizedPiece, Error>
where M.Result: SpeechTextResult {
    AsyncThrowingStream { continuation in
        let task = Task {
            do {
                for try await result in module.results { continuation.yield(recognizedPiece(result)) }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
    }
}

/// Which language to recognise, and with which of Apple's two transcribers.
///
/// `SpeechTranscriber` is the better one and is preferred, but `DictationTranscriber`
/// covers languages it lacks — Russian, Polish, Dutch, Turkish and Arabic among
/// them — so between them macOS 27 reaches far more than either alone.
private struct RecognitionTarget {
    let locale: Locale
    let useDictation: Bool
    var id: String { locale.identifier(.bcp47) }
    var engine: String { useDictation ? "dictation" : "transcriber" }
}

@available(macOS 26.0, *)
private func recognitionTarget(_ spec: String?) async throws -> RecognitionTarget {
    let transcriberOK = SpeechTranscriber.isAvailable
    let main = transcriberOK ? await SpeechTranscriber.supportedLocales : []
    let dictation = await DictationTranscriber.supportedLocales

    func exact(_ wanted: Locale) async -> RecognitionTarget? {
        if transcriberOK, let found = await SpeechTranscriber.supportedLocale(equivalentTo: wanted) {
            return RecognitionTarget(locale: found, useDictation: false)
        }
        if let found = await DictationTranscriber.supportedLocale(equivalentTo: wanted) {
            return RecognitionTarget(locale: found, useDictation: true)
        }
        return nil
    }
    /// "hi" has no equivalent on its own: find its region, the speaker's first.
    func family(_ language: String) -> RecognitionTarget? {
        let code = language.split(separator: "-").first.map { $0.lowercased() } ?? language.lowercased()
        let likely = Locale.Language(identifier: language).maximalIdentifier.split(separator: "-").last.map(String.init)
        let order = [Locale.current.region?.identifier, likely].compactMap { $0 }
        for (list, useDictation) in [(main, false), (dictation, true)] {
            let matches = list.filter { $0.language.languageCode?.identifier.lowercased() == code }
            guard !matches.isEmpty else { continue }
            let preferred = order.compactMap { region in matches.first { $0.region?.identifier == region } }.first
            return RecognitionTarget(locale: preferred ?? matches[0], useDictation: useDictation)
        }
        return nil
    }

    if let spec, !spec.isEmpty {
        let normal = spec.replacingOccurrences(of: "_", with: "-")
        if let found = await exact(Locale(identifier: normal)), found.id.lowercased() == normal.lowercased() {
            return found
        }
        if let found = family(normal) { return found }
        throw SpeechFailure.unsupported(
            "macOS cannot recognise \"\(spec)\". Run `apple-llm languages` to see what it can.")
    }
    if let found = await exact(Locale.current) { return found }
    if let code = Locale.current.language.languageCode?.identifier, let found = family(code) { return found }
    if let found = await exact(Locale(identifier: "en-US")) { return found }
    throw SpeechFailure.unsupported("Speech recognition is not available on this Mac.")
}

@available(macOS 26.0, *)
private func installedLanguages() async -> (main: Set<String>, dictation: Set<String>) {
    let main = await SpeechTranscriber.installedLocales.map { $0.identifier(.bcp47) }
    let dictation = await DictationTranscriber.installedLocales.map { $0.identifier(.bcp47) }
    return (Set(main), Set(dictation))
}

/// Download a language's model if it is missing.
///
/// The check is `installedLocales`, not `AssetInventory.status`: that reported
/// `.supported` for an installed en-US model that worked. Without a model the
/// framework fails with "Audio format is not supported", which sends you looking
/// in entirely the wrong place, so this runs before any audio is touched.
@available(macOS 26.0, *)
private func ensureInstalled(
    _ target: RecognitionTarget, module: any SpeechModule, allowDownload: Bool
) async throws {
    let installed = await installedLanguages()
    if (target.useDictation ? installed.dictation : installed.main).contains(target.id) { return }
    guard allowDownload else {
        throw SpeechFailure.notInstalled(
            language: target.id,
            message: "The speech model for \(target.id) is not installed. "
                + "Run `apple-llm languages install \(target.id)`, or allow the download.")
    }
    let request: AssetInstallationRequest?
    do {
        request = try await AssetInventory.assetInstallationRequest(supporting: [module])
    } catch let error as SFSpeechError where error.code == .tooManyAssetLocalesAllocated {
        // An app may reserve only a few languages; make room rather than fail.
        for locale in await AssetInventory.reservedLocales where locale.identifier(.bcp47) != target.id {
            if await AssetInventory.release(reservedLocale: locale) { break }
        }
        request = try await AssetInventory.assetInstallationRequest(supporting: [module])
    }
    guard let request else { return }
    let reporter = Task {
        var last = -1.0
        while !Task.isCancelled {
            let fraction = request.progress.fractionCompleted
            if fraction != last {
                emit(["ok": true, "done": false, "download": ["language": target.id, "fraction": fraction] as [String: Any]])
                last = fraction
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
    }
    defer { reporter.cancel() }
    try await withTaskCancellationHandler {
        try await request.downloadAndInstall()
    } onCancel: {
        request.progress.cancel()
    }
}

/// The transcriber module and its results, whichever transcriber it is.
@available(macOS 26.0, *)
private func makeRecognizer(
    _ target: RecognitionTarget, live: Bool, alternatives: Bool
) -> (module: any SpeechModule, pieces: AsyncThrowingStream<RecognizedPiece, Error>) {
    if target.useDictation {
        var reporting: Set<DictationTranscriber.ReportingOption> = []
        if live { reporting.formUnion([.volatileResults, .frequentFinalization]) }
        if alternatives { reporting.insert(.alternativeTranscriptions) }
        let module = DictationTranscriber(
            locale: target.locale, contentHints: [], transcriptionOptions: [.punctuation],
            reportingOptions: reporting, attributeOptions: [.audioTimeRange, .transcriptionConfidence])
        return (module, recognizedStream(module))
    }
    var reporting: Set<SpeechTranscriber.ReportingOption> = []
    if live { reporting.formUnion([.volatileResults, .fastResults]) }
    if alternatives { reporting.insert(.alternativeTranscriptions) }
    let module = SpeechTranscriber(
        locale: target.locale, transcriptionOptions: [], reportingOptions: reporting,
        attributeOptions: [.audioTimeRange, .transcriptionConfidence])
    return (module, recognizedStream(module))
}

@available(macOS 26.0, *)
private func makeAnalyzer(modules: [any SpeechModule], hints: [String]) async throws -> SpeechAnalyzer {
    // Lingering keeps the model loaded between requests on this long-lived helper.
    let analyzer = SpeechAnalyzer(
        modules: modules, options: SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .lingering))
    if !hints.isEmpty {
        let context = AnalysisContext()
        context.contextualStrings[.general] = hints
        try await analyzer.setContext(context)
    }
    return analyzer
}

/// The final payload, shared by file and microphone.
private func recognitionPayload(
    _ pieces: [RecognizedPiece], target: RecognitionTarget, duration: Double, includeWords: Bool
) -> [String: Any] {
    [
        "ok": true,
        "text": pieces.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines),
        "segments": pieces.map { $0.json(includeWords: includeWords) },
        "language": target.id, "engine": target.engine, "duration": rounded3(duration),
    ]
}

@available(macOS 26.0, *)
private func transcribeFile(_ envelope: [String: Any]) async throws -> [String: Any] {
    guard let path = envelope["path"] as? String, !path.isEmpty else {
        throw SpeechFailure.unsupported("transcribe needs a file path.")
    }
    guard FileManager.default.fileExists(atPath: path) else {
        throw SpeechFailure.unsupported("Audio file not found: \(path)")
    }
    let file: AVAudioFile
    do {
        file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
    } catch {
        throw SpeechFailure.unsupported(
            "macOS could not read audio from \(path). It reads WAV, AIFF, CAF, MP3, M4A, AAC, FLAC and MP4 or MOV "
            + "video, but not WebM, Ogg or MKV; convert those with ffmpeg first.")
    }
    let duration = Double(file.length) / max(file.fileFormat.sampleRate, 1)
    let target = try await recognitionTarget(envelope["language"] as? String)
    let includeWords = envelope["words"] as? Bool ?? false
    let hints = stringArray(envelope["hints"])
    let (module, pieces) = makeRecognizer(
        target, live: false, alternatives: envelope["alternatives"] as? Bool ?? false)
    try await ensureInstalled(target, module: module, allowDownload: envelope["download"] as? Bool ?? true)

    emit(["ok": true, "done": false, "started": [
        "language": target.id, "engine": target.engine, "duration": rounded3(duration),
    ] as [String: Any]])
    let collector = Task { () throws -> [RecognizedPiece] in
        var finals: [RecognizedPiece] = []
        for try await piece in pieces {
            finals.append(piece)
            emit(["ok": true, "done": false, "segment": piece.json(includeWords: includeWords)])
        }
        return finals
    }
    let analyzer = try await makeAnalyzer(modules: [module], hints: hints)
    let canceller = Canceller()
    canceller.set { Task { await analyzer.cancelAndFinishNow() } }
    do {
        try await withTaskCancellationHandler {
            if let last = try await analyzer.analyzeSequence(from: file) {
                try await analyzer.finalizeAndFinish(through: last)
            } else {
                await analyzer.cancelAndFinishNow()
            }
        } onCancel: {
            canceller.cancel()
        }
        let finals = try await collector.value
        try Task.checkCancellation()
        return recognitionPayload(finals, target: target, duration: duration, includeWords: includeWords)
    } catch {
        collector.cancel()
        try Task.checkCancellation()
        throw error
    }
}

/// What a running `listen` has heard, for deciding when it is over.
private final class ListenState: @unchecked Sendable {
    private let lock = NSLock()
    private var lastChange = Date()
    private var heard = false
    private var latest = ""

    func saw(_ text: String) {
        lock.withLock {
            guard text != latest else { return }
            latest = text
            lastChange = Date()
            if !text.isEmpty { heard = true }
        }
    }

    func quietFor(atLeast silence: Double) -> Bool {
        lock.withLock { heard && Date().timeIntervalSince(lastChange) >= silence }
    }
}

@available(macOS 26.0, *)
private func requireMicrophone() async throws {
    let fix = "Allow your terminal app under System Settings › Privacy & Security › Microphone, then try again."
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return
    case .notDetermined:
        // Answered by whichever app launched this helper: Terminal, VS Code.
        if await AVCaptureDevice.requestAccess(for: .audio) { return }
        throw SpeechFailure.permission("Microphone access was refused. \(fix)")
    default:
        throw SpeechFailure.permission("This process is not allowed to use the microphone. \(fix)")
    }
}

@available(macOS 26.0, *)
private func listenToMicrophone(_ envelope: [String: Any]) async throws -> [String: Any] {
    let silence = num(envelope["silence"]).flatMap { $0 > 0 ? $0 : nil }
    let maxSeconds = num(envelope["maxSeconds"]).flatMap { $0 > 0 ? $0 : nil }
    let includeWords = envelope["words"] as? Bool ?? false
    let target = try await recognitionTarget(envelope["language"] as? String)
    let (module, pieces) = makeRecognizer(target, live: true, alternatives: false)
    try await ensureInstalled(target, module: module, allowDownload: envelope["download"] as? Bool ?? true)
    try await requireMicrophone()

    let engine = AVAudioEngine()
    let input = engine.inputNode
    let micFormat = input.outputFormat(forBus: 0)
    guard micFormat.sampleRate > 0, micFormat.channelCount > 0 else {
        throw SpeechFailure.unsupported("No microphone is available.")
    }
    let wanted = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [module], considering: micFormat)
        ?? micFormat
    guard let converter = AVAudioConverter(from: micFormat, to: wanted) else {
        throw SpeechFailure.failed("macOS cannot convert the microphone's audio to what recognition needs.")
    }

    let (inputs, feed) = AsyncStream<AnalyzerInput>.makeStream()
    input.installTap(onBus: 0, bufferSize: 4096, format: micFormat) { buffer, _ in
        let capacity = AVAudioFrameCount((Double(buffer.frameLength) * wanted.sampleRate / micFormat.sampleRate).rounded(.up)) + 16
        guard let output = AVAudioPCMBuffer(pcmFormat: wanted, frameCapacity: capacity) else { return }
        let once = Once()
        var problem: NSError?
        converter.convert(to: output, error: &problem) { _, status in
            var given = false
            once.run { given = true }
            status.pointee = given ? .haveData : .noDataNow
            return given ? buffer : nil
        }
        if problem == nil, output.frameLength > 0 { feed.yield(AnalyzerInput(buffer: output)) }
    }

    let state = ListenState()
    let (stops, stopFeed) = AsyncStream<Void>.makeStream()
    let stop: () -> Void = { _ = stopFeed.yield(()) }
    Control.shared.onEndInput(stop)
    defer { Control.shared.onEndInput(nil) }

    let collector = Task { () throws -> [RecognizedPiece] in
        var finals: [RecognizedPiece] = []
        for try await piece in pieces {
            let text = piece.text.trimmingCharacters(in: .whitespacesAndNewlines)
            state.saw(piece.isFinal ? "\(finals.count)|\(text)" : text)
            var event = piece.json(includeWords: piece.isFinal && includeWords)
            event["final"] = piece.isFinal
            if piece.isFinal { finals.append(piece) }
            emit(["ok": true, "done": false, "transcript": event])
        }
        return finals
    }

    let started = Date()
    let analyzer = try await makeAnalyzer(modules: [module], hints: stringArray(envelope["hints"]))
    do {
        try await analyzer.start(inputSequence: inputs)
        engine.prepare()
        try engine.start()
    } catch {
        input.removeTap(onBus: 0)
        feed.finish()
        await analyzer.cancelAndFinishNow()
        collector.cancel()
        throw SpeechFailure.failed("Could not start the microphone: \(error.localizedDescription)")
    }
    emit(["ok": true, "done": false, "listening": true, "language": target.id])

    let watcher = Task {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 100_000_000)
            if let maxSeconds, Date().timeIntervalSince(started) >= maxSeconds { break }
            if let silence, state.quietFor(atLeast: silence) { break }
        }
        if !Task.isCancelled { stop() }
    }
    let canceller = Canceller()
    await withTaskCancellationHandler {
        canceller.set(stop)
        for await _ in stops { break }
    } onCancel: {
        canceller.cancel()
    }
    watcher.cancel()
    engine.stop()
    input.removeTap(onBus: 0)
    feed.finish()
    stopFeed.finish()

    if Task.isCancelled {
        await analyzer.cancelAndFinishNow()
        collector.cancel()
        throw CancellationError()
    }
    // The finish runs outside the request's task tree in spirit: stopping is what
    // the caller asked for, and the words already heard are what they want back.
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    let finals = try await collector.value
    return recognitionPayload(
        finals, target: target, duration: Date().timeIntervalSince(started), includeWords: includeWords)
}

@available(macOS 26.0, *)
private func languagesPayload() async throws -> [String: Any] {
    let main = SpeechTranscriber.isAvailable ? await SpeechTranscriber.supportedLocales : []
    let dictation = await DictationTranscriber.supportedLocales
    let installed = await installedLanguages()
    var seen = Set<String>()
    var languages: [[String: Any]] = []
    for (list, useDictation) in [(main, false), (dictation, true)] {
        for locale in list {
            let id = locale.identifier(.bcp47)
            guard seen.insert(id).inserted else { continue }
            languages.append([
                "id": id,
                "name": Locale.current.localizedString(forIdentifier: id) ?? id,
                "installed": (useDictation ? installed.dictation : installed.main).contains(id),
                "engine": useDictation ? "dictation" : "transcriber",
            ])
        }
    }
    languages.sort { ($0["id"] as? String ?? "") < ($1["id"] as? String ?? "") }
    let current = try? await recognitionTarget(nil)
    return ["ok": true, "languages": languages, "default": current?.id ?? "en-US"]
}

@available(macOS 26.0, *)
private func installLanguage(_ envelope: [String: Any]) async throws -> [String: Any] {
    guard let language = envelope["language"] as? String, !language.isEmpty else {
        throw SpeechFailure.unsupported("installSpeechLanguage needs a language.")
    }
    let target = try await recognitionTarget(language)
    let (module, _) = makeRecognizer(target, live: false, alternatives: false)
    try await ensureInstalled(target, module: module, allowDownload: true)
    return ["ok": true, "language": target.id, "engine": target.engine]
}

/// Give a downloaded language back. macOS removes the files when it wants the
/// space; this only says this helper no longer needs them.
@available(macOS 26.0, *)
private func removeLanguage(_ envelope: [String: Any]) async -> [String: Any] {
    let language = (envelope["language"] as? String ?? "").replacingOccurrences(of: "_", with: "-").lowercased()
    for locale in await AssetInventory.reservedLocales where locale.identifier(.bcp47).lowercased() == language {
        return ["ok": true, "removed": await AssetInventory.release(reservedLocale: locale)]
    }
    return ["ok": true, "removed": false]
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
                "speech": true,
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
            // Ctrl-C reaches the whole foreground process group. The parent
            // decides what it means: a `listen` that is stopped this way still
            // has to finish, and hand back what it heard, in this process.
            signal(SIGINT, SIG_IGN)
            let requests = AsyncStream<String> { continuation in
                let reader = Thread {
                    while let line = readLine(strippingNewline: true) {
                        if line.isEmpty || Control.shared.intercept(line) { continue }
                        continuation.yield(line)
                    }
                    Control.shared.stdinClosed()
                    continuation.finish()
                    // Nobody is left to read a reply. A request that ignores
                    // cancellation must not leave this process running on, and
                    // ignoring SIGINT has removed the other way it would die.
                    DispatchQueue.global().asyncAfter(deadline: .now() + 5) { exit(0) }
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
