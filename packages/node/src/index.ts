/**
 * apple-llm — one library for Apple's on-device and Private Cloud Compute models.
 *
 * macOS 26+ on Apple Silicon only. No API key, no account, no developer program.
 */
export { AppleLLM, Conversation, captureScreenshot, probe } from './client.js';
export type {
  AppleLLMOptions,
  DeepPartial,
  GenerateResult,
  Input,
  JsonOptions,
  ObjectStream,
  ProbeResult,
  StreamOptions,
  TextOptions,
  TextStream,
  Tier,
} from './client.js';
export { ResultStream } from './stream.js';
export { tool, normalizeTools, toolOutputText } from './tools.js';
export type { FunctionTool, ToolExecutionContext, ToolSet } from './tools.js';
export { splitMessages, ReplayBook } from './messages.js';
export type { ChatMessage, ChatToolCall, HistoryEntry } from './messages.js';
export { isStandardSchema, resolveSchema, restoreNulls } from './standard-schema.js';
export type { InferSchema, SchemaLike, StandardSchemaV1, StandardJSONSchemaV1 } from './standard-schema.js';
export {
  DeviceClient,
  probeDevice,
  DEFAULT_TEMPERATURE,
  DEFAULT_MAX_TOKENS,
  DEFAULT_MAX_TOOL_CALLS,
  assertTools,
  parseImageFlag,
  withDocuments,
} from './device.js';
export type {
  DeviceProbe,
  DeviceRequest,
  DeviceClientOptions,
  DeviceOutcome,
  RunHooks,
  ModelCapabilities,
  HelperFeatures,
  CloudQuota,
  CloudModelInfo,
  BuiltInTool,
  HistoryTurn,
  ImageAttachment,
  SamplingMode,
  UseCase,
  Guardrails,
  Usage,
  FinishReason,
  ToolCallRecord,
} from './device.js';
export {
  CloudClient,
  probeCloud,
  installCloudShortcut,
  shortcutDefinition,
  cloudSetupHint,
  CLOUD_SHORTCUT_NAME,
  CLOUD_SHORTCUT_NAME_WEB,
  CLOUD_CONTEXT_TOKENS,
} from './cloud.js';
export type { CloudProbe, CloudRequest } from './cloud.js';
export { toAppleSchema } from './schema.js';
export type { JsonSchema } from './schema.js';
export { parseLlmJson, stripCodeFences, extractJsonSpan } from './json-recovery.js';
export { isAppleSiliconMac, targetTripleFrom, hostTarget } from './target.js';
export { cacheDir, fingerprint, helperSource, ensureBinary } from './compile.js';
export type { Progress, OnProgress } from './compile.js';
export {
  AppleLLMError,
  ModelUnavailableError,
  SchemaRejectedError,
  SchemaValidationError,
  ContextLengthError,
  QuotaError,
  TimeoutError,
  AbortError,
  SetupRequiredError,
  RefusalError,
  ToolExecutionError,
  ModelBusyError,
  UnsupportedError,
} from './errors.js';
export type { UnavailableReason, ErrorCode } from './errors.js';
