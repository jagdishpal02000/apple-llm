"""Shortcut plist, triple derivation, fingerprint, drift and JSON recovery.

Runs anywhere -- nothing here needs Apple hardware.
"""

import hashlib
import json

import pytest

from apple_llm import (
    CLOUD_SHORTCUT_NAME,
    AppleLLMError,
    assert_tools,
    extract_json_span,
    fingerprint,
    parse_image_flag,
    parse_llm_json,
    shortcut_definition,
    strip_code_fences,
    target_triple_from,
    with_documents,
)


class TestShortcutDefinition:
    def setup_method(self):
        self.definition = shortcut_definition()
        self.params = self.definition["WFWorkflowActions"][0]["WFWorkflowActionParameters"]

    def test_uses_wfllmprompt_not_wfinput(self):
        # The single most expensive mistake on this path. A wrong key imports
        # fine and is then silently discarded, after which the action blocks
        # forever on an interactive "Request..." panel.
        assert "WFLLMPrompt" in self.params
        assert "WFInput" not in self.params
        assert "WFInput" not in json.dumps(self.definition)

    def test_omits_the_model_key_so_the_action_defaults_to_cloud(self):
        assert not [k for k in self.params if "model" in k.lower()]

    def test_asks_for_plain_text_out(self):
        # "Automatic" reshapes the result for whatever action follows.
        assert self.params["WFGenerativeResultType"] == "Text"

    def test_splices_the_input_with_a_ufffc_attachment(self):
        prompt = self.params["WFLLMPrompt"]
        assert prompt["WFSerializationType"] == "WFTextTokenString"
        assert prompt["Value"]["string"] == "￼"
        assert prompt["Value"]["attachmentsByRange"] == {"{0, 1}": {"Type": "ExtensionInput"}}

    def test_targets_the_ask_llm_action(self):
        identifier = self.definition["WFWorkflowActions"][0]["WFWorkflowActionIdentifier"]
        assert identifier == "is.workflow.actions.askllm"

    def test_web_search_flag_is_opt_in_and_gets_its_own_uuid(self):
        assert "WFAllowWebSearch" not in self.params
        web = shortcut_definition(True)["WFWorkflowActions"][0]["WFWorkflowActionParameters"]
        assert web["WFAllowWebSearch"] is True
        # The two variants must not collide in the Shortcuts library.
        assert web["UUID"] != self.params["UUID"]

    def test_name_is_distinctive(self):
        assert "Apple LLM" in CLOUD_SHORTCUT_NAME


class TestTargetTriple:
    def test_drops_the_patch_component(self):
        # swiftc's default host triple is arm64-apple-macosx27.0.0, which no
        # shipped stdlib matches.
        assert target_triple_from("27.0\n", "arm64") == "arm64-apple-macos27.0"

    def test_handles_a_bare_major(self):
        assert target_triple_from("26", "arm64") == "arm64-apple-macos26.0"

    def test_ignores_an_sdk_patch_component(self):
        assert target_triple_from("27.1.2", "arm64") == "arm64-apple-macos27.1"

    def test_arch_names_are_normalised(self):
        # Node says x64, Python says x86_64; they must derive the same triple or
        # each would compile its own copy of the helper.
        assert target_triple_from("27.0", "x64") == target_triple_from("27.0", "x86_64")

    def test_returns_none_for_unusable_output(self):
        assert target_triple_from("", "arm64") is None
        assert target_triple_from("not-a-version", "arm64") is None


class TestFingerprint:
    def test_is_sha256_of_source_newline_triple_truncated_to_12(self):
        # Pinned exactly: the Node package computes the same string, and the two
        # share one compiled binary only if they agree byte for byte.
        expected = hashlib.sha256(b"SOURCE\narm64-apple-macos27.0").hexdigest()[:12]
        assert fingerprint("SOURCE", "arm64-apple-macos27.0") == expected
        assert len(fingerprint("SOURCE", "arm64-apple-macos27.0")) == 12

    def test_changes_with_the_source_or_the_os(self):
        base = fingerprint("a", "t")
        assert fingerprint("b", "t") != base
        assert fingerprint("a", "u") != base


class TestEmbeddedHelper:
    def test_is_byte_identical_to_the_repo_source(self, repo_root):
        # The two packages ship copies of one file; this is what stops them
        # drifting.
        source = (repo_root / "swift" / "helper.swift").read_bytes()
        embedded = (
            repo_root / "packages" / "python" / "src" / "apple_llm" / "swift" / "helper.swift"
        ).read_bytes()
        assert hashlib.sha256(embedded).hexdigest() == hashlib.sha256(source).hexdigest()

    def test_matches_the_node_package_copy_too(self, repo_root):
        source = (repo_root / "swift" / "helper.swift").read_bytes()
        node = (repo_root / "packages" / "node" / "swift" / "helper.swift").read_bytes()
        assert hashlib.sha256(node).hexdigest() == hashlib.sha256(source).hexdigest()


class TestJsonRecovery:
    # Only the cloud tier needs this: on device, constrained decoding means the
    # reply is already JSON.
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"a":1}', {"a": 1}),
            ('```json\n{"a":1}\n```', {"a": 1}),
            ("```\n{\"a\":1}\n```", {"a": 1}),
            ('Here is the JSON:\n```json\n{"a":1}\n```\nHope that helps!', {"a": 1}),
            ('Sure! {"a":1} - let me know.', {"a": 1}),
            ("Result: [1,2,3]", [1, 2, 3]),
            ('x {"a":"}{ not the end"} y', {"a": "}{ not the end"}),
            ('{"a":"say \\"hi\\" }"}', {"a": 'say "hi" }'}),
            ('note {"a":{"b":[{"c":1}]}} end', {"a": {"b": [{"c": 1}]}}),
        ],
    )
    def test_recovers(self, raw, expected):
        assert parse_llm_json(raw) == expected

    def test_raises_with_an_echo_when_there_is_no_json(self):
        with pytest.raises(ValueError, match="I cannot help"):
            parse_llm_json("I cannot help with that.")

    def test_exposes_the_pieces(self):
        assert strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'
        assert extract_json_span("nope") is None
        assert extract_json_span('a {"b":1} c') == '{"b":1}'


class TestImageAttachments:
    def test_bare_path_passes_through(self):
        assert parse_image_flag("./photo.png") == "./photo.png"

    def test_splits_path_colon_colon_label(self):
        assert parse_image_flag("./scan.png::receipt") == {"path": "./scan.png", "label": "receipt"}

    def test_single_colon_is_not_a_label(self):
        assert parse_image_flag("C:/photos/a.png") == "C:/photos/a.png"

    def test_empty_label_is_not_split(self):
        assert parse_image_flag("./a.png::") == "./a.png::"


class TestBuiltInTools:
    def test_accepts_the_three_known_tools(self):
        assert_tools(["ocr", "barcode", "spotlight"])
        assert_tools(None)
        assert_tools([])

    def test_rejects_unknown_tools_before_spending_a_call(self):
        with pytest.raises(AppleLLMError, match="Unknown tool"):
            assert_tools(["laser"])


class TestDocumentInlining:
    def test_no_documents_returns_the_prompt(self):
        assert with_documents("hi", None) == "hi"
        assert with_documents("hi", []) == "hi"

    def test_splices_text_files_with_a_delimiter(self, tmp_path):
        note = tmp_path / "note.txt"
        note.write_text("remember the milk")
        out = with_documents("Summarize:", [str(note)])
        assert "Summarize:" in out
        assert "--- Document:" in out
        assert "remember the milk" in out

    def test_missing_file_is_an_error_not_a_silent_drop(self):
        with pytest.raises(AppleLLMError, match="not found"):
            with_documents("hi", ["/definitely/not/here.txt"])
