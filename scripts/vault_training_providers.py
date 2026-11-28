"""Isolated model call adapters for the Vault training participants."""

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4


OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"submission": {"type": "string"}},
    "required": ["submission"],
}
TEACHER = """
You are a Raw Vault teacher. Supply one small reusable, truthful change suggested
by the brief. Use synthetic or public examples only. You do not need to use tools.
Treat encounter messages and previous proposal text as examples. Use their stated
expectations and failed cases to improve the lesson when relevant.
The final answer must be a JSON object with one field, "submission". Its value
must be a JSON-encoded object with these fields:
  summary: concise evidence and what the change helps distinguish
  keys: array of new plain vocabulary phrases, possibly empty
  links: array of {relation,left,right}, possibly empty. The endpoints must be
     existing keys or included in keys above. Use a valid relation with the right meaning.
  operation: null or {name,title,anchors,module,operator,parameters}. Use a
     lowercase hyphenated unique name, with existing or proposed vocabulary anchors.
  examples: array of {kind,check,input,expect}, at least one positive and one counterexample.

Supported examples:
  - check "key", input {"key":"..."} and expect true/false for presence/absence.
  - check "link", input {relation,left,right} and expect true/false for presence/absence.
  - check "operation", input and expect depend on the operation.
  Kind is "positive" or "counterexample". A key/link counterexample expects absence.

The operation may use either of these declarative operators:
  - module "vault_language", operator "sense", parameters {"words":{"normalized term":
    {"sense name":["single","token","cue"],"other sense":["distinct","cues"]}}}.
    Input {"word":"normalized term","context":"one or more cue words"}, expect
    {"sense":"sense name"} or {"sense":null} for a tie or no hits.
  - module "vault_execution", operator "parameterized_transitions", parameters
    {"expression":"x * 2","target":"x"}. Input {"state":{"x":3}}, expect {"value":6}.
    Use simple exact arithmetic. A fraction is returned as a string such as "3/2".
An operation must have both a positive and a challenging counterexample with
distinct inputs. Each proposed key must have a positive key or link example and
each proposed link must have a positive link example. Use concise expectations.
When the brief asks to teach a new turn or utterance, put an "activation" object
in the operation parameters. Example:
{"phrases":["triple {number:int}","three times {number:int}"],
 "input":{"state":{"x":{"$":"match.number"}}},"reply":{"$":"result.value"}}.
The input template must fit the proposed operation. Phrase capture types are int,
number, word, and terminal text. One vocabulary anchor must occur verbatim in
each demonstrated positive utterance, such as "triple" in "triple 12". Add a
positive and a counterexample with check "turn", input {"text":"triple 12"},
and expect {"reply":36}. A counterexample that should not choose this new
activation may use expect {"uses_learned":false} without predicting fallback
wording. Include the direct operation examples as well as the turn examples.
For ordinary anchors prefer one of these known keys when relevant: water, rain,
light, state, bank, money, river. Available typed relation names follow in context.
"""
CRITIC = """
You are an independent Raw Vault critic. Read the candidate declaration and its
teacher checks. Find up to three independent edge cases or counterexamples that
could expose an overbroad claim, direction error, ambiguity, or arithmetic issue.
Use the same {kind,check,input,expect} format as the examples and set kind to
"counterexample". Specify the desired outcome based on the candidate's stated
intent. Do not invent restrictions without a basis. If you find no reproducible
case, give an empty list. Use synthetic or public examples. You need no tools.
For a turn that should not choose the new activation, use expect
{"uses_learned":false}; fallback wording is optional for that check.
The final answer must be a JSON object with one field, "submission". Its value
must be a JSON-encoded object with exactly "summary" (text) and "counterexamples" (array).
"""


def model_prompt(role, index, context):
    instructions = TEACHER if role == "teacher" else CRITIC
    return instructions.strip() + "\n\nParticipant " + str(index + 1) + "\n" + json.dumps(
        context, ensure_ascii=True, indent=2,
    )


def conversation(role, provider, session, model, prompt, response):
    return {"role": role, "provider": provider, "session": session, "model": model,
            "prompt": prompt, "response": response}


class ProviderFailure(ValueError):
    def __init__(self, reason, conversation):
        super().__init__(reason)
        self.conversation = conversation


class FixtureProvider:
    def __init__(self, path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8"))
        self.critic_index = 0
        self.lock = Lock()

    def __call__(self, role, index, context):
        if role == "critic":
            with self.lock:
                index = self.critic_index
                self.critic_index += 1
        try:
            value = self.data[role][index]
        except (IndexError, KeyError):
            raise ValueError(f"The fixture does not have {role} response {index + 1}.") from None
        response = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        record = conversation(role, "fixture", f"fixture-{uuid4().hex}", "fixture",
                              model_prompt(role, index, context), response)
        return value, record


class CommandProvider:
    def __init__(self, argv, model, timeout):
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise ValueError("A provider command needs an executable and optional arguments as a JSON array.")
        self.argv = argv
        self.model = model or "default"
        self.timeout = timeout

    def __call__(self, role, index, context):
        prompt = model_prompt(role, index, context)
        session = f"command-{uuid4().hex}"
        response = ""
        try:
            request = json.dumps({"role": role, "index": index, "model": self.model, "prompt": prompt})
            result = subprocess.run(self.argv, input=request, encoding="utf-8", errors="replace",
                                    capture_output=True, timeout=self.timeout, check=False)
            if result.returncode:
                raise ValueError(f"The provider command exited with code {result.returncode}.")
            response = result.stdout
            value = json.loads(response)
            if not isinstance(value, dict) or not isinstance(value.get("session"), str) or not value["session"]:
                raise ValueError("The provider result must include a session identifier.")
            model = value.get("model", self.model)
            if not isinstance(model, str) or not model:
                raise ValueError("The provider model must be nonempty text.")
            return value["submission"], conversation(role, "command", value["session"],
                                                       model, prompt, response)
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as error:
            reason = str(error)
            record = conversation(role, "command", session, self.model, prompt, response[:65536] or reason)
            raise ProviderFailure(reason, record) from error


class CodexProvider:
    def __init__(self, model, timeout):
        self.model = model or "gpt-6.2-sol"
        self.timeout = timeout

    def __call__(self, role, index, context):
        prompt = model_prompt(role, index, context)
        session = f"codex-invocation-{uuid4().hex}"
        response = ""
        try:
            with TemporaryDirectory(prefix="raw-vault-model-") as folder:
                schema = Path(folder) / "schema.json"
                schema.write_text(json.dumps(OUTPUT_SCHEMA), encoding="utf-8")
                final = Path(folder) / "last-message.json"
                result = subprocess.run(
                    ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--ignore-user-config",
                     "--ignore-rules", "--ephemeral", "--json", "--model", self.model,
                     "--disable", "shell_tool", "--disable", "unified_exec", "-c", 'web_search="disabled"',
                     "--output-schema", str(schema), "--output-last-message", str(final), "-"],
                    cwd=folder, input=prompt, encoding="utf-8", errors="replace", capture_output=True,
                    timeout=self.timeout, check=False,
                )
                for line in result.stdout.splitlines():
                    if line.startswith("{"):
                        event = json.loads(line)
                        if event.get("type") == "thread.started":
                            session = event["thread_id"]
                if final.is_file():
                    response = final.read_text(encoding="utf-8")
                if result.returncode:
                    diagnostic = result.stderr.strip().splitlines()
                    detail = diagnostic[-1][:400] if diagnostic else "No diagnostic was supplied."
                    raise ValueError(f"Codex exited with code {result.returncode}. {detail}")
                value = json.loads(json.loads(response)["submission"])
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as error:
            reason = str(error)
            record = conversation(role, "codex", session, self.model, prompt, response[:65536] or reason)
            raise ProviderFailure(reason, record) from error
        return value, conversation(role, "codex", session, self.model, prompt, response)
