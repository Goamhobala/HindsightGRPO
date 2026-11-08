"""Standalone standard-library worker. The parent must isolate this process.

Trusted execution helpers and precompiled tests are captured before running a
candidate. This blocks simple builtins/JSON monkeypatching and early sys.exit
from manufacturing a passing result. This is not a proof against adversarial
Python interpreter introspection; the external namespace sandbox protects the
host independently of scoring logic.
"""
import builtins
import io
import json
import os
import sys


class DiscardOutput(io.TextIOBase):
    def write(self, text):
        return len(text)

    def flush(self):
        return None


def main():
    with open(sys.argv[1], encoding="utf-8") as handle:
        payload = json.load(handle)
    # Capture helpers and test bytecode before executing untrusted Python. Keep
    # scoring state in function locals, outside the generated globals dict.
    execute = exec
    catch_exception = BaseException
    exception_type = type
    original_builtins = vars(builtins).copy()
    copy_dict = dict
    stringify = str
    write_result = os.write
    result_fd = os.dup(1)
    name_length = len
    sys.stdout = DiscardOutput()
    sys.stderr = DiscardOutput()
    code = compile(payload["code"], "<generated>", "exec")
    setup = compile(payload["setup"], "<setup>", "exec")
    tests = tuple(compile(test, "<test>", "exec") for test in payload["tests"])
    error_bytes = []
    passed = 0
    for test in tests:
        namespace = {"__name__": "__generated__", "__builtins__": copy_dict(original_builtins)}
        try:
            execute(setup, namespace)
            execute(code, namespace)
            execute(test, namespace)
        except catch_exception as error:
            name = exception_type(error).__name__
            if not name.isidentifier() or name_length(name) > 64:
                name = "ExecutionError"
            error_bytes.append(b'"' + name.encode("ascii", "replace") + b'"')
        else:
            passed += 1
            error_bytes.append(b"null")
    first_error = b"null"
    for error in error_bytes:
        if error != b"null":
            first_error = error
            break
    # Avoid mutable global serializers and sys.stdout after candidate execution.
    result = (b'{"tests_passed":' + stringify(passed).encode("ascii")
              + b',"test_errors":[' + b",".join(error_bytes)
              + b'],"error":' + first_error + b',"timeout":false}\n')
    write_result(result_fd, result)


if __name__ == "__main__":
    main()
