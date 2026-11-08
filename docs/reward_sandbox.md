# Dataset, extraction, and reward isolation

The dataset is `google-research-datasets/mbpp`, configuration `full`, pinned to
`4bb6404fdc6cacfda99d4ac4205087b89d32030c`. The loader checks train=374,
validation=90, test=500, prompt=10, unique task IDs, and disjoint split IDs. It
records per-split Hugging Face fingerprints and SHA-256 hashes of the actual
loaded content. The prompt split is preserved and never mixed into training.

Primary references:

- [Official MBPP dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp/blob/4bb6404fdc6cacfda99d4ac4205087b89d32030c/README.md)
- [Pinned full-data conversion commit](https://huggingface.co/datasets/google-research-datasets/mbpp/commit/4bb6404fdc6cacfda99d4ac4205087b89d32030c)
- [Google Research source](https://github.com/google-research/google-research/tree/master/mbpp)

`prompt_for` uses the same chat template for all arms, includes the task and
public `test_list` examples, and never includes the reference solution. Reward
uses every test in the task's official `test_list`, plus `test_setup_code`.
Optional `challenge_test_list` entries do not alter the primary MBPP reward.

Reward decoding ends at the first EOS/padding token. The sampler still records
all 256 generated tokens; post-stop samples cannot be joined into reward code.
One extraction rule is shared between reward and commitment eligibility: use the
first Python code fence if present, otherwise locate the first implementation
statement; exclude trailing generated top-level assertions, assertion-based `test_` functions,
`unittest.TestCase` classes, and an `__main__` test harness when identifiable. Invalid Python
is preserved and receives syntax validity zero, rather than being repaired to
manufacture a passing implementation. Token eligibility requires exact
original-token/decode/re-encode alignment and full containment in the code span.
An alignment failure returns no eligible tokens and emits a warning. Whitespace
inside the implementation is eligible; padding, stop tokens, fences, prose, and
reliably identifiable test suffixes are excluded.

`RewardRunner` requires a SLURM allocation and a functioning Linux `bwrap`
installation supporting `--disable-userns`, `--size`, and `--remount-ro`. Its preflight executes a fixed harmless probe inside the exact
sandbox configuration. The full test process receives:

- Separate user, PID, IPC, network, and UTS namespaces, no network access,
  dropped capabilities, disabled nested user namespaces, and a new session that
  dies with the parent.
- Read-only runtime mounts (`/usr`, `/bin`, `/lib`, `/lib64`, and the exact Python
  base installation when needed). It receives no home, project, model cache,
  scratch tree, scheduler environment, or W&B credentials.
- Read-only root, `/dev`, and `/proc` mounts. A separate 16 MiB writable temporary
  filesystem, a read-only worker script, and one
  read-only JSON input containing only implementation, test setup, and tests.
- 512 MiB address-space limit, 5 CPU seconds, a hard 6-second wall timeout,
  64 file descriptors, 32 processes, 64 KiB output-file limit, and no core dumps.

Each test executes with a fresh globals namespace. Normal candidate stdout and
stderr are discarded. The parent bounds all process output with regular files
and parses structured results. A crash, malformed output, resource-limit exit,
or timeout earns zero binary reward and records the reason. Ordinary test
exceptions retain the measured partial pass fraction.

The runner never falls back to unrestricted host Python. If user namespaces are
blocked or bubblewrap is missing, the job must stop before collecting rewards.
The site's scheduler administrator must provide an equivalent isolation facility
before training can launch. No generated code is executed during macOS unit
tests; these test extraction, metadata boundaries, fail-closed behavior, and
result aggregation using controlled mocks.

The worker captures execution helpers, copies the original builtins for each
test, compiles tests before running a candidate, catches `BaseException`
(including `SystemExit`), and writes results through a captured file descriptor.
This prevents simple `sys.exit`, `builtins.exec`, or JSON serializer mutation from
turning a failing program into a pass. Python interpreter introspection is not an
adversarially secure scoring boundary; the external namespace sandbox protects
the host independently, and reward records remain auditable.

The filesystem configuration follows the [official bubblewrap options](https://github.com/containers/bubblewrap/blob/main/bwrap.xml):
`--disable-userns` prevents a candidate from creating a new user namespace and
replacing the bounded temporary filesystem. Nonrecursive `--remount-ro /` leaves
the separate `/tmp` mount writable while closing writes to the root tmpfs. An
older bubblewrap or kernel that cannot enforce these settings fails preflight.

## Compatibility with older cluster installations

`scripts/setup.sh` first checks `bwrap --help` and `bwrap --version`. Missing
options are reported before any worker starts. Setup reuses a capable installed
binary or builds bubblewrap **0.13.0** and a static libcap **2.75** under
`runtime/`, using official source archives verified by hard-coded SHA-256 hashes.
The build uses its own Meson 1.9.1/Ninja 1.13.0 environment, with no sudo or system
libcap development package required. A C compiler, make, ar, ranlib, and pkg-config
must be available. No isolation option is dropped to accommodate an older binary.

The build manifest is `runtime/metadata/bubblewrap.json`. Selection prefers the
explicit `HINDSIGHT_BWRAP` override, then `runtime/bin/bwrap`, then the site's
binary. `scripts/install_bwrap.py --check` checks capabilities only and is safe
on the login node; actual namespace/worker execution is still verified in the
compute allocation. If a compiler/build-tool module is needed, load the site's
provided tools and rerun setup.

The pinned sources are the [official bubblewrap release](https://github.com/containers/bubblewrap/releases/tag/v0.13.0)
and [official libcap source archive](https://www.kernel.org/pub/linux/libs/security/linux-privs/libcap2/libcap-2.75.tar.xz).
