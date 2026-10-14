# Contributing to vsr-mobile

Issues and pull requests are welcome. For a large change, open an issue first so we can agree on the approach.

## Set-up

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train,export,dev]"
cargo build -p vsr-runtime --features onnx
```

You also need `ffmpeg` and `ffprobe` on the `PATH`, with libx264 and libaom-av1 for the codec tests.

## Tests and lint

```bash
pytest
ruff check .
cargo fmt --all --check
cargo clippy --all-targets --features onnx -- -D warnings
cargo test
examples/smoke.sh          # the whole path: train, export, play back
```

CI runs all of these on every push to `main` and on every pull request. The TensorRT plugin (`plugins/`) and the iOS
demo (`ios/VSRDemo`) are not built in CI; if you change them, say in the pull request on which device you built and
ran them.

## What a pull request needs

- A test for the behaviour it adds or fixes, written against a public function or the command line, not internals.
- A line under a new heading at the top of `CHANGELOG.md`.
- Commit messages in the style of the history: `type(scope): summary`, lowercase, for example
  `feat(runtime): …`, `fix(export): …`, `test(train): …`, `docs: …`.
- Exported models keep exactly two inputs, `frame` and `state`; the runtime and the iOS demo depend on it.

## Security

Please do not open a public issue for a security problem. Follow
[hanaML's security policy](https://github.com/hanaml-lab/.github/blob/main/SECURITY.md).

## Licence

vsr-mobile is licensed under Apache-2.0. By contributing, you agree that your contributions are licensed under the
same terms (inbound = outbound).
