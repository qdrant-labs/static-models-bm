# static-models-bm

Bare-metal runner for the [potion-2M](https://huggingface.co/minishlab) static embedding model on an ESP32-S3 (`no_std`, Rust).

The model (7.2 MiB, F32, 29,528 tokens x 64 dims) and the tokenizer vocabulary are embedded in the firmware image and read in place from flash. Only small per-request scratch data uses the heap.

## How it works

- `build.rs` compiles `model/tokenizer.json` into a sorted binary vocabulary and reads the shape of `model/model.safetensors`. It fails the build if the model is not F32 or the sizes don't match.
- `src/tokenizer.rs` is a `no_std` BERT WordPiece tokenizer: normalize (clean text, CJK spacing, accent stripping, lowercase), split on whitespace and punctuation, then greedy longest-match WordPiece.
- `src/embedder.rs` looks up the token rows, mean-pools them, and optionally L2-normalizes (`normalize`, default `true`).
- `src/bin/main.rs` embeds a few sample sentences and logs timings, vector heads and cosine similarities.

The tokenizer and pooling match the [`statembed`](https://github.com/AstraBert/statembed) crate exactly on a set of tricky inputs (accents, CJK, emoji, control characters, truncation).

## Requirements

- An ESP32-S3 with 16 MB of flash (the image is about 8.2 MB), connected over USB.
- The Xtensa Rust toolchain installed with [`espup`](https://github.com/esp-rs/espup).
- `espflash`.
- The model files in `model/`: `model.safetensors` and `tokenizer.json`.

## Run

```bash
source ~/export-esp.sh
cargo run --release
```

This builds, flashes, and opens the serial monitor. Press `Ctrl+R` in the monitor to reset the board and print the output again, or `Ctrl+C` to exit. To watch the output without reflashing, run `espflash monitor`.

Expected output:

```
potion ready: 29528 tokens x 64 dims, table 7382 KiB in flash
10 tokens, 2720 us: "The quick brown fox jumps over the lazy dog." -> [-0.2936, -0.1589, 0.1046, -0.2228, ...]
...
cosine(fox, fox paraphrase) = 0.5557
cosine(fox, revenue)        = 0.0089
cosine(fox, qdrant)         = -0.1157
```

## Troubleshooting

- **Permission denied on `/dev/ttyACM0`:** run `sudo usermod -aG dialout $USER`, then log out and in.
- **Board not listed or `Connecting...` fails:** use a data-capable USB cable. Hold BOOT, tap RESET, release BOOT, then try again.
- **`waiting for download` after flashing:** release BOOT and tap RESET, or unplug and replug the board.
- **Flash size errors:** add `--flash-size 16mb` to the runner in `.cargo/config.toml`.

## Use it in your code

```rust
let embedder = static_models_bm::embedder();
let vector = embedder.embed("Qdrant is a vector search engine written in Rust.");
```

The embedder is cheap to create because nothing is copied. Texts are truncated at 512 tokens.
