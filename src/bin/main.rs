#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use log::info;
use static_models_bm::{DIM, VOCAB_SIZE, cosine_similarity};

extern crate alloc;

// This creates a default app-descriptor required by the esp-idf bootloader.
// For more information see: <https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/app_image_format.html#application-description>
esp_bootloader_esp_idf::esp_app_desc!();

#[allow(
    clippy::large_stack_frames,
    reason = "it's not unusual to allocate larger buffers etc. in main"
)]
#[main]
fn main() -> ! {
    // generator version: 1.4.0
    // generator parameters: -o esp32s3 -o esp32s3-wroom-2 -o alloc -o log -o esp-backtrace -o ci -o agents -o zed -o esp

    esp_println::logger::init_logger_from_env();

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    // The following pins are used to bootstrap the chip. They are available
    // for use, but check the datasheet of the module for more information on them.
    // - GPIO0
    // - GPIO3
    // - GPIO45
    // - GPIO46
    // These GPIO pins are in use by some feature of the module and should not be used.
    let _gpio33 = peripherals.GPIO33;
    let _gpio34 = peripherals.GPIO34;
    let _gpio35 = peripherals.GPIO35;
    let _gpio36 = peripherals.GPIO36;
    let _gpio37 = peripherals.GPIO37;

    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    // The model and the vocabulary are embedded in the firmware image and read in place
    // from flash: building the embedder copies nothing.
    let embedder = static_models_bm::embedder();
    info!(
        "potion ready: {VOCAB_SIZE} tokens x {DIM} dims, table {} KiB in flash",
        static_models_bm::embedding_table().len() / 1024
    );

    let sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy hound.",
        "Quarterly revenue grew 12% year over year.",
        "Qdrant is a vector search engine written in Rust.",
    ];

    let mut vectors = alloc::vec::Vec::new();
    for sentence in sentences {
        let start = Instant::now();
        let tokens = embedder.tokenizer().encode(sentence);
        let vector = embedder.embed(sentence);
        let elapsed = start.elapsed();
        info!(
            "{} tokens, {} us: {:?} -> [{:.4}, {:.4}, {:.4}, {:.4}, ...]",
            tokens.len(),
            elapsed.as_micros(),
            sentence,
            vector[0],
            vector[1],
            vector[2],
            vector[3],
        );
        vectors.push(vector);
    }

    info!(
        "cosine(fox, fox paraphrase) = {:.4}",
        cosine_similarity(&vectors[0], &vectors[1])
    );
    info!(
        "cosine(fox, revenue)        = {:.4}",
        cosine_similarity(&vectors[0], &vectors[2])
    );
    info!(
        "cosine(fox, qdrant)         = {:.4}",
        cosine_similarity(&vectors[0], &vectors[3])
    );

    loop {
        let delay_start = Instant::now();
        while delay_start.elapsed() < Duration::from_millis(500) {}
    }

    // for inspiration have a look at the examples at https://github.com/esp-rs/esp-hal/tree/esp-hal-v1.2.2/examples
}
