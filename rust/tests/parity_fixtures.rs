//! Emits compressed output for shared fixtures so the Python side can diff
//! against it. Run via: `cargo test --test parity_fixtures -- --nocapture`
//! or through `scripts/check_parity.py`.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use token_defense_proxy::{
    json_to_minimal_yaml, lossless_code_compressor, lossless_terminal_cleaner, CodeOptions,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("fixtures")
}

fn out_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("fixtures").join("rust_out")
}

#[test]
fn emit_rust_outputs() {
    let fx = fixtures_dir();
    let out = out_dir();
    fs::create_dir_all(&out).expect("create output dir");

    let code = fs::read_to_string(fx.join("sample_code.py")).expect("read sample_code.py");
    fs::write(
        out.join("code.txt"),
        lossless_code_compressor(&code, &CodeOptions::default()),
    )
    .expect("write code output");

    let logs = fs::read_to_string(fx.join("sample_logs.txt")).expect("read sample_logs.txt");
    fs::write(out.join("logs.txt"), lossless_terminal_cleaner(&logs, true, true))
        .expect("write logs output");

    let data_raw = fs::read_to_string(fx.join("sample_data.json")).expect("read sample_data.json");
    let data: Value = serde_json::from_str(&data_raw).expect("parse sample_data.json");
    fs::write(out.join("data.txt"), json_to_minimal_yaml(&data)).expect("write data output");

    println!("wrote rust fixture outputs to {}", out.display());
}
