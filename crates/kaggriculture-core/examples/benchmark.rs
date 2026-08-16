use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use kaggriculture_core::{Action, Config, Sim};

const DEFAULT_EPISODES: u32 = 10_000;
const ACTING_TRANSITIONS: usize = 719;

fn parse_episode_count() -> Result<u32, String> {
    let mut arguments = std::env::args();
    let program = arguments.next().unwrap_or_else(|| "benchmark".to_owned());
    let raw = arguments
        .next()
        .unwrap_or_else(|| DEFAULT_EPISODES.to_string());
    if arguments.next().is_some() {
        return Err(format!("usage: {program} [positive_episode_count]"));
    }

    match raw.parse::<u32>() {
        Ok(episodes) if episodes > 0 => Ok(episodes),
        _ => Err(format!(
            "episode count must be a positive integer (received {raw:?})"
        )),
    }
}

fn run_episode(seed: u64, pass: &Action) -> Sim {
    let mut sim = Sim::new(Config {
        seed,
        ..Config::default()
    });
    for _ in 0..ACTING_TRANSITIONS {
        sim.step([pass, pass]);
    }
    sim
}

fn main() -> ExitCode {
    if cfg!(debug_assertions) {
        eprintln!(
            "benchmark refused a debug build; rerun with: \
             cargo run --release -p kaggriculture-core --example benchmark -- [episodes]"
        );
        return ExitCode::FAILURE;
    }

    let episodes = match parse_episode_count() {
        Ok(episodes) => episodes,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };
    let pass = Action::pass();

    // Compile/lazy-allocation warm-up is deliberately outside the timed region.
    let warmup = run_episode(u64::from(episodes), &pass);
    assert_eq!(warmup.state.step, 719);
    assert!(warmup.state.done);
    black_box(warmup);

    let started = Instant::now();
    let mut checksum = 0_u64;
    for seed in 0..u64::from(episodes) {
        // Varying seeds exercise deterministic end-of-day randomness. Passing
        // the complete result through black_box keeps the simulation observable
        // to the optimizer even though pass/pass leaves both rewards unchanged.
        let sim = black_box(run_episode(black_box(seed), &pass));
        checksum = checksum.wrapping_add(u64::from(sim.state.step));
    }
    let elapsed = started.elapsed();
    black_box(checksum);

    let seconds = elapsed.as_secs_f64();
    let episodes_per_second = f64::from(episodes) / seconds;
    let milliseconds_per_episode = seconds * 1_000.0 / f64::from(episodes);
    println!(
        "Rust typed core (Sim::new + 719 typed pass/pass steps, release): \
         {episodes} episodes in {seconds:.3}s = {episodes_per_second:.1} episodes/sec, \
         {milliseconds_per_episode:.3} ms/episode"
    );
    println!(
        "Scope: typed rules core only; this is not apples-to-apples with the Python full-framework benchmark."
    );

    ExitCode::SUCCESS
}
