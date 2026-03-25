use crate::parameter::Parameter;
use crate::renderer::{OrthographicRenderer, Renderer, TrainingTopologyCache};
use crate::scene::Scene;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::time::{Duration, Instant};

const DEFAULT_PROFILE_CENTROIDS: usize = 4_000;
const DEFAULT_PROFILE_STEPS: usize = 8;
const DEFAULT_PROFILE_WIDTH: u32 = 192;
const DEFAULT_PROFILE_HEIGHT: u32 = 192;
const PROFILE_SEED: u64 = 0x5eed_1234_abcd_9876;

#[derive(Clone, Copy, Debug)]
struct ProfileConfig {
    centroids: usize,
    steps: usize,
    width: u32,
    height: u32,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            centroids: DEFAULT_PROFILE_CENTROIDS,
            steps: DEFAULT_PROFILE_STEPS,
            width: DEFAULT_PROFILE_WIDTH,
            height: DEFAULT_PROFILE_HEIGHT,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ProfileReport {
    target_neighbor_build: Duration,
    target_render: Duration,
    train_steps: Duration,
    final_neighbor_refresh: Duration,
}

pub fn run_profile(args: &[String]) -> Result<(), String> {
    let config = parse_profile_config(args)?;
    println!(
        "profiling centroids={} steps={} size={}x{}",
        config.centroids, config.steps, config.width, config.height
    );

    let scene = profile_scene(config.centroids, 5.0);
    let renderer = profile_renderer(config.width, config.height);
    let mut report = ProfileReport::default();

    let mut target_scene = scene.clone();
    let (neighbor_build, ()) = timed(|| target_scene.compute_neighbors())
        .map_err(|error| format!("target neighbor build failed: {error:?}"))?;
    report.target_neighbor_build = neighbor_build;
    println!(
        "profile phase=target_neighbor_build elapsed_ms={:.3}",
        elapsed_ms(report.target_neighbor_build)
    );

    let (target_render, target) = timed(|| renderer.render(&target_scene))
        .map_err(|error| format!("target render failed: {error:?}"))?;
    report.target_render = target_render;
    println!(
        "profile phase=target_render elapsed_ms={:.3}",
        elapsed_ms(report.target_render)
    );

    let mut scene = scene;
    let mut topology_cache = TrainingTopologyCache::default();
    for step in 0..config.steps {
        let (elapsed, result) = timed(|| {
            renderer.train_step_with_cache_without_neighbor_refresh(
                &mut scene,
                &target,
                &mut topology_cache,
            )
        })
        .map_err(|error| format!("training step failed: {error:?}"))?;
        report.train_steps += elapsed;
        println!(
            "profile phase=train_step step={} elapsed_ms={:.3} loss={:.6}",
            step + 1,
            elapsed_ms(elapsed),
            result.loss,
        );
    }

    let (final_neighbor_refresh, ()) = timed(|| scene.compute_neighbors())
        .map_err(|error| format!("final neighbor refresh failed: {error:?}"))?;
    report.final_neighbor_refresh = final_neighbor_refresh;
    println!(
        "profile phase=final_neighbor_refresh elapsed_ms={:.3}",
        elapsed_ms(report.final_neighbor_refresh)
    );
    println!(
        "profile summary total_ms={:.3} deferred_training_pipeline_ms={:.3} eager_training_pipeline_ms={:.3} target_neighbor_build_ms={:.3} target_render_ms={:.3} train_steps_ms={:.3} avg_train_step_ms={:.3} final_neighbor_refresh_ms={:.3}",
        elapsed_ms(
            report.target_neighbor_build
                + report.target_render
                + report.train_steps
                + report.final_neighbor_refresh
        ),
        elapsed_ms(report.train_steps + report.final_neighbor_refresh),
        elapsed_ms(
            report.target_neighbor_build + report.train_steps + report.final_neighbor_refresh
        ),
        elapsed_ms(report.target_neighbor_build),
        elapsed_ms(report.target_render),
        elapsed_ms(report.train_steps),
        elapsed_ms(report.train_steps) / config.steps as f64,
        elapsed_ms(report.final_neighbor_refresh),
    );

    Ok(())
}

fn profile_scene(count: usize, scale: f64) -> Scene {
    let mut rng = StdRng::seed_from_u64(PROFILE_SEED);
    let mut x = Vec::with_capacity(count);
    let mut y = Vec::with_capacity(count);
    let mut z = Vec::with_capacity(count);
    let mut opacity = Vec::with_capacity(count);
    let mut r = Vec::with_capacity(count);
    let mut g = Vec::with_capacity(count);
    let mut b = Vec::with_capacity(count);

    for _ in 0..count {
        x.push(rng.random_range(-scale..=scale));
        y.push(rng.random_range(-scale..=scale));
        z.push(rng.random_range(-scale..=scale));
        opacity.push(-3.0);
        r.push(rng.random_range(0.0..=1.0));
        g.push(rng.random_range(0.0..=1.0));
        b.push(rng.random_range(0.0..=1.0));
    }

    Scene {
        centroid_x: Parameter::new(x, 1e-3, 0.9, 0.999),
        centroid_y: Parameter::new(y, 1e-3, 0.9, 0.999),
        centroid_z: Parameter::new(z, 1e-3, 0.9, 0.999),
        centroid_opacity: Parameter::new(opacity, 1e-3, 0.9, 0.999),
        centroid_r: Parameter::new(r, 1e-3, 0.9, 0.999),
        centroid_g: Parameter::new(g, 1e-3, 0.9, 0.999),
        centroid_b: Parameter::new(b, 1e-3, 0.9, 0.999),
        centroid_neighbors: vec![Vec::new(); count],
    }
}

fn parse_profile_config(args: &[String]) -> Result<ProfileConfig, String> {
    let mut config = ProfileConfig::default();
    let mut values = args.iter().skip(2);

    if let Some(value) = values.next() {
        config.centroids = parse_usize_arg("centroids", value)?;
    }
    if let Some(value) = values.next() {
        config.steps = parse_usize_arg("steps", value)?;
    }
    if let Some(value) = values.next() {
        config.width = parse_u32_arg("width", value)?;
    }
    if let Some(value) = values.next() {
        config.height = parse_u32_arg("height", value)?;
    }
    if values.next().is_some() {
        return Err("usage: radiant-foam profile [centroids] [steps] [width] [height]".to_string());
    }

    if config.steps == 0 {
        return Err("profile steps must be greater than zero".to_string());
    }

    Ok(config)
}

fn parse_usize_arg(name: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid {name}: {value}"))
}

fn parse_u32_arg(name: &str, value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("invalid {name}: {value}"))
}

fn profile_renderer(width: u32, height: u32) -> OrthographicRenderer {
    let mut renderer = OrthographicRenderer::new(width, height);
    renderer.world_x_min = -2.0;
    renderer.world_x_max = 2.0;
    renderer.world_y_min = -2.0;
    renderer.world_y_max = 2.0;
    renderer.ray_start_z = 0.0;
    renderer.ray_direction = [0.0, 0.0, 1.0];
    renderer
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn timed<T, E>(f: impl FnOnce() -> Result<T, E>) -> Result<(Duration, T), E> {
    let start = Instant::now();
    let value = f()?;
    Ok((start.elapsed(), value))
}
