pub mod parameter;
pub mod renderer;
pub mod scene;
pub mod video;
pub mod viewer;

use crate::renderer::{OrthographicRenderer, Renderer};
use crate::scene::Scene;
use crate::video::{ColmapVideoInitializer, VideoInitOptions};
use crate::viewer::run_scene_viewer;
use image::RgbImage;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() == 2 {
        let input_path = Path::new(&args[1]);
        if input_path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            if let Err(error) = run_scene_viewer(input_path) {
                eprintln!("{error}");
                std::process::exit(1);
            }
        } else {
            if let Err(error) = run_video_training(input_path) {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.get(1).map(String::as_str) == Some("init-video") {
        run_video_initialization(&args);
        return;
    }
    if args.get(1).map(String::as_str) == Some("train-video") {
        run_explicit_video_training(&args);
        return;
    }
    if args.get(1).map(String::as_str) == Some("view-scene") {
        if args.len() != 3 {
            eprintln!("usage: radiant-foam view-scene <scene-json>");
            std::process::exit(1);
        }
        if let Err(error) = run_scene_viewer(Path::new(&args[2])) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }

    let mut scene = Scene::new_random(1000, 5.0).expect("scale must be non-negative");
    scene
        .compute_neighbors()
        .expect("failed to compute centroid neighbors");

    let mut renderer = OrthographicRenderer::new(512, 512);
    renderer.world_x_min = -2.0;
    renderer.world_x_max = 2.0;
    renderer.world_y_min = -2.0;
    renderer.world_y_max = 2.0;
    renderer.ray_start_z = 0.0;
    renderer.ray_direction = [0.0, 0.0, 1.0];

    let image_data = renderer.render(&scene).expect("rendering failed");

    let image = RgbImage::from_raw(image_data.width, image_data.height, image_data.pixels)
        .expect("pixel buffer size must match image dimensions");
    image
        .save("output.png")
        .expect("failed to write output.png");
}

fn run_video_initialization(args: &[String]) {
    if args.len() != 4 && args.len() != 5 {
        eprintln!("usage: radiant-foam init-video <video-path> <scene-json> [workspace-dir]");
        std::process::exit(1);
    }

    let video_path = Path::new(&args[2]);
    let scene_path = Path::new(&args[3]);
    let workspace = args
        .get(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| scene_path.with_extension("colmap"));

    if let Err(error) = run_video_initialization_impl(video_path, scene_path, &workspace) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run_explicit_video_training(args: &[String]) {
    if args.len() < 4 || args.len() > 6 {
        eprintln!("usage: radiant-foam train-video <video-path> <scene-json> [workspace-dir] [fps]");
        std::process::exit(1);
    }

    let video_path = Path::new(&args[2]);
    let scene_path = Path::new(&args[3]);
    let workspace = args
        .get(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| scene_path.with_extension("colmap"));
    let fps = args
        .get(5)
        .or_else(|| {
            args.get(4)
                .filter(|value| value.parse::<f64>().is_ok())
        })
        .and_then(|value| value.parse::<f64>().ok());

    let options = fps.map(|fps| VideoInitOptions {
        fps,
        ..VideoInitOptions::default()
    });

    if let Err(error) = run_video_training_impl(video_path, scene_path, &workspace, options) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run_video_training(video_path: &Path) -> Result<(), String> {
    let scene_path = video_path.with_extension("scene.json");
    let workspace = video_path.with_extension("radiant-foam");

    run_video_training_impl(video_path, &scene_path, &workspace, None)
}

fn run_video_initialization_impl(
    video_path: &Path,
    scene_path: &Path,
    workspace: &Path,
) -> Result<(), String> {
    let mut initializer = ColmapVideoInitializer::default();
    let scene = initializer
        .initialize_scene_from_video(video_path, workspace)
        .map_err(|error| format!("video initialization failed: {error:?}"))?;
    scene
        .save_to_json(scene_path)
        .map_err(|error| format!("failed to save scene {}: {error:?}", scene_path.display()))?;
    Ok(())
}

fn run_video_training_impl(
    video_path: &Path,
    scene_path: &Path,
    workspace: &Path,
    options: Option<VideoInitOptions>,
) -> Result<(), String> {
    let mut initializer = ColmapVideoInitializer::new(
        crate::video::SystemCommandRunner,
        options.unwrap_or_default(),
    );
    let scene = initializer
        .initialize_and_train_from_video(video_path, workspace)
        .map_err(|error| format!("video training failed: {error:?}"))?;
    scene
        .save_to_json(scene_path)
        .map_err(|error| format!("failed to save scene {}: {error:?}", scene_path.display()))?;
    Ok(())
}
