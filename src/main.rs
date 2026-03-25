pub mod parameter;
pub mod renderer;
pub mod scene;
pub mod video;

use crate::renderer::{OrthographicRenderer, Renderer};
use crate::scene::Scene;
use crate::video::ColmapVideoInitializer;
use image::RgbImage;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("init-video") {
        run_video_initialization(&args);
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

    let mut initializer = ColmapVideoInitializer::default();
    let scene = initializer
        .initialize_scene_from_video(video_path, &workspace)
        .expect("video initialization should succeed");
    scene
        .save_to_json(scene_path)
        .expect("scene should serialize");
}
