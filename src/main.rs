pub mod parameter;
pub mod renderer;
pub mod scene;

use crate::renderer::{OrthographicRenderer, Renderer};
use crate::scene::Scene;
use image::RgbImage;

fn main() {
    let mut scene = Scene::new_random(100, 5.0).expect("scale must be non-negative");
    scene
        .compute_neighbors()
        .expect("failed to compute centroid neighbors");

    let mut renderer = OrthographicRenderer::new(512, 512);
    renderer.world_x_min = -5.0;
    renderer.world_x_max = 5.0;
    renderer.world_y_min = -5.0;
    renderer.world_y_max = 5.0;
    renderer.ray_start_z = -10.0;
    renderer.ray_direction = [0.0, 0.0, 1.0];

    let image_data = renderer.render(&scene).expect("rendering failed");

    let image = RgbImage::from_raw(image_data.width, image_data.height, image_data.pixels)
        .expect("pixel buffer size must match image dimensions");
    image.save("output.png").expect("failed to write output.png");
}
