use crate::renderer::{PerspectiveRenderer, TrainingTopologyCache};
use crate::scene::Scene;

use super::colmap::TrainingFrame;
use super::{VideoInitError, VideoInitOptions};

pub(super) fn train_scene_on_frames(
    options: &VideoInitOptions,
    scene: &mut Scene,
    frames: &[TrainingFrame],
) -> Result<(), VideoInitError> {
    println!(
        "training on {} registered frames for {} epochs",
        frames.len(),
        options.train_epochs,
    );
    let mut topology_cache = TrainingTopologyCache::default();
    for epoch_index in 0..options.train_epochs {
        let mut epoch_total_loss = 0.0_f64;
        let mut epoch_rgb_loss = 0.0_f64;
        let mut epoch_distortion_loss = 0.0_f64;

        for (frame_index, frame) in frames.iter().enumerate() {
            let renderer = PerspectiveRenderer::with_distortion(
                frame.camera.clone(),
                options.distortion_lambda,
            );
            let result = renderer.train_step_with_cache_without_neighbor_refresh(
                scene,
                &frame.target,
                &mut topology_cache,
            )?;
            epoch_total_loss += result.loss;
            epoch_rgb_loss += result.rgb_loss;
            epoch_distortion_loss += result.distortion_loss;

            println!(
                "epoch {}/{} step {}/{} frame={} loss={:.6} rgb_loss={:.6} distortion_loss={:.6}",
                epoch_index + 1,
                options.train_epochs,
                frame_index + 1,
                frames.len(),
                frame.name,
                result.loss,
                result.rgb_loss,
                result.distortion_loss,
            );
        }

        let frame_count = frames.len() as f64;
        println!(
            "epoch {}/{} avg_loss={:.6} avg_rgb_loss={:.6} avg_distortion_loss={:.6} centroids={}",
            epoch_index + 1,
            options.train_epochs,
            epoch_total_loss / frame_count,
            epoch_rgb_loss / frame_count,
            epoch_distortion_loss / frame_count,
            scene.centroid_x.len(),
        );
    }
    scene.compute_neighbors()?;
    Ok(())
}
