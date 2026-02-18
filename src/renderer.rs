use crate::scene::{Scene, SceneError};
use rayon::prelude::*;

#[derive(Debug)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Debug)]
pub enum RendererError {
    Scene(SceneError),
}

impl From<SceneError> for RendererError {
    fn from(value: SceneError) -> Self {
        Self::Scene(value)
    }
}

pub trait Renderer {
    fn render(&self, scene: &Scene) -> Result<ImageData, RendererError>;
}

#[derive(Debug)]
pub struct OrthographicRenderer {
    pub width: u32,
    pub height: u32,
    pub world_x_min: f64,
    pub world_x_max: f64,
    pub world_y_min: f64,
    pub world_y_max: f64,
    pub ray_start_z: f64,
    pub ray_direction: [f64; 3],
}

impl OrthographicRenderer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            world_x_min: -1.0,
            world_x_max: 1.0,
            world_y_min: -1.0,
            world_y_max: 1.0,
            ray_start_z: -10.0,
            ray_direction: [0.0, 0.0, 1.0],
        }
    }

    fn pixel_to_world(&self, px: u32, py: u32) -> [f64; 3] {
        let nx = if self.width > 1 {
            px as f64 / (self.width - 1) as f64
        } else {
            0.5
        };
        let ny = if self.height > 1 {
            py as f64 / (self.height - 1) as f64
        } else {
            0.5
        };

        let x = self.world_x_min + nx * (self.world_x_max - self.world_x_min);
        // Flip Y so row 0 is top of image.
        let y = self.world_y_max - ny * (self.world_y_max - self.world_y_min);
        [x, y, self.ray_start_z]
    }
}

impl Renderer for OrthographicRenderer {
    fn render(&self, scene: &Scene) -> Result<ImageData, RendererError> {
        let mut pixels = vec![0_u8; (self.width as usize) * (self.height as usize) * 3];
        let row_stride = (self.width as usize) * 3;

        pixels.par_chunks_mut(row_stride).enumerate().try_for_each(
            |(py, row)| -> Result<(), RendererError> {
                let py = py as u32;
                for px in 0..self.width {
                    let start = self.pixel_to_world(px, py);
                    let color = scene.render(start, self.ray_direction)?;

                    let idx = (px as usize) * 3;
                    row[idx] = (color[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                    row[idx + 1] = (color[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                    row[idx + 2] = (color[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
                Ok(())
            },
        )?;

        Ok(ImageData {
            width: self.width,
            height: self.height,
            pixels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{OrthographicRenderer, Renderer};
    use crate::parameter::Parameter;
    use crate::scene::Scene;

    #[test]
    fn orthographic_renderer_outputs_expected_shape() {
        let scene = Scene {
            centroid_x: Parameter::new(vec![0.0], 1e-3, 0.9, 0.999),
            centroid_y: Parameter::new(vec![0.0], 1e-3, 0.9, 0.999),
            centroid_z: Parameter::new(vec![0.0], 1e-3, 0.9, 0.999),
            centroid_opacity: Parameter::new(vec![1.0], 1e-3, 0.9, 0.999),
            centroid_r: Parameter::new(vec![0.2], 1e-3, 0.9, 0.999),
            centroid_g: Parameter::new(vec![0.4], 1e-3, 0.9, 0.999),
            centroid_b: Parameter::new(vec![0.8], 1e-3, 0.9, 0.999),
            centroid_neighbors: vec![vec![]],
        };

        let renderer = OrthographicRenderer::new(16, 8);
        let image = renderer.render(&scene).expect("render should succeed");

        assert_eq!(image.width, 16);
        assert_eq!(image.height, 8);
        assert_eq!(image.pixels.len(), 16 * 8 * 3);
    }
}
