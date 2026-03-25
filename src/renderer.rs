use crate::parameter::ParameterError;
use crate::scene::{Scene, SceneError};
use rayon::prelude::*;

const RAY_ADVANCE_EPSILON: f64 = 1e-9;
const TRANSMITTANCE_EPSILON: f64 = 1e-6;

#[derive(Clone, Debug)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Debug)]
pub enum RendererError {
    Scene(SceneError),
    Parameter(ParameterError),
    TargetSizeMismatch {
        expected_width: u32,
        expected_height: u32,
        got_width: u32,
        got_height: u32,
    },
}

impl From<SceneError> for RendererError {
    fn from(value: SceneError) -> Self {
        Self::Scene(value)
    }
}

impl From<ParameterError> for RendererError {
    fn from(value: ParameterError) -> Self {
        Self::Parameter(value)
    }
}

#[derive(Clone, Debug)]
pub struct SceneGradients {
    pub centroid_x: Vec<f64>,
    pub centroid_y: Vec<f64>,
    pub centroid_z: Vec<f64>,
    pub centroid_opacity: Vec<f64>,
    pub centroid_r: Vec<f64>,
    pub centroid_g: Vec<f64>,
    pub centroid_b: Vec<f64>,
}

impl SceneGradients {
    fn zeros(count: usize) -> Self {
        Self {
            centroid_x: vec![0.0; count],
            centroid_y: vec![0.0; count],
            centroid_z: vec![0.0; count],
            centroid_opacity: vec![0.0; count],
            centroid_r: vec![0.0; count],
            centroid_g: vec![0.0; count],
            centroid_b: vec![0.0; count],
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainStepResult {
    pub loss: f64,
    pub gradients: SceneGradients,
}

#[derive(Clone, Debug)]
struct RaySegment {
    centroid_index: usize,
    next_index: Option<usize>,
    remaining_before: f64,
    alpha: f64,
    segment_length: Option<f64>,
    sigma: f64,
}

pub trait Renderer {
    fn render(&self, scene: &Scene) -> Result<ImageData, RendererError>;
    fn train_step(
        &self,
        scene: &mut Scene,
        target: &ImageData,
    ) -> Result<TrainStepResult, RendererError>;
}

#[derive(Clone, Debug)]
pub struct PerspectiveCamera {
    pub width: u32,
    pub height: u32,
    pub focal_x: f64,
    pub focal_y: f64,
    pub principal_x: f64,
    pub principal_y: f64,
    pub origin: [f64; 3],
    pub camera_to_world: [[f64; 3]; 3],
}

impl PerspectiveCamera {
    pub fn pixel_to_world_ray(&self, px: u32, py: u32) -> ([f64; 3], [f64; 3]) {
        let x = (px as f64 + 0.5 - self.principal_x) / self.focal_x;
        let y = (py as f64 + 0.5 - self.principal_y) / self.focal_y;
        let direction_camera = normalize([x, y, 1.0]);
        let direction_world = normalize(mat3_mul_vec3(self.camera_to_world, direction_camera));
        (self.origin, direction_world)
    }
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

    fn validate_target(&self, target: &ImageData) -> Result<(), RendererError> {
        validate_target_dimensions(self.width, self.height, target)
    }

    fn render_linear(&self, scene: &Scene) -> Result<Vec<f64>, RendererError> {
        let mut pixels = vec![0.0_f64; (self.width as usize) * (self.height as usize) * 3];
        let row_stride = (self.width as usize) * 3;

        pixels.par_chunks_mut(row_stride).enumerate().try_for_each(
            |(py, row)| -> Result<(), RendererError> {
                let py = py as u32;
                for px in 0..self.width {
                    let start = self.pixel_to_world(px, py);
                    let color = scene.render(start, self.ray_direction)?;

                    let idx = (px as usize) * 3;
                    row[idx] = color[0];
                    row[idx + 1] = color[1];
                    row[idx + 2] = color[2];
                }
                Ok(())
            },
        )?;

        Ok(pixels)
    }

    fn mse_loss(&self, scene: &Scene, target: &ImageData) -> Result<f64, RendererError> {
        self.validate_target(target)?;
        let rendered = self.render_linear(scene)?;
        let target_scale = 1.0 / 255.0;
        let mse = rendered
            .iter()
            .zip(&target.pixels)
            .map(|(&predicted, &target)| {
                let diff = predicted - (target as f64 * target_scale);
                diff * diff
            })
            .sum::<f64>()
            / rendered.len() as f64;
        Ok(mse)
    }

    pub fn compute_gradients(
        &self,
        scene: &Scene,
        target: &ImageData,
    ) -> Result<SceneGradients, RendererError> {
        self.validate_target(target)?;
        let count = validate_scene(scene)?;
        let direction = normalize(self.ray_direction);
        if direction == [0.0, 0.0, 0.0] {
            return Err(SceneError::InvalidRayDirection.into());
        }

        let mut gradients = SceneGradients::zeros(count);
        let normalizer = 2.0 / ((self.width as usize * self.height as usize * 3) as f64);
        let target_scale = 1.0 / 255.0;

        for py in 0..self.height {
            for px in 0..self.width {
                let start = self.pixel_to_world(px, py);
                let (pixel_color, segments) = trace_ray(scene, start, direction)?;
                let pixel_index = ((py * self.width + px) as usize) * 3;
                let output_grad = [
                    normalizer * (pixel_color[0] - target.pixels[pixel_index] as f64 * target_scale),
                    normalizer
                        * (pixel_color[1] - target.pixels[pixel_index + 1] as f64 * target_scale),
                    normalizer
                        * (pixel_color[2] - target.pixels[pixel_index + 2] as f64 * target_scale),
                ];

                accumulate_trace_gradients(
                    scene,
                    &segments,
                    start,
                    direction,
                    output_grad,
                    &mut gradients,
                );
            }
        }

        Ok(gradients)
    }
}

#[derive(Clone, Debug)]
pub struct PerspectiveRenderer {
    pub camera: PerspectiveCamera,
}

impl PerspectiveRenderer {
    pub fn new(camera: PerspectiveCamera) -> Self {
        Self { camera }
    }

    fn validate_target(&self, target: &ImageData) -> Result<(), RendererError> {
        validate_target_dimensions(self.camera.width, self.camera.height, target)
    }

    fn render_linear(&self, scene: &Scene) -> Result<Vec<f64>, RendererError> {
        let mut pixels =
            vec![0.0_f64; (self.camera.width as usize) * (self.camera.height as usize) * 3];
        let row_stride = (self.camera.width as usize) * 3;

        pixels.par_chunks_mut(row_stride).enumerate().try_for_each(
            |(py, row)| -> Result<(), RendererError> {
                let py = py as u32;
                for px in 0..self.camera.width {
                    let (start, direction) = self.camera.pixel_to_world_ray(px, py);
                    let color = trace_ray(scene, start, direction)?.0;

                    let idx = (px as usize) * 3;
                    row[idx] = color[0];
                    row[idx + 1] = color[1];
                    row[idx + 2] = color[2];
                }
                Ok(())
            },
        )?;

        Ok(pixels)
    }

    fn mse_loss(&self, scene: &Scene, target: &ImageData) -> Result<f64, RendererError> {
        self.validate_target(target)?;
        compute_mse_loss(&self.render_linear(scene)?, target)
    }

    pub fn compute_gradients(
        &self,
        scene: &Scene,
        target: &ImageData,
    ) -> Result<SceneGradients, RendererError> {
        self.validate_target(target)?;
        let count = validate_scene(scene)?;

        let mut gradients = SceneGradients::zeros(count);
        let normalizer =
            2.0 / ((self.camera.width as usize * self.camera.height as usize * 3) as f64);
        let target_scale = 1.0 / 255.0;

        for py in 0..self.camera.height {
            for px in 0..self.camera.width {
                let (start, direction) = self.camera.pixel_to_world_ray(px, py);
                let (pixel_color, segments) = trace_ray(scene, start, direction)?;
                let pixel_index = ((py * self.camera.width + px) as usize) * 3;
                let output_grad = [
                    normalizer * (pixel_color[0] - target.pixels[pixel_index] as f64 * target_scale),
                    normalizer
                        * (pixel_color[1] - target.pixels[pixel_index + 1] as f64 * target_scale),
                    normalizer
                        * (pixel_color[2] - target.pixels[pixel_index + 2] as f64 * target_scale),
                ];

                accumulate_trace_gradients(
                    scene,
                    &segments,
                    start,
                    direction,
                    output_grad,
                    &mut gradients,
                );
            }
        }

        Ok(gradients)
    }
}

impl Renderer for OrthographicRenderer {
    fn render(&self, scene: &Scene) -> Result<ImageData, RendererError> {
        let rendered = self.render_linear(scene)?;
        let pixels = rendered
            .chunks_exact(3)
            .flat_map(|rgb| {
                rgb.iter()
                    .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
            })
            .collect();

        Ok(ImageData {
            width: self.width,
            height: self.height,
            pixels,
        })
    }

    fn train_step(
        &self,
        scene: &mut Scene,
        target: &ImageData,
    ) -> Result<TrainStepResult, RendererError> {
        let loss = self.mse_loss(scene, target)?;
        let gradients = self.compute_gradients(scene, target)?;

        scene.centroid_x.update_adam(&gradients.centroid_x)?;
        scene.centroid_y.update_adam(&gradients.centroid_y)?;
        scene.centroid_z.update_adam(&gradients.centroid_z)?;
        scene
            .centroid_opacity
            .update_adam(&gradients.centroid_opacity)?;
        scene.centroid_r.update_adam(&gradients.centroid_r)?;
        scene.centroid_g.update_adam(&gradients.centroid_g)?;
        scene.centroid_b.update_adam(&gradients.centroid_b)?;
        scene.compute_neighbors()?;

        Ok(TrainStepResult { loss, gradients })
    }
}

impl Renderer for PerspectiveRenderer {
    fn render(&self, scene: &Scene) -> Result<ImageData, RendererError> {
        let rendered = self.render_linear(scene)?;
        let pixels = rendered
            .chunks_exact(3)
            .flat_map(|rgb| {
                rgb.iter()
                    .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
            })
            .collect();

        Ok(ImageData {
            width: self.camera.width,
            height: self.camera.height,
            pixels,
        })
    }

    fn train_step(
        &self,
        scene: &mut Scene,
        target: &ImageData,
    ) -> Result<TrainStepResult, RendererError> {
        let loss = self.mse_loss(scene, target)?;
        let gradients = self.compute_gradients(scene, target)?;

        scene.centroid_x.update_adam(&gradients.centroid_x)?;
        scene.centroid_y.update_adam(&gradients.centroid_y)?;
        scene.centroid_z.update_adam(&gradients.centroid_z)?;
        scene
            .centroid_opacity
            .update_adam(&gradients.centroid_opacity)?;
        scene.centroid_r.update_adam(&gradients.centroid_r)?;
        scene.centroid_g.update_adam(&gradients.centroid_g)?;
        scene.centroid_b.update_adam(&gradients.centroid_b)?;
        scene.compute_neighbors()?;

        Ok(TrainStepResult { loss, gradients })
    }
}

fn validate_target_dimensions(
    expected_width: u32,
    expected_height: u32,
    target: &ImageData,
) -> Result<(), RendererError> {
    if target.width == expected_width && target.height == expected_height {
        Ok(())
    } else {
        Err(RendererError::TargetSizeMismatch {
            expected_width,
            expected_height,
            got_width: target.width,
            got_height: target.height,
        })
    }
}

fn compute_mse_loss(rendered: &[f64], target: &ImageData) -> Result<f64, RendererError> {
    let target_scale = 1.0 / 255.0;
    let mse = rendered
        .iter()
        .zip(&target.pixels)
        .map(|(&predicted, &target)| {
            let diff = predicted - (target as f64 * target_scale);
            diff * diff
        })
        .sum::<f64>()
        / rendered.len() as f64;
    Ok(mse)
}

fn validate_scene(scene: &Scene) -> Result<usize, SceneError> {
    let x = scene.centroid_x.len();
    let y = scene.centroid_y.len();
    let z = scene.centroid_z.len();
    let opacity = scene.centroid_opacity.len();
    let r = scene.centroid_r.len();
    let g = scene.centroid_g.len();
    let b = scene.centroid_b.len();

    if !(x == y && x == z && x == opacity && x == r && x == g && x == b) {
        return Err(SceneError::InconsistentCentroidData {
            x,
            y,
            z,
            opacity,
            r,
            g,
            b,
        });
    }
    if x == 0 {
        return Err(SceneError::EmptyScene);
    }
    if scene.centroid_neighbors.len() != x {
        return Err(SceneError::InconsistentNeighborData {
            expected: x,
            got: scene.centroid_neighbors.len(),
        });
    }

    Ok(x)
}

fn trace_ray(
    scene: &Scene,
    ray_origin: [f64; 3],
    direction: [f64; 3],
) -> Result<([f64; 3], Vec<RaySegment>), RendererError> {
    validate_scene(scene)?;

    let mut color = [0.0, 0.0, 0.0];
    let mut remaining = 1.0_f64;
    let mut current = closest_centroid_at_point(scene, ray_origin);
    let mut t0 = 0.0_f64;
    let mut segments = Vec::new();

    while remaining > TRANSMITTANCE_EPSILON {
        let next = next_centroid_along_ray(scene, current, ray_origin, direction, t0);
        let centroid_color = centroid_color(scene, current);
        let log_density = scene.centroid_opacity.values[current];
        let sigma = log_density.exp();
        let segment_length = next.map(|(_, t1)| (t1 - t0).max(0.0));
        let alpha = if let Some(length) = segment_length {
            1.0 - (-(sigma * length)).exp()
        } else {
            1.0
        };
        let weight = remaining * alpha;
        color[0] += centroid_color[0] * weight;
        color[1] += centroid_color[1] * weight;
        color[2] += centroid_color[2] * weight;

        segments.push(RaySegment {
            centroid_index: current,
            next_index: next.map(|(next_index, _)| next_index),
            remaining_before: remaining,
            alpha,
            segment_length,
            sigma,
        });

        remaining *= 1.0 - alpha;

        let Some((next_index, t_cross)) = next else {
            break;
        };

        current = next_index;
        t0 = t_cross;
    }

    Ok((color, segments))
}

fn accumulate_trace_gradients(
    scene: &Scene,
    segments: &[RaySegment],
    ray_origin: [f64; 3],
    direction: [f64; 3],
    output_grad: [f64; 3],
    gradients: &mut SceneGradients,
) {
    let mut grad_remaining = 0.0_f64;
    let mut boundary_grads = vec![0.0_f64; segments.len()];

    for segment_index in (0..segments.len()).rev() {
        let segment = &segments[segment_index];
        let centroid_index = segment.centroid_index;
        let centroid_color = centroid_color(scene, centroid_index);
        let weight = segment.remaining_before * segment.alpha;
        let grad_weight = dot(output_grad, centroid_color);
        let grad_alpha =
            grad_weight * segment.remaining_before - grad_remaining * segment.remaining_before;

        gradients.centroid_r[centroid_index] +=
            output_grad[0] * weight * clamp_derivative(scene.centroid_r.values[centroid_index]);
        gradients.centroid_g[centroid_index] +=
            output_grad[1] * weight * clamp_derivative(scene.centroid_g.values[centroid_index]);
        gradients.centroid_b[centroid_index] +=
            output_grad[2] * weight * clamp_derivative(scene.centroid_b.values[centroid_index]);

        if let Some(length) = segment.segment_length {
            let transmission = 1.0 - segment.alpha;
            gradients.centroid_opacity[centroid_index] +=
                grad_alpha * segment.sigma * length * transmission;

            let grad_length = grad_alpha * segment.sigma * transmission;
            boundary_grads[segment_index] += grad_length;
            if segment_index > 0 {
                boundary_grads[segment_index - 1] -= grad_length;
            }
        }

        grad_remaining = grad_weight * segment.alpha + grad_remaining * (1.0 - segment.alpha);
    }

    for (boundary_index, &grad_t) in boundary_grads.iter().enumerate() {
        if grad_t == 0.0 {
            continue;
        }
        let segment = &segments[boundary_index];
        let Some(next_index) = segment.next_index else {
            continue;
        };
        accumulate_boundary_time_gradient(
            scene,
            segment.centroid_index,
            next_index,
            ray_origin,
            direction,
            grad_t,
            gradients,
        );
    }
}

fn accumulate_boundary_time_gradient(
    scene: &Scene,
    current_index: usize,
    next_index: usize,
    ray_origin: [f64; 3],
    direction: [f64; 3],
    grad_t: f64,
    gradients: &mut SceneGradients,
) {
    let pi = point(scene, current_index);
    let pj = point(scene, next_index);
    let delta = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
    let numerator = dot(pj, pj) - dot(pi, pi) - 2.0 * dot(delta, ray_origin);
    let denominator = 2.0 * dot(delta, direction);
    let denominator_sq = denominator * denominator;

    let numerator_grad_i = [
        2.0 * (ray_origin[0] - pi[0]),
        2.0 * (ray_origin[1] - pi[1]),
        2.0 * (ray_origin[2] - pi[2]),
    ];
    let numerator_grad_j = [
        2.0 * (pj[0] - ray_origin[0]),
        2.0 * (pj[1] - ray_origin[1]),
        2.0 * (pj[2] - ray_origin[2]),
    ];
    let denominator_grad_i = [
        -2.0 * direction[0],
        -2.0 * direction[1],
        -2.0 * direction[2],
    ];
    let denominator_grad_j = [
        2.0 * direction[0],
        2.0 * direction[1],
        2.0 * direction[2],
    ];

    let time_grad_i = quotient_gradient(
        numerator,
        denominator,
        denominator_sq,
        numerator_grad_i,
        denominator_grad_i,
    );
    let time_grad_j = quotient_gradient(
        numerator,
        denominator,
        denominator_sq,
        numerator_grad_j,
        denominator_grad_j,
    );

    gradients.centroid_x[current_index] += grad_t * time_grad_i[0];
    gradients.centroid_y[current_index] += grad_t * time_grad_i[1];
    gradients.centroid_z[current_index] += grad_t * time_grad_i[2];
    gradients.centroid_x[next_index] += grad_t * time_grad_j[0];
    gradients.centroid_y[next_index] += grad_t * time_grad_j[1];
    gradients.centroid_z[next_index] += grad_t * time_grad_j[2];
}

fn quotient_gradient(
    numerator: f64,
    denominator: f64,
    denominator_sq: f64,
    numerator_gradient: [f64; 3],
    denominator_gradient: [f64; 3],
) -> [f64; 3] {
    [
        (numerator_gradient[0] * denominator - numerator * denominator_gradient[0])
            / denominator_sq,
        (numerator_gradient[1] * denominator - numerator * denominator_gradient[1])
            / denominator_sq,
        (numerator_gradient[2] * denominator - numerator * denominator_gradient[2])
            / denominator_sq,
    ]
}

fn point(scene: &Scene, idx: usize) -> [f64; 3] {
    [
        scene.centroid_x.values[idx],
        scene.centroid_y.values[idx],
        scene.centroid_z.values[idx],
    ]
}

fn centroid_color(scene: &Scene, idx: usize) -> [f64; 3] {
    [
        scene.centroid_r.values[idx].clamp(0.0, 1.0),
        scene.centroid_g.values[idx].clamp(0.0, 1.0),
        scene.centroid_b.values[idx].clamp(0.0, 1.0),
    ]
}

fn closest_centroid_at_point(scene: &Scene, q: [f64; 3]) -> usize {
    let mut best_idx = 0usize;
    let mut best_dist = dist_sq(q, point(scene, 0));
    for index in 1..scene.centroid_x.len() {
        let distance = dist_sq(q, point(scene, index));
        if distance < best_dist {
            best_dist = distance;
            best_idx = index;
        }
    }
    best_idx
}

fn next_centroid_along_ray(
    scene: &Scene,
    current: usize,
    ray_origin: [f64; 3],
    direction: [f64; 3],
    t0: f64,
) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    let pi = point(scene, current);
    let oi = dot(pi, pi);

    for &neighbor in &scene.centroid_neighbors[current] {
        let pj = point(scene, neighbor);
        let rhs = dot(pj, pj) - oi;
        let delta = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
        let denominator = 2.0 * dot(delta, direction);
        if denominator.abs() < 1e-8 || denominator <= 0.0 {
            continue;
        }

        let numerator = rhs - 2.0 * dot(delta, ray_origin);
        let t = numerator / denominator;
        if t <= t0 + RAY_ADVANCE_EPSILON {
            continue;
        }

        match best {
            None => best = Some((neighbor, t)),
            Some((_, best_t)) if t < best_t => best = Some((neighbor, t)),
            _ => {}
        }
    }

    best
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mat3_mul_vec3(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn dist_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let norm = dot(v, v).sqrt();
    if norm == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

fn clamp_derivative(value: f64) -> f64 {
    if (0.0..1.0).contains(&value) { 1.0 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::{
        ImageData, OrthographicRenderer, PerspectiveCamera, PerspectiveRenderer, Renderer,
        RendererError,
    };
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

    #[test]
    fn perspective_renderer_outputs_expected_shape() {
        let scene = Scene {
            centroid_x: Parameter::new(vec![0.0], 1e-3, 0.9, 0.999),
            centroid_y: Parameter::new(vec![0.0], 1e-3, 0.9, 0.999),
            centroid_z: Parameter::new(vec![3.0], 1e-3, 0.9, 0.999),
            centroid_opacity: Parameter::new(vec![1.0], 1e-3, 0.9, 0.999),
            centroid_r: Parameter::new(vec![0.2], 1e-3, 0.9, 0.999),
            centroid_g: Parameter::new(vec![0.4], 1e-3, 0.9, 0.999),
            centroid_b: Parameter::new(vec![0.8], 1e-3, 0.9, 0.999),
            centroid_neighbors: vec![vec![]],
        };
        let renderer = PerspectiveRenderer::new(PerspectiveCamera {
            width: 4,
            height: 3,
            focal_x: 3.0,
            focal_y: 3.0,
            principal_x: 2.0,
            principal_y: 1.5,
            origin: [0.0, 0.0, 0.0],
            camera_to_world: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        });

        let image = renderer.render(&scene).expect("render should succeed");

        assert_eq!(image.width, 4);
        assert_eq!(image.height, 3);
        assert_eq!(image.pixels.len(), 4 * 3 * 3);
    }

    #[test]
    fn train_step_updates_all_parameter_groups() {
        let mut scene = gradient_test_scene();
        let target = target_image(&gradient_test_renderer());
        let renderer = gradient_test_renderer();

        let result = renderer
            .train_step(&mut scene, &target)
            .expect("train step should succeed");

        assert!(result.loss.is_finite());
        assert_eq!(scene.centroid_x.step, 1);
        assert_eq!(scene.centroid_y.step, 1);
        assert_eq!(scene.centroid_z.step, 1);
        assert_eq!(scene.centroid_opacity.step, 1);
        assert_eq!(scene.centroid_r.step, 1);
        assert_eq!(scene.centroid_g.step, 1);
        assert_eq!(scene.centroid_b.step, 1);
    }

    #[test]
    fn compute_gradients_matches_delta_rule_for_each_parameter_kind() {
        let scene = gradient_test_scene();
        let renderer = gradient_test_renderer();
        let target = target_image(&renderer);
        let gradients = renderer
            .compute_gradients(&scene, &target)
            .expect("gradient calculation should succeed");
        let delta = 5e-5;

        assert_gradient_close(
            gradients.centroid_x[0],
            approximate_partial_derivative(&scene, &renderer, &target, delta, |scene| {
                &mut scene.centroid_x.values[0]
            })
            .expect("x derivative should succeed"),
        );
        assert_gradient_close(
            gradients.centroid_y[0],
            approximate_partial_derivative(&scene, &renderer, &target, delta, |scene| {
                &mut scene.centroid_y.values[0]
            })
            .expect("y derivative should succeed"),
        );
        assert_gradient_close(
            gradients.centroid_z[0],
            approximate_partial_derivative(&scene, &renderer, &target, delta, |scene| {
                &mut scene.centroid_z.values[0]
            })
            .expect("z derivative should succeed"),
        );
        assert_gradient_close(
            gradients.centroid_opacity[0],
            approximate_partial_derivative(&scene, &renderer, &target, delta, |scene| {
                &mut scene.centroid_opacity.values[0]
            })
            .expect("opacity derivative should succeed"),
        );
        assert_gradient_close(
            gradients.centroid_r[0],
            approximate_partial_derivative(&scene, &renderer, &target, delta, |scene| {
                &mut scene.centroid_r.values[0]
            })
            .expect("r derivative should succeed"),
        );
        assert_gradient_close(
            gradients.centroid_g[0],
            approximate_partial_derivative(&scene, &renderer, &target, delta, |scene| {
                &mut scene.centroid_g.values[0]
            })
            .expect("g derivative should succeed"),
        );
        assert_gradient_close(
            gradients.centroid_b[0],
            approximate_partial_derivative(&scene, &renderer, &target, delta, |scene| {
                &mut scene.centroid_b.values[0]
            })
            .expect("b derivative should succeed"),
        );
    }

    fn gradient_test_scene() -> Scene {
        let mut scene = Scene {
            centroid_x: Parameter::new(vec![-0.35, 0.45], 1e-2, 0.9, 0.999),
            centroid_y: Parameter::new(vec![0.25, -0.15], 1e-2, 0.9, 0.999),
            centroid_z: Parameter::new(vec![-0.2, 0.3], 1e-2, 0.9, 0.999),
            centroid_opacity: Parameter::new(vec![-0.7, -0.2], 1e-2, 0.9, 0.999),
            centroid_r: Parameter::new(vec![0.25, 0.7], 1e-2, 0.9, 0.999),
            centroid_g: Parameter::new(vec![0.55, 0.2], 1e-2, 0.9, 0.999),
            centroid_b: Parameter::new(vec![0.35, 0.85], 1e-2, 0.9, 0.999),
            centroid_neighbors: vec![Vec::new(); 2],
        };
        scene.compute_neighbors().expect("neighbors should compute");
        scene
    }

    fn gradient_test_renderer() -> OrthographicRenderer {
        let mut renderer = OrthographicRenderer::new(3, 2);
        renderer.world_x_min = -0.8;
        renderer.world_x_max = 0.8;
        renderer.world_y_min = -0.6;
        renderer.world_y_max = 0.6;
        renderer.ray_start_z = -1.5;
        renderer.ray_direction = [0.25, -0.1, 1.0];
        renderer
    }

    fn target_image(renderer: &OrthographicRenderer) -> ImageData {
        let mut target_scene = gradient_test_scene();
        target_scene.centroid_x.values[0] += 0.06;
        target_scene.centroid_y.values[0] -= 0.04;
        target_scene.centroid_z.values[0] += 0.05;
        target_scene.centroid_opacity.values[0] += 0.1;
        target_scene.centroid_r.values[0] += 0.08;
        target_scene.centroid_g.values[0] -= 0.06;
        target_scene.centroid_b.values[0] += 0.07;
        target_scene
            .compute_neighbors()
            .expect("target neighbors should compute");
        renderer
            .render(&target_scene)
            .expect("target render should succeed")
    }

    fn approximate_partial_derivative(
        scene: &Scene,
        renderer: &OrthographicRenderer,
        target: &ImageData,
        delta: f64,
        accessor: impl Fn(&mut Scene) -> &mut f64,
    ) -> Result<f64, RendererError> {
        let baseline = renderer.mse_loss(scene, target)?;
        let mut shifted = scene.clone();
        *accessor(&mut shifted) += delta;
        shifted.compute_neighbors()?;
        let shifted_loss = renderer.mse_loss(&shifted, target)?;
        Ok((shifted_loss - baseline) / delta)
    }

    fn assert_gradient_close(actual: f64, expected: f64) {
        let scale = 1.0_f64.max(actual.abs()).max(expected.abs());
        let tolerance = 2e-2 * scale;
        assert!(
            (actual - expected).abs() <= tolerance,
            "gradient mismatch: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}
