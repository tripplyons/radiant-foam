use crate::parameter::{Parameter, ParameterError};
use qhull_enhanced::Qh;
use rand::RngExt;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

const SPLIT_FALLBACK_OFFSET_MAGNITUDE: f64 = 1e-3;
const SPLIT_SAMPLING_ATTEMPTS: usize = 64;
const RAY_ADVANCE_EPSILON: f64 = 1e-9;
const TRANSMITTANCE_EPSILON: f64 = 1e-6;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Scene {
    pub centroid_x: Parameter,
    pub centroid_y: Parameter,
    pub centroid_z: Parameter,
    pub centroid_opacity: Parameter,
    pub centroid_r: Parameter,
    pub centroid_g: Parameter,
    pub centroid_b: Parameter,
    pub centroid_neighbors: Vec<Vec<usize>>,
}

#[derive(Debug)]
pub enum SceneError {
    NegativeScale(f64),
    InconsistentCentroidData {
        x: usize,
        y: usize,
        z: usize,
        opacity: usize,
        r: usize,
        g: usize,
        b: usize,
    },
    Parameter(ParameterError),
    InvalidCentroidIndex {
        index: usize,
        len: usize,
    },
    InconsistentNeighborData {
        expected: usize,
        got: usize,
    },
    EmptyScene,
    InvalidRayDirection,
    Io(std::io::Error),
    SerializationFailed,
}

impl Scene {
    pub fn new_random(count: usize, scale: f64) -> Result<Self, SceneError> {
        if scale < 0.0 {
            return Err(SceneError::NegativeScale(scale));
        }

        let mut rng = rand::rng();
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

        let learning_rate = 1e-3;
        let beta1 = 0.9;
        let beta2 = 0.999;

        Ok(Self {
            centroid_x: Parameter::new(x, learning_rate, beta1, beta2),
            centroid_y: Parameter::new(y, learning_rate, beta1, beta2),
            centroid_z: Parameter::new(z, learning_rate, beta1, beta2),
            centroid_opacity: Parameter::new(opacity, learning_rate, beta1, beta2),
            centroid_r: Parameter::new(r, learning_rate, beta1, beta2),
            centroid_g: Parameter::new(g, learning_rate, beta1, beta2),
            centroid_b: Parameter::new(b, learning_rate, beta1, beta2),
            centroid_neighbors: vec![Vec::new(); count],
        })
    }

    pub fn save_to_json(&self, path: &Path) -> Result<(), SceneError> {
        let json =
            serde_json::to_string_pretty(self).map_err(|_| SceneError::SerializationFailed)?;
        fs::write(path, json).map_err(SceneError::Io)?;
        Ok(())
    }

    pub fn load_from_json(path: &Path) -> Result<Self, SceneError> {
        let json = fs::read_to_string(path).map_err(SceneError::Io)?;
        serde_json::from_str(&json).map_err(|_| SceneError::SerializationFailed)
    }

    pub fn compute_neighbors(&mut self) -> Result<(), SceneError> {
        self.validate_lengths()?;
        let points = self.points();
        self.centroid_neighbors = exact_neighbors_for_points(&points)
            .unwrap_or_else(|| sampled_neighbors_for_points(&points));
        Ok(())
    }

    pub fn split_centroid(&mut self, index: usize) -> Result<usize, SceneError> {
        self.validate_lengths()?;
        let count = self.centroid_x.len();
        if index >= count {
            return Err(SceneError::InvalidCentroidIndex { index, len: count });
        }
        if self.centroid_neighbors.len() != count {
            return Err(SceneError::InconsistentNeighborData {
                expected: count,
                got: self.centroid_neighbors.len(),
            });
        }

        let old_neighbors = self.centroid_neighbors[index].clone();

        let orig_x = self.centroid_x.values[index];
        let orig_y = self.centroid_y.values[index];
        let orig_z = self.centroid_z.values[index];
        let origin = [orig_x, orig_y, orig_z];
        let mut rng = rand::rng();
        let new_point = self.sample_split_point(index, &old_neighbors, origin, &mut rng);
        let new_x = new_point[0];
        let new_y = new_point[1];
        let new_z = new_point[2];

        self.centroid_x.values.push(new_x);
        self.centroid_y.values.push(new_y);
        self.centroid_z.values.push(new_z);
        self.centroid_opacity
            .values
            .push(self.centroid_opacity.values[index]);
        self.centroid_r.values.push(self.centroid_r.values[index]);
        self.centroid_g.values.push(self.centroid_g.values[index]);
        self.centroid_b.values.push(self.centroid_b.values[index]);

        self.centroid_x.m.push(self.centroid_x.m[index]);
        self.centroid_y.m.push(self.centroid_y.m[index]);
        self.centroid_z.m.push(self.centroid_z.m[index]);
        self.centroid_opacity.m.push(self.centroid_opacity.m[index]);
        self.centroid_r.m.push(self.centroid_r.m[index]);
        self.centroid_g.m.push(self.centroid_g.m[index]);
        self.centroid_b.m.push(self.centroid_b.m[index]);

        self.centroid_x.v.push(self.centroid_x.v[index]);
        self.centroid_y.v.push(self.centroid_y.v[index]);
        self.centroid_z.v.push(self.centroid_z.v[index]);
        self.centroid_opacity.v.push(self.centroid_opacity.v[index]);
        self.centroid_r.v.push(self.centroid_r.v[index]);
        self.centroid_g.v.push(self.centroid_g.v[index]);
        self.centroid_b.v.push(self.centroid_b.v[index]);

        let new_index = count;
        self.centroid_neighbors.push(Vec::new());
        let points = self.points();
        let is_neighbor = are_neighbors_for_points(&points, index, new_index);
        self.set_neighbor_pair(index, new_index, is_neighbor);

        for neighbor in old_neighbors {
            let original_pair = are_neighbors_for_points(&points, index, neighbor);
            self.set_neighbor_pair(index, neighbor, original_pair);

            let new_pair = are_neighbors_for_points(&points, new_index, neighbor);
            self.set_neighbor_pair(new_index, neighbor, new_pair);
        }

        Ok(new_index)
    }

    pub fn render(
        &self,
        start_position: [f64; 3],
        direction: [f64; 3],
    ) -> Result<[f64; 3], SceneError> {
        self.validate_lengths()?;
        if self.centroid_x.is_empty() {
            return Err(SceneError::EmptyScene);
        }
        if self.centroid_neighbors.len() != self.centroid_x.len() {
            return Err(SceneError::InconsistentNeighborData {
                expected: self.centroid_x.len(),
                got: self.centroid_neighbors.len(),
            });
        }

        let direction = normalize(direction);
        if direction == [0.0, 0.0, 0.0] {
            return Err(SceneError::InvalidRayDirection);
        }

        let mut color = [0.0, 0.0, 0.0];
        let mut remaining = 1.0_f64;
        let mut current = self.closest_centroid_at_point(start_position);
        let ray_origin = start_position;
        let mut t0 = 0.0_f64;

        while remaining > TRANSMITTANCE_EPSILON {
            let next = self.next_centroid_along_ray(current, ray_origin, direction, t0);
            let is_terminal = next.is_none();
            let centroid_color = self.centroid_color(current);
            let log_density = self.centroid_opacity.values[current];
            let extinction_per_unit = log_density.exp();
            let segment_length = if let Some((_, t1)) = next {
                (t1 - t0).max(0.0)
            } else {
                f64::INFINITY
            };
            let alpha = if segment_length.is_finite() {
                // Opacity is modeled as log-density; extinction is exp(log_density) per unit length.
                // Over a segment of length L, alpha = 1 - exp(-exp(log_density) * L).
                1.0 - (-(extinction_per_unit * segment_length)).exp()
            } else {
                // Terminal region extends indefinitely in this ray formulation and exp(log_density) > 0.
                1.0
            };

            let weight = remaining * alpha;
            color[0] += centroid_color[0] * weight;
            color[1] += centroid_color[1] * weight;
            color[2] += centroid_color[2] * weight;
            remaining *= 1.0 - alpha;

            if is_terminal {
                break;
            }

            let Some((next_index, t_cross)) = next else {
                break;
            };

            t0 = t_cross;
            current = next_index;
        }

        Ok(color)
    }

    fn validate_lengths(&self) -> Result<(), SceneError> {
        let x = self.centroid_x.len();
        let y = self.centroid_y.len();
        let z = self.centroid_z.len();
        let opacity = self.centroid_opacity.len();
        let r = self.centroid_r.len();
        let g = self.centroid_g.len();
        let b = self.centroid_b.len();

        if x == y && x == z && x == opacity && x == r && x == g && x == b {
            Ok(())
        } else {
            Err(SceneError::InconsistentCentroidData {
                x,
                y,
                z,
                opacity,
                r,
                g,
                b,
            })
        }
    }

    #[cfg(test)]
    fn are_neighbors(&self, i: usize, j: usize) -> bool {
        let points = self.points();
        are_neighbors_for_points(&points, i, j)
    }

    fn points(&self) -> Vec<[f64; 3]> {
        (0..self.centroid_x.len())
            .map(|idx| {
                [
                    self.centroid_x.values[idx],
                    self.centroid_y.values[idx],
                    self.centroid_z.values[idx],
                ]
            })
            .collect()
    }
}

fn are_neighbors_for_points(points: &[[f64; 3]], i: usize, j: usize) -> bool {
    let pi = points[i];
    let pj = points[j];
    let d = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
    let d_norm_sq = dot(d, d);
    if d_norm_sq == 0.0 {
        return false;
    }

    let midpoint = [
        (pi[0] + pj[0]) * 0.5,
        (pi[1] + pj[1]) * 0.5,
        (pi[2] + pj[2]) * 0.5,
    ];
    let (basis0, basis1) = bisector_basis(d);
    let base_radius = d_norm_sq.sqrt() * 0.5;
    let radii = [0.0, 0.5, 1.0, 2.0, 4.0].map(|s| s * base_radius);
    let angles = unit_circle_samples();

    for &radius in &radii {
        for &[cos_theta, sin_theta] in angles {
            let offset = [
                radius * (cos_theta * basis0[0] + sin_theta * basis1[0]),
                radius * (cos_theta * basis0[1] + sin_theta * basis1[1]),
                radius * (cos_theta * basis0[2] + sin_theta * basis1[2]),
            ];
            let q = [
                midpoint[0] + offset[0],
                midpoint[1] + offset[1],
                midpoint[2] + offset[2],
            ];
            if is_shared_closest_point_for_points(points, q, i, j) {
                return true;
            }
        }
    }

    false
}

fn is_shared_closest_point_for_points(
    points: &[[f64; 3]],
    q: [f64; 3],
    i: usize,
    j: usize,
) -> bool {
    let dij = dist_sq(q, points[i]);
    let eps = 1e-5 * (1.0 + dij.abs());
    for (k, &point) in points.iter().enumerate() {
        if k == i || k == j {
            continue;
        }
        let dk = dist_sq(q, point);
        if dk < dij - eps {
            return false;
        }
    }
    true
}

fn exact_neighbors_for_points(points: &[[f64; 3]]) -> Option<Vec<Vec<usize>>> {
    let affine = AffineProjection::for_points(points);
    match affine.dim {
        0 => Some(vec![Vec::new(); points.len()]),
        1 => Some(exact_neighbors_for_points_1d(points, &affine)),
        2 => exact_neighbors_for_projected_points::<2>(points, &affine),
        3 => exact_neighbors_for_projected_points::<3>(points, &affine),
        _ => None,
    }
}

fn exact_neighbors_for_points_1d(
    points: &[[f64; 3]],
    affine: &AffineProjection,
) -> Vec<Vec<usize>> {
    let mut projected = (0..points.len())
        .map(|index| (index, affine.project_scalar(points[index], 0)))
        .collect::<Vec<_>>();
    projected.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));

    let mut neighbors = vec![Vec::new(); points.len()];
    let eps = affine.epsilon;
    let mut groups = Vec::<Vec<usize>>::new();
    for (index, value) in projected {
        if let Some(last_group) = groups.last_mut() {
            let representative = last_group[0];
            let representative_value = affine.project_scalar(points[representative], 0);
            if (value - representative_value).abs() <= eps {
                last_group.push(index);
                continue;
            }
        }
        groups.push(vec![index]);
    }

    for pair in groups.windows(2) {
        for &left in &pair[0] {
            for &right in &pair[1] {
                add_unique_neighbor(&mut neighbors[left], right);
                add_unique_neighbor(&mut neighbors[right], left);
            }
        }
    }

    neighbors
}

fn exact_neighbors_for_projected_points<const D: usize>(
    points: &[[f64; 3]],
    affine: &AffineProjection,
) -> Option<Vec<Vec<usize>>> {
    let projected = points
        .iter()
        .map(|&point| affine.project(point))
        .map(|coords| std::array::from_fn(|axis| coords[axis]))
        .collect::<Vec<[f64; D]>>();
    let qh = Qh::new_delaunay(projected).ok()?;
    let mut neighbors = vec![Vec::new(); points.len()];

    for simplex in qh
        .simplices()
        .filter(|facet| !facet.is_sentinel() && !facet.upper_delaunay())
    {
        let vertices = simplex
            .vertices()?
            .iter()
            .filter_map(|vertex| vertex.index(&qh))
            .collect::<Vec<_>>();
        for left in 0..vertices.len() {
            for right in (left + 1)..vertices.len() {
                add_unique_neighbor(&mut neighbors[vertices[left]], vertices[right]);
                add_unique_neighbor(&mut neighbors[vertices[right]], vertices[left]);
            }
        }
    }

    Some(neighbors)
}

fn sampled_neighbors_for_points(points: &[[f64; 3]]) -> Vec<Vec<usize>> {
    let count = points.len();
    let forward_neighbors = (0..count)
        .into_par_iter()
        .map(|i| {
            let mut row = Vec::new();
            for j in (i + 1)..count {
                if are_neighbors_for_points(points, i, j) {
                    row.push(j);
                }
            }
            row
        })
        .collect::<Vec<_>>();

    let mut neighbors = vec![Vec::new(); count];
    for (i, row) in forward_neighbors.into_iter().enumerate() {
        for j in row {
            neighbors[i].push(j);
            neighbors[j].push(i);
        }
    }

    neighbors
}

fn unit_circle_samples() -> &'static [[f64; 2]; 24] {
    static SAMPLES: OnceLock<[[f64; 2]; 24]> = OnceLock::new();
    SAMPLES.get_or_init(|| {
        std::array::from_fn(|index| {
            let angle = TAU * (index as f64 / 24.0);
            [angle.cos(), angle.sin()]
        })
    })
}

struct AffineProjection {
    dim: usize,
    origin: [f64; 3],
    basis: [[f64; 3]; 3],
    epsilon: f64,
}

impl AffineProjection {
    fn for_points(points: &[[f64; 3]]) -> Self {
        let origin = points.first().copied().unwrap_or([0.0, 0.0, 0.0]);
        let extent = max_extent(points).max(1.0);
        let epsilon = extent * 1e-9;

        let Some(axis0_seed) = points
            .iter()
            .map(|&point| sub(point, origin))
            .find(|&delta| norm(delta) > epsilon)
        else {
            return Self {
                dim: 0,
                origin,
                basis: [[0.0; 3]; 3],
                epsilon,
            };
        };
        let axis0 = normalize(axis0_seed);

        let Some(axis1_seed) = points
            .iter()
            .map(|&point| sub(point, origin))
            .map(|delta| sub(delta, scale(axis0, dot(delta, axis0))))
            .find(|&delta| norm(delta) > epsilon)
        else {
            return Self {
                dim: 1,
                origin,
                basis: [axis0, [0.0; 3], [0.0; 3]],
                epsilon,
            };
        };
        let axis1 = normalize(axis1_seed);
        let axis2 = normalize(cross(axis0, axis1));

        if points
            .iter()
            .map(|&point| sub(point, origin))
            .any(|delta| dot(delta, axis2).abs() > epsilon)
        {
            Self {
                dim: 3,
                origin,
                basis: [axis0, axis1, axis2],
                epsilon,
            }
        } else {
            Self {
                dim: 2,
                origin,
                basis: [axis0, axis1, [0.0; 3]],
                epsilon,
            }
        }
    }

    fn project(&self, point: [f64; 3]) -> [f64; 3] {
        let delta = sub(point, self.origin);
        [
            dot(delta, self.basis[0]),
            dot(delta, self.basis[1]),
            dot(delta, self.basis[2]),
        ]
    }

    fn project_scalar(&self, point: [f64; 3], axis: usize) -> f64 {
        let delta = sub(point, self.origin);
        dot(delta, self.basis[axis])
    }
}
impl Scene {
    fn point(&self, idx: usize) -> [f64; 3] {
        [
            self.centroid_x.values[idx],
            self.centroid_y.values[idx],
            self.centroid_z.values[idx],
        ]
    }

    fn sample_split_point(
        &self,
        index: usize,
        neighbors: &[usize],
        origin: [f64; 3],
        rng: &mut rand::rngs::ThreadRng,
    ) -> [f64; 3] {
        if neighbors.is_empty() {
            let offset = random_unit_vector(rng);
            return [
                origin[0] + offset[0] * SPLIT_FALLBACK_OFFSET_MAGNITUDE,
                origin[1] + offset[1] * SPLIT_FALLBACK_OFFSET_MAGNITUDE,
                origin[2] + offset[2] * SPLIT_FALLBACK_OFFSET_MAGNITUDE,
            ];
        }

        let mut neighborhood_radius = 0.0_f64;
        let mut nearest_neighbor_dist = f64::INFINITY;
        for &neighbor in neighbors {
            let d = dist_sq(origin, self.point(neighbor)).sqrt();
            neighborhood_radius = neighborhood_radius.max(d);
            nearest_neighbor_dist = nearest_neighbor_dist.min(d);
        }
        if neighborhood_radius <= 0.0 || !neighborhood_radius.is_finite() {
            neighborhood_radius = SPLIT_FALLBACK_OFFSET_MAGNITUDE;
        }

        for _ in 0..SPLIT_SAMPLING_ATTEMPTS {
            let candidate = sample_point_in_sphere(origin, neighborhood_radius, rng);
            if self.is_split_candidate_acceptable(candidate, index, neighbors) {
                return candidate;
            }
        }

        let guaranteed_radius = (nearest_neighbor_dist * 0.25).max(SPLIT_FALLBACK_OFFSET_MAGNITUDE);
        let offset = random_unit_vector(rng);
        [
            origin[0] + offset[0] * guaranteed_radius,
            origin[1] + offset[1] * guaranteed_radius,
            origin[2] + offset[2] * guaranteed_radius,
        ]
    }

    fn is_split_candidate_acceptable(
        &self,
        candidate: [f64; 3],
        original_index: usize,
        neighbors: &[usize],
    ) -> bool {
        let d_orig = dist_sq(candidate, self.point(original_index));
        neighbors
            .iter()
            .all(|&neighbor| d_orig < dist_sq(candidate, self.point(neighbor)))
    }

    fn set_neighbor_pair(&mut self, a: usize, b: usize, is_neighbor: bool) {
        if is_neighbor {
            add_unique_neighbor(&mut self.centroid_neighbors[a], b);
            add_unique_neighbor(&mut self.centroid_neighbors[b], a);
        } else {
            remove_neighbor(&mut self.centroid_neighbors[a], b);
            remove_neighbor(&mut self.centroid_neighbors[b], a);
        }
    }

    fn closest_centroid_at_point(&self, q: [f64; 3]) -> usize {
        let mut best_idx = 0usize;
        let mut best_dist = dist_sq(q, self.point(0));
        for i in 1..self.centroid_x.len() {
            let d = dist_sq(q, self.point(i));
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }

    fn next_centroid_along_ray(
        &self,
        current: usize,
        ray_origin: [f64; 3],
        direction: [f64; 3],
        t0: f64,
    ) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        let pi = self.point(current);
        let oi = dot(pi, pi);

        for &neighbor in &self.centroid_neighbors[current] {
            let pj = self.point(neighbor);
            let rhs = dot(pj, pj) - oi;
            let d = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
            let denom = 2.0 * dot(d, direction);
            if denom.abs() < 1e-8 {
                continue;
            }
            // "front" condition from the algorithm: moving along the ray toward neighbor j.
            if denom <= 0.0 {
                continue;
            }
            let num = rhs - 2.0 * dot(d, ray_origin);
            let t = num / denom;
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

    fn centroid_color(&self, idx: usize) -> [f64; 3] {
        [
            self.centroid_r.values[idx].clamp(0.0, 1.0),
            self.centroid_g.values[idx].clamp(0.0, 1.0),
            self.centroid_b.values[idx].clamp(0.0, 1.0),
        ]
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: [f64; 3], factor: f64) -> [f64; 3] {
    [v[0] * factor, v[1] * factor, v[2] * factor]
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn dist_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = dot(v, v).sqrt();
    if n == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn max_extent(points: &[[f64; 3]]) -> f64 {
    let mut min = points.first().copied().unwrap_or([0.0; 3]);
    let mut max = min;
    for &point in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    (0..3).map(|axis| max[axis] - min[axis]).fold(0.0, f64::max)
}

fn bisector_basis(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let n = normalize(normal);
    let helper = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let e1 = normalize(cross(n, helper));
    let e2 = normalize(cross(n, e1));
    (e1, e2)
}

fn random_unit_vector(rng: &mut rand::rngs::ThreadRng) -> [f64; 3] {
    loop {
        let x: f64 = rng.random_range(-1.0..=1.0);
        let y: f64 = rng.random_range(-1.0..=1.0);
        let z: f64 = rng.random_range(-1.0..=1.0);
        let n2 = x * x + y * y + z * z;
        if n2 > 0.0 {
            let n = n2.sqrt();
            return [x / n, y / n, z / n];
        }
    }
}

fn sample_point_in_sphere(
    center: [f64; 3],
    radius: f64,
    rng: &mut rand::rngs::ThreadRng,
) -> [f64; 3] {
    let direction = random_unit_vector(rng);
    let scale = radius * rng.random::<f64>().cbrt();
    [
        center[0] + direction[0] * scale,
        center[1] + direction[1] * scale,
        center[2] + direction[2] * scale,
    ]
}

fn add_unique_neighbor(neighbors: &mut Vec<usize>, value: usize) {
    if !neighbors.contains(&value) {
        neighbors.push(value);
        neighbors.sort_unstable();
    }
}

fn remove_neighbor(neighbors: &mut Vec<usize>, value: usize) {
    neighbors.retain(|&n| n != value);
}

#[cfg(test)]
mod tests {
    use super::{Scene, SceneError, dist_sq};
    use crate::parameter::Parameter;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn new_random_creates_expected_number_of_centroids() {
        let count = 128;
        let scene = Scene::new_random(count, 5.0).expect("scale is valid");

        assert_eq!(scene.centroid_x.len(), count);
        assert_eq!(scene.centroid_y.len(), count);
        assert_eq!(scene.centroid_z.len(), count);
        assert_eq!(scene.centroid_opacity.len(), count);
        assert_eq!(scene.centroid_r.len(), count);
        assert_eq!(scene.centroid_g.len(), count);
        assert_eq!(scene.centroid_b.len(), count);
        assert_eq!(scene.centroid_neighbors.len(), count);
        assert!(scene.centroid_neighbors.iter().all(|v| v.is_empty()));
    }

    #[test]
    fn new_random_values_stay_in_bounds() {
        let scale = 4.0;

        let scene = Scene::new_random(2048, scale).expect("scale is valid");

        for &x in &scene.centroid_x.values {
            assert!((-scale..=scale).contains(&x));
        }
        for &y in &scene.centroid_y.values {
            assert!((-scale..=scale).contains(&y));
        }
        for &z in &scene.centroid_z.values {
            assert!((-scale..=scale).contains(&z));
        }
        for &opacity in &scene.centroid_opacity.values {
            assert!(opacity.is_finite());
        }
        for &r in &scene.centroid_r.values {
            assert!((0.0..=1.0).contains(&r));
        }
        for &g in &scene.centroid_g.values {
            assert!((0.0..=1.0).contains(&g));
        }
        for &b in &scene.centroid_b.values {
            assert!((0.0..=1.0).contains(&b));
        }
    }

    #[test]
    fn new_random_with_zero_count_creates_empty_vectors() {
        let scene = Scene::new_random(0, 1.0).expect("scale is valid");

        assert!(scene.centroid_x.is_empty());
        assert!(scene.centroid_y.is_empty());
        assert!(scene.centroid_z.is_empty());
        assert!(scene.centroid_opacity.is_empty());
        assert!(scene.centroid_r.is_empty());
        assert!(scene.centroid_g.is_empty());
        assert!(scene.centroid_b.is_empty());
        assert!(scene.centroid_neighbors.is_empty());
    }

    #[test]
    fn new_random_returns_error_on_negative_scales() {
        let result = Scene::new_random(10, -1.0);

        assert!(matches!(result, Err(SceneError::NegativeScale(-1.0))));
    }

    #[test]
    fn compute_neighbors_for_three_in_line() {
        let mut scene = Scene {
            centroid_x: Parameter::new(vec![-1.0, 0.0, 1.0], 1e-3, 0.9, 0.999),
            centroid_y: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_z: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_opacity: Parameter::new(vec![1.0, 1.0, 1.0], 1e-3, 0.9, 0.999),
            centroid_r: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_g: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_b: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_neighbors: vec![Vec::new(); 3],
        };

        scene.compute_neighbors().expect("lengths are consistent");

        assert_eq!(scene.centroid_neighbors[0], vec![1]);
        assert_eq!(scene.centroid_neighbors[1], vec![0, 2]);
        assert_eq!(scene.centroid_neighbors[2], vec![1]);
    }

    #[test]
    fn compute_neighbors_for_tetrahedron() {
        let mut scene = Scene {
            centroid_x: Parameter::new(vec![0.0, 1.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_y: Parameter::new(vec![0.0, 0.0, 1.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_z: Parameter::new(vec![0.0, 0.0, 0.0, 1.0], 1e-3, 0.9, 0.999),
            centroid_opacity: Parameter::new(vec![1.0; 4], 1e-3, 0.9, 0.999),
            centroid_r: Parameter::new(vec![0.0; 4], 1e-3, 0.9, 0.999),
            centroid_g: Parameter::new(vec![0.0; 4], 1e-3, 0.9, 0.999),
            centroid_b: Parameter::new(vec![0.0; 4], 1e-3, 0.9, 0.999),
            centroid_neighbors: vec![Vec::new(); 4],
        };

        scene.compute_neighbors().expect("lengths are consistent");

        assert_eq!(scene.centroid_neighbors[0], vec![1, 2, 3]);
        assert_eq!(scene.centroid_neighbors[1], vec![0, 2, 3]);
        assert_eq!(scene.centroid_neighbors[2], vec![0, 1, 3]);
        assert_eq!(scene.centroid_neighbors[3], vec![0, 1, 2]);
    }

    #[test]
    fn compute_neighbors_returns_error_for_mismatched_lengths() {
        let mut scene = Scene {
            centroid_x: Parameter::new(vec![0.0, 1.0], 1e-3, 0.9, 0.999),
            centroid_y: Parameter::new(vec![0.0], 1e-3, 0.9, 0.999),
            centroid_z: Parameter::new(vec![0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_opacity: Parameter::new(vec![1.0, 1.0], 1e-3, 0.9, 0.999),
            centroid_r: Parameter::new(vec![0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_g: Parameter::new(vec![0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_b: Parameter::new(vec![0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_neighbors: vec![],
        };

        let result = scene.compute_neighbors();
        assert!(matches!(
            result,
            Err(SceneError::InconsistentCentroidData { .. })
        ));
    }

    #[test]
    fn split_centroid_keeps_original_and_copies_non_position_properties() {
        let mut scene = Scene::new_random(4, 3.0).expect("valid scene");
        scene.compute_neighbors().expect("neighbors are valid");

        let index = 2usize;
        let orig_x = scene.centroid_x.values[index];
        let orig_y = scene.centroid_y.values[index];
        let orig_z = scene.centroid_z.values[index];
        let orig_opacity = scene.centroid_opacity.values[index];
        let orig_r = scene.centroid_r.values[index];
        let orig_g = scene.centroid_g.values[index];
        let orig_b = scene.centroid_b.values[index];

        let new_index = scene.split_centroid(index).expect("split should succeed");

        assert_eq!(scene.centroid_x.len(), 5);
        assert_eq!(scene.centroid_neighbors.len(), 5);
        assert_eq!(scene.centroid_opacity.values[new_index], orig_opacity);
        assert_eq!(scene.centroid_r.values[new_index], orig_r);
        assert_eq!(scene.centroid_g.values[new_index], orig_g);
        assert_eq!(scene.centroid_b.values[new_index], orig_b);
        assert_eq!(scene.centroid_x.values[index], orig_x);
        assert_eq!(scene.centroid_y.values[index], orig_y);
        assert_eq!(scene.centroid_z.values[index], orig_z);

        let dx = scene.centroid_x.values[new_index] - orig_x;
        let dy = scene.centroid_y.values[new_index] - orig_y;
        let dz = scene.centroid_z.values[new_index] - orig_z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(dist > 0.0);
    }

    #[test]
    fn split_centroid_incrementally_updates_relevant_neighbors() {
        let mut scene = Scene {
            centroid_x: Parameter::new(vec![-1.0, 0.0, 1.0], 1e-3, 0.9, 0.999),
            centroid_y: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_z: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_opacity: Parameter::new(vec![1.0, 1.0, 1.0], 1e-3, 0.9, 0.999),
            centroid_r: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_g: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_b: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_neighbors: vec![Vec::new(); 3],
        };
        scene.compute_neighbors().expect("valid neighbors");

        let index = 1usize;
        let old_neighbors = scene.centroid_neighbors[index].clone();
        let new_index = scene.split_centroid(index).expect("split should succeed");

        for &n in &old_neighbors {
            assert_eq!(
                scene.centroid_neighbors[index].contains(&n),
                scene.are_neighbors(index, n)
            );
            assert_eq!(
                scene.centroid_neighbors[new_index].contains(&n),
                scene.are_neighbors(new_index, n)
            );
        }

        assert_eq!(
            scene.centroid_neighbors[index].contains(&new_index),
            scene.are_neighbors(index, new_index)
        );
        assert!(
            scene.centroid_neighbors[new_index].contains(&index)
                == scene.centroid_neighbors[index].contains(&new_index)
        );
    }

    #[test]
    fn split_centroid_samples_point_closer_to_split_centroid_than_old_neighbors() {
        let mut scene = Scene {
            centroid_x: Parameter::new(vec![-1.0, 0.0, 1.0], 1e-3, 0.9, 0.999),
            centroid_y: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_z: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_opacity: Parameter::new(vec![1.0, 1.0, 1.0], 1e-3, 0.9, 0.999),
            centroid_r: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_g: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_b: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_neighbors: vec![Vec::new(); 3],
        };
        scene.compute_neighbors().expect("valid neighbors");

        let index = 1usize;
        let old_neighbors = scene.centroid_neighbors[index].clone();
        let new_index = scene.split_centroid(index).expect("split should succeed");
        let candidate = scene.point(new_index);
        let d_to_split = dist_sq(candidate, scene.point(index));

        assert!(!old_neighbors.is_empty());
        for neighbor in old_neighbors {
            assert!(d_to_split < dist_sq(candidate, scene.point(neighbor)));
        }
    }

    #[test]
    fn render_fills_with_terminal_centroid_color() {
        let scene = Scene {
            centroid_x: Parameter::new(vec![-1.0, 0.0, 1.0], 1e-3, 0.9, 0.999),
            centroid_y: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_z: Parameter::new(vec![0.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_opacity: Parameter::new(vec![0.25, 0.5, 0.2], 1e-3, 0.9, 0.999),
            centroid_r: Parameter::new(vec![1.0, 0.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_g: Parameter::new(vec![0.0, 1.0, 0.0], 1e-3, 0.9, 0.999),
            centroid_b: Parameter::new(vec![0.0, 0.0, 1.0], 1e-3, 0.9, 0.999),
            centroid_neighbors: vec![vec![1], vec![0, 2], vec![1]],
        };

        let color = scene
            .render([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            .expect("render should succeed");

        let alpha0 = 1.0 - (-(0.25_f64.exp() * 0.5)).exp();
        let alpha1 = 1.0 - (-(0.5_f64.exp() * 1.0)).exp();
        let expected_r = alpha0;
        let expected_g = (1.0 - alpha0) * alpha1;
        let expected_b = (1.0 - alpha0) * (1.0 - alpha1);

        assert!((color[0] - expected_r).abs() < 1e-4);
        assert!((color[1] - expected_g).abs() < 1e-4);
        assert!((color[2] - expected_b).abs() < 1e-4);
    }

    #[test]
    fn render_returns_error_for_zero_direction() {
        let scene = Scene::new_random(1, 1.0).expect("valid");
        let result = scene.render([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(matches!(result, Err(SceneError::InvalidRayDirection)));
    }

    #[test]
    fn scene_json_round_trip_preserves_centroid_data() {
        let mut scene = Scene::new_random(3, 1.0).expect("valid");
        scene.compute_neighbors().expect("neighbors should compute");
        let path = std::env::temp_dir().join(format!(
            "radiant-foam-scene-{}.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should advance")
                .as_nanos()
        ));

        scene.save_to_json(&path).expect("scene should save");
        let loaded = Scene::load_from_json(&path).expect("scene should load");

        assert!(
            scene
                .centroid_x
                .values
                .iter()
                .zip(&loaded.centroid_x.values)
                .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12)
        );
        assert!(
            scene
                .centroid_y
                .values
                .iter()
                .zip(&loaded.centroid_y.values)
                .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12)
        );
        assert!(
            scene
                .centroid_z
                .values
                .iter()
                .zip(&loaded.centroid_z.values)
                .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12)
        );
        assert_eq!(scene.centroid_neighbors, loaded.centroid_neighbors);

        fs::remove_file(path).expect("json should clean up");
    }
}
