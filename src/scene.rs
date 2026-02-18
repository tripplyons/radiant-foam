use crate::parameter::{Parameter, ParameterError};
use rand::RngExt;
use std::f32::consts::TAU;

const SPLIT_OFFSET_MAGNITUDE: f32 = 1e-3;

#[derive(Debug)]
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

#[derive(Debug, PartialEq)]
pub enum SceneError {
    NegativeScale(f32),
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
    InvalidCentroidIndex { index: usize, len: usize },
    InconsistentNeighborData { expected: usize, got: usize },
    EmptyScene,
    InvalidRayDirection,
}

impl Scene {
    pub fn new_random(count: usize, scale: f32) -> Result<Self, SceneError> {
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
            opacity.push(0.1);
            r.push(rng.random_range(0.0..=1.0));
            g.push(rng.random_range(0.0..=1.0));
            b.push(rng.random_range(0.0..=1.0));
        }

        Ok(Self {
            centroid_x: Parameter::new(x, 1e-3, 0.9, 0.999),
            centroid_y: Parameter::new(y, 1e-3, 0.9, 0.999),
            centroid_z: Parameter::new(z, 1e-3, 0.9, 0.999),
            centroid_opacity: Parameter::new(opacity, 1e-3, 0.9, 0.999),
            centroid_r: Parameter::new(r, 1e-3, 0.9, 0.999),
            centroid_g: Parameter::new(g, 1e-3, 0.9, 0.999),
            centroid_b: Parameter::new(b, 1e-3, 0.9, 0.999),
            centroid_neighbors: vec![Vec::new(); count],
        })
    }

    pub fn compute_neighbors(&mut self) -> Result<(), SceneError> {
        self.validate_lengths()?;
        let count = self.centroid_x.len();
        let mut neighbors = vec![Vec::new(); count];

        for i in 0..count {
            for j in (i + 1)..count {
                if self.are_neighbors(i, j) {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                }
            }
        }

        self.centroid_neighbors = neighbors;
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

        let mut rng = rand::rng();
        let direction = random_unit_vector(&mut rng);
        let dx = direction[0] * SPLIT_OFFSET_MAGNITUDE;
        let dy = direction[1] * SPLIT_OFFSET_MAGNITUDE;
        let dz = direction[2] * SPLIT_OFFSET_MAGNITUDE;

        let orig_x = self.centroid_x.values[index];
        let orig_y = self.centroid_y.values[index];
        let orig_z = self.centroid_z.values[index];

        let new_x = orig_x + dx;
        let new_y = orig_y + dy;
        let new_z = orig_z + dz;

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

        let is_neighbor = self.are_neighbors(index, new_index);
        self.set_neighbor_pair(index, new_index, is_neighbor);

        for neighbor in old_neighbors {
            let original_pair = self.are_neighbors(index, neighbor);
            self.set_neighbor_pair(index, neighbor, original_pair);

            let new_pair = self.are_neighbors(new_index, neighbor);
            self.set_neighbor_pair(new_index, neighbor, new_pair);
        }

        Ok(new_index)
    }

    pub fn render(
        &self,
        start_position: [f32; 3],
        direction: [f32; 3],
    ) -> Result<[f32; 3], SceneError> {
        self.validate_lengths()?;
        let count = self.centroid_x.len();
        if count == 0 {
            return Err(SceneError::EmptyScene);
        }
        if self.centroid_neighbors.len() != count {
            return Err(SceneError::InconsistentNeighborData {
                expected: count,
                got: self.centroid_neighbors.len(),
            });
        }

        let direction = normalize(direction);
        if direction == [0.0, 0.0, 0.0] {
            return Err(SceneError::InvalidRayDirection);
        }

        let mut color = [0.0, 0.0, 0.0];
        let mut remaining = 1.0_f32;
        let mut current = self.closest_centroid_at_point(start_position);
        let mut origin = start_position;
        let mut visited = vec![false; count];

        for _ in 0..count {
            if visited[current] {
                break;
            }
            visited[current] = true;

            let next = self.next_centroid_along_ray(current, origin, direction);
            let is_terminal = next.is_none();
            let centroid_color = self.centroid_color(current);
            if is_terminal {
                color[0] += centroid_color[0] * remaining;
                color[1] += centroid_color[1] * remaining;
                color[2] += centroid_color[2] * remaining;
                remaining = 0.0;
            } else {
                let alpha = self.centroid_opacity.values[current].clamp(0.0, 1.0);
                let weight = remaining * alpha;
                color[0] += centroid_color[0] * weight;
                color[1] += centroid_color[1] * weight;
                color[2] += centroid_color[2] * weight;
                remaining *= 1.0 - alpha;
            }

            let Some((next_index, t_cross)) = next else {
                break;
            };

            let advance = (t_cross + 1e-4).max(1e-4);
            origin = [
                origin[0] + direction[0] * advance,
                origin[1] + direction[1] * advance,
                origin[2] + direction[2] * advance,
            ];
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

    fn are_neighbors(&self, i: usize, j: usize) -> bool {
        let pi = self.point(i);
        let pj = self.point(j);
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
        let basis = bisector_basis(d);
        let base_radius = d_norm_sq.sqrt() * 0.5;
        let radii = [0.0, 0.5, 1.0, 2.0, 4.0].map(|s| s * base_radius);
        let steps = 24_usize;

        for &radius in &radii {
            for t in 0..steps {
                let angle = TAU * (t as f32 / steps as f32);
                let offset = [
                    radius * (angle.cos() * basis.0[0] + angle.sin() * basis.1[0]),
                    radius * (angle.cos() * basis.0[1] + angle.sin() * basis.1[1]),
                    radius * (angle.cos() * basis.0[2] + angle.sin() * basis.1[2]),
                ];
                let q = [
                    midpoint[0] + offset[0],
                    midpoint[1] + offset[1],
                    midpoint[2] + offset[2],
                ];
                if self.is_shared_closest_point(q, i, j) {
                    return true;
                }
            }
        }

        false
    }

    fn is_shared_closest_point(&self, q: [f32; 3], i: usize, j: usize) -> bool {
        let dij = dist_sq(q, self.point(i));
        let eps = 1e-5 * (1.0 + dij.abs());

        for k in 0..self.centroid_x.len() {
            if k == i || k == j {
                continue;
            }
            let dk = dist_sq(q, self.point(k));
            if dk < dij - eps {
                return false;
            }
        }
        true
    }

    fn point(&self, idx: usize) -> [f32; 3] {
        [
            self.centroid_x.values[idx],
            self.centroid_y.values[idx],
            self.centroid_z.values[idx],
        ]
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

    fn closest_centroid_at_point(&self, q: [f32; 3]) -> usize {
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
        origin: [f32; 3],
        direction: [f32; 3],
    ) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
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
            let num = rhs - 2.0 * dot(d, origin);
            let t = num / denom;
            if t <= 1e-6 {
                continue;
            }

            let q_after = [
                origin[0] + direction[0] * (t + 1e-4),
                origin[1] + direction[1] * (t + 1e-4),
                origin[2] + direction[2] * (t + 1e-4),
            ];
            if self.closest_centroid_at_point(q_after) != neighbor {
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

    fn centroid_color(&self, idx: usize) -> [f32; 3] {
        [
            self.centroid_r.values[idx].clamp(0.0, 1.0),
            self.centroid_g.values[idx].clamp(0.0, 1.0),
            self.centroid_b.values[idx].clamp(0.0, 1.0),
        ]
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dist_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let n = dot(v, v).sqrt();
    if n == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn bisector_basis(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
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

fn random_unit_vector(rng: &mut rand::rngs::ThreadRng) -> [f32; 3] {
    loop {
        let x: f32 = rng.random_range(-1.0..=1.0);
        let y: f32 = rng.random_range(-1.0..=1.0);
        let z: f32 = rng.random_range(-1.0..=1.0);
        let n2 = x * x + y * y + z * z;
        if n2 > 0.0 {
            let n = n2.sqrt();
            return [x / n, y / n, z / n];
        }
    }
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
    use super::{Scene, SceneError};
    use crate::parameter::Parameter;

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
            assert!((0.0..=1.0).contains(&opacity));
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
        assert!(scene.centroid_neighbors[new_index].contains(&index)
            == scene.centroid_neighbors[index].contains(&new_index));
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

        assert!((color[0] - 0.25).abs() < 1e-5);
        assert!((color[1] - 0.375).abs() < 1e-5);
        assert!((color[2] - 0.375).abs() < 1e-5);
    }

    #[test]
    fn render_returns_error_for_zero_direction() {
        let scene = Scene::new_random(1, 1.0).expect("valid");
        let result = scene.render([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(matches!(result, Err(SceneError::InvalidRayDirection)));
    }
}
