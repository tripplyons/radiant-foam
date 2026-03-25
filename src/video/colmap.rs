use crate::parameter::Parameter;
use crate::renderer::{ImageData, PerspectiveCamera};
use crate::scene::Scene;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::{
    DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_LEARNING_RATE, VideoInitError, VideoInitOptions,
};

pub(super) fn load_training_frames(
    text_dir: &Path,
    frames_dir: &Path,
) -> Result<Vec<TrainingFrame>, VideoInitError> {
    let cameras = load_cameras(&text_dir.join("cameras.txt"))?;
    let images = load_registered_images(&text_dir.join("images.txt"))?;
    let extracted_frames = count_extracted_frames(frames_dir)?;

    println!(
        "colmap registered {} / {} extracted frames",
        images.len(),
        extracted_frames,
    );

    images
        .into_iter()
        .map(|image| {
            let camera_model = cameras
                .get(&image.camera_id)
                .ok_or(VideoInitError::MissingCameraId(image.camera_id))?;
            let image_path = frames_dir.join(&image.name);
            let target = load_target_image(&image_path)?;
            Ok(TrainingFrame {
                name: image.name,
                camera: camera_model
                    .to_perspective_camera(image.world_to_camera_rotation, image.translation),
                target,
            })
        })
        .collect()
}

pub fn load_scene_from_colmap_text_model(
    text_dir: &Path,
    options: &VideoInitOptions,
) -> Result<Scene, VideoInitError> {
    let points_path = text_dir.join("points3D.txt");
    let contents = fs::read_to_string(&points_path)
        .map_err(|_| VideoInitError::MissingPointsFile(points_path.clone()))?;
    let mut points = contents
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(parse_point_record)
        .collect::<Result<Vec<_>, _>>()?;

    if let Some(max_points) = options.max_points {
        if points.len() > max_points {
            let stride = points.len() as f64 / max_points as f64;
            points = (0..max_points)
                .map(|index| points[(index as f64 * stride).floor() as usize])
                .collect();
        }
    }

    if points.is_empty() {
        return Err(VideoInitError::NoPoints(points_path));
    }

    let count = points.len();
    let mut x = Vec::with_capacity(count);
    let mut y = Vec::with_capacity(count);
    let mut z = Vec::with_capacity(count);
    let mut opacity = Vec::with_capacity(count);
    let mut r = Vec::with_capacity(count);
    let mut g = Vec::with_capacity(count);
    let mut b = Vec::with_capacity(count);

    for point in points {
        x.push(point.position[0]);
        y.push(point.position[1]);
        z.push(point.position[2]);
        opacity.push(options.initial_log_density);
        r.push(point.color[0]);
        g.push(point.color[1]);
        b.push(point.color[2]);
    }

    let mut scene = Scene {
        centroid_x: Parameter::new(x, DEFAULT_LEARNING_RATE, DEFAULT_BETA1, DEFAULT_BETA2),
        centroid_y: Parameter::new(y, DEFAULT_LEARNING_RATE, DEFAULT_BETA1, DEFAULT_BETA2),
        centroid_z: Parameter::new(z, DEFAULT_LEARNING_RATE, DEFAULT_BETA1, DEFAULT_BETA2),
        centroid_opacity: Parameter::new(
            opacity,
            DEFAULT_LEARNING_RATE,
            DEFAULT_BETA1,
            DEFAULT_BETA2,
        ),
        centroid_r: Parameter::new(r, DEFAULT_LEARNING_RATE, DEFAULT_BETA1, DEFAULT_BETA2),
        centroid_g: Parameter::new(g, DEFAULT_LEARNING_RATE, DEFAULT_BETA1, DEFAULT_BETA2),
        centroid_b: Parameter::new(b, DEFAULT_LEARNING_RATE, DEFAULT_BETA1, DEFAULT_BETA2),
        centroid_neighbors: vec![Vec::new(); count],
    };
    scene.compute_neighbors()?;
    Ok(scene)
}

pub fn load_cameras(path: &Path) -> Result<HashMap<u32, ColmapCameraModel>, VideoInitError> {
    let contents = fs::read_to_string(path)
        .map_err(|_| VideoInitError::MissingCameraFile(path.to_path_buf()))?;
    contents
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(parse_camera_record)
        .map(|result| result.map(|camera| (camera.id, camera)))
        .collect()
}

pub fn load_registered_images(path: &Path) -> Result<Vec<ColmapImageRecord>, VideoInitError> {
    let contents = fs::read_to_string(path)
        .map_err(|_| VideoInitError::MissingImageFile(path.to_path_buf()))?;
    let mut images = Vec::new();
    let mut lines = contents.lines().peekable();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        images.push(parse_image_record(trimmed)?);
        let _ = lines.next();
    }

    Ok(images)
}

#[derive(Clone, Copy, Debug)]
struct SparsePoint {
    position: [f64; 3],
    color: [f64; 3],
}

#[derive(Clone, Debug)]
pub(super) struct TrainingFrame {
    pub(super) name: String,
    pub(super) camera: PerspectiveCamera,
    pub(super) target: ImageData,
}

#[derive(Clone, Debug)]
pub struct ColmapCameraModel {
    id: u32,
    width: u32,
    height: u32,
    focal_x: f64,
    focal_y: f64,
    principal_x: f64,
    principal_y: f64,
}

impl ColmapCameraModel {
    fn to_perspective_camera(
        &self,
        world_to_camera_rotation: [[f64; 3]; 3],
        translation: [f64; 3],
    ) -> PerspectiveCamera {
        let camera_to_world = transpose(world_to_camera_rotation);
        let origin_camera = [-translation[0], -translation[1], -translation[2]];
        let origin = mat3_mul_vec3(camera_to_world, origin_camera);

        PerspectiveCamera {
            width: self.width,
            height: self.height,
            focal_x: self.focal_x,
            focal_y: self.focal_y,
            principal_x: self.principal_x,
            principal_y: self.principal_y,
            origin,
            camera_to_world,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColmapImageRecord {
    pub camera_id: u32,
    pub name: String,
    world_to_camera_rotation: [[f64; 3]; 3],
    translation: [f64; 3],
}

fn count_extracted_frames(frames_dir: &Path) -> Result<usize, VideoInitError> {
    Ok(fs::read_dir(frames_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("png"))
        .count())
}

fn load_target_image(path: &Path) -> Result<ImageData, VideoInitError> {
    if !path.exists() {
        return Err(VideoInitError::MissingFrameImage(path.to_path_buf()));
    }
    let image = image::open(path)?.to_rgb8();
    Ok(ImageData {
        width: image.width(),
        height: image.height(),
        pixels: image.into_raw(),
    })
}

fn parse_point_record(line: &str) -> Result<SparsePoint, VideoInitError> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 7 {
        return Err(VideoInitError::InvalidPointRecord(line.to_string()));
    }

    let x = fields[1]
        .parse::<f64>()
        .map_err(|_| VideoInitError::InvalidPointRecord(line.to_string()))?;
    let y = fields[2]
        .parse::<f64>()
        .map_err(|_| VideoInitError::InvalidPointRecord(line.to_string()))?;
    let z = fields[3]
        .parse::<f64>()
        .map_err(|_| VideoInitError::InvalidPointRecord(line.to_string()))?;
    let r = fields[4]
        .parse::<f64>()
        .map_err(|_| VideoInitError::InvalidPointRecord(line.to_string()))?;
    let g = fields[5]
        .parse::<f64>()
        .map_err(|_| VideoInitError::InvalidPointRecord(line.to_string()))?;
    let b = fields[6]
        .parse::<f64>()
        .map_err(|_| VideoInitError::InvalidPointRecord(line.to_string()))?;

    Ok(SparsePoint {
        position: [x, y, z],
        color: [r / 255.0, g / 255.0, b / 255.0],
    })
}

fn parse_camera_record(line: &str) -> Result<ColmapCameraModel, VideoInitError> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 5 {
        return Err(VideoInitError::InvalidCameraRecord(line.to_string()));
    }

    let id = parse_field::<u32>(&fields, 0, line, VideoInitError::InvalidCameraRecord)?;
    let model = fields[1];
    let width = parse_field::<u32>(&fields, 2, line, VideoInitError::InvalidCameraRecord)?;
    let height = parse_field::<u32>(&fields, 3, line, VideoInitError::InvalidCameraRecord)?;
    let params = fields[4..]
        .iter()
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|_| VideoInitError::InvalidCameraRecord(line.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (focal_x, focal_y, principal_x, principal_y) = match model {
        "SIMPLE_PINHOLE" | "SIMPLE_RADIAL" | "RADIAL" => {
            if params.len() < 3 {
                return Err(VideoInitError::InvalidCameraRecord(line.to_string()));
            }
            (params[0], params[0], params[1], params[2])
        }
        "PINHOLE" | "OPENCV" | "FULL_OPENCV" => {
            if params.len() < 4 {
                return Err(VideoInitError::InvalidCameraRecord(line.to_string()));
            }
            (params[0], params[1], params[2], params[3])
        }
        other => return Err(VideoInitError::UnsupportedCameraModel(other.to_string())),
    };

    Ok(ColmapCameraModel {
        id,
        width,
        height,
        focal_x,
        focal_y,
        principal_x,
        principal_y,
    })
}

fn parse_image_record(line: &str) -> Result<ColmapImageRecord, VideoInitError> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 10 {
        return Err(VideoInitError::InvalidImageRecord(line.to_string()));
    }

    let qw = parse_field::<f64>(&fields, 1, line, VideoInitError::InvalidImageRecord)?;
    let qx = parse_field::<f64>(&fields, 2, line, VideoInitError::InvalidImageRecord)?;
    let qy = parse_field::<f64>(&fields, 3, line, VideoInitError::InvalidImageRecord)?;
    let qz = parse_field::<f64>(&fields, 4, line, VideoInitError::InvalidImageRecord)?;
    let tx = parse_field::<f64>(&fields, 5, line, VideoInitError::InvalidImageRecord)?;
    let ty = parse_field::<f64>(&fields, 6, line, VideoInitError::InvalidImageRecord)?;
    let tz = parse_field::<f64>(&fields, 7, line, VideoInitError::InvalidImageRecord)?;
    let camera_id = parse_field::<u32>(&fields, 8, line, VideoInitError::InvalidImageRecord)?;
    let name = fields[9].to_string();

    Ok(ColmapImageRecord {
        camera_id,
        name,
        world_to_camera_rotation: quaternion_to_rotation_matrix([qw, qx, qy, qz]),
        translation: [tx, ty, tz],
    })
}

fn parse_field<T>(
    fields: &[&str],
    index: usize,
    line: &str,
    error_builder: fn(String) -> VideoInitError,
) -> Result<T, VideoInitError>
where
    T: std::str::FromStr,
{
    fields[index]
        .parse::<T>()
        .map_err(|_| error_builder(line.to_string()))
}

fn quaternion_to_rotation_matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [qw, qx, qy, qz] = q;
    let norm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
    let qw = qw / norm;
    let qx = qx / norm;
    let qy = qy / norm;
    let qz = qz / norm;

    [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]
}

fn transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn mat3_mul_vec3(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}
