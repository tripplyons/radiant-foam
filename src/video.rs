use crate::parameter::Parameter;
use crate::renderer::{ImageData, PerspectiveCamera, PerspectiveRenderer, RendererError};
use crate::scene::{Scene, SceneError};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_FPS: f64 = 8.0;
const DEFAULT_LOG_DENSITY: f64 = -3.0;
const DEFAULT_DISTORTION_LAMBDA: f64 = 1e-2;
const DEFAULT_LEARNING_RATE: f64 = 1e-3;
const DEFAULT_BETA1: f64 = 0.9;
const DEFAULT_BETA2: f64 = 0.999;
const DEFAULT_TRAIN_EPOCHS: usize = 10;

#[derive(Clone, Debug)]
pub struct VideoInitOptions {
    pub fps: f64,
    pub max_points: Option<usize>,
    pub initial_log_density: f64,
    pub train_epochs: usize,
    pub distortion_lambda: f64,
}

impl Default for VideoInitOptions {
    fn default() -> Self {
        Self {
            fps: DEFAULT_FPS,
            max_points: Some(10_000),
            initial_log_density: DEFAULT_LOG_DENSITY,
            train_epochs: DEFAULT_TRAIN_EPOCHS,
            distortion_lambda: DEFAULT_DISTORTION_LAMBDA,
        }
    }
}

#[derive(Debug)]
pub enum VideoInitError {
    Io(std::io::Error),
    Scene(SceneError),
    CommandFailed {
        program: String,
        args: Vec<String>,
        status: std::process::ExitStatus,
    },
    MissingSparseModel(PathBuf),
    MissingPointsFile(PathBuf),
    NoPoints(PathBuf),
    InvalidPointRecord(String),
    MissingCameraFile(PathBuf),
    MissingImageFile(PathBuf),
    InvalidCameraRecord(String),
    InvalidImageRecord(String),
    MissingFrameImage(PathBuf),
    UnsupportedCameraModel(String),
    Renderer(RendererError),
    Image(image::ImageError),
    MissingCameraId(u32),
}

impl From<std::io::Error> for VideoInitError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<SceneError> for VideoInitError {
    fn from(value: SceneError) -> Self {
        Self::Scene(value)
    }
}

impl From<RendererError> for VideoInitError {
    fn from(value: RendererError) -> Self {
        Self::Renderer(value)
    }
}

impl From<image::ImageError> for VideoInitError {
    fn from(value: image::ImageError) -> Self {
        Self::Image(value)
    }
}

pub trait CommandRunner {
    fn run(&mut self, program: &str, args: &[String]) -> Result<(), VideoInitError>;
}

#[derive(Debug, Default)]
pub struct SystemCommandRunner;

impl CommandRunner for SystemCommandRunner {
    fn run(&mut self, program: &str, args: &[String]) -> Result<(), VideoInitError> {
        let status = Command::new(program).args(args).status()?;
        if status.success() {
            Ok(())
        } else {
            Err(VideoInitError::CommandFailed {
                program: program.to_string(),
                args: args.to_vec(),
                status,
            })
        }
    }
}

#[derive(Debug)]
pub struct ColmapVideoInitializer<R> {
    runner: R,
    options: VideoInitOptions,
}

impl Default for ColmapVideoInitializer<SystemCommandRunner> {
    fn default() -> Self {
        Self::new(SystemCommandRunner, VideoInitOptions::default())
    }
}

impl<R: CommandRunner> ColmapVideoInitializer<R> {
    pub fn new(runner: R, options: VideoInitOptions) -> Self {
        Self { runner, options }
    }

    pub fn initialize_scene_from_video(
        &mut self,
        video_path: &Path,
        workspace: &Path,
    ) -> Result<Scene, VideoInitError> {
        let workspace_layout = self.prepare_workspace(video_path, workspace)?;
        load_scene_from_colmap_text_model(&workspace_layout.text_dir, &self.options)
    }

    pub fn initialize_and_train_from_video(
        &mut self,
        video_path: &Path,
        workspace: &Path,
    ) -> Result<Scene, VideoInitError> {
        let workspace_layout = self.prepare_workspace(video_path, workspace)?;
        let mut scene = load_scene_from_colmap_text_model(&workspace_layout.text_dir, &self.options)?;
        let frames = load_training_frames(&workspace_layout.text_dir, &workspace_layout.frames_dir)?;
        self.train_scene_on_frames(&mut scene, &frames)?;
        Ok(scene)
    }

    fn prepare_workspace(
        &mut self,
        video_path: &Path,
        workspace: &Path,
    ) -> Result<WorkspaceLayout, VideoInitError> {
        let frames_dir = workspace.join("frames");
        let sparse_dir = workspace.join("sparse");
        let refined_dir = workspace.join("refined");
        let text_dir = workspace.join("text");
        let database_path = workspace.join("database.db");

        if workspace.exists() {
            fs::remove_dir_all(workspace)?;
        }
        fs::create_dir_all(workspace)?;
        fs::create_dir_all(&frames_dir)?;
        fs::create_dir_all(&sparse_dir)?;
        fs::create_dir_all(&refined_dir)?;
        fs::create_dir_all(&text_dir)?;

        self.extract_frames(video_path, &frames_dir)?;
        self.run_feature_extractor(&database_path, &frames_dir)?;
        self.run_exhaustive_matcher(&database_path)?;
        self.run_mapper(&database_path, &frames_dir, &sparse_dir)?;

        let sparse_model = first_sparse_model_dir(&sparse_dir)?;
        self.run_image_registrator(&database_path, &sparse_model, &refined_dir)?;
        self.run_point_triangulator(&database_path, &frames_dir, &refined_dir)?;
        self.run_bundle_adjuster(&refined_dir)?;
        self.run_model_converter(&refined_dir, &text_dir)?;
        Ok(WorkspaceLayout { frames_dir, text_dir })
    }

    fn extract_frames(
        &mut self,
        video_path: &Path,
        frames_dir: &Path,
    ) -> Result<(), VideoInitError> {
        let frame_pattern = frames_dir.join("frame_%05d.png");
        self.runner.run(
            "ffmpeg",
            &[
                "-y".to_string(),
                "-i".to_string(),
                path_arg(video_path),
                "-vf".to_string(),
                format!("fps={}", self.options.fps),
                path_arg(&frame_pattern),
            ],
        )
    }

    fn run_feature_extractor(
        &mut self,
        database_path: &Path,
        frames_dir: &Path,
    ) -> Result<(), VideoInitError> {
        self.runner.run(
            "colmap",
            &[
                "feature_extractor".to_string(),
                "--database_path".to_string(),
                path_arg(database_path),
                "--image_path".to_string(),
                path_arg(frames_dir),
                "--ImageReader.single_camera".to_string(),
                "1".to_string(),
                "--SiftExtraction.max_num_features".to_string(),
                "16384".to_string(),
                "--SiftExtraction.estimate_affine_shape".to_string(),
                "1".to_string(),
                "--SiftExtraction.domain_size_pooling".to_string(),
                "1".to_string(),
            ],
        )
    }

    fn run_exhaustive_matcher(&mut self, database_path: &Path) -> Result<(), VideoInitError> {
        self.runner.run(
            "colmap",
            &[
                "exhaustive_matcher".to_string(),
                "--database_path".to_string(),
                path_arg(database_path),
                "--FeatureMatching.guided_matching".to_string(),
                "1".to_string(),
                "--FeatureMatching.max_num_matches".to_string(),
                "65536".to_string(),
            ],
        )?;

        self.runner.run(
            "colmap",
            &[
                "transitive_matcher".to_string(),
                "--database_path".to_string(),
                path_arg(database_path),
                "--FeatureMatching.guided_matching".to_string(),
                "1".to_string(),
                "--TransitiveMatching.num_iterations".to_string(),
                "5".to_string(),
            ],
        )
    }

    fn run_mapper(
        &mut self,
        database_path: &Path,
        frames_dir: &Path,
        sparse_dir: &Path,
    ) -> Result<(), VideoInitError> {
        self.runner.run(
            "colmap",
            &[
                "mapper".to_string(),
                "--database_path".to_string(),
                path_arg(database_path),
                "--image_path".to_string(),
                path_arg(frames_dir),
                "--output_path".to_string(),
                path_arg(sparse_dir),
                "--Mapper.multiple_models".to_string(),
                "0".to_string(),
                "--Mapper.max_num_models".to_string(),
                "1".to_string(),
                "--Mapper.min_model_size".to_string(),
                "2".to_string(),
                "--Mapper.init_min_num_inliers".to_string(),
                "30".to_string(),
                "--Mapper.init_max_error".to_string(),
                "12".to_string(),
                "--Mapper.init_min_tri_angle".to_string(),
                "4".to_string(),
                "--Mapper.init_max_reg_trials".to_string(),
                "20".to_string(),
                "--Mapper.abs_pose_min_num_inliers".to_string(),
                "8".to_string(),
                "--Mapper.abs_pose_min_inlier_ratio".to_string(),
                "0.02".to_string(),
                "--Mapper.abs_pose_max_error".to_string(),
                "24".to_string(),
                "--Mapper.max_reg_trials".to_string(),
                "20".to_string(),
                "--Mapper.filter_min_tri_angle".to_string(),
                "0.5".to_string(),
                "--Mapper.tri_min_angle".to_string(),
                "0.5".to_string(),
            ],
        )
    }

    fn run_model_converter(
        &mut self,
        sparse_model: &Path,
        text_dir: &Path,
    ) -> Result<(), VideoInitError> {
        self.runner.run(
            "colmap",
            &[
                "model_converter".to_string(),
                "--input_path".to_string(),
                path_arg(sparse_model),
                "--output_path".to_string(),
                path_arg(text_dir),
                "--output_type".to_string(),
                "TXT".to_string(),
            ],
        )
    }

    fn run_image_registrator(
        &mut self,
        database_path: &Path,
        sparse_model: &Path,
        refined_dir: &Path,
    ) -> Result<(), VideoInitError> {
        copy_dir_all(sparse_model, refined_dir)?;
        self.runner.run(
            "colmap",
            &[
                "image_registrator".to_string(),
                "--database_path".to_string(),
                path_arg(database_path),
                "--input_path".to_string(),
                path_arg(refined_dir),
                "--output_path".to_string(),
                path_arg(refined_dir),
                "--Mapper.fix_existing_frames".to_string(),
                "1".to_string(),
                "--Mapper.min_num_matches".to_string(),
                "6".to_string(),
                "--Mapper.abs_pose_min_num_inliers".to_string(),
                "8".to_string(),
                "--Mapper.abs_pose_min_inlier_ratio".to_string(),
                "0.02".to_string(),
                "--Mapper.abs_pose_max_error".to_string(),
                "24".to_string(),
                "--Mapper.max_reg_trials".to_string(),
                "20".to_string(),
            ],
        )
    }

    fn run_point_triangulator(
        &mut self,
        database_path: &Path,
        frames_dir: &Path,
        refined_dir: &Path,
    ) -> Result<(), VideoInitError> {
        self.runner.run(
            "colmap",
            &[
                "point_triangulator".to_string(),
                "--database_path".to_string(),
                path_arg(database_path),
                "--image_path".to_string(),
                path_arg(frames_dir),
                "--input_path".to_string(),
                path_arg(refined_dir),
                "--output_path".to_string(),
                path_arg(refined_dir),
                "--clear_points".to_string(),
                "0".to_string(),
                "--Mapper.filter_min_tri_angle".to_string(),
                "0.5".to_string(),
                "--Mapper.tri_min_angle".to_string(),
                "0.5".to_string(),
            ],
        )
    }

    fn run_bundle_adjuster(&mut self, refined_dir: &Path) -> Result<(), VideoInitError> {
        self.runner.run(
            "colmap",
            &[
                "bundle_adjuster".to_string(),
                "--input_path".to_string(),
                path_arg(refined_dir),
                "--output_path".to_string(),
                path_arg(refined_dir),
            ],
        )
    }

    fn train_scene_on_frames(
        &self,
        scene: &mut Scene,
        frames: &[TrainingFrame],
    ) -> Result<(), VideoInitError> {
        println!(
            "training on {} registered frames for {} epochs",
            frames.len(),
            self.options.train_epochs,
        );
        for epoch_index in 0..self.options.train_epochs {
            let mut epoch_total_loss = 0.0_f64;
            let mut epoch_rgb_loss = 0.0_f64;
            let mut epoch_distortion_loss = 0.0_f64;

            for (frame_index, frame) in frames.iter().enumerate() {
                let renderer = PerspectiveRenderer::with_distortion(
                    frame.camera.clone(),
                    self.options.distortion_lambda,
                );
                let result = renderer.train_step_without_neighbor_refresh(scene, &frame.target)?;
                epoch_total_loss += result.loss;
                epoch_rgb_loss += result.rgb_loss;
                epoch_distortion_loss += result.distortion_loss;

                println!(
                    "epoch {}/{} step {}/{} frame={} loss={:.6} rgb_loss={:.6} distortion_loss={:.6}",
                    epoch_index + 1,
                    self.options.train_epochs,
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
                "epoch {}/{} avg_loss={:.6} avg_rgb_loss={:.6} avg_distortion_loss={:.6}",
                epoch_index + 1,
                self.options.train_epochs,
                epoch_total_loss / frame_count,
                epoch_rgb_loss / frame_count,
                epoch_distortion_loss / frame_count,
            );
        }
        scene.compute_neighbors()?;
        Ok(())
    }
}

#[derive(Debug)]
struct WorkspaceLayout {
    frames_dir: PathBuf,
    text_dir: PathBuf,
}

fn path_arg(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn first_sparse_model_dir(sparse_root: &Path) -> Result<PathBuf, VideoInitError> {
    let mut candidates = fs::read_dir(sparse_root)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| VideoInitError::MissingSparseModel(sparse_root.to_path_buf()))
}

fn copy_dir_all(from: &Path, to: &Path) -> Result<(), VideoInitError> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let source = entry.path();
        let destination = to.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&source, &destination)?;
        } else {
            fs::copy(source, destination)?;
        }
    }
    Ok(())
}

fn load_scene_from_colmap_text_model(
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

fn load_training_frames(
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
                camera: camera_model.to_perspective_camera(image.world_to_camera_rotation, image.translation),
                target,
            })
        })
        .collect()
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

fn load_cameras(path: &Path) -> Result<HashMap<u32, ColmapCameraModel>, VideoInitError> {
    let contents =
        fs::read_to_string(path).map_err(|_| VideoInitError::MissingCameraFile(path.to_path_buf()))?;
    contents
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(parse_camera_record)
        .map(|result| result.map(|camera| (camera.id, camera)))
        .collect()
}

fn load_registered_images(path: &Path) -> Result<Vec<ColmapImageRecord>, VideoInitError> {
    let contents =
        fs::read_to_string(path).map_err(|_| VideoInitError::MissingImageFile(path.to_path_buf()))?;
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
struct TrainingFrame {
    name: String,
    camera: PerspectiveCamera,
    target: ImageData,
}

#[derive(Clone, Debug)]
struct ColmapCameraModel {
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
struct ColmapImageRecord {
    camera_id: u32,
    name: String,
    world_to_camera_rotation: [[f64; 3]; 3],
    translation: [f64; 3],
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

#[cfg(test)]
mod tests {
    use super::{
        ColmapVideoInitializer, CommandRunner, VideoInitError, VideoInitOptions,
        load_cameras, load_registered_images, load_scene_from_colmap_text_model,
    };
    use image::RgbImage;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Debug)]
    struct FakeRunner {
        calls: Vec<(String, Vec<String>)>,
    }

    impl FakeRunner {
        fn new() -> Self {
            Self { calls: Vec::new() }
        }
    }

    impl CommandRunner for FakeRunner {
        fn run(&mut self, program: &str, args: &[String]) -> Result<(), VideoInitError> {
            self.calls.push((program.to_string(), args.to_vec()));

            if program == "ffmpeg" {
                let frame_pattern = PathBuf::from(args.last().expect("frame pattern should exist"));
                let frames_dir = frame_pattern
                    .parent()
                    .expect("frame output should have parent")
                    .to_path_buf();
                fs::create_dir_all(&frames_dir)?;
                for frame_name in [
                    "frame_00001.png",
                    "frame_00005.png",
                    "frame_00009.png",
                    "frame_00013.png",
                    "frame_00017.png",
                ] {
                    RgbImage::from_raw(2, 2, vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0])
                        .expect("image should construct")
                        .save(frames_dir.join(frame_name))?;
                }
            }

            if program == "colmap" && args.first().map(String::as_str) == Some("mapper") {
                let sparse_dir = PathBuf::from(argument_value(args, "--output_path"));
                fs::create_dir_all(sparse_dir.join("0"))?;
                fs::write(sparse_dir.join("0").join("placeholder.bin"), "model")
                    .expect("placeholder model should write");
            }

            if program == "colmap" && args.first().map(String::as_str) == Some("image_registrator") {
                let refined_dir = PathBuf::from(argument_value(args, "--output_path"));
                fs::write(
                    refined_dir.join("registered_count.txt"),
                    "5",
                )?;
            }

            if program == "colmap" && args.first().map(String::as_str) == Some("model_converter") {
                let text_dir = PathBuf::from(argument_value(args, "--output_path"));
                let input_dir = PathBuf::from(argument_value(args, "--input_path"));
                fs::create_dir_all(&text_dir)?;
                let registered_count = fs::read_to_string(input_dir.join("registered_count.txt"))
                    .ok()
                    .and_then(|contents| contents.trim().parse::<usize>().ok())
                    .unwrap_or(3);
                let images_txt = (0..registered_count)
                    .map(|index| {
                        format!(
                            "{} 1 0 0 0 0 0 -{} 1 frame_{:05}.png\n\n",
                            index + 1,
                            index as f64 * 0.1,
                            1 + index * 4
                        )
                    })
                    .collect::<String>();
                fs::write(
                    text_dir.join("points3D.txt"),
                    "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n1 0.0 0.0 3.0 255 0 0 0.1\n2 0.5 0.1 4.0 255 0 0 0.2\n",
                )?;
                fs::write(
                    text_dir.join("cameras.txt"),
                    "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n1 PINHOLE 2 2 2.0 2.0 1.0 1.0\n",
                )?;
                fs::write(
                    text_dir.join("images.txt"),
                    format!("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n{images_txt}"),
                )?;
            }

            Ok(())
        }
    }

    #[test]
    fn load_scene_from_colmap_points_assigns_positions_and_colors() {
        let temp_dir = make_temp_dir("colmap-points");
        fs::write(
            temp_dir.join("points3D.txt"),
            "# header\n1 1.0 2.0 3.0 255 0 128 0.5\n2 -1.0 0.0 4.0 0 255 64 0.25\n",
        )
        .expect("point file should write");

        let scene = load_scene_from_colmap_text_model(&temp_dir, &VideoInitOptions::default())
            .expect("scene should parse");

        assert_eq!(scene.centroid_x.values, vec![1.0, -1.0]);
        assert_eq!(scene.centroid_y.values, vec![2.0, 0.0]);
        assert_eq!(scene.centroid_z.values, vec![3.0, 4.0]);
        assert!((scene.centroid_r.values[0] - 1.0).abs() < 1e-9);
        assert!((scene.centroid_g.values[1] - 1.0).abs() < 1e-9);
        assert!((scene.centroid_b.values[0] - (128.0 / 255.0)).abs() < 1e-9);

        fs::remove_dir_all(temp_dir).expect("temp dir should clean up");
    }

    #[test]
    fn initializer_runs_ffmpeg_and_colmap_pipeline() {
        let temp_dir = make_temp_dir("video-init");
        let video_path = temp_dir.join("scene.mp4");
        fs::write(&video_path, "placeholder").expect("video placeholder should write");

        let runner = FakeRunner::new();
        let mut initializer = ColmapVideoInitializer::new(runner, VideoInitOptions::default());
        let scene = initializer
            .initialize_scene_from_video(&video_path, &temp_dir.join("workspace"))
            .expect("initialization should succeed");

        assert_eq!(scene.centroid_x.len(), 2);
        assert_eq!(initializer.runner.calls.len(), 9);
        assert_eq!(initializer.runner.calls[0].0, "ffmpeg");
        assert_eq!(initializer.runner.calls[1].0, "colmap");
        assert_eq!(initializer.runner.calls[1].1[0], "feature_extractor");
        assert_eq!(initializer.runner.calls[2].1[0], "exhaustive_matcher");
        assert_eq!(initializer.runner.calls[3].1[0], "transitive_matcher");
        assert_eq!(initializer.runner.calls[4].1[0], "mapper");
        assert_eq!(initializer.runner.calls[5].1[0], "image_registrator");
        assert_eq!(initializer.runner.calls[6].1[0], "point_triangulator");
        assert_eq!(initializer.runner.calls[7].1[0], "bundle_adjuster");
        assert_eq!(initializer.runner.calls[8].1[0], "model_converter");
        assert!(initializer.runner.calls[2].1.contains(&"--FeatureMatching.guided_matching".to_string()));
        assert!(initializer.runner.calls[3].1.contains(&"--TransitiveMatching.num_iterations".to_string()));
        assert!(initializer.runner.calls[4].1.contains(&"--Mapper.multiple_models".to_string()));
        assert!(initializer.runner.calls[4].1.contains(&"0".to_string()));

        fs::remove_dir_all(temp_dir).expect("temp dir should clean up");
    }

    #[test]
    fn colmap_text_model_loads_cameras_and_registered_images() {
        let temp_dir = make_temp_dir("colmap-camera-load");
        fs::write(
            temp_dir.join("cameras.txt"),
            "# header\n1 SIMPLE_RADIAL 640 480 500 320 240 0.1\n",
        )
        .expect("camera file should write");
        fs::write(
            temp_dir.join("images.txt"),
            "# header\n1 1 0 0 0 0 0 -1 1 frame_00001.png\n0.0 0.0 -1\n",
        )
        .expect("images file should write");

        let cameras = load_cameras(&temp_dir.join("cameras.txt")).expect("cameras should parse");
        let images =
            load_registered_images(&temp_dir.join("images.txt")).expect("images should parse");

        assert_eq!(cameras.len(), 1);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].name, "frame_00001.png");
        assert_eq!(images[0].camera_id, 1);

        fs::remove_dir_all(temp_dir).expect("temp dir should clean up");
    }

    #[test]
    fn initializer_can_train_scene_from_registered_frames() {
        let temp_dir = make_temp_dir("video-train");
        let video_path = temp_dir.join("scene.mp4");
        fs::write(&video_path, "placeholder").expect("video placeholder should write");

        let mut options = VideoInitOptions::default();
        options.train_epochs = 1;
        let runner = FakeRunner::new();
        let mut initializer = ColmapVideoInitializer::new(runner, options);
        let scene = initializer
            .initialize_and_train_from_video(&video_path, &temp_dir.join("workspace"))
            .expect("video training should succeed");

        assert_eq!(scene.centroid_x.step, 5);
        assert_eq!(scene.centroid_r.step, 5);

        fs::remove_dir_all(temp_dir).expect("temp dir should clean up");
    }


    fn argument_value<'a>(args: &'a [String], flag: &str) -> &'a str {
        let index = args
            .iter()
            .position(|arg| arg == flag)
            .expect("flag should exist");
        &args[index + 1]
    }

    fn make_temp_dir(name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should advance")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("radiant-foam-{name}-{suffix}"));
        fs::create_dir_all(&path).expect("temp dir should create");
        path
    }
}
