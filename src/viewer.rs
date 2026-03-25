use crate::renderer::{PerspectiveCamera, PerspectiveRenderer, Renderer};
use crate::scene::Scene;
use minifb::{Key, MouseMode, Window, WindowOptions};
use std::path::Path;
use std::time::Instant;

const VIEW_WIDTH: usize = 320;
const VIEW_HEIGHT: usize = 180;
const MOVE_SPEED: f64 = 3.0;
const MOUSE_SENSITIVITY: f64 = 0.003;
const FOCAL_SCALE: f64 = 0.9;
const PITCH_LIMIT: f64 = 1.5533430342749532;

pub fn run_scene_viewer(scene_path: &Path) -> Result<(), String> {
    let scene = Scene::load_from_json(scene_path)
        .map_err(|error| format!("failed to load scene {}: {error:?}", scene_path.display()))?;
    let mut viewer = Viewer::new(scene)?;
    viewer.run()
}

struct Viewer {
    scene: Scene,
    controller: CameraController,
    window: Window,
    frame_buffer: Vec<u32>,
    previous_mouse_position: Option<(f32, f32)>,
}

impl Viewer {
    fn new(scene: Scene) -> Result<Self, String> {
        let mut window = Window::new(
            "radiant foam viewer",
            VIEW_WIDTH,
            VIEW_HEIGHT,
            WindowOptions::default(),
        )
        .map_err(|error| format!("failed to create viewer window: {error}"))?;
        window.set_target_fps(60);
        window.set_cursor_visibility(false);

        Ok(Self {
            scene,
            controller: CameraController::default(),
            window,
            frame_buffer: vec![0; VIEW_WIDTH * VIEW_HEIGHT],
            previous_mouse_position: None,
        })
    }

    fn run(&mut self) -> Result<(), String> {
        let mut previous_frame = Instant::now();

        while self.window.is_open() && !self.window.is_key_down(Key::Escape) {
            let now = Instant::now();
            let delta_seconds = (now - previous_frame).as_secs_f64().min(0.05);
            previous_frame = now;

            self.update_controls(delta_seconds);
            self.render_frame()?;
        }

        Ok(())
    }

    fn update_controls(&mut self, delta_seconds: f64) {
        if let Some((mouse_x, mouse_y)) = self.window.get_mouse_pos(MouseMode::Discard) {
            if let Some((previous_x, previous_y)) = self.previous_mouse_position {
                self.controller
                    .turn((mouse_x - previous_x) as f64, (mouse_y - previous_y) as f64);
            }
            self.previous_mouse_position = Some((mouse_x, mouse_y));
        } else {
            self.previous_mouse_position = None;
        }

        let movement = MovementInput {
            forward: axis(
                self.window.is_key_down(Key::W),
                self.window.is_key_down(Key::S),
            ),
            strafe: axis(
                self.window.is_key_down(Key::D),
                self.window.is_key_down(Key::A),
            ),
            vertical: axis(
                self.window.is_key_down(Key::Space),
                self.window.is_key_down(Key::LeftShift)
                    || self.window.is_key_down(Key::RightShift),
            ),
        };
        self.controller.step(movement, delta_seconds);
    }

    fn render_frame(&mut self) -> Result<(), String> {
        let renderer = PerspectiveRenderer::new(self.controller.to_camera());
        let image = renderer
            .render(&self.scene)
            .map_err(|error| format!("failed to render scene: {error:?}"))?;

        for (dst, rgb) in self.frame_buffer.iter_mut().zip(image.pixels.chunks_exact(3)) {
            *dst = ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | rgb[2] as u32;
        }

        self.window
            .update_with_buffer(&self.frame_buffer, VIEW_WIDTH, VIEW_HEIGHT)
            .map_err(|error| format!("failed to update viewer window: {error}"))
    }
}

fn axis(positive: bool, negative: bool) -> f64 {
    match (positive, negative) {
        (true, false) => 1.0,
        (false, true) => -1.0,
        _ => 0.0,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct MovementInput {
    forward: f64,
    strafe: f64,
    vertical: f64,
}

#[derive(Clone, Debug)]
struct CameraController {
    position: [f64; 3],
    yaw: f64,
    pitch: f64,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, -4.0],
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

impl CameraController {
    fn step(&mut self, movement: MovementInput, delta_seconds: f64) {
        let flat_forward = [self.yaw.sin(), 0.0, self.yaw.cos()];
        let flat_right = [flat_forward[2], 0.0, -flat_forward[0]];
        let speed = MOVE_SPEED * delta_seconds;

        self.position[0] += speed * (movement.forward * flat_forward[0] + movement.strafe * flat_right[0]);
        self.position[1] += speed * movement.vertical;
        self.position[2] += speed * (movement.forward * flat_forward[2] + movement.strafe * flat_right[2]);
    }

    fn turn(&mut self, delta_x: f64, delta_y: f64) {
        self.yaw += delta_x * MOUSE_SENSITIVITY;
        self.pitch = (self.pitch - delta_y * MOUSE_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    fn to_camera(&self) -> PerspectiveCamera {
        let forward = [
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        ];
        let right = normalize(cross([0.0, 1.0, 0.0], forward));
        let up = normalize(cross(forward, right));
        let width = VIEW_WIDTH as u32;
        let height = VIEW_HEIGHT as u32;
        let focal_x = width as f64 * FOCAL_SCALE;
        let focal_y = height as f64 * FOCAL_SCALE;

        PerspectiveCamera {
            width,
            height,
            focal_x,
            focal_y,
            principal_x: width as f64 * 0.5,
            principal_y: height as f64 * 0.5,
            origin: self.position,
            camera_to_world: [
                [right[0], up[0], forward[0]],
                [right[1], up[1], forward[1]],
                [right[2], up[2], forward[2]],
            ],
        }
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::{CameraController, MovementInput};

    #[test]
    fn camera_controller_moves_forward_flat_on_w() {
        let mut controller = CameraController::default();

        controller.step(
            MovementInput {
                forward: 1.0,
                strafe: 0.0,
                vertical: 0.0,
            },
            1.0,
        );

        assert!(controller.position[2] > -4.0);
        assert_eq!(controller.position[1], 0.0);
    }

    #[test]
    fn camera_controller_vertical_motion_is_decoupled_from_pitch() {
        let mut controller = CameraController::default();
        controller.turn(0.0, -200.0);

        controller.step(
            MovementInput {
                forward: 0.0,
                strafe: 0.0,
                vertical: 1.0,
            },
            1.0,
        );

        assert!(controller.position[1] > 0.0);
        assert_eq!(controller.position[0], 0.0);
    }

    #[test]
    fn camera_controller_turn_changes_forward_direction() {
        let mut controller = CameraController::default();

        controller.turn(200.0, 0.0);
        let camera = controller.to_camera();

        assert!(camera.camera_to_world[0][2].abs() > 0.0);
    }
}
