use yakf::lie::{
    se3::SE3,
    so3::{Vec3, SO3},
};

#[derive(Debug, Copy, Clone)]
pub struct Camera {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    baseline: f64,
    pub pose: SE3,
}

impl Camera {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, baseline: f64, pose: SE3) -> Camera {
        Camera {
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            baseline: baseline,
            pose: pose,
        }
    }

    pub fn intrinsic_matrix(&self) -> nalgebra::Matrix3::<f64> {
        nalgebra::Matrix3::<f64>::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    // coordinate transform functions
    pub fn world_to_camera(&self, p_w: &Vec3, t_c_w: &SE3) -> Vec3 {
        self.pose.act_v(t_c_w.act_v(*p_w))
    }
    pub fn world_to_pixel(&self, p_w: &Vec3, t_c_w: &SE3) -> Vec3 {
        self.camera_to_pixel(&self.world_to_camera(p_w, t_c_w))
    }
    pub fn pixel_to_camera(&self, p_p: &yakf::linalg::Vector2<f64>, depth: f64) -> Vec3 {
        Vec3::new(
            (p_p[0] - self.cx) * depth / self.fx,
            (p_p[1] - self.cy) * depth / self.fy,
            depth,
        )
    }
    pub fn camera_to_pixel(&self, p_c: &Vec3) -> Vec3 {
        Vec3::new(
            self.fx * p_c.x / p_c.z + self.cx,
            self.fy * p_c.y / p_c.z + self.cy,
            0.0,
        )
    }
}
