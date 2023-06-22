use yakf::lie::{
    se3::SE3,
    so2::Vec2,
    so3::{Vec3, SO3},
};

#[derive(Copy, Clone)]
pub struct Camera {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    baseline: f64,
    pose: SO3,
}

impl Camera {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, baseline: f64, pose: SO3) -> Camera {
        Camera {
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            baseline: baseline,
            pose: pose,
        }
    }

    // coordinate transform functions
    pub fn world_to_camera(&self, p_w: &Vec3, t_c_w: &SE3) -> Vec3 {
        self.pose.act_v(t_c_w.act_v(*p_w))
    }
    pub fn pixel_to_camera(&self, p_p: &Vec2, depth: f64) -> Vec3 {
        Vec3::new(
            (p_p[0] - self.cx) * depth / self.fx,
            (p_p[1] - self.cy) * depth / self.fy,
            depth,
        )
    }
}
