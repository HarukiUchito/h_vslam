use yakf::lie::so3::SO3;

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
}
