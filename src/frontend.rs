use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

use crate::camera::Camera;
use crate::error::SLAMError;
use crate::frame::{Feature, Frame};
use crate::kitti_dataset;
use crate::map::Map;
use anyhow::Result;

use opencv::core::{KeyPoint, Mat, Point2f};
use opencv::imgproc::INTER_LINEAR;

use log::debug;

#[derive(Debug)]
enum FrontendStatus {
    INITIALIZATION,
    TRACKING,
    LOST,
}

pub struct FrontEnd {
    status: FrontendStatus,
    last_frame: Option<Rc<RefCell<Frame>>>,
    current_frame: Option<Rc<RefCell<Frame>>>,
    left_camera: Option<Rc<Camera>>,
    right_camera: Option<Rc<Camera>>,
    image_output: Mat,
    pub map: Map,

    relative_motion: yakf::lie::se3::SE3,
}

impl FrontEnd {
    pub fn new() -> FrontEnd {
        FrontEnd {
            status: FrontendStatus::INITIALIZATION,
            last_frame: None,
            current_frame: None,
            left_camera: None,
            right_camera: None,
            image_output: Mat::default(),
            map: Map::new(),
            relative_motion: yakf::lie::se3::SE3::from_r_t(
                yakf::linalg::Matrix3::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                yakf::linalg::Vector3::<f64>::new(0.0, 0.0, 0.0),
            ),
        }
    }

    pub fn set_cameras(&mut self, left_camera: Rc<Camera>, right_camera: Rc<Camera>) {
        self.left_camera = Some(left_camera);
        self.right_camera = Some(right_camera);
    }

    pub fn update(&mut self, new_frame: &Rc<RefCell<Frame>>) -> Result<()> {
        debug!("[frontend update]");
        debug!("status: {:?}", self.status);
        self.last_frame = self.current_frame.clone();
        //self.last_frame = Some(Rc::clone(&self.current_frame.as_ref().unwrap()));
        self.current_frame = Some(Rc::clone(new_frame));
        match self.status {
            FrontendStatus::INITIALIZATION => {
                self.initialize()?;
            }
            FrontendStatus::TRACKING => self.track(),
            FrontendStatus::LOST => (),
        }

        let current_frame = Rc::clone(new_frame);
        //current_frame.deref().borrow_mut().find_keypoints()?;
        //current_frame.ok_or(SLAMError::new("set frame before get_image"))?;

        //    let mut rgb_img2 = opencv::core::Mat::default();
        //    opencv::imgproc::cvt_color(&img2, &mut rgb_img2, opencv::imgproc::COLOR_GRAY2RGB, 0)?;

        // concat left, right images
        let mut lr_img = opencv::core::Mat::default();
        let mut vec = opencv::types::VectorOfMat::new();
        vec.push(current_frame.borrow().left_image_kps.clone());
        vec.push(current_frame.borrow().right_image_kps.clone());

        opencv::core::hconcat(&vec, &mut lr_img)?;
        //println!("hcon w: {}, h: {}", lr_img.cols(), lr_img.rows());

        // resize
        opencv::imgproc::resize(
            &lr_img,
            &mut self.image_output,
            opencv::core::Size::new(0, 0),
            0.8,
            0.8,
            INTER_LINEAR,
        )?;

        Ok(())
    }

    fn initialize(&mut self) -> Result<()> {
        if self.left_camera.is_none() || self.right_camera.is_none() {
            return Err(SLAMError::new("set camera object before initialization").into());
        }
        if let Some(current_frame) = &self.current_frame {
            //.as_deref()
            //.ok_or(SLAMError::new("set frame before initialization"))?;
            current_frame.deref().borrow_mut().find_keypoints()?;

            current_frame.deref().borrow_mut().set_as_keyframe(0)?;
            let num_landmarks = initialize_map(
                &mut self.map,
                &Rc::clone(&self.current_frame.as_ref().unwrap()),
                &self.left_camera.as_ref().unwrap().as_ref(),
                &self.right_camera.as_ref().unwrap().as_ref(),
            )?;

            self.status = FrontendStatus::TRACKING;
        }

        Ok(())
    }

    fn track(&self) {
        if let Some(last_frame) = &self.last_frame {
            if let Some(current_frame) = &self.current_frame {
                current_frame.deref().borrow_mut().pose =
                    self.relative_motion.act_g(last_frame.borrow().pose);
            }
        }

        self.track_last_frame();

        unimplemented!();
    }

    fn track_last_frame(&self) -> Result<()> {
        // prepare float keypoints for optical-flow
        let mut last_kps = opencv::types::VectorOfPoint2f::new();
        let mut current_kps = opencv::types::VectorOfPoint2f::new();

        let last_frame = Rc::clone(&self.last_frame.as_ref().unwrap());
        let last_frame = last_frame.borrow();
        let current_frame = Rc::clone(&self.current_frame.as_ref().unwrap());
        let current_frame = current_frame.borrow();
        debug!("last f len: {:?}", last_frame.left_features.len());
        for kp in last_frame.left_features.iter() {
            let kp = kp.borrow();
            let left_camera = self.left_camera.as_ref().unwrap();
            //println!("kp: {}, {}", kp.position.pt.x, kp.position.pt.y);
            if let Some(mp_id) = kp.map_point_id {
                let mp = &self.map.landmarks[&mp_id];
                let px = left_camera.world_to_pixel(&mp.position, &current_frame.pose);
                //self.left_camera.unwrap().world_to_camera(p_w, t_c_w)
                last_kps.push(kp.position.pt); // just push the keypoint in mat1
                current_kps.push(Point2f::new(px.x as f32, px.y as f32));
                //println!("px: {}, {}", px.x, px.y);
            } else {
                last_kps.push(kp.position.pt);
                current_kps.push(kp.position.pt);
            }
        }

        debug!(
            "kps len last: {}, current: {}",
            last_kps.len(),
            current_kps.len()
        );

        let mut err = Mat::default();
        let mut status: opencv::core::Vector<u8> = Vec::new().into();
        opencv::video::calc_optical_flow_pyr_lk(
            &last_frame.left_image,
            &current_frame.left_image,
            &mut last_kps,
            &mut current_kps,
            &mut status,
            &mut err,
            opencv::core::Size::new(11, 11),
            3,
            opencv::core::TermCriteria::new(
                opencv::core::TermCriteria_Type::COUNT as i32
                    + opencv::core::TermCriteria_Type::EPS as i32,
                30,
                0.01,
            )?,
            opencv::video::OPTFLOW_USE_INITIAL_FLOW,
            1e-4,
        )?;

        let mut features = Vec::new();
        let mut cnt = 0;
        for i in 0..status.len() {
            let s = status.get(i)?;
            if s != 0 {
                cnt += 1;
                let kp = current_kps.get(i)?;
                features.push(Some(Rc::new(RefCell::new(Feature::new(
                    &KeyPoint::new_point(kp, 7.0, -1.0, 0.0, 0, -1)?,
                )))));
            } else {
                features.push(None);
            }
        }
        debug!(
            "number of keypoints in last image: {}, status len: {}, features len: {}",
            cnt,
            status.len(),
            features.len(),
        );

        Ok(())
    }

    pub fn get_image(&self) -> Result<Mat> {
        Ok(self.image_output.clone())
    }
}

pub fn triangulation(
    poses: &Vec<yakf::lie::se3::SE3>,
    points: &Vec<yakf::linalg::Vector3<f64>>,
) -> Result<yakf::linalg::Vector3<f64>> {
    let l = 2 * poses.len();
    let mut mat_a = nalgebra::DMatrix::<f64>::from_iterator(l, 4, vec![0.0; l * 4]);

    for i in 0..poses.len() {
        let m_pair = poses[i].to_r_t();
        let mut m = m_pair.0.insert_column(3, 0.);
        m.column_mut(3).copy_from(&m_pair.1);

        let na_mat_m =
            nalgebra::DMatrix::from_iterator(m.shape().0, m.shape().1, m.as_slice().to_vec());
        let ps =
            nalgebra::DVector::<f64>::from_iterator(points[i].len(), points[i].as_slice().to_vec());
        mat_a
            .row_mut(2 * i)
            .copy_from(&(ps[0] * na_mat_m.row(2) - na_mat_m.row(0)));
        mat_a
            .row_mut(2 * i + 1)
            .copy_from(&(ps[1] * na_mat_m.row(2) - na_mat_m.row(1)));
    }

    //println!("mat_a\n{}", &mat_a);

    let svd = nalgebra::linalg::SVD::new(mat_a, true, true);
    if svd.singular_values[3] / svd.singular_values[2] >= 1e-2 {
        return Err(SLAMError::new("bad singular value on triangulation").into());
    }

    let v_t = svd.v_t.unwrap();
    //println!("{}", &v_t);
    //println!("{}", &svd.u.unwrap());

    Ok(unsafe {
        std::mem::transmute::<nalgebra::Vector3<f64>, yakf::linalg::Vector3<f64>>(
            nalgebra::Vector3::from_column_slice(
                (v_t.transpose().column(3) / v_t[(3, 3)])
                    .fixed_rows::<3>(0)
                    .column(0)
                    .as_slice(),
            ),
        )
    })
}

#[test]
fn test_triangulation() -> Result<()> {
    let mut dataset = kitti_dataset::KITTIDataset::new(std::path::PathBuf::from("./test/"));
    dataset.load_calib_file()?;

    let mut first_frame = dataset.get_frame()?;
    first_frame.find_keypoints()?;

    let left_camera = *dataset.get_camera(0).as_ref();
    let right_camera = *dataset.get_camera(1).as_ref();

    let poses = vec![left_camera.pose, right_camera.pose];

    let index = 0; // just check triangulation result for the first feature

    let f_left = &first_frame.left_features[index];
    let f_right = match &first_frame.right_features[index] {
        None => return Err(SLAMError::new("corresponding right-side feature not found").into()),
        Some(f) => f,
    };

    let left_pos = yakf::linalg::Vector2::from_vec(vec![
        f_left.borrow().position.pt.x as f64,
        f_left.borrow().position.pt.y as f64,
    ]);
    let right_pos = yakf::linalg::Vector2::from_vec(vec![
        f_right.borrow().position.pt.x as f64,
        f_right.borrow().position.pt.y as f64,
    ]);
    let points = vec![
        left_camera.pixel_to_camera(&left_pos, 1.),
        right_camera.pixel_to_camera(&right_pos, 1.),
    ];

    let ans = yakf::linalg::Vector3::from_vec(vec![
        8.830560218052856,
        -2.2832203135803057,
        19.752454592502367,
    ]);

    match triangulation(&poses, &points) {
        Ok(pt_world) => approx::assert_relative_eq!(pt_world, ans, epsilon = 1e-4),
        Err(e) => return Err(e),
    }

    Ok(())
}

fn initialize_map(
    map: &mut Map,
    frame: &Rc<RefCell<Frame>>,
    left_camera: &Camera,
    right_camera: &Camera,
) -> Result<usize> {
    let poses = vec![left_camera.pose, right_camera.pose];
    let mut num_landmarks = 0;
    let frame_b = frame.borrow();
    for i in 0..frame_b.left_features.len() {
        let f_left = &frame_b.left_features[i];
        let f_right = match &frame_b.right_features[i] {
            None => continue,
            Some(f) => f,
        };

        let left_pos = yakf::linalg::Vector2::from_vec(vec![
            f_left.borrow().position.pt.x as f64,
            f_left.borrow().position.pt.y as f64,
        ]);
        let right_pos = yakf::linalg::Vector2::from_vec(vec![
            f_right.borrow().position.pt.x as f64,
            f_right.borrow().position.pt.y as f64,
        ]);
        let points = vec![
            left_camera.pixel_to_camera(&left_pos, 1.),
            right_camera.pixel_to_camera(&right_pos, 1.),
        ];

        if let Ok(pt_world) = triangulation(&poses, &points) {
            //println!("ptw: {}", pt_world);
            num_landmarks += 1;
            let new_id = map.add_new_map_point(&pt_world);
            map.add_observation(new_id, f_left)?;
            map.add_observation(new_id, f_right)?;

            frame_b.left_features[i].deref().borrow_mut().map_point_id = Some(new_id);
            if let Some(r_feature) = &frame_b.right_features[i] {
                r_feature.deref().borrow_mut().map_point_id = Some(new_id);
            }
        }
    }
    println!("initial map created with {} map points", num_landmarks);
    //let points = vec![left_camera.pixel_to_camera(yakf::so2::Vec2::, depth)];

    map.add_keyframe(&Rc::clone(frame))?;
    Ok(num_landmarks)
}

#[test]
fn test_map_initialization() -> Result<()> {
    let mut dataset = kitti_dataset::KITTIDataset::new(std::path::PathBuf::from("./test/"));
    dataset.load_calib_file()?;

    let mut first_frame = dataset.get_frame()?;
    first_frame.find_keypoints()?;

    let left_camera = *dataset.get_camera(0).as_ref();
    let right_camera = *dataset.get_camera(1).as_ref();

    let mut map = Map::new();
    first_frame.set_as_keyframe(0)?;
    let num_landmarks = initialize_map(
        &mut map,
        &Rc::new(RefCell::new(first_frame)),
        &left_camera,
        &right_camera,
    )?;

    assert_eq!(num_landmarks, 79);

    Ok(())
}
