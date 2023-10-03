use std::borrow::Borrow;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

use anyhow::Ok;
use opencv::core::Mat;
use opencv::core::Point2f;
use opencv::core::Scalar;
use opencv::features2d::*;
use opencv::flann::feature_index;
use opencv::imgcodecs;
use opencv::prelude::Feature2DTrait;
use opencv::prelude::MatTraitConstManual;
use opencv::types::VectorOfKeyPoint;
use opencv::types::VectorOfPoint2f;

use opencv::core::KeyPoint;

use anyhow::Result;
use log::debug;

use crate::error::SLAMError;
use crate::kitti_dataset;
use crate::map::Map;
use crate::map_point::MapPoint;

#[derive(Clone)]
pub struct Feature {
    pub position: KeyPoint,
    pub map_point_id: Option<usize>,
    pub is_outlier: bool,
}

impl Feature {
    pub fn new(kp: &KeyPoint) -> Feature {
        Feature {
            position: *kp,
            map_point_id: None,
            is_outlier: false,
        }
    }
}

#[derive(Clone)]
pub struct Frame {
    pub left_image: Mat,
    pub right_image: Mat,

    pub left_features: Vec<Rc<RefCell<Feature>>>,
    pub right_features: Vec<Option<Rc<RefCell<Feature>>>>,

    pub left_image_kps: Mat,
    pub right_image_kps: Mat,

    pub key_frame_id: usize,
    pub is_key_frame: bool,

    pub pose: yakf::lie::se3::SE3,
}

impl Default for Frame {
    fn default() -> Frame {
        Frame {
            left_image: Mat::default(),
            right_image: Mat::default(),
            left_features: Vec::default(),
            right_features: Vec::default(),
            left_image_kps: Mat::default(),
            right_image_kps: Mat::default(),
            key_frame_id: 0,
            is_key_frame: false,
            pose: yakf::lie::se3::SE3::from_r_t(
                yakf::linalg::Matrix3::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                yakf::linalg::Vector3::<f64>::new(0.0, 0.0, 0.0),
            ),
        }
    }
}

impl Frame {
    pub fn load_image(mut self, left_img_path: &str, right_img_path: &str) -> Result<Self> {
        let left_img = imgcodecs::imread(left_img_path, opencv::imgcodecs::IMREAD_GRAYSCALE)?;
        let right_img = imgcodecs::imread(right_img_path, opencv::imgcodecs::IMREAD_GRAYSCALE)?;

        opencv::imgproc::resize(
            &left_img,
            &mut self.left_image,
            opencv::core::Size::new(0, 0),
            0.5,
            0.5,
            opencv::imgproc::INTER_NEAREST,
        )?;
        opencv::imgproc::resize(
            &right_img,
            &mut self.right_image,
            opencv::core::Size::new(0, 0),
            0.5,
            0.5,
            opencv::imgproc::INTER_NEAREST,
        )?;

        Ok(self)
    }

    pub fn set_as_keyframe(&mut self, id: usize) -> Result<()> {
        self.is_key_frame = true;
        self.key_frame_id = id;
        Ok(())
    }

    pub fn find_keypoints(&mut self) -> Result<()> {
        self.find_left_keypoints()?;
        self.find_right_keypoints()?;

        assert!(self.left_features.len() == self.right_features.len());

        self.draw_keypoints_from_features(true)?;
        self.draw_keypoints_from_features(false)?;

        Ok(())
    }

    pub fn find_left_keypoints(&mut self) -> Result<()> {
        let left_kps = detect_features(&self.left_image, &None)?;
        for kp in left_kps.iter() {
            self.left_features
                .push(Rc::new(RefCell::new(Feature::new(&kp))));
        }
        Ok(())
    }

    pub fn draw_keypoints_from_features(&mut self, is_left: bool) -> Result<()> {
        let mut keypoints = VectorOfKeyPoint::new();
        if is_left {
            for feat in self.left_features.iter() {
                keypoints.push(feat.clone().deref().borrow().position.clone());
            }
            self.left_image_kps = draw_keypoints(&self.left_image, &keypoints)?;
        } else {
            for feat in self.right_features.iter() {
                if let Some(feat) = feat.as_ref() {
                    keypoints.push(feat.clone().deref().borrow().position.clone());
                }
            }
            self.right_image_kps = draw_keypoints(&self.right_image, &keypoints)?;
        }
        Ok(())
    }

    pub fn find_right_keypoints(&mut self) -> Result<()> {
        let right_kps_img = detect_features(&self.right_image, &Some(&self.left_features))?;
        self.right_features =
            detect_feature_movement(&self.left_features, &self.left_image, &self.right_image)?;
        Ok(())
    }
}

fn detect_features(
    mat: &Mat,
    features: &Option<&Vec<Rc<RefCell<Feature>>>>,
) -> Result<VectorOfKeyPoint> {
    let num_features = 150;
    let mut gftt =
        <dyn opencv::features2d::GFTTDetector>::create(num_features, 0.01, 20.0, 3, false, 0.04)?;
    let mut mask = Mat::new_size_with_default(
        mat.size().unwrap_or_default(),
        opencv::core::CV_8UC1,
        opencv::core::Scalar::new(255.0, 255.0, 255.0, 255.0),
    )?;
    if let Some(features) = features {
        for f in features.iter() {
            let f = f.deref().borrow();
            let p1 = f.position.pt - Point2f::new(10.0, 10.0);
            opencv::imgproc::rectangle(
                &mut mask,
                opencv::core::Rect::new(p1.x as i32, p1.y as i32, 10, 10),
                opencv::core::Scalar::new(0.0, 0.0, 0.0, 0.0),
                1,
                opencv::imgproc::LINE_8,
                0,
            )?;
        }
    }

    let mut keypoints = VectorOfKeyPoint::new();
    gftt.detect(&mat, &mut keypoints, &mask)?;

    debug!("max features: {}", gftt.get_max_features()?);
    debug!("num keypoints: {}", keypoints.len());
    //println!("{:#?}", keypoints);
    //for kp in keypoints.iter() {}
    return Ok(keypoints);
}

fn detect_feature_movement(
    features: &Vec<Rc<RefCell<Feature>>>,
    mat1: &Mat,
    mat2: &Mat,
) -> Result<Vec<Option<Rc<RefCell<Feature>>>>> {
    // prepare float keypoints for optical-flow
    let mut fkps1 = VectorOfPoint2f::new();
    let mut fkps2 = VectorOfPoint2f::new();
    for kp in features.iter() {
        let kp = kp.deref().borrow();
        fkps1.push(kp.position.pt.clone()); // just push the keypoint in mat1
        fkps2.push(kp.position.pt);
    }

    let mut err = Mat::default();
    let mut status: opencv::core::Vector<u8> = Vec::new().into();
    opencv::video::calc_optical_flow_pyr_lk(
        &mat1,
        &mat2,
        &mut fkps1,
        &mut fkps2,
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
        if s > 0 {
            cnt += 1;
            let kp = fkps2.get(i)?;
            features.push(Some(Rc::new(RefCell::new(Feature::new(
                &KeyPoint::new_point(kp, 7.0, -1.0, 0.0, 0, -1)?,
            )))));
        } else {
            features.push(None);
        }
    }
    debug!(
        "number of keypoints in right image: {}, status len: {}, features len: {}",
        cnt,
        status.len(),
        features.len(),
    );

    Ok(features)
}

fn draw_keypoints(mat: &Mat, keypoints: &VectorOfKeyPoint) -> Result<Mat> {
    let mut out_image = Mat::default();
    opencv::features2d::draw_keypoints(
        &mat,
        &keypoints,
        &mut out_image,
        Scalar::all(-1.0),
        DrawMatchesFlags::DEFAULT,
    )?;

    Ok(out_image)
}
