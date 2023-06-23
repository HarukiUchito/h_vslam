use opencv::core::Mat;
use opencv::core::Point2f;
use opencv::core::Scalar;
use opencv::features2d::*;
use opencv::imgcodecs;
use opencv::prelude::Feature2DTrait;
use opencv::prelude::MatTraitConstManual;
use opencv::types::VectorOfKeyPoint;
use opencv::types::VectorOfPoint2f;

use opencv::core::KeyPoint;

use log::debug;

#[derive(Clone)]
pub struct Feature {
    pub position: KeyPoint,
}

impl Feature {
    pub fn new(kp: &KeyPoint) -> Feature {
        Feature { position: *kp }
    }
}

#[derive(Clone)]
pub struct Frame {
    pub left_image: Mat,
    pub right_image: Mat,
    pub left_features: Vec<Feature>,
    pub left_image_kps: Mat,
    pub right_image_kps: Mat,
}

impl Frame {
    pub fn new(left_img_path: &str, right_img_path: &str) -> Frame {
        let left_img =
            imgcodecs::imread(left_img_path, opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();
        let right_img =
            imgcodecs::imread(right_img_path, opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();

        let mut left_resized = opencv::core::Mat::default();
        let mut right_resized = opencv::core::Mat::default();
        opencv::imgproc::resize(
            &left_img,
            &mut left_resized,
            opencv::core::Size::new(0, 0),
            0.5,
            0.5,
            opencv::imgproc::INTER_NEAREST,
        )
        .unwrap();
        opencv::imgproc::resize(
            &right_img,
            &mut right_resized,
            opencv::core::Size::new(0, 0),
            0.5,
            0.5,
            opencv::imgproc::INTER_NEAREST,
        )
        .unwrap();
        Frame {
            left_image: left_resized,
            right_image: right_resized,
            left_features: Vec::new(),
            left_image_kps: Mat::default(),
            right_image_kps: Mat::default(),
        }
    }

    pub fn find_keypoints(&mut self) {
        let left_kps = detect_features(&self.left_image, &None).unwrap();
        for kp in left_kps.iter() {
            self.left_features.push(Feature::new(&kp));
        }

        let right_kps_img = detect_features(&self.right_image, &Some(&self.left_features)).unwrap();
        let right_kps =
            detect_feature_movement(&self.left_features, &self.left_image, &self.right_image);
        for kp in right_kps.unwrap().iter() {}

        self.left_image_kps = draw_keypoints(&self.left_image, &left_kps).unwrap();
        self.right_image_kps = draw_keypoints(&self.right_image, &right_kps_img).unwrap();
    }
}

fn detect_features(
    mat: &Mat,
    features: &Option<&Vec<Feature>>,
) -> Result<VectorOfKeyPoint, opencv::Error> {
    let num_features = 150;
    let mut gftt =
        <dyn opencv::features2d::GFTTDetector>::create(num_features, 0.01, 20.0, 3, false, 0.04)?;
    let sz = mat.size().unwrap();
    let mut mask = unsafe {
        Mat::new_size_with_default(
            sz,
            opencv::core::CV_8UC1,
            opencv::core::Scalar::new(255.0, 255.0, 255.0, 255.0),
        )
    }
    .unwrap();
    if let Some(features) = features {
        for f in features.iter() {
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
    features: &Vec<Feature>,
    mat1: &Mat,
    mat2: &Mat,
) -> Result<Vec<Option<Feature>>, opencv::Error> {
    // prepare float keypoints for optical-flow
    let mut fkps1 = VectorOfPoint2f::new();
    let mut fkps2 = VectorOfPoint2f::new();
    for kp in features.iter() {
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
        )
        .unwrap(),
        opencv::video::OPTFLOW_USE_INITIAL_FLOW,
        1e-4,
    );

    let mut features = Vec::new();
    let mut cnt = 0;
    for i in 0..status.len() {
        let s = status.get(i).unwrap();
        if s > 0 {
            cnt += 1;
            let kp = fkps2.get(i).unwrap();
            features.push(Some(Feature::new(
                &KeyPoint::new_point(kp, 7.0, -1.0, 0.0, 0, -1).unwrap(),
            )));
        } else {
            features.push(None);
        }
    }
    debug!("number of keypoints in right image: {}", cnt);

    Ok(features)
}

fn draw_keypoints(mat: &Mat, keypoints: &VectorOfKeyPoint) -> Result<Mat, opencv::Error> {
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
