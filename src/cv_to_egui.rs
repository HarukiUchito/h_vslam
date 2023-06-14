use opencv::core;
use opencv::core::Mat;
use opencv::core::Scalar;
use opencv::core::Vector;
use opencv::features2d::*;
use opencv::imgcodecs;
use opencv::imgproc::INTER_LINEAR;
use opencv::prelude::Feature2DTrait;
use opencv::types::VectorOfKeyPoint;

use opencv::core::KeyPoint;

use env_logger;
use log::debug;
use opencv::types::VectorOfPoint2f;
use opencv::types::VectorOfVectorOfPoint2f;

pub struct FrontEnd {
    last_frame: Option<Frame>,
    current_frame: Option<Frame>,
}

impl FrontEnd {
    pub fn new() -> FrontEnd {
        FrontEnd {
            last_frame: None,
            current_frame: None,
        }
    }
    pub fn addFrame(&mut self, new_frame: &Frame) {
        self.last_frame = self.current_frame.clone();
        self.current_frame = Some(new_frame.clone());
    }

    pub fn track(&self) {
        let Some(last_frame) = self.last_frame.as_ref() else { return };
        let Some(current_frame) = self.current_frame.as_ref() else { return };

        let mut last_keypoints = VectorOfPoint2f::new();
        for kp in last_frame.left_features.iter() {
            last_keypoints.push(opencv::core::Point2f::new(
                kp.position.pt.x,
                kp.position.pt.y,
            ));
        }
        let mut curr_keypoints = VectorOfPoint2f::new();
        for kp in current_frame.left_features.iter() {
            curr_keypoints.push(opencv::core::Point2f::new(
                kp.position.pt.x,
                kp.position.pt.y,
            ));
        }

        let mut err = Mat::default();
        let mut status: opencv::core::Vector<u8> = Vec::new().into();
        opencv::video::calc_optical_flow_pyr_lk(
            &last_frame.left_image,
            &current_frame.left_image,
            &mut last_keypoints,
            &mut curr_keypoints,
            &mut status,
            &mut err,
            opencv::core::Size::new(11, 11),
            3,
            opencv::core::TermCriteria::new(
                core::TermCriteria_Type::COUNT as i32 + core::TermCriteria_Type::EPS as i32,
                30,
                0.01,
            )
            .unwrap(),
            opencv::video::OPTFLOW_USE_INITIAL_FLOW,
            1e-4,
        );

        debug!("{:?}", status);
    }

    pub fn getImage(&mut self) -> Result<Mat, opencv::Error> {
        let mut current_frame = self.current_frame.as_mut().unwrap();
        current_frame.find_keypoints();

        //    let mut rgb_img2 = opencv::core::Mat::default();
        //    opencv::imgproc::cvt_color(&img2, &mut rgb_img2, opencv::imgproc::COLOR_GRAY2RGB, 0)?;

        // concat left, right images
        let mut lr_img = opencv::core::Mat::default();
        let mut vec = opencv::types::VectorOfMat::new();
        vec.push(current_frame.left_image_kps.clone());
        vec.push(current_frame.right_image_kps.clone());

        opencv::core::hconcat(&vec, &mut lr_img).unwrap();
        //println!("hcon w: {}, h: {}", lr_img.cols(), lr_img.rows());

        // resize
        let mut resized = opencv::core::Mat::default();
        opencv::imgproc::resize(
            &lr_img,
            &mut resized,
            opencv::core::Size::new(0, 0),
            0.4,
            0.4,
            INTER_LINEAR,
        )?;

        Ok(resized)
    }
}

#[derive(Clone)]
struct Feature {
    position: KeyPoint,
}

impl Feature {
    fn new(kp: &KeyPoint) -> Feature {
        Feature { position: *kp }
    }
}

#[derive(Clone)]
pub struct Frame {
    left_image: Mat,
    right_image: Mat,
    left_features: Vec<Feature>,
    left_image_kps: Mat,
    right_image_kps: Mat,
}

impl Frame {
    pub fn new(left_img_path: &str, right_img_path: &str) -> Frame {
        Frame {
            left_image: imgcodecs::imread(left_img_path, 0).unwrap(),
            right_image: imgcodecs::imread(right_img_path, 0).unwrap(),
            left_features: Vec::new(),
            left_image_kps: Mat::default(),
            right_image_kps: Mat::default(),
        }
    }

    pub fn find_keypoints(&mut self) {
        let left_kps = detect_features(&self.left_image).unwrap();
        let right_kps = detect_features(&self.right_image).unwrap();

        for kp in left_kps.iter() {
            self.left_features.push(Feature::new(&kp));
        }

        self.left_image_kps = draw_keypoints(&self.left_image, &left_kps).unwrap();
        self.right_image_kps = draw_keypoints(&self.right_image, &right_kps).unwrap();
    }
}

fn detect_features(mat: &Mat) -> Result<VectorOfKeyPoint, opencv::Error> {
    let num_features = 100000;
    let mut gftt =
        <dyn opencv::features2d::GFTTDetector>::create(num_features, 0.01, 1.0, 3, false, 0.04)?;
    let mask = Mat::default();
    let mut keypoints = VectorOfKeyPoint::new();
    gftt.detect(&mat, &mut keypoints, &mask)?;

    debug!("max features: {}", gftt.get_max_features()?);
    debug!("num keypoints: {}", keypoints.len());
    //println!("{:#?}", keypoints);
    //for kp in keypoints.iter() {}
    return Ok(keypoints);
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

pub fn image_vector(img: &Mat) -> Result<Vector<u8>, opencv::Error> {
    let mut image_vector = Vector::<u8>::new();
    let params = Vector::from_slice(&[0, 95]);
    imgcodecs::imencode(".png", &img, &mut image_vector, &params)?;
    Ok(image_vector)
}
