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

struct Feature {
    position: KeyPoint,
}

impl Feature {
    fn new(kp: &KeyPoint) -> Feature {
        Feature { position: *kp }
    }
}

struct Frame {
    left_features: Vec<Feature>,
}

impl Frame {
    fn new() -> Frame {
        Frame {
            left_features: Vec::new(),
        }
    }
}

fn draw_keypoints(mat: &Mat) -> Result<Mat, opencv::Error> {
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

    let mut frame = Frame::new();
    for kp in keypoints.iter() {
        frame.left_features.push(Feature::new(&kp));
    }

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

pub fn image_vector(
    left_image_path: &str,
    right_image_path: &str,
) -> Result<Vector<u8>, opencv::Error> {
    let img1 = imgcodecs::imread(left_image_path, 0).unwrap();
    let img2 = imgcodecs::imread(right_image_path, 0).unwrap();

    let k_img1 = draw_keypoints(&img1).unwrap();
    let k_img2 = draw_keypoints(&img2).unwrap();

    //    let mut rgb_img2 = opencv::core::Mat::default();
    //    opencv::imgproc::cvt_color(&img2, &mut rgb_img2, opencv::imgproc::COLOR_GRAY2RGB, 0)?;

    // concat left, right images
    let mut lr_img = opencv::core::Mat::default();
    let mut vec = opencv::types::VectorOfMat::new();
    vec.push(k_img1);
    vec.push(k_img2);

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

    let mut image_vector = Vector::<u8>::new();
    let params = Vector::from_slice(&[0, 95]);
    imgcodecs::imencode(".png", &resized, &mut image_vector, &params)?;
    Ok(image_vector)
}
