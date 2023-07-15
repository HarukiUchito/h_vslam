use std::rc::Rc;

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

use anyhow::Result;
use log::debug;

use crate::error::SLAMError;
use crate::kitti_dataset;
use crate::map::Map;
use crate::map_point::MapPoint;

#[derive(Clone)]
pub struct Feature {
    pub position: KeyPoint,
}

impl Feature {
    pub fn new(kp: &KeyPoint) -> Feature {
        Feature { position: *kp }
    }
}

#[derive(Clone, Default)]
pub struct Frame {
    pub left_image: Mat,
    pub right_image: Mat,
    pub left_features: Vec<Rc<Feature>>,
    pub right_features: Vec<Option<Rc<Feature>>>,
    pub left_image_kps: Mat,
    pub right_image_kps: Mat,
}

impl Frame {
    pub fn load_image(
        mut self,
        left_img_path: &str,
        right_img_path: &str,
    ) -> Result<Self, opencv::Error> {
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

    pub fn find_keypoints(&mut self) -> Result<(), opencv::Error> {
        let left_kps = detect_features(&self.left_image, &None)?;
        for kp in left_kps.iter() {
            self.left_features.push(Rc::new(Feature::new(&kp)));
        }

        let right_kps_img = detect_features(&self.right_image, &Some(&self.left_features))?;
        self.right_features =
            detect_feature_movement(&self.left_features, &self.left_image, &self.right_image)?;

        assert!(self.left_features.len() == self.right_features.len());

        self.left_image_kps = draw_keypoints(&self.left_image, &left_kps)?;
        self.right_image_kps = draw_keypoints(&self.right_image, &right_kps_img)?;

        Ok(())
    }
}

fn detect_features(
    mat: &Mat,
    features: &Option<&Vec<Rc<Feature>>>,
) -> Result<VectorOfKeyPoint, opencv::Error> {
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
    features: &Vec<Rc<Feature>>,
    mat1: &Mat,
    mat2: &Mat,
) -> Result<Vec<Option<Rc<Feature>>>, opencv::Error> {
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
            features.push(Some(Rc::new(Feature::new(&KeyPoint::new_point(
                kp, 7.0, -1.0, 0.0, 0, -1,
            )?))));
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

fn triangulation(
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
        f_left.position.pt.x as f64,
        f_left.position.pt.y as f64,
    ]);
    let right_pos = yakf::linalg::Vector2::from_vec(vec![
        f_right.position.pt.x as f64,
        f_right.position.pt.y as f64,
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

#[test]
fn test_map_initialization() -> Result<()> {
    let mut dataset = kitti_dataset::KITTIDataset::new(std::path::PathBuf::from("./test/"));
    dataset.load_calib_file()?;

    let mut first_frame = dataset.get_frame()?;
    first_frame.find_keypoints()?;

    let left_camera = *dataset.get_camera(0).as_ref();
    let right_camera = *dataset.get_camera(1).as_ref();

    let poses = vec![left_camera.pose, right_camera.pose];

    let mut map = Map::new();

    let mut num_landmarks = 0;
    for i in 0..first_frame.left_features.len() {
        let f_left = &first_frame.left_features[i];
        let f_right = match &first_frame.right_features[i] {
            None => continue,
            Some(f) => f,
        };

        let left_pos = yakf::linalg::Vector2::from_vec(vec![
            f_left.position.pt.x as f64,
            f_left.position.pt.y as f64,
        ]);
        let right_pos = yakf::linalg::Vector2::from_vec(vec![
            f_right.position.pt.x as f64,
            f_right.position.pt.y as f64,
        ]);
        let points = vec![
            left_camera.pixel_to_camera(&left_pos, 1.),
            right_camera.pixel_to_camera(&right_pos, 1.),
        ];

        if let Ok(pt_world) = triangulation(&poses, &points) {
            println!("ptw: {}", pt_world);
            num_landmarks += 1;
            let new_id = map.add_new_map_point(&pt_world);
            map.add_observation(new_id, f_left)?;
            map.add_observation(new_id, f_right)?;
        }
    }
    println!("initial map created with {} map points", num_landmarks);
    //let points = vec![left_camera.pixel_to_camera(yakf::so2::Vec2::, depth)];

    Ok(())
}