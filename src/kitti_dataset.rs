use crate::camera::Camera;
use crate::error::SLAMError;
use crate::frame::Frame;
use anyhow::Result;
use log::debug;
use std::rc::Rc;

pub struct KITTIDataset {
    dataset_path: std::path::PathBuf,
    cameras: Vec<Camera>,
    img_index: usize,
}

impl KITTIDataset {
    pub fn new(dataset_path: std::path::PathBuf) -> KITTIDataset {
        KITTIDataset {
            dataset_path: dataset_path,
            cameras: Vec::new(),
            img_index: 0,
        }
    }

    pub fn load_calib_file(&mut self) -> Result<(), SLAMError> {
        debug!("load calib.txt");

        let calib_file = self.dataset_path.join("calib.txt");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b' ')
            .flexible(true)
            .from_reader(std::fs::File::open(calib_file).unwrap());

        for result in rdr.records() {
            let mut first = true;
            let mut vs = Vec::new();
            for vstr in result.unwrap().iter() {
                if first {
                    first = false;
                    continue;
                }
                vs.push(vstr.parse::<f64>().unwrap());
            }

            let mut k_mat = yakf::linalg::Matrix3::<f64>::new(
                vs[0], vs[1], vs[2], vs[4], vs[5], vs[6], vs[8], vs[9], vs[10],
            );
            let t_vec = k_mat.try_inverse().unwrap()
                * yakf::linalg::Vector3::<f64>::new(vs[3], vs[7], vs[11]);
            k_mat *= 0.5;

            let pose = yakf::lie::so3::SO3::from_vec(t_vec);
            debug!("pose: {:?}", &pose);
            let pose = yakf::lie::se3::SE3::from_r_t(
                yakf::linalg::Matrix3::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                t_vec,
            );
            self.cameras.push(Camera::new(
                k_mat[(0, 0)],
                k_mat[(1, 1)],
                k_mat[(0, 2)],
                k_mat[(1, 2)],
                t_vec.norm(),
                pose,
            ));

            //debug!("{:.1}", &k_mat);
            //debug!("{:.5}", &t_vec);
        }
        Ok(())
    }

    pub fn get_frame(&self) -> Result<Frame, opencv::Error> {
        let left_image_path = self
            .dataset_path
            .join("image_0")
            .join(format!("{:06}.png", self.img_index));
        let right_image_path = self
            .dataset_path
            .join("image_1")
            .join(format!("{:06}.png", self.img_index));
        let frame = Frame::default().load_image(
            left_image_path.to_str().unwrap(),
            right_image_path.to_str().unwrap(),
        )?;
        Ok(frame)
    }

    pub fn get_camera(&self, index: usize) -> Rc<Camera> {
        Rc::new(self.cameras[index])
    }

    pub fn get_img_index(&self) -> usize {
        self.img_index
    }

    pub fn reset(&mut self) {
        self.img_index = 0;
    }

    pub fn next_frame(&mut self) {
        self.img_index += 1;
    }
}

#[test]
fn test_kitti_dataset() -> Result<()> {
    let mut dataset = KITTIDataset::new(std::path::PathBuf::from("./test/"));
    dataset.load_calib_file()?;

    let no_rotation =
        yakf::linalg::Matrix3::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let p0 = yakf::lie::se3::SE3::from_r_t(
        no_rotation,
        yakf::linalg::Vector3::from_vec(vec![0., 0., 0.]),
    );
    let p1 = yakf::lie::se3::SE3::from_r_t(
        no_rotation,
        yakf::linalg::Vector3::from_vec(vec![-0.537151, 0., 0.]),
    );
    let p2 = yakf::lie::se3::SE3::from_r_t(
        no_rotation,
        yakf::linalg::Vector3::from_vec(vec![0.0610306, -0.00143972, 0.00620322]),
    );
    let p3 = yakf::lie::se3::SE3::from_r_t(
        no_rotation,
        yakf::linalg::Vector3::from_vec(vec![-0.474418, 0.00187031, 0.0033185]),
    );

    approx::assert_relative_eq!(dataset.get_camera(0).pose.adj(), p0.adj(), epsilon = 1e-4);
    approx::assert_relative_eq!(dataset.get_camera(1).pose.adj(), p1.adj(), epsilon = 1e-4);
    approx::assert_relative_eq!(dataset.get_camera(2).pose.adj(), p2.adj(), epsilon = 1e-4);
    approx::assert_relative_eq!(dataset.get_camera(3).pose.adj(), p3.adj(), epsilon = 1e-4);

    Ok(())
}
