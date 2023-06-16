use crate::camera::Camera;
use log::debug;
use std::{fmt::Debug, io};

pub struct KITTIDataset {
    dataset_path: std::path::PathBuf,
    cameras: Vec<Camera>,
    img_index: usize,
}

impl KITTIDataset {
    pub fn new(dataset_path: std::path::PathBuf) -> KITTIDataset {
        debug!("load calib.txt");

        let calib_file = dataset_path.join("calib.txt");
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b' ')
            .flexible(true)
            .from_reader(std::fs::File::open(calib_file).unwrap());

        let mut cameras = Vec::new();
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
            cameras.push(Camera::new(
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

        KITTIDataset {
            dataset_path: dataset_path,
            cameras: cameras,
            img_index: 0,
        }
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
