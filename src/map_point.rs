use std::rc::Rc;

use crate::frame::Feature;

pub struct MapPoint {
    id: usize,
    position: yakf::linalg::Vector3<f64>,
    observations: Vec<Rc<Feature>>,
}

impl MapPoint {
    pub fn new(id: usize, position: &yakf::linalg::Vector3<f64>) -> MapPoint {
        MapPoint {
            id: id,
            position: *position,
            observations: Vec::new(),
        }
    }

    pub fn add_observation(&mut self, feature_reference: Rc<Feature>) {
        self.observations.push(feature_reference);
    }
}
