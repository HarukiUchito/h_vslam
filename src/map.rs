use anyhow::Result;
use std::{collections::HashMap, rc::Rc};

use crate::{error::SLAMError, frame::Feature, map_point::MapPoint};

pub struct Map {
    landmarks: HashMap<usize, MapPoint>, // id -> MapPoint
}

impl Map {
    pub fn new() -> Map {
        Map {
            landmarks: HashMap::new(),
        }
    }

    pub fn add_new_map_point(&mut self, position: &yakf::linalg::Vector3<f64>) -> usize {
        let new_id = self.landmarks.len();
        let new_map_point = MapPoint::new(new_id, position);
        self.landmarks.insert(new_id, new_map_point);
        new_id
    }

    pub fn add_observation(&mut self, id: usize, obs: &Rc<Feature>) -> Result<()> {
        match self.landmarks.get_mut(&id) {
            Some(l) => {
                l.add_observation(Rc::clone(obs));
                Ok(())
            }
            None => Err(SLAMError::new("no corresponding landmark").into()),
        }
    }
}
