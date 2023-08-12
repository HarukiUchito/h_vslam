use anyhow::Result;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::{error::SLAMError, frame::Feature, frame::Frame, map_point::MapPoint};

pub struct Map {
    pub landmarks: HashMap<usize, MapPoint>, // id -> MapPoint
    keyframes: HashMap<usize, Rc<RefCell<Frame>>>,
    current_frame: Option<Rc<RefCell<Frame>>>,
}

impl Map {
    pub fn new() -> Map {
        Map {
            landmarks: HashMap::new(),
            keyframes: HashMap::new(),
            current_frame: None,
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

    pub fn add_keyframe(&mut self, frame: &Rc<RefCell<Frame>>) -> Result<()> {
        self.current_frame = Some(Rc::clone(frame));
        self.keyframes
            .insert(frame.borrow().key_frame_id, Rc::clone(frame));
        Ok(())
    }
}
