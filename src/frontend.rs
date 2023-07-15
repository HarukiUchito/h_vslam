use std::rc::Rc;

use crate::camera::Camera;
use crate::error::SLAMError;
use crate::frame::Frame;
use anyhow::Result;

use opencv::core::Mat;
use opencv::imgproc::INTER_LINEAR;

use log::debug;

#[derive(Debug)]
enum FrontendStatus {
    INITIALIZATION,
    TRACKING,
    LOST,
}

pub struct FrontEnd {
    status: FrontendStatus,
    last_frame: Option<Frame>,
    current_frame: Option<Frame>,
    left_camera: Option<Rc<Camera>>,
    right_camera: Option<Rc<Camera>>,
    image_output: Mat,
}

impl FrontEnd {
    pub fn new() -> FrontEnd {
        FrontEnd {
            status: FrontendStatus::INITIALIZATION,
            last_frame: None,
            current_frame: None,
            left_camera: None,
            right_camera: None,
            image_output: Mat::default(),
        }
    }

    pub fn set_cameras(&mut self, left_camera: Rc<Camera>, right_camera: Rc<Camera>) {
        self.left_camera = Some(left_camera);
        self.right_camera = Some(right_camera);
    }

    pub fn update(&mut self, new_frame: &Frame) -> Result<()> {
        debug!("[frontend update]");
        debug!("status: {:?}", self.status);
        self.last_frame = self.current_frame.clone();
        self.current_frame = Some(new_frame.clone());
        match self.status {
            FrontendStatus::INITIALIZATION => {
                self.initialize()?;
                self.status = FrontendStatus::TRACKING;
            }
            FrontendStatus::TRACKING => (), //self.track(),
            FrontendStatus::LOST => (),
        }

        let current_frame = self
            .current_frame
            .as_mut()
            .ok_or(SLAMError::new("set frame before get_image"))?;
        current_frame.find_keypoints()?;

        //    let mut rgb_img2 = opencv::core::Mat::default();
        //    opencv::imgproc::cvt_color(&img2, &mut rgb_img2, opencv::imgproc::COLOR_GRAY2RGB, 0)?;

        // concat left, right images
        let mut lr_img = opencv::core::Mat::default();
        let mut vec = opencv::types::VectorOfMat::new();
        vec.push(current_frame.left_image_kps.clone());
        vec.push(current_frame.right_image_kps.clone());

        opencv::core::hconcat(&vec, &mut lr_img)?;
        //println!("hcon w: {}, h: {}", lr_img.cols(), lr_img.rows());

        // resize
        opencv::imgproc::resize(
            &lr_img,
            &mut self.image_output,
            opencv::core::Size::new(0, 0),
            0.8,
            0.8,
            INTER_LINEAR,
        )?;
        Ok(())
    }

    fn initialize(&mut self) -> Result<()> {
        if self.left_camera.is_none() || self.right_camera.is_none() {
            return Err(SLAMError::new("set camera object before initialization").into());
        }
        let current_frame = self
            .current_frame
            .as_mut()
            .ok_or(SLAMError::new("set frame before initialization"))?;
        current_frame.find_keypoints()?;

        Ok(())
    }

    fn track(&self) {
        unimplemented!();
    }

    pub fn get_image(&self) -> Result<Mat> {
        Ok(self.image_output.clone())
    }
}
