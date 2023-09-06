use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::future;
use futures::stream::StreamExt;
use opencv::prelude::{MatTraitConst, MatTraitManual};
use r2r::QosProfile;

use futures::executor::LocalPool;
use futures::task::LocalSpawnExt;

mod camera;
mod cv_to_egui;
mod error;
mod frame;
mod frontend;
mod kitti_dataset;
mod map;
mod map_point;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let seq_dir_path =
        std::path::PathBuf::from("/home/xoke/Downloads/data_odometry_gray/dataset/sequences/05");
    let mut dataset = kitti_dataset::KITTIDataset::new(seq_dir_path.clone());
    dataset.load_calib_file()?;

    let mut frontend = frontend::FrontEnd::new();
    frontend.set_cameras(dataset.get_camera(0), dataset.get_camera(1));
    let new_frame = dataset.get_frame()?;
    frontend.update(&Rc::new(RefCell::new(new_frame)))?;
    dataset.next_frame();
    let new_frame = dataset.get_frame()?;
    frontend.update(&Rc::new(RefCell::new(new_frame)))?;

    let dataset_arc = Arc::new(Mutex::new(dataset));

    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "testnode", "")?;
    let duration = std::time::Duration::from_millis(100);

    let mut pool = LocalPool::new();
    let spawner = pool.spawner();

    // for sub2 we just print the data
    let dataset_arc0 = Arc::clone(&dataset_arc);
    let sub2 =
        node.subscribe::<r2r::std_msgs::msg::Bool>("/button_pushed", QosProfile::default())?;
    spawner.spawn_local(async move {
        sub2.for_each(|msg| {
            dataset_arc0.lock().unwrap().next_frame();
            println!("topic2: new msg: {}", msg.data);
            future::ready(())
        })
        .await
    })?;

    let mut timer = node.create_wall_timer(duration)?;
    let publisher =
        node.create_publisher::<r2r::std_msgs::msg::String>("/topic", QosProfile::default())?;
    let pub2 =
        node.create_publisher::<r2r::sensor_msgs::msg::Image>("/img", QosProfile::default())?;

    let pub3 =
        node.create_publisher::<r2r::sensor_msgs::msg::PointCloud2>("/pcs", QosProfile::default())?;

    // following instantiation must be there for some reason [TODO]
    let handle = tokio::task::spawn_blocking(move || loop {
        node.spin_once(std::time::Duration::from_millis(100));
    });

    let mut i = 0;
    let dataset_arc1 = Arc::clone(&dataset_arc);
    loop {
        let new_frame = dataset_arc1.lock().unwrap().get_frame()?;
        frontend.update(&Rc::new(RefCell::new(new_frame)))?;

        i += 1;
        timer.tick().await?;
        let msg = r2r::std_msgs::msg::String {
            data: "hello from r2r".to_string() + &i.to_string(),
        };
        publisher.publish(&msg)?;

        let mut img = r2r::sensor_msgs::msg::Image::default();
        img.encoding = "8UC3".to_string();
        let fimg = frontend.get_image().unwrap();
        img.data = Vec::<u8>::from(fimg.clone().data_bytes_mut().unwrap());
        img.width = fimg.cols() as u32;
        img.height = fimg.rows() as u32;
        img.step = img.width * 8;
        pub2.publish(&img)?;

        let mut cloud = Vec::new();
        for lm in frontend.map.landmarks.values() {
            cloud.push(ros_pointcloud2::pcl_utils::PointXYZ {
                x: lm.position[0] as f32,
                y: lm.position[1] as f32,
                z: lm.position[2] as f32,
            });
        }
        let internal_cloud: ros_pointcloud2::ros_types::PointCloud2Msg =
            ros_pointcloud2::ConvertXYZ::try_from(cloud)
                .unwrap()
                .try_into()
                .unwrap();
        let mut msg_cloud: r2r::sensor_msgs::msg::PointCloud2 = internal_cloud.into();
        msg_cloud.header.frame_id = "map".to_string();
        pub3.publish(&msg_cloud)?;

        pool.run_until_stalled();
    }

    handle.await?;

    Ok(())
}
