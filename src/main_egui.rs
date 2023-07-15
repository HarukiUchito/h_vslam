//use std::fs;

use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use opencv::prelude::{MatTraitConst, MatTraitManual};
use tokio;

use anyhow::Result;
use env_logger;
use std::env;

mod camera;
mod cv_to_egui;
mod error;
mod frame;
mod frontend;
mod kitti_dataset;
mod map;
mod map_point;

struct SharedData {
    seq_dir_path: std::path::PathBuf,
    cnt: i32,
    img_bytes: Vec<u8>,
    color_image: egui::ColorImage,
    retained_image: std::sync::Arc<Texture2D>,
    status_text: String,
    frontend: frontend::FrontEnd,
    kitti_dataset: kitti_dataset::KITTIDataset,
}

impl SharedData {
    fn new(cnt: i32, context: &Context) -> Self {
        let seq_dir_path = std::path::PathBuf::from(
            "/home/xoke/Downloads/data_odometry_gray/dataset/sequences/05",
        );
        let mut dataset = kitti_dataset::KITTIDataset::new(seq_dir_path.clone());
        dataset.load_calib_file();

        let mut frontend = frontend::FrontEnd::new();
        frontend.set_cameras(dataset.get_camera(0), dataset.get_camera(1));

        let mut loaded = three_d_asset::io::load(&["test/image_0/000000.png"]).unwrap();
        let image = std::sync::Arc::new(Texture2D::new(
            &context,
            &loaded.deserialize("000000.png").unwrap(),
        ));

        SharedData {
            seq_dir_path: seq_dir_path.clone(),
            cnt: cnt,
            img_bytes: Vec::new(),
            color_image: egui::ColorImage::default(),
            retained_image: image,
            status_text: "none".to_string(),
            frontend: frontend,
            kitti_dataset: dataset,
        }
    }

    fn update(&mut self, context: &Context) -> Result<()> {
        self.cnt += 1;

        let new_frame = self.kitti_dataset.get_frame()?;
        self.frontend.update(&new_frame)?;
        let img = self.frontend.get_image()?;

        let mut cpimg = img.clone();
        cpimg = cpimg
            .reshape(1, img.total() as i32 * img.channels())
            .unwrap();
        let cpimg_v = cpimg.data_bytes_mut().unwrap();
        println!("cpimg_v: {}", cpimg_v.len());

        let sz = [img.cols() as usize, img.rows() as usize];
        println!("sz: {:?}", sz.clone());
        self.color_image = egui::ColorImage::from_rgb(sz, &cpimg_v);

        match cv_to_egui::image_vector(&img) {
            Ok(v) => {
                self.img_bytes.clear();
                for val in v {
                    self.img_bytes.push(val);
                }
                let mut loaded = three_d_asset::io::RawAssets::new();
                loaded.insert("cv.png", self.img_bytes.clone());
                self.img_bytes.clear();
                self.retained_image = std::sync::Arc::new(Texture2D::new(
                    &context,
                    &loaded.deserialize("cv.png").unwrap(),
                ))
            }
            Err(..) => self.status_text = "image update failed".to_string(),
        }

        Ok(())
    }
}

use three_d::core::*;
use three_d::*;

#[tokio::main]
async fn main() -> Result<()> {
    env::set_var("RUST_LOG", "debug");
    env_logger::init();

    let window = three_d::Window::new(three_d::WindowSettings {
        title: "Screen!".to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();
    //let (tx, rx) = mpsc::channel();

    let mut shared_data = SharedData::new(0, &context);
    //return;
    shared_data.update(&context)?;

    let shared_arc = Arc::new(Mutex::new(shared_data));
    //let shared_arc_main = Arc::clone(&shared_arc);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(10));

        //let ctx = rx.recv().unwrap();
        let mut repaint_cnt = 0;
        loop {
            println!("repaint cnt: {}", repaint_cnt);
            repaint_cnt += 1;
            if repaint_cnt > 9 {
                repaint_cnt = 0;
                //three_d::egui::Context::request_repaint(&ctx);
            }

            let ten_millis = std::time::Duration::from_millis(1000);
            std::thread::sleep(ten_millis);

            interval.tick().await; // ticks after 10ms
        }
    });

    let mut camera = three_d::Camera::new_perspective(
        window.viewport(),
        vec3(0.0, 0.0, 1.3),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        10.0,
    );

    // Load point cloud
    let mut loaded = three_d_asset::io::load_async(&["hand.pcd"]).await.unwrap();
    let cpu_point_cloud: three_d_asset::geometry::PointCloud =
        loaded.deserialize("hand.pcd").unwrap();

    let mut point_mesh = three_d::CpuMesh::sphere(4);
    point_mesh
        .transform(&three_d::Mat4::from_scale(0.001))
        .unwrap();
    let mut point_cloud = three_d::Gm {
        geometry: three_d::InstancedMesh::new(&context, &cpu_point_cloud.into(), &point_mesh),
        material: three_d::ColorMaterial::default(),
    };
    let c = -point_cloud.aabb().center();
    point_cloud.set_transformation(three_d::Mat4::from_translation(c));

    let mut control = OrbitControl::new(*camera.target(), 0.1, 3.0);

    let mut gui = three_d::GUI::new(&context);
    let mut ctx_sent = false;
    let mut repaint_cnt = 0;
    window.render_loop(move |mut frame_input| {
        let mut panel_width = 0.0;
        let mut center_rect = three_d::egui::Rect::NOTHING;
        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                use three_d::egui::*;
                Window::new("window")
                    .min_width(1000.0)
                    .min_height(100.0)
                    .show(gui_context, |ui| {
                        ui.label("Hello world!");
                        ui.allocate_space(ui.available_size());
                    });
                //ui.set_pixels_per_point(1.2); // zoom in
                containers::TopBottomPanel::top(Id::new("header")).show(gui_context, |ui| {
                    Grid::new("my_grid")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            //let guard = self.shared_data_arc.lock().expect("Couldn't get lock");
                            ui.label("seq path");
                            ui.label(WidgetText::from(
                                "test", //guard.seq_dir_path.to_str().unwrap_or("empty"),
                            ));
                            ui.end_row();
                            ui.label("status");
                            //ui.label(guard.status_text.as_str());
                        });
                });

                egui::containers::SidePanel::left(egui::Id::new("info")).show(gui_context, |ui| {
                    let guard = shared_arc.lock().expect("Couldn't get lock");
                    ui.heading("Info");
                    egui::Grid::new("my_grid")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("Time");
                            ui.label(WidgetText::from(
                                (guard.cnt as f64 / 100.0).to_string().as_str(),
                            ));
                            ui.end_row();

                            ui.label("Image Index");
                            ui.label(WidgetText::from(
                                "test",
                                //guard.kitti_dataset.get_img_index().to_string().as_str(),
                            ));
                            ui.end_row();

                            ui.label("Loading Progress");
                            let progress = 180.0 / 360.0;
                            let progress_bar = egui::ProgressBar::new(progress)
                                .show_percentage()
                                .animate(true);
                            ui.add(progress_bar)
                                .on_hover_text("The progress bar can be animated!")
                                .hovered();
                            ui.end_row();
                        });
                });
                egui::containers::CentralPanel::default().show(gui_context, |ui| {
                    egui::containers::TopBottomPanel::top(egui::Id::new("camera")).show(
                        gui_context,
                        |ui| {
                            ui.group(|ui| {
                                ui.heading("Camera");
                                let mut guard = shared_arc.lock().expect("Couldn't get lock");
                                let texture: egui::TextureHandle = ui.ctx().load_texture(
                                    "my-image",
                                    guard.color_image.clone(),
                                    Default::default(),
                                );
                                ui.image(&texture, texture.size_vec2());
                                //guard.retained_image.show(ui);
                                ui.set_min_width(200.0);
                            });

                            ui.set_height(200.0);
                            center_rect = ui.min_rect();
                            //println!("center used rect: {:?}", &center_rect);
                        },
                    );
                    egui::containers::TopBottomPanel::bottom(egui::Id::new("other")).show(
                        gui_context,
                        |ui| {
                            ui.heading("Control");
                            ui.horizontal(|ui| -> Result<()> {
                                //let mut guard =
                                //  self.shared_data_arc.lock().expect("Couldn't get lock");
                                let mut reload = false;
                                if ui.button("reset").clicked() {
                                    //guard.kitti_dataset.reset();
                                    reload = true;
                                }
                                if ui.button("next frame").clicked() {
                                    //guard.kitti_dataset.next_frame();
                                    reload = true;
                                }
                                if reload {
                                    //guard.update()?;
                                }
                                Ok(())
                            });

                            ui.group(|ui| {
                                ui.label("Within a frame");
                                ui.set_min_height(30.0);
                                ui.set_min_width(100.0);
                            });
                        },
                    );
                    //ui.allocate_space(ui.available_size());
                });
                if !ctx_sent {

                    //tx.send(.clone()).unwrap(); // (非同期)送信
                }
                panel_width = gui_context.used_rect().width();
                //println!("used rect: {:?}", gui_context.used_rect());
                panel_width = 10.0;
            },
        );
        let viewport = Viewport {
            x: ((222.3) * frame_input.device_pixel_ratio) as i32,
            y: (0.0 * frame_input.device_pixel_ratio) as i32,
            width: ((1000.0) * frame_input.device_pixel_ratio) as u32,
            height: ((200.0) * frame_input.device_pixel_ratio) as u32,
        };

        /*
        let mut texture_transform_scale = 1.0;
        let mut texture_transform_x = 0.0;
        let mut texture_transform_y = 0.0;
        {
            let guard = shared_arc.lock().expect("Couldn't get lock");
            let material = ColorMaterial {
                texture: Some(Texture2DRef {
                    texture: guard.retained_image.clone(),
                    transformation: Mat3::from_scale(texture_transform_scale)
                        * Mat3::from_translation(vec2(texture_transform_x, texture_transform_y)),
                }),
                ..Default::default()
            };
            frame_input
                .screen()
                .clear(ClearState::color_and_depth(1.0, 1.0, 1.0, 1.0, 1.0))
                //.render_partially(scissor_box_zoomed, &camera, &model, &[])
                .write(|| gui.render())
                .apply_screen_material(&material, &camera2d(viewport), &[]);
        }
        return FrameOutput {
            ..Default::default()
        };*/
        //println!("viewport: {:?}", frame_input.viewport);
        let viewport = Viewport {
            x: ((222.3) * frame_input.device_pixel_ratio) as i32,
            y: (0.0 * frame_input.device_pixel_ratio) as i32,
            width: ((1000.0) * frame_input.device_pixel_ratio) as u32,
            height: ((frame_input.viewport.height as f32 - 45.0) * frame_input.device_pixel_ratio)
                as u32,
        };
        //println!("viewport: {:?}", viewport);

        let viewport_zoomed = zoom(1.0, viewport);
        //let scissor_box_zoomed = zoom(scissor_zoom, viewport).into();
        camera.set_viewport(viewport_zoomed);
        frame_input
            .screen()
            .clear(ClearState::color_and_depth(1.0, 1.0, 1.0, 1.0, 1.0))
            //.render_partially(scissor_box_zoomed, &camera, &model, &[])
            .write(|| gui.render());

        let mut redraw = frame_input.first_frame;
        //redraw |= camera.set_viewport(frame_input.viewport);
        redraw |= camera.set_viewport(viewport);
        redraw |= control.handle_events(&mut camera, &mut frame_input.events);

        repaint_cnt += 1;
        if repaint_cnt > 9 {
            repaint_cnt = 0;
            redraw |= true;
        }

        frame_input
            .screen()
            .clear_partially(viewport.into(), ClearState::color(0.3, 0.3, 0.3, 1.0))
            .render_partially(
                viewport.into(),
                &camera,
                point_cloud
                    .into_iter()
                    .chain(&Axes::new(&context, 0.01, 0.1)),
                &[],
            );

        // Returns default frame output to end the frame
        FrameOutput {
            swap_buffers: redraw,
            ..Default::default()
        }
    });

    Ok(())
}

/*
#[tokio::main]
async fn main() -> Result<()> {
    env::set_var("RUST_LOG", "debug");
    env_logger::init();

    let (tx, rx) = mpsc::channel();

    let mut shared_data = SharedData::new(0);
    //return;
    shared_data.update()?;
    let shared_arc = Arc::new(Mutex::new(shared_data));
    //let shared_arc_main = Arc::clone(&shared_arc);
    let app = MyApp::new(tx, shared_arc);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(10));

        let ctx = rx.recv().unwrap();
        let mut repaint_cnt = 0;
        loop {
            repaint_cnt += 1;
            if repaint_cnt > 9 {
                repaint_cnt = 0;
                egui::Context::request_repaint(&ctx);
            }

            interval.tick().await; // ticks after 10ms
        }
    });

    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(1500.0, 500.0)),
        maximized: true,
        centered: true,
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    match eframe::run_native("SLAM viewer", options, Box::new(|_cc| Box::new(app))) {
        Ok(_) => (),
        Err(_) => (),
    }
    Ok(())
}
*/
fn zoom(zoom: f32, viewport: Viewport) -> Viewport {
    let width = (viewport.width as f32 * zoom) as u32;
    let height = (viewport.height as f32 * zoom) as u32;
    Viewport {
        x: ((viewport.width - width) / 2 + viewport.x as u32) as i32,
        y: ((viewport.height - height) / 2 + viewport.y as u32) as i32,
        width,
        height,
    }
}
