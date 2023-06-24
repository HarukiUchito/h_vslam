use eframe::egui::WidgetText;
use eframe::epaint::{Color32, ColorImage};
//use std::fs;

use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui;
use egui_extras::RetainedImage;
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

struct SharedData {
    seq_dir_path: std::path::PathBuf,
    cnt: i32,
    retained_image: RetainedImage,
    status_text: String,
    frontend: frontend::FrontEnd,
    kitti_dataset: kitti_dataset::KITTIDataset,
}

impl SharedData {
    fn new(cnt: i32) -> Self {
        let seq_dir_path = std::path::PathBuf::from(
            "/home/xoke/Downloads/data_odometry_gray/dataset/sequences/05",
        );
        let dataset = kitti_dataset::KITTIDataset::new(seq_dir_path.clone());

        let mut frontend = frontend::FrontEnd::new();
        frontend.set_cameras(dataset.get_camera(0), dataset.get_camera(1));

        let black = Color32::from_rgb(255, 255, 255);
        let black_image = ColorImage::new([2482, 376], black);
        SharedData {
            seq_dir_path: seq_dir_path.clone(),
            cnt: cnt,
            retained_image: RetainedImage::from_color_image("", black_image),
            status_text: "none".to_string(),
            frontend: frontend,
            kitti_dataset: dataset,
        }
    }

    fn update(&mut self) -> Result<()> {
        self.cnt += 1;

        let new_frame = self.kitti_dataset.get_frame()?;
        self.frontend.update(&new_frame)?;
        let img = self.frontend.get_image()?;

        match cv_to_egui::image_vector(&img) {
            Ok(v) => {
                self.retained_image = RetainedImage::from_image_bytes("img", &v.as_slice()).unwrap()
            }
            Err(..) => self.status_text = "image update failed".to_string(),
        }

        Ok(())
    }
}

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
        ..Default::default()
    };
    match eframe::run_native("SLAM viewer", options, Box::new(|_cc| Box::new(app))) {
        Ok(_) => (),
        Err(_) => (),
    }
    Ok(())
}

struct MyApp {
    tx: Sender<egui::Context>,
    ctx_sent: bool,
    shared_data_arc: Arc<Mutex<SharedData>>,
}

impl MyApp {
    fn new(tx: Sender<egui::Context>, shared_data_arc: Arc<Mutex<SharedData>>) -> Self {
        Self {
            tx: tx,
            ctx_sent: false,
            shared_data_arc: shared_data_arc,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(1.2); // zoom in
        egui::containers::TopBottomPanel::top(egui::Id::new("header")).show(ctx, |ui| {
            egui::Grid::new("my_grid")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    let guard = self.shared_data_arc.lock().expect("Couldn't get lock");
                    ui.label("seq path");
                    ui.label(WidgetText::from(
                        guard.seq_dir_path.to_str().unwrap_or("empty"),
                    ));
                    ui.end_row();
                    ui.label("status");
                    ui.label(guard.status_text.as_str());
                });
        });
        egui::containers::SidePanel::left(egui::Id::new("info")).show(ctx, |ui| {
            let guard = self.shared_data_arc.lock().expect("Couldn't get lock");
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
                        guard.kitti_dataset.get_img_index().to_string().as_str(),
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
        egui::containers::CentralPanel::default().show(ctx, |ui| {
            egui::containers::TopBottomPanel::top(egui::Id::new("camera")).show(ctx, |ui| {
                ui.group(|ui| {
                    ui.heading("Camera");
                    let guard = self.shared_data_arc.lock().expect("Couldn't get lock");
                    guard.retained_image.show(ui);
                    //ui.set_min_width(200.0);
                });
            });
            egui::containers::TopBottomPanel::bottom(egui::Id::new("other")).show(ctx, |ui| {
                ui.heading("Control");
                ui.horizontal(|ui| -> Result<()> {
                    let mut guard = self.shared_data_arc.lock().expect("Couldn't get lock");
                    let mut reload = false;
                    if ui.button("reset").clicked() {
                        guard.kitti_dataset.reset();
                        reload = true;
                    }
                    if ui.button("next frame").clicked() {
                        guard.kitti_dataset.next_frame();
                        reload = true;
                    }
                    if reload {
                        guard.update()?;
                    }
                    Ok(())
                });

                ui.group(|ui| {
                    ui.label("Within a frame");
                    ui.set_min_height(30.0);
                    ui.set_min_width(100.0);
                });
            });
            ui.allocate_space(ui.available_size());
        });
        if !self.ctx_sent {
            self.tx.send(ctx.clone()).unwrap(); // (非同期)送信
        }
    }
}
