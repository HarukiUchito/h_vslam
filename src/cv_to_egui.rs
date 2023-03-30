use opencv::core::Vector;
use opencv::imgcodecs;
use opencv::imgproc::INTER_LINEAR;
use opencv::prelude::MatTraitConst;

pub fn image_vector(
    left_image_path: &str,
    right_image_path: &str,
) -> Result<Vector<u8>, opencv::Error> {
    let img1 = imgcodecs::imread(left_image_path, 0).unwrap();
    let img2 = imgcodecs::imread(right_image_path, 0).unwrap();

    let mut lr_img = opencv::core::Mat::default();
    println!("img1 w: {}, h: {}", img1.cols(), img1.rows());
    let mut vec = opencv::types::VectorOfMat::new();
    vec.push(img1);
    vec.push(img2);
    opencv::core::hconcat(&vec, &mut lr_img)?;
    println!("hcon w: {}, h: {}", lr_img.cols(), lr_img.rows());

    let mut resized = opencv::core::Mat::default();
    opencv::imgproc::resize(
        &lr_img,
        &mut resized,
        opencv::core::Size::new(0, 0),
        0.4,
        0.4,
        INTER_LINEAR,
    )?;

    let mut image_vector = Vector::<u8>::new();
    let params = Vector::from_slice(&[0, 95]);
    imgcodecs::imencode(".png", &resized, &mut image_vector, &params)?;
    Ok(image_vector)
}
