use opencv::core::Mat;
use opencv::core::Vector;
use opencv::imgcodecs;

pub fn image_vector(img: &Mat) -> Result<Vector<u8>, opencv::Error> {
    let mut image_vector = Vector::<u8>::new();
    let params = Vector::from_slice(&[0, 95]);
    imgcodecs::imencode(".png", &img, &mut image_vector, &params)?;
    Ok(image_vector)
}
