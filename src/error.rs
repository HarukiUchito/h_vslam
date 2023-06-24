use std::error;
use std::fmt;

#[derive(Debug, Clone)]
pub struct SLAMError {
    pub message: String,
}

impl SLAMError {
    #[inline]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SLAMError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SLAM error: {}", self.message)
    }
}

// This is important for other errors to wrap this one.
impl error::Error for SLAMError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        // Generic error, underlying cause isn't tracked.
        // 基本となるエラー、原因は記録されていない。
        None
    }
}
