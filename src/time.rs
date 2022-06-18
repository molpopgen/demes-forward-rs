#[derive(Debug, Copy, Clone)]
pub struct ForwardTime(f64);

impl ForwardTime {
    pub fn valid(&self) -> bool {
        self.0.is_finite() && self.0.is_sign_positive()
    }

    pub fn new<F: Into<ForwardTime>>(value: F) -> Self {
        value.into()
    }
}

impl<T> From<T> for ForwardTime
where
    T: Into<f64>,
{
    fn from(value: T) -> Self {
        Self(value.into())
    }
}
