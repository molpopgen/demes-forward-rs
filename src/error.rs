use thiserror::Error;

#[derive(Error, Debug)]
pub enum DemesForwardError {
    #[error("{0:?}")]
    DemesError(demes::DemesError),
    #[error("{0:?}")]
    TimeError(String),
}

impl From<demes::DemesError> for DemesForwardError {
    fn from(error: demes::DemesError) -> Self {
        Self::DemesError(error)
    }
}
