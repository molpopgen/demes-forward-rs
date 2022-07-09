//! # Forward-time traversal of demes models.
//!
//! ## Re-exports
//!
//! This crate re-exports `demes`.
//! Client code does not have to list `demes`
//! as a cargo dependency, guaranteeing that
//! a compatible version is avalable.
//!
//! ```{rust}
//! use demes_forward::demes;
//!
//! let yaml = "
//! time_units: generations
//! demes:
//!  - name: a_deme
//!    epochs:
//!     - start_size: 100
//! ";
//! assert!(demes::loads(yaml).is_ok());
//! ```


mod error;
mod graph;
mod time;

pub use demes;
pub use error::DemesForwardError;
pub use graph::ForwardGraph;
pub use time::ForwardTime;
pub use time::IntoForwardTime;
