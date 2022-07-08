use std::ops::{Add, Sub};

use crate::DemesForwardError;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct ForwardTime(f64);

impl ForwardTime {
    pub fn valid(&self) -> bool {
        self.0.is_finite() && self.0.is_sign_positive()
    }

    pub fn new<F: Into<ForwardTime>>(value: F) -> Self {
        value.into()
    }

    pub fn value(&self) -> f64 {
        self.0
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

impl Sub for ForwardTime {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        (self.0 - rhs.0).into()
    }
}

impl Add for ForwardTime {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        (self.0 + rhs.0).into()
    }
}

pub trait IntoForwardTime: Into<ForwardTime> + std::fmt::Debug + Copy {}

impl<T> IntoForwardTime for T where T: Into<ForwardTime> + std::fmt::Debug + Copy {}

pub(crate) struct TimeIterator {
    current_time: ForwardTime,
    final_time: ForwardTime,
}

impl Iterator for TimeIterator {
    type Item = ForwardTime;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_time.0 < self.final_time.0 - 1.0 {
            self.current_time = self.current_time + 1.0.into();
            Some(self.current_time)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct ModelTime {
    model_start_time: demes::Time,
    model_duration: f64,
    burnin_generation: f64,
}

impl ModelTime {
    pub(crate) fn convert(
        &self,
        time: ForwardTime,
    ) -> Result<Option<demes::Time>, DemesForwardError> {
        if time.value() < self.model_duration + self.burnin_generation {
            Ok(Some(
                (self.burnin_generation + self.model_duration - 1.0 - time.value()).into(),
            ))
        } else {
            Ok(None)
        }
    }
}

fn get_model_start_time(graph: &demes::Graph) -> demes::Time {
    // first end time of all demes with start time of infinity
    let mut times = graph
        .demes()
        .iter()
        .filter(|deme| deme.start_time() == f64::INFINITY)
        .map(|deme| deme.epochs()[0].end_time())
        .collect::<Vec<_>>();
    // NOTE: ends.is_empty() is an Error here!!!

    // start times of all demes whose start time is not infinity
    times.extend(
        graph
            .demes()
            .iter()
            .filter(|deme| deme.start_time() != f64::INFINITY)
            .map(|deme| deme.start_time()),
    );

    times.extend(
        graph
            .migrations()
            .iter()
            .filter(|migration| migration.start_time() != f64::INFINITY)
            .map(|migration| migration.start_time()),
    );

    times.extend(
        graph
            .migrations()
            .iter()
            .filter(|migration| migration.start_time() != f64::INFINITY)
            .map(|migration| migration.end_time()),
    );

    times.extend(graph.pulses().iter().map(|pulse| pulse.time()));

    debug_assert!(!times.is_empty());

    demes::Time::from(f64::from(*times.iter().max().unwrap()) + 1.0)
}

impl ModelTime {
    pub(crate) fn new_from_graph(
        burnin_time_length: crate::ForwardTime,
        graph: &demes::Graph,
    ) -> Result<Self, crate::DemesForwardError> {
        // The logic here is lifted from the fwdpy11
        // demes import code by Aaron Ragsdale.
        let most_ancient_deme_start = graph
            .demes()
            .iter()
            .map(|deme| deme.start_time())
            .collect::<Vec<_>>()
            .into_iter()
            .max()
            .unwrap();

        // Thus MUST be true!
        let model_start_time = get_model_start_time(graph);

        let most_recent_deme_end = graph
            .demes()
            .iter()
            .map(|deme| deme.end_time())
            .collect::<Vec<_>>()
            .into_iter()
            .min()
            .unwrap();
        let model_duration = if most_recent_deme_end > 0.0 {
            f64::from(model_start_time) - f64::from(most_recent_deme_end)
        } else {
            f64::from(model_start_time)
        };

        let burnin_generation = burnin_time_length.value();
        Ok(Self {
            model_start_time,
            model_duration,
            burnin_generation,
        })
    }

    pub(crate) fn burnin_generation(&self) -> f64 {
        self.burnin_generation
    }

    pub(crate) fn model_duration(&self) -> f64 {
        self.model_duration
    }

    pub(crate) fn time_iterator(&self) -> TimeIterator {
        TimeIterator {
            current_time: (-1.0).into(),
            final_time: (self.burnin_generation() + self.model_duration()).into(),
        }
    }
}

// FIXME: delete at some point
// when we have a full public API in place.
// These tests tests NON-PUBLIC bits
// of the API. They are for design musings
// only and should be deleted prior to merge.
#[cfg(test)]
mod delete_before_merge {
    use super::*;

    fn two_epoch_model() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 200
      end_time: 50
    - start_size: 100
";
        demes::loads(yaml).unwrap()
    }

    #[test]
    fn test_forwards_to_backwards_time_conversion() {
        let g = two_epoch_model();
        let graph = crate::graph::ForwardGraph::new(g, 100, None).unwrap();
        assert_eq!(
            graph
                .model_times
                .convert(ForwardTime::from(0))
                .unwrap()
                .unwrap(),
            150.
        );
        assert_eq!(
            graph
                .model_times
                .convert(ForwardTime::from(150.))
                .unwrap()
                .unwrap(),
            0.
        );
    }
}
