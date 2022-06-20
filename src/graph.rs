use std::fmt::Debug;

use crate::time::ModelTime;
use crate::DemesForwardError;
use crate::ForwardTime;
use crate::IntoForwardTime;

#[derive(Debug)]
pub struct CurrentEpochData {
    epoch: demes::Epoch,
}

impl CurrentEpochData {
    pub fn end_time(&self) -> demes::Time {
        self.epoch.end_time()
    }
}

#[derive(Debug)]
pub struct CurrentEpochs {
    time: ForwardTime,
    epochs: Vec<CurrentEpochData>,
}

impl CurrentEpochs {
    fn iter(&self) -> std::slice::Iter<CurrentEpochData> {
        self.epochs.iter()
    }
}

pub struct ForwardGraph {
    graph: demes::Graph,
    pub(crate) model_times: ModelTime, // FIXME: this should be private to this module.
    child_epochs: Option<CurrentEpochs>,
    parent_epochs: Option<CurrentEpochs>,
}

impl ForwardGraph {
    fn calculate_current_epochs(
        &mut self,
        generation_time: ForwardTime,
    ) -> Result<Option<CurrentEpochs>, DemesForwardError> {
        let backwards_time = self.model_times.convert(generation_time)?;
        match backwards_time {
            Some(backwards_time) => {
                let mut e = vec![];
                for deme in self.graph.demes() {
                    if backwards_time < deme.start_time() {
                        if let Some(epoch) = deme
                            .epochs()
                            .iter()
                            .find(|epoch| epoch.end_time() <= backwards_time)
                        {
                            e.push(CurrentEpochData { epoch: *epoch });
                        }
                    }
                }
                Ok(Some(CurrentEpochs {
                    time: generation_time,
                    epochs: e,
                }))
            }
            None => Ok(None),
        }
    }

    fn update_parental_deme_epochs(
        &mut self,
        parental_generation_time: ForwardTime,
    ) -> Result<(), DemesForwardError> {
        self.parent_epochs = self.calculate_current_epochs(parental_generation_time)?;
        Ok(())
    }

    fn update_child_deme_epochs(
        &mut self,
        parental_generation_time: ForwardTime,
    ) -> Result<(), DemesForwardError> {
        let child_generation_time = ForwardTime::from(parental_generation_time.value() + 1.);
        self.child_epochs = self.calculate_current_epochs(child_generation_time)?;
        Ok(())
    }

    pub fn new<F: IntoForwardTime>(
        graph: demes::Graph,
        burnin_time: F,
        rounding: Option<demes::RoundTimeToInteger>,
    ) -> Result<Self, crate::DemesForwardError> {
        let burnin_time = burnin_time.into();
        if !burnin_time.valid() {
            return Err(DemesForwardError::TimeError(format!(
                "invalid time value: {:?}",
                burnin_time
            )));
        }
        let graph = match rounding {
            Some(r) => graph.to_integer_generations(r)?,
            None => graph.to_generations()?,
        };
        let model_times = ModelTime::new_from_graph(burnin_time, &graph)?;
        let child_epochs = Option::<CurrentEpochs>::default();
        let parent_epochs = Option::<CurrentEpochs>::default();
        Ok(Self {
            graph,
            model_times,
            child_epochs,
            parent_epochs,
        })
    }

    // NOTE: is this a birth time or a parental time?
    // Semantically, we want this function to:
    // * Update BOTH parental and offspring info.
    //   * What epoch are we in for both parent/offspring demes?
    //   * What size is each parental/offspring deme?
    // * with respect to offspring demes:
    //   * What Pulse events are happening "now"?
    //   * What migrations are going on?
    // NOTE: we are going w/the decision to make
    // this a PARENTAL generation time.
    //
    // NOTE: to improve performance we may
    // * cache the index of the current Epoch for each extant Deme
    // * Use that index as for Iterator::skip()
    // * Doing so will break some of our existing tests?
    pub fn update_state<F: IntoForwardTime>(
        &mut self,
        parental_generation_time: F,
    ) -> Result<(), DemesForwardError> {
        let parental_generation_time = parental_generation_time.into();
        self.update_parental_deme_epochs(parental_generation_time)?;
        self.update_child_deme_epochs(parental_generation_time)
    }

    pub fn parental_epochs(&self) -> Option<&CurrentEpochs> {
        self.parent_epochs.as_ref()
    }

    pub fn child_epochs(&self) -> Option<&CurrentEpochs> {
        self.child_epochs.as_ref()
    }
}

#[cfg(test)]
mod graph_tests {
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

    fn two_epoch_model_invalid_conversion_to_generations() -> demes::Graph {
        let yaml = "
time_units: years
description:
  50/1000 = 0.05, rounds to zero.
  Thus, the second epoch has length zero.
generation_time: 1000.0
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
    fn one_deme_two_epochs() {
        let demes_graph = two_epoch_model();
        let mut graph = ForwardGraph::new(demes_graph, 100_u32, None).unwrap();
        graph.update_state(125_i32).unwrap();
        assert_eq!(graph.parental_epochs().unwrap().iter().count(), 1);
        for epoch in graph.parental_epochs().unwrap().iter() {
            // Access the underlying demes::Time
            assert_eq!(epoch.end_time(), 0.0);
        }
        graph.update_state(75_i32).unwrap();
        assert_eq!(graph.parental_epochs().unwrap().iter().count(), 1);
        for epoch in graph.parental_epochs().unwrap().iter() {
            // Access the underlying demes::Time
            assert_eq!(epoch.end_time(), 50.0);
        }

        // The last generation
        graph.update_state(150_i32).unwrap();
        for epoch in graph.parental_epochs().unwrap().iter() {
            // Access the underlying demes::Time
            assert_eq!(epoch.end_time(), 0.0);
        }
        assert!(graph.child_epochs().is_none());

        // One past the last generation
        graph.update_state(151_i32).unwrap();
        assert!(graph.parental_epochs().is_none());
        assert!(graph.child_epochs().is_none());
    }

    #[test]
    fn invalid_conversion_error() {
        let demes_graph = two_epoch_model_invalid_conversion_to_generations();
        let result = ForwardGraph::new(demes_graph, 100.0, Some(demes::RoundTimeToInteger::F64));
        assert!(matches!(
            result,
            Err(crate::DemesForwardError::DemesError(
                demes::DemesError::EpochError(_)
            ))
        ));
    }

    #[test]
    fn invalid_forward_time() {
        {
            let x = ForwardTime::new(-1_i32);
            assert!(!x.valid());
        }
        {
            let x = ForwardTime::from(-1_f64);
            assert!(!x.valid());
        }
        {
            let x = ForwardTime::from(f64::INFINITY);
            assert!(!x.valid());
        }
        {
            let x = ForwardTime::from(f64::NAN);
            assert!(!x.valid());
            let graph = two_epoch_model();
            assert!(ForwardGraph::new(graph, x, None).is_err());
        }
    }
}
