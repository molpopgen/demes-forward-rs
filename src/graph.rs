use std::fmt::Debug;

use crate::time::ModelTime;
use crate::DemesForwardError;
use crate::ForwardTime;
use crate::IntoForwardTime;

#[derive(Debug)]
pub struct Deme {
    deme: demes::Deme,
    status: DemeStatus,
}

#[derive(Debug)]
enum DemeStatus {
    /// Before the deme first appears.
    /// (Moving forwards in time.)
    Before,
    /// During the deme's Epochs
    During(usize),
    /// After the deme ceases to exist.
    /// (Moving forwards in time.)
    After,
}

impl Deme {
    fn new(deme: demes::Deme) -> Self {
        Self {
            deme,
            status: DemeStatus::Before,
        }
    }

    fn is_extant(&self) -> bool {
        matches!(self.status, DemeStatus::During(_))
    }

    fn is_before(&self) -> bool {
        matches!(self.status, DemeStatus::Before)
    }

    fn is_after(&self) -> bool {
        matches!(self.status, DemeStatus::After)
    }

    fn epoch_index_for_update(&self) -> usize {
        match self.status {
            DemeStatus::Before => 0,
            DemeStatus::During(x) => x,
            DemeStatus::After => self.deme.num_epochs(),
        }
    }
}

pub struct ForwardGraph {
    graph: demes::Graph,
    pub(crate) model_times: ModelTime, // FIXME: this should be private to this module.
    parent_demes: Option<Vec<Deme>>,
    child_demes: Option<Vec<Deme>>,
    last_time_updated: Option<ForwardTime>,
}

impl ForwardGraph {
    fn update_current_demes(
        &mut self,
        generation_time: ForwardTime,
        current_demes: &mut Vec<Deme>,
    ) -> Result<(), DemesForwardError> {
        let backwards_time = self.model_times.convert(generation_time)?;
        if backwards_time.is_none() {
            current_demes.clear();
            return Ok(());
        }
        if current_demes.is_empty() {
            for deme in self.graph.demes() {
                current_demes.push(Deme::new(deme.clone()));
            }
        }
        let backwards_time = backwards_time.unwrap();
        for deme in current_demes {
            if backwards_time < deme.deme.start_time() {
                let i = deme.epoch_index_for_update();

                // NOTE: by having enumerate BEFORE
                // skip, the j value is the offset
                // from .epoch()[0]!!!
                if let Some((j, _epoch)) = deme
                    .deme
                    .epochs()
                    .iter()
                    .enumerate()
                    .skip(i)
                    .find(|index_epoch| backwards_time >= index_epoch.1.end_time())
                {
                    assert!(
                        j < deme.deme.epochs().len(),
                        "{} {} {:?}",
                        j,
                        backwards_time,
                        generation_time
                    );
                    deme.status = DemeStatus::During(j);
                } else {
                    deme.status = DemeStatus::After;
                }
            }
        }
        Ok(())
    }

    fn update_demes(
        &mut self,
        generation_time: ForwardTime,
        current_demes: Vec<Deme>,
    ) -> Result<Option<Vec<Deme>>, DemesForwardError> {
        let mut current_demes = current_demes;
        self.update_current_demes(generation_time, &mut current_demes)?;
        if current_demes.is_empty() {
            Ok(None)
        } else {
            Ok(Some(current_demes))
        }
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
        let child_demes = Option::<Vec<Deme>>::default();
        let parent_demes = Option::<Vec<Deme>>::default();
        Ok(Self {
            graph,
            model_times,
            parent_demes,
            child_demes,
            last_time_updated: None,
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
        match self.last_time_updated {
            Some(time) => {
                if parental_generation_time < time {
                    // gotta reset...
                    self.parent_demes = None;
                    self.child_demes = None;
                }
            }
            None => (),
        }
        //self.update_parental_demes(parental_generation_time)?;
        let demes = self.parent_demes.take().unwrap_or_default();
        self.parent_demes = self.update_demes(parental_generation_time, demes)?;
        let demes = self.child_demes.take().unwrap_or_default();
        let child_generation_time = ForwardTime::from(parental_generation_time.value() + 1.0);
        self.child_demes = self.update_demes(child_generation_time, demes)?;
        // self.update_child_demes(parental_generation_time)?;
        self.last_time_updated = Some(parental_generation_time);

        Ok(())
    }

    pub fn num_demes_in_model(&self) -> usize {
        self.graph.num_demes()
    }

    pub fn parental_demes(&self) -> Option<&[Deme]> {
        match &self.parent_demes {
            Some(x) => Some(x.as_slice()),
            None => None,
        }
    }

    pub fn child_demes(&self) -> Option<&[Deme]> {
        match &self.child_demes {
            Some(x) => Some(x.as_slice()),
            None => None,
        }
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

    fn three_deme_model() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 100
      end_time: 50
 - name: B
   ancestors: [A]
   epochs:
    - start_size: 100
 - name: C
   ancestors: [A]
   epochs:
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
        assert_eq!(graph.parental_demes().unwrap().iter().count(), 1);
        for deme in graph.parental_demes().unwrap().iter() {
            // Access the underlying demes::Time
            assert!(deme.is_extant());
        }
        graph.update_state(75_i32).unwrap();
        assert_eq!(graph.parental_demes().unwrap().iter().count(), 1);
        for deme in graph.parental_demes().unwrap().iter() {
            // Access the underlying demes::Time
            assert!(deme.is_extant());
        }

        // The last generation
        graph.update_state(150_i32).unwrap();
        for deme in graph.parental_demes().unwrap().iter() {
            assert!(deme.is_extant());
        }
        assert!(graph.child_demes().is_none());

        // One past the last generation
        graph.update_state(151_i32).unwrap();
        assert!(graph.parental_demes().is_none());
        assert!(graph.child_demes().is_none());

        // Test what happens as we "evolve through"
        // an epoch boundary.
        graph.update_state(99_i32).unwrap();
        graph.update_state(100_i32).unwrap();
        graph.update_state(101_i32).unwrap();
        graph.update_state(102_i32).unwrap();
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

    #[test]
    fn test_three_deme_model() {
        let demes_graph = three_deme_model();
        let mut g = ForwardGraph::new(demes_graph, 100, None).unwrap();
        assert!(g.parental_demes().is_none()); // Not initialized -- may break this later!
        g.update_state(0).unwrap(); // first generation of model
        assert_eq!(g.num_demes_in_model(), 3); // There are 3 demes in the model
        assert_eq!(g.parental_demes().unwrap().iter().count(), 3); // There are 3 demes in the model
        for (i, deme) in g.parental_demes().unwrap().iter().enumerate() {
            if i == 0 {
                assert!(deme.is_extant());
            } else {
                assert!(!deme.is_extant());
                assert!(deme.is_before());
            }
        }

        g.update_state(100).unwrap(); // Last generation of ancestor deme
        for (i, deme) in g.parental_demes().unwrap().iter().enumerate() {
            if i == 0 {
                assert!(deme.is_extant());
            } else {
                assert!(!deme.is_extant());
                assert!(deme.is_before());
            }
        }
        for (i, deme) in g.child_demes().unwrap().iter().enumerate() {
            if i == 0 {
                assert!(!deme.is_extant());
                assert!(deme.is_after());
            } else {
                assert!(deme.is_extant());
            }
        }

        g.update_state(101).unwrap(); // Last generation of ancestor deme
        for (i, deme) in g.parental_demes().unwrap().iter().enumerate() {
            if i == 0 {
                assert!(!deme.is_extant());
                assert!(deme.is_after());
            } else {
                assert!(deme.is_extant());
            }
        }
        for (i, deme) in g.child_demes().unwrap().iter().enumerate() {
            if i == 0 {
                assert!(!deme.is_extant());
                assert!(deme.is_after());
            } else {
                assert!(deme.is_extant());
            }
        }
    }
}
