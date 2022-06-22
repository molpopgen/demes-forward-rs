use std::fmt::Debug;

use crate::time::ModelTime;
use crate::DemesForwardError;
use crate::ForwardTime;
use crate::IntoForwardTime;

fn time_minus_1(time: demes::Time) -> demes::Time {
    demes::Time::from(f64::from(time) - 1.0)
}

// NOTE: can refactor this when demes-rs/#134 is closed
fn get_epoch_start_time_discrete_time_model(
    deme: &demes::Deme,
    epoch_index: usize,
) -> Result<demes::Time, DemesForwardError> {
    // NOTE: is an error because
    // * We are calling this when size_function != Constant.
    // * The size_function for the first epoch of any deme
    //   MUST BE Constant
    if epoch_index == 0 {
        Err(DemesForwardError::InternalError(format!(
            "attempted to get start_time from epoch {} of deme {}",
            epoch_index,
            deme.name()
        )))
    } else {
        match deme.get_epoch(epoch_index - 1) {
            Some(epoch) => Ok(time_minus_1(epoch.end_time())),
            None => Err(DemesForwardError::InternalError(format!(
                "could not obtain epoch {} from deme {}",
                epoch_index - 1,
                deme.name()
            ))),
        }
    }
}

// #[derive(Copy, Clone)]
struct SizeFunctionDetails {
    epoch_start_time: demes::Time,
    epoch_end_time: demes::Time,
    epoch_start_size: demes::DemeSize,
    epoch_end_size: demes::DemeSize,
    backwards_time: demes::Time,
}

impl SizeFunctionDetails {
    fn duration(&self) -> f64 {
        f64::from(self.epoch_start_time) - f64::from(self.epoch_end_time)
    }

    fn time_from_epoch_start(&self) -> f64 {
        f64::from(self.epoch_start_time) - f64::from(self.backwards_time)
    }
}

macro_rules! fast_return {
    ($details: expr) => {
        if !($details.epoch_start_time > $details.backwards_time) {
            return $details.epoch_start_size.into();
        }
        if !($details.epoch_end_time < $details.backwards_time) {
            return $details.epoch_end_size.into();
        }
    };
}

fn linear_size_change(details: SizeFunctionDetails) -> f64 {
    fast_return!(details);
    let duration = details.duration();
    let x = details.time_from_epoch_start();
    let size_diff = f64::from(details.epoch_end_size) - f64::from(details.epoch_start_size);
    (f64::from(details.epoch_start_size) + (x / duration) * size_diff).round()
}

fn exponential_size_change(details: SizeFunctionDetails) -> f64 {
    fast_return!(details);
    let duration = details.duration();
    let nt = f64::from(details.epoch_end_size).round();
    let n0 = f64::from(details.epoch_start_size).round();
    let growth_rate = (nt.ln() - n0.ln()) / duration;
    let x = details.time_from_epoch_start();
    (n0 * (growth_rate * x).exp()).round()
}

fn apply_size_function(
    deme: &demes::Deme,
    epoch_index: usize,
    backwards_time: Option<demes::Time>,
    size_function_details: impl Fn(SizeFunctionDetails) -> f64,
) -> Result<Option<demes::DemeSize>, DemesForwardError> {
    match backwards_time {
        Some(btime) => {
            let epoch_start_time = get_epoch_start_time_discrete_time_model(deme, epoch_index)?;
            let current_epoch = match deme.get_epoch(epoch_index) {
                Some(epoch) => epoch,
                None => {
                    return Err(DemesForwardError::InternalError(format!(
                        "could not retrieve epoch {} from deme {}",
                        epoch_index,
                        deme.name()
                    )))
                }
            };

            let epoch_end_time = current_epoch.end_time();
            let epoch_start_size = current_epoch.start_size();
            let epoch_end_size = current_epoch.end_size();

            let size: f64 = size_function_details(SizeFunctionDetails {
                epoch_start_time,
                epoch_end_time,
                epoch_start_size,
                epoch_end_size,
                backwards_time: btime,
            });

            if !size.gt(&0.0) || !size.is_finite() {
                Err(DemesForwardError::InvalidDemeSize(size.into()))
            } else {
                Ok(Some(demes::DemeSize::from(size)))
            }
        }
        None => Ok(None),
    }
}

#[derive(Debug)]
pub struct Deme {
    deme: demes::Deme,
    status: DemeStatus,
    backwards_time: Option<demes::Time>,
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
            backwards_time: None,
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

    // return None if !self.is_extant()
    fn current_size(&self) -> Result<Option<demes::DemeSize>, DemesForwardError> {
        match self.status {
            DemeStatus::During(epoch_index) => match self.deme.get_epoch(epoch_index) {
                Some(epoch) => match epoch.size_function() {
                    demes::SizeFunction::Constant => Ok(Some(epoch.start_size())),
                    demes::SizeFunction::Linear => apply_size_function(
                        &self.deme,
                        epoch_index,
                        self.backwards_time,
                        linear_size_change,
                    ),
                    demes::SizeFunction::Exponential => apply_size_function(
                        &self.deme,
                        epoch_index,
                        self.backwards_time,
                        exponential_size_change,
                    ),
                },
                None => panic!("fatal error: epoch_index out of range"),
            },
            _ => Ok(None),
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
                    deme.backwards_time = Some(backwards_time);
                } else {
                    deme.status = DemeStatus::After;
                    deme.backwards_time = None;
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

    pub fn get_parental_deme(&self, index: usize) -> Option<&Deme> {
        match &self.parent_demes {
            Some(demes) => demes.get(index),
            None => None,
        }
    }

    pub fn get_child_deme(&self, index: usize) -> Option<&Deme> {
        match &self.child_demes {
            Some(demes) => demes.get(index),
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
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(100.))
            );
        }
        graph.update_state(75_i32).unwrap();
        assert_eq!(graph.parental_demes().unwrap().iter().count(), 1);
        for deme in graph.parental_demes().unwrap().iter() {
            // Access the underlying demes::Time
            assert!(deme.is_extant());
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
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
        let expected_sizes = |generation: i32| -> (f64, f64) {
            if generation < 100 {
                (200.0, 200.0)
            } else if generation < 101 {
                (200.0, 100.0)
            } else {
                (100.0, 100.0)
            }
        };
        for generation in [99_i32, 100, 101, 102] {
            graph.update_state(generation).unwrap();
            let expected = expected_sizes(generation);
            for deme in graph.parental_demes().unwrap().iter() {
                // Access the underlying demes::Time
                assert!(deme.is_extant());
                assert_eq!(
                    deme.current_size().unwrap(),
                    Some(demes::DemeSize::from(expected.0))
                );
            }
            for deme in graph.child_demes().unwrap().iter() {
                // Access the underlying demes::Time
                assert!(deme.is_extant());
                assert_eq!(
                    deme.current_size().unwrap(),
                    Some(demes::DemeSize::from(expected.1))
                );
            }
        }
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

#[cfg(test)]
mod test_nonlinear_size_changes {
    use super::*;

    fn two_epoch_model_linear_growth() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 200
      end_time: 50
    - start_size: 100
      end_size: 200
      size_function: linear
";
        demes::loads(yaml).unwrap()
    }

    fn two_epoch_model_linear_decline() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 200
      end_time: 50
    - start_size: 200
      end_size: 100
      size_function: linear
";
        demes::loads(yaml).unwrap()
    }

    fn two_epoch_model_exponential_growth() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 200
      end_time: 50
    - start_size: 100
      end_size: 200
      size_function: exponential
";
        demes::loads(yaml).unwrap()
    }

    #[test]
    fn test_two_epoch_model_linear_growth() {
        let demes_graph = two_epoch_model_linear_growth();
        let mut graph = ForwardGraph::new(demes_graph, 100_u32, None).unwrap();
        graph.update_state(100).unwrap(); // last generation of the 1st epoch
        if let Some(deme) = graph.get_parental_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        println!("child...");
        if let Some(deme) = graph.get_child_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(100.))
            );
        } else {
            panic!();
        }
        // one generation before end
        graph.update_state(149).unwrap();
        if let Some(deme) = graph.get_child_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        graph.update_state(150).unwrap(); // last gen
        assert!(graph.get_child_deme(0).is_none());
        if let Some(deme) = graph.get_parental_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }

        // 1/2-way into the final epoch
        graph.update_state(125).unwrap();
        for deme in graph.parental_demes().unwrap().iter() {
            let expected_size: f64 = 100. + ((49. - 25.) / (49.)) * (200. - 100.);
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(expected_size.round()))
            );
        }
        for deme in graph.child_demes().unwrap().iter() {
            let expected_size: f64 = 100. + ((49. - 24.) / (49.)) * (200. - 100.);
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(expected_size.round()))
            );
        }
    }

    #[test]
    fn test_two_epoch_model_linear_decline() {
        let demes_graph = two_epoch_model_linear_decline();
        let mut graph = ForwardGraph::new(demes_graph, 100_u32, None).unwrap();
        graph.update_state(100).unwrap(); // last generation of the 1st epoch
        if let Some(deme) = graph.get_parental_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        println!("child...");
        if let Some(deme) = graph.get_child_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        // one generation before end
        graph.update_state(149).unwrap();
        if let Some(deme) = graph.get_child_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(100.))
            );
        } else {
            panic!();
        }
        graph.update_state(150).unwrap(); // last gen
        assert!(graph.get_child_deme(0).is_none());
        if let Some(deme) = graph.get_parental_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(100.))
            );
        } else {
            panic!();
        }

        // 1/2-way into the final epoch
        graph.update_state(125).unwrap();
        for deme in graph.parental_demes().unwrap().iter() {
            let expected_size: f64 = 200. + ((49. - 25.) / (49.)) * (100. - 200.);
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(expected_size.round()))
            );
        }
        for deme in graph.child_demes().unwrap().iter() {
            let expected_size: f64 = 200. + ((49. - 24.) / (49.)) * (100. - 200.);
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(expected_size.round()))
            );
        }
    }

    #[test]
    fn test_two_epoch_model_exponential_growth() {
        let demes_graph = two_epoch_model_exponential_growth();
        let mut graph = ForwardGraph::new(demes_graph, 100_u32, None).unwrap();
        graph.update_state(100).unwrap(); // last generation of the 1st epoch
        if let Some(deme) = graph.get_parental_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        println!("child...");
        if let Some(deme) = graph.get_child_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(100.))
            );
        } else {
            panic!();
        }
        // one generation before end
        graph.update_state(149).unwrap();
        if let Some(deme) = graph.get_child_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        graph.update_state(150).unwrap(); // last gen
        assert!(graph.get_child_deme(0).is_none());
        if let Some(deme) = graph.get_parental_deme(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }

        // 1/2-way into the final epoch
        graph.update_state(125).unwrap();
        for deme in graph.parental_demes().unwrap().iter() {
            let g = (200_f64.ln() - 100_f64.ln()) / 49.0;
            let expected_size: f64 = 100.0 * (g * 24.0).exp();
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(expected_size.round()))
            );
        }
        for deme in graph.child_demes().unwrap().iter() {
            let g = (200_f64.ln() - 100_f64.ln()) / 49.0;
            let expected_size: f64 = 100.0 * (g * 25.0).exp();
            println!("{} {}", g, expected_size);
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(expected_size.round()))
            );
        }
    }
}
