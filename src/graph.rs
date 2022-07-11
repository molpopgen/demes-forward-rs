use crate::time::ModelTime;
use crate::DemesForwardError;
use crate::ForwardTime;
use crate::IntoForwardTime;

enum Generation {
    Parent,
    Child,
}

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

fn time_is_rounded(time: demes::Time, level: &str) -> Result<(), DemesForwardError> {
    let f = f64::from(time);
    if f.fract() == 0.0 || f.is_infinite() {
        Ok(())
    } else {
        Err(DemesForwardError::TimeError(format!(
            "{} not rounded to integer: {}",
            level, time
        )))
    }
}

fn validate_model_times(graph: &demes::Graph) -> Result<(), DemesForwardError> {
    graph
        .demes()
        .iter()
        .try_for_each(|deme| time_is_rounded(deme.start_time(), "Deme start_time"))?;
    graph.demes().iter().try_for_each(|deme| {
        deme.epochs()
            .iter()
            .try_for_each(|epoch| time_is_rounded(epoch.end_time(), "Epoch end_time"))
    })?;
    graph
        .pulses()
        .iter()
        .try_for_each(|pulse| time_is_rounded(pulse.time(), "Pulse time"))?;
    graph.migrations().iter().try_for_each(|migration| {
        time_is_rounded(migration.start_time(), "Migration start_time")?;
        time_is_rounded(migration.end_time(), "Migration end_time")
    })?;
    Ok(())
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
struct Deme {
    deme: demes::Deme,
    status: DemeStatus,
    backwards_time: Option<demes::Time>,
    ancestors: Vec<usize>,
    proportions: Vec<demes::Proportion>,
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
            ancestors: vec![],
            proportions: vec![],
        }
    }

    fn is_extant(&self) -> bool {
        matches!(self.status, DemeStatus::During(_))
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

    fn ancestors(&self) -> &[usize] {
        &self.ancestors
    }

    fn proportions(&self) -> &[demes::Proportion] {
        &self.proportions
    }

    fn update(
        &mut self,
        time: demes::Time,
        update_ancestors: bool,
        deme_to_index: &std::collections::HashMap<String, usize>,
    ) -> Result<demes::DemeSize, DemesForwardError> {
        self.ancestors.clear();
        self.proportions.clear();
        let mut current_size = demes::DemeSize::from(0.0);
        if time < self.deme.start_time() {
            let i = self.epoch_index_for_update();

            // NOTE: by having enumerate BEFORE
            // skip, the j value is the offset
            // from .epoch()[0]!!!
            if let Some((j, _epoch)) = self
                .deme
                .epochs()
                .iter()
                .enumerate()
                .skip(i)
                .find(|index_epoch| time >= index_epoch.1.end_time())
            {
                self.status = DemeStatus::During(j);
                self.backwards_time = Some(time);
                if update_ancestors {
                    let generation_to_check_ancestors =
                        demes::Time::from(f64::from(self.deme.start_time()) - 2.0);
                    if time > generation_to_check_ancestors {
                        for (name, proportion) in self
                            .deme
                            .ancestor_names()
                            .iter()
                            .zip(self.deme.proportions().iter())
                        {
                            self.ancestors.push(*deme_to_index.get(name).unwrap());
                            self.proportions.push(*proportion);
                        }
                    }
                }
                let deme_size = self.current_size()?;
                current_size = deme_size.unwrap();
            } else {
                self.status = DemeStatus::After;
                self.backwards_time = None;
            }
        }
        Ok(current_size)
    }
}

fn update_demes(
    backwards_time: Option<demes::Time>,
    update_ancestors: bool,
    deme_to_index: &std::collections::HashMap<String, usize>,
    graph: &demes::Graph,
    demes: &mut Vec<Deme>,
    sizes: &mut Vec<demes::DemeSize>,
) -> Result<(), DemesForwardError> {
    match backwards_time {
        Some(time) => {
            if demes.is_empty() {
                sizes.clear();
                for deme in graph.demes().iter() {
                    demes.push(Deme::new(deme.clone()));
                    sizes.push(demes::DemeSize::from(0.0));
                }
            }

            demes.iter_mut().enumerate().try_for_each(
                |(i, deme)| -> Result<(), DemesForwardError> {
                    let size = deme.update(time, update_ancestors, deme_to_index)?;
                    sizes[i] = size;
                    Ok(())
                },
            )?;
        }
        None => demes.clear(),
    }
    Ok(())
}

#[derive(Debug)]
pub struct ForwardGraph {
    graph: demes::Graph,
    pub(crate) model_times: ModelTime, // FIXME: this should be private to this module.
    parent_demes: Vec<Deme>,
    child_demes: Vec<Deme>,
    last_time_updated: Option<ForwardTime>,
    deme_to_index: std::collections::HashMap<String, usize>,
    pulses: Vec<demes::Pulse>,
    migrations: Vec<demes::AsymmetricMigration>,
    ancestry_proportions: ndarray::Array<f64, ndarray::Ix2>,
    migration_matrix: ndarray::Array<f64, ndarray::Ix2>,
    cloning_rates: Vec<demes::CloningRate>,
    selfing_rates: Vec<demes::SelfingRate>,
    parental_deme_sizes: Vec<demes::DemeSize>,
    child_deme_sizes: Vec<demes::DemeSize>,
}

impl ForwardGraph {
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

        validate_model_times(&graph)?;

        let model_times = ModelTime::new_from_graph(burnin_time, &graph)?;
        let child_demes = vec![];
        let parent_demes = vec![];
        let mut deme_to_index = std::collections::HashMap::default();
        for (i, deme) in graph.demes().iter().enumerate() {
            deme_to_index.insert(deme.name().to_string(), i);
        }
        let pulses = vec![];
        let ancestry_proportions =
            ndarray::Array2::<f64>::zeros((deme_to_index.len(), deme_to_index.len()));
        let migration_matrix =
            ndarray::Array2::<f64>::zeros((deme_to_index.len(), deme_to_index.len()));
        Ok(Self {
            graph,
            model_times,
            parent_demes,
            child_demes,
            last_time_updated: None,
            deme_to_index,
            pulses,
            migrations: vec![],
            ancestry_proportions,
            migration_matrix,
            cloning_rates: vec![],
            selfing_rates: vec![],
            parental_deme_sizes: vec![],
            child_deme_sizes: vec![],
        })
    }

    fn update_pulses(&mut self, backwards_time: Option<demes::Time>) {
        self.pulses.clear();
        match backwards_time {
            None => (),
            Some(time) => self.graph.pulses().iter().for_each(|pulse| {
                if !(time > pulse.time() || time < pulse.time()) {
                    self.pulses.push(pulse.clone());
                }
            }),
        }
    }

    // NOTE: performance here is poop emoji.
    // Migrations tend to occur over long epochs
    // and we are figuring this out from scratch each time.
    // This may not be a "big deal" so this note is here in
    // case of future profiling.
    // Alternative:
    // * Maintain a vec of current epochs that are (index, Mig)
    // * Remove epochs no longer needed
    // * Only add epochs not already there.
    fn update_migrations(&mut self, backwards_time: Option<demes::Time>) {
        self.migrations.clear();
        match backwards_time {
            None => (),
            Some(time) => self.graph.migrations().iter().for_each(|migration| {
                if time > migration.end_time() && time <= migration.start_time() {
                    self.migrations.push(migration.clone());
                }
            }),
        }
    }

    fn initialize_ancestry_proportions(&mut self) {
        self.ancestry_proportions.fill(0.0);
        for (i, deme) in self.child_demes.iter().enumerate() {
            if deme.is_extant() {
                if deme.ancestors().is_empty() {
                    self.ancestry_proportions[[i, i]] = 1.0;
                } else {
                    deme.ancestors()
                        .iter()
                        .zip(deme.proportions().iter())
                        .for_each(|(ancestor, proportion)| {
                            self.ancestry_proportions[[i, *ancestor]] = f64::from(*proportion)
                        });
                }
            }
        }
    }

    // NOTE: the borrow checker "made" us go fully functional
    fn update_ancestry_proportions_from_pulses(&mut self) -> Result<(), DemesForwardError> {
        let mut sources = vec![];
        let mut proportions = vec![];
        self.pulses.iter().try_for_each(|pulse| {
            sources.clear();
            proportions.clear();
            let dest = *self.deme_to_index.get(pulse.dest()).unwrap();
            if !self.child_demes[dest].is_extant() {
                return Err(DemesForwardError::InternalError(format!(
                    "pulse dest deme {} is extinct",
                    dest
                )));
            }
            let mut sum = 0.0;
            pulse
                .sources()
                .iter()
                .zip(pulse.proportions().iter())
                .try_for_each(|(source, proportion)| {
                    let index: usize = *self.deme_to_index.get(source).unwrap();
                    if !self.parent_demes[index].is_extant() {
                        return Err(DemesForwardError::InternalError(format!(
                            "pulse source deme {} is extinct",
                            index
                        )));
                    }
                    sources.push(index);
                    let p = f64::from(*proportion);
                    sum += p;
                    proportions.push(p);
                    Ok(())
                })?;
            self.ancestry_proportions
                .row_mut(dest)
                .iter_mut()
                .for_each(|v| *v *= 1. - sum);
            sources
                .iter()
                .zip(proportions.iter())
                .for_each(|(source, proportion)| {
                    self.ancestry_proportions[[dest, *source]] += proportion;
                });
            Ok(())
        })?;
        Ok(())
    }

    fn update_migration_matrix(&mut self) -> Result<(), DemesForwardError> {
        self.migration_matrix.fill(0.0);
        self.migrations.iter().try_for_each(|migration| {
            let source = self.deme_to_index.get(migration.source()).unwrap();
            let dest = self.deme_to_index.get(migration.dest()).unwrap();
            if !self.parent_demes[*source].is_extant() {
                return Err(DemesForwardError::InternalError(format!(
                    "migration source deme {} is extinct",
                    source
                )));
            }
            if !self.child_demes[*dest].is_extant() {
                return Err(DemesForwardError::InternalError(format!(
                    "migration dest deme {} is extinct",
                    source
                )));
            }
            self.migration_matrix[[*dest, *source]] = migration.rate().into();
            Ok(())
        })?;
        Ok(())
    }

    // NOTE: this doesn't Err b/c:
    // It is called after update_migration_matrix, which
    // does the extant/extinct checks already
    fn update_ancestry_proportions_from_migration_matrix(&mut self) {
        self.ancestry_proportions
            .outer_iter_mut()
            .zip(self.migration_matrix.outer_iter())
            .for_each(|(mut p, m)| {
                let sum = m.sum();
                p *= 1. - sum;
                p.iter_mut().zip(m.iter()).for_each(|(a, b)| {
                    *a += b;
                })
            });
    }

    fn deme_slice(&self, generation: Generation) -> &[Deme] {
        match generation {
            Generation::Parent => self.parent_demes.as_slice(),
            Generation::Child => self.child_demes.as_slice(),
        }
    }

    fn any_extant_demes(&self, generation: Generation) -> bool {
        let s = self.deme_slice(generation);
        s.iter().any(|deme| deme.is_extant())
    }

    fn num_extant_demes(&self, generation: Generation) -> usize {
        let s = self.deme_slice(generation);
        s.iter().filter(|deme| deme.is_extant()).count()
    }

    fn get_slice_if<'a, T: Sized>(&'a self, generation: Generation, s: &'a [T]) -> Option<&[T]> {
        let is_empty = match generation {
            Generation::Parent => self.parent_demes.is_empty(),
            Generation::Child => self.child_demes.is_empty(),
        };
        if !is_empty {
            Some(s)
        } else {
            None
        }
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
                    self.parent_demes.clear();
                    self.child_demes.clear();
                }
            }
            None => (),
        }
        let backwards_time = self.model_times.convert(parental_generation_time)?;
        update_demes(
            backwards_time,
            false,
            &self.deme_to_index,
            &self.graph,
            &mut self.parent_demes,
            &mut self.parental_deme_sizes,
        )?;
        self.update_pulses(backwards_time);
        self.update_migrations(backwards_time);
        let child_generation_time = ForwardTime::from(parental_generation_time.value() + 1.0);
        let backwards_time = self.model_times.convert(child_generation_time)?;
        update_demes(
            backwards_time,
            true,
            &self.deme_to_index,
            &self.graph,
            &mut self.child_demes,
            &mut self.child_deme_sizes,
        )?;

        self.selfing_rates.clear();
        self.cloning_rates.clear();
        for deme in &self.child_demes {
            match deme.status {
                DemeStatus::During(x) => {
                    self.cloning_rates
                        .push(deme.deme.epochs()[x].cloning_rate());
                    self.selfing_rates
                        .push(deme.deme.epochs()[x].selfing_rate());
                }
                _ => {
                    self.cloning_rates.push(demes::CloningRate::from(0.0));
                    self.selfing_rates.push(demes::SelfingRate::from(0.0));
                }
            }
        }

        self.initialize_ancestry_proportions();
        self.update_ancestry_proportions_from_pulses()?;
        self.update_migration_matrix()?;
        self.update_ancestry_proportions_from_migration_matrix();
        self.last_time_updated = Some(parental_generation_time);

        Ok(())
    }

    pub fn num_demes_in_model(&self) -> usize {
        self.graph.num_demes()
    }

    pub fn ancestry_proportions(&self, child_deme: usize) -> Option<&[f64]> {
        if !self.child_demes.is_empty() {
            let start = child_deme * self.child_demes.len();
            let stop = start + self.child_demes.len();
            Some(&self.ancestry_proportions.as_slice().unwrap()[start..stop])
        } else {
            None
        }
    }

    pub fn cloning_rates(&self) -> Option<&[demes::CloningRate]> {
        self.get_slice_if(Generation::Child, self.cloning_rates.as_slice())
    }

    pub fn selfing_rates(&self) -> Option<&[demes::SelfingRate]> {
        self.get_slice_if(Generation::Child, self.selfing_rates.as_slice())
    }

    pub fn last_time_updated(&self) -> Option<ForwardTime> {
        self.last_time_updated
    }

    pub fn end_time(&self) -> ForwardTime {
        let burnin_gen = self.model_times.burnin_generation();
        let model_duration = self.model_times.model_duration();

        (burnin_gen + model_duration).into()
    }

    pub fn time_iterator(&self) -> impl Iterator<Item = ForwardTime> {
        self.model_times.time_iterator(self.last_time_updated)
    }

    pub fn parental_deme_sizes(&self) -> Option<&[demes::DemeSize]> {
        self.get_slice_if(Generation::Parent, self.parental_deme_sizes.as_slice())
    }

    pub fn child_deme_sizes(&self) -> Option<&[demes::DemeSize]> {
        self.get_slice_if(Generation::Child, self.child_deme_sizes.as_slice())
    }

    pub fn any_extant_parental_demes(&self) -> bool {
        self.any_extant_demes(Generation::Parent)
    }

    pub fn any_extant_child_demes(&self) -> bool {
        self.any_extant_demes(Generation::Child)
    }

    pub fn num_extant_parental_demes(&self) -> usize {
        self.num_extant_demes(Generation::Parent)
    }

    pub fn num_extant_child_demes(&self) -> usize {
        self.num_extant_demes(Generation::Child)
    }
}

#[cfg(test)]
mod test_functions {
    use super::*;

    pub fn update_ancestry_proportions(
        sources: &[usize],
        source_proportions: &[f64],
        ancestry_proportions: &mut [f64],
    ) {
        assert_eq!(sources.len(), source_proportions.len());
        let sum = source_proportions.iter().fold(0.0, |a, b| a + b);
        ancestry_proportions.iter_mut().for_each(|a| *a *= 1. - sum);
        sources
            .iter()
            .zip(source_proportions.iter())
            .for_each(|(source, proportion)| ancestry_proportions[*source] += proportion);
    }

    pub fn ancestry_proportions_from_graph(
        graph: &ForwardGraph,
        child_deme: usize,
    ) -> Option<Vec<f64>> {
        graph.child_deme_sizes()?;

        let mut rv = vec![0.0; graph.child_deme_sizes().unwrap().len()];

        let deme = graph.child_demes.get(child_deme).unwrap();

        if !deme.ancestors().is_empty() {
            for (a, p) in deme.ancestors().iter().zip(deme.proportions().iter()) {
                rv[*a] = f64::from(*p);
            }
        } else {
            rv[child_deme] = 1.0;
        }

        let mut sources: Vec<usize> = vec![];
        let mut source_proportions: Vec<f64> = vec![];

        // FIXME: a bit of an anti-pattern here:
        // * We are assumming the state of graph is correct!
        // * We should, instead, use the last updated time
        //   to get the pulses from the underlying demes::Graph.
        for p in &graph.pulses {
            sources.clear();
            source_proportions.clear();
            let dest = *graph.deme_to_index.get(p.dest()).unwrap();
            if dest == child_deme {
                for (s, d) in p.sources().iter().zip(p.proportions().iter()) {
                    let source = *graph.deme_to_index.get(s).unwrap();
                    sources.push(source);
                    source_proportions.push(f64::from(*d));
                }
                update_ancestry_proportions(&sources, &source_proportions, &mut rv);
            }
        }

        sources.clear();
        source_proportions.clear();

        for m in &graph.migrations {
            let d = *graph.deme_to_index.get(m.dest()).unwrap();
            if d == child_deme {
                let s = *graph.deme_to_index.get(m.source()).unwrap();
                sources.push(s);
                source_proportions.push(f64::from(m.rate()));
            }
        }
        update_ancestry_proportions(&sources, &source_proportions, &mut rv);

        Some(rv)
    }

    pub fn test_model_duration(graph: &mut ForwardGraph) {
        for time in graph.time_iterator() {
            graph.update_state(time).unwrap();
            // assert!(graph.parental_demes().is_some(), "{}", time);
            assert!(
                graph.num_extant_parental_demes() > 0,
                "{:?} {:?}",
                time,
                graph.end_time()
            );
            assert!(graph
                .parental_deme_sizes()
                .unwrap()
                .iter()
                .any(|size| size > &0.0));
            if time == graph.end_time() - 1.0.into() {
                assert!(graph.child_deme_sizes().is_none(), "time = {:?}", time);
            } else {
                assert!(graph.child_deme_sizes().is_some(), "time = {:?}", time);
            }
        }
    }
}

#[cfg(test)]
mod graphs_for_testing {
    pub fn three_deme_model() -> demes::Graph {
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

    pub fn four_deme_model() -> demes::Graph {
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
      end_time: 49
 - name: D
   ancestors: [C, B]
   proportions: [0.5, 0.5]
   start_time: 49
   epochs:
    - start_size: 50
";
        demes::loads(yaml).unwrap()
    }

    pub fn one_generation_model() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 50
      end_time: 1
    - start_size: 100
";
        demes::loads(yaml).unwrap()
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
        assert_eq!(graph.num_extant_parental_demes(), 1);
        assert_eq!(
            graph.parental_deme_sizes().unwrap().get(0),
            Some(&demes::DemeSize::from(100.0))
        );
        graph.update_state(75_i32).unwrap();
        // assert_eq!(graph.parental_demes().unwrap().iter().count(), 1);
        assert_eq!(graph.num_extant_parental_demes(), 1);
        assert_eq!(
            graph.parental_deme_sizes().unwrap().get(0),
            Some(&demes::DemeSize::from(200.0))
        );

        // The last generation
        graph.update_state(150_i32).unwrap();
        assert_eq!(graph.num_extant_parental_demes(), 1);
        assert_eq!(graph.num_extant_child_demes(), 0);
        assert!(graph.ancestry_proportions(0).is_none());

        // One past the last generation
        graph.update_state(151_i32).unwrap();
        //assert!(graph.parental_demes().is_none());
        assert_eq!(graph.num_extant_parental_demes(), 0);
        assert_eq!(graph.num_extant_child_demes(), 0);
        assert!(graph.child_deme_sizes().is_none());

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

            assert!(graph
                .parental_deme_sizes()
                .unwrap()
                .iter()
                .all(|size| size == &expected.0));

            assert!(graph
                .child_deme_sizes()
                .unwrap()
                .iter()
                .all(|size| size == &expected.1));
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
        let demes_graph = graphs_for_testing::three_deme_model();
        let mut g = ForwardGraph::new(demes_graph, 100, None).unwrap();
        g.update_state(0).unwrap(); // first generation of model
        assert!(g
            .parental_deme_sizes()
            .unwrap()
            .iter()
            .any(|size| size > &0.0));
        assert_eq!(g.num_demes_in_model(), 3); // There are 3 demes in the model
        assert_eq!(g.parental_deme_sizes().unwrap().len(), 3); // There are 3 demes in the model
        for (i, size) in g.parental_deme_sizes().unwrap().iter().enumerate() {
            if i == 0 {
                assert!(size > &0.0);
            } else {
                assert!(size <= &0.0);
            }
        }

        g.update_state(100).unwrap(); // Last generation of ancestor deme
        for (i, size) in g.parental_deme_sizes().unwrap().iter().enumerate() {
            if i == 0 {
                assert!(size > &0.0);
            } else {
                assert!(size <= &0.0);
            }
        }
        for (i, size) in g.child_deme_sizes().unwrap().iter().enumerate() {
            if i == 0 {
                assert!(size <= &0.0);
            } else {
                assert!(size > &0.0);
            }
        }

        g.update_state(101).unwrap(); // Last generation of ancestor deme
        for (i, size) in g.parental_deme_sizes().unwrap().iter().enumerate() {
            if i == 0 {
                assert!(size <= &0.0);
            } else {
                assert!(size > &0.0);
            }
        }
        for (i, size) in g.child_deme_sizes().unwrap().iter().enumerate() {
            if i == 0 {
                assert!(size <= &0.0);
            } else {
                assert!(size > &0.0);
            }
        }
    }

    #[test]
    fn test_one_generation_model() {
        {
            let demes_graph = graphs_for_testing::one_generation_model();
            let mut graph = ForwardGraph::new(demes_graph, 0, None).unwrap();
            assert_eq!(graph.end_time(), 2.0.into());
            for i in graph.time_iterator() {
                graph.update_state(i).unwrap();
                assert!(graph
                    .parental_deme_sizes()
                    .unwrap()
                    .iter()
                    .any(|size| size > &0.0));
                if i == graph.end_time() - 1.0.into() {
                    assert!(graph.child_deme_sizes().is_none(), "time = {:?}", i);
                } else {
                    assert!(graph.child_deme_sizes().is_some(), "time = {:?}", i);
                }
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
        if let Some(deme) = graph.parent_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        if let Some(deme) = graph.child_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(100.))
            );
        } else {
            panic!();
        }
        // one generation before end
        graph.update_state(149).unwrap();
        if let Some(deme) = graph.child_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        graph.update_state(150).unwrap(); // last gen
        assert!(graph.child_demes.get(0).is_none());
        if let Some(deme) = graph.parent_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }

        // 1/2-way into the final epoch
        graph.update_state(125).unwrap();
        let expected_size: f64 = 100. + ((49. - 25.) / (49.)) * (200. - 100.);
        let expected_size = demes::DemeSize::from(expected_size.round());
        assert_eq!(
            graph.parental_deme_sizes().unwrap().get(0),
            Some(&expected_size)
        );

        let expected_size: f64 = 100. + ((49. - 24.) / (49.)) * (200. - 100.);
        let expected_size = demes::DemeSize::from(expected_size.round());
        assert_eq!(
            graph.child_deme_sizes().unwrap().get(0),
            Some(&expected_size)
        );
    }

    #[test]
    fn test_two_epoch_model_linear_decline() {
        let demes_graph = two_epoch_model_linear_decline();
        let mut graph = ForwardGraph::new(demes_graph, 100_u32, None).unwrap();
        graph.update_state(100).unwrap(); // last generation of the 1st epoch
        if let Some(deme) = graph.parent_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        if let Some(deme) = graph.child_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        // one generation before end
        graph.update_state(149).unwrap();
        if let Some(deme) = graph.child_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(100.))
            );
        } else {
            panic!();
        }
        graph.update_state(150).unwrap(); // last gen
        assert!(graph.child_demes.get(0).is_none());
        if let Some(deme) = graph.parent_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(100.))
            );
        } else {
            panic!();
        }

        // 1/2-way into the final epoch
        graph.update_state(125).unwrap();
        let expected_size: demes::DemeSize = (200_f64 + ((49. - 25.) / (49.)) * (100. - 200.))
            .round()
            .into();
        assert_eq!(
            graph.parental_deme_sizes().unwrap().get(0),
            Some(&expected_size)
        );
        let expected_size =
            demes::DemeSize::from((200_f64 + ((49. - 24.) / (49.)) * (100. - 200.)).round());
        assert_eq!(
            graph.child_deme_sizes().unwrap().get(0),
            Some(&expected_size)
        );
    }

    #[test]
    fn test_two_epoch_model_exponential_growth() {
        let demes_graph = two_epoch_model_exponential_growth();
        let mut graph = ForwardGraph::new(demes_graph, 100_u32, None).unwrap();
        graph.update_state(100).unwrap(); // last generation of the 1st epoch
        if let Some(deme) = graph.parent_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        if let Some(deme) = graph.child_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(100.))
            );
        } else {
            panic!();
        }
        // one generation before end
        graph.update_state(149).unwrap();
        if let Some(deme) = graph.child_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }
        graph.update_state(150).unwrap(); // last gen
        assert!(graph.child_demes.get(0).is_none());
        if let Some(deme) = graph.parent_demes.get(0) {
            assert_eq!(
                deme.current_size().unwrap(),
                Some(demes::DemeSize::from(200.))
            );
        } else {
            panic!();
        }

        // 1/2-way into the final epoch
        graph.update_state(125).unwrap();
        let g = (200_f64.ln() - 100_f64.ln()) / 49.0;
        let expected_size = demes::DemeSize::from((100.0 * (g * 24.0).exp()).round());
        assert_eq!(
            graph.parental_deme_sizes().unwrap().get(0),
            Some(&expected_size)
        );
        let g = (200_f64.ln() - 100_f64.ln()) / 49.0;
        let expected_size = demes::DemeSize::from((100.0 * (g * 25.0).exp()).round());
        assert_eq!(
            graph.child_deme_sizes().unwrap().get(0),
            Some(&expected_size)
        );
        test_functions::test_model_duration(&mut graph);
    }
}

#[cfg(test)]
mod test_deme_ancestors {
    use super::{test_functions::test_model_duration, *};

    #[test]
    fn test_three_deme_model() {
        let demes_graph = graphs_for_testing::three_deme_model();
        let mut graph =
            ForwardGraph::new(demes_graph, 100, Some(demes::RoundTimeToInteger::F64)).unwrap();

        {
            graph.update_state(0).unwrap();
            let deme = graph.parent_demes.get(0).unwrap();
            assert!(deme.is_extant());
            assert_eq!(deme.ancestors().len(), 0);
        }

        {
            graph.update_state(100).unwrap();
            let deme = graph.parent_demes.get(0).unwrap();
            assert!(deme.is_extant());
            assert_eq!(deme.ancestors().len(), 0);

            for descendant_deme in [1_usize, 2] {
                let deme = graph.child_demes.get(descendant_deme).unwrap();
                assert!(deme.is_extant());
                assert_eq!(deme.ancestors().len(), 1);
                assert_eq!(deme.ancestors()[0], 0_usize);
            }
        }

        // Runs to 149, which is last generation
        // that has a child gen
        for generation in 101_i32..150 {
            graph.update_state(generation).unwrap();
            let deme = graph.parent_demes.get(0).unwrap();
            assert_eq!(deme.ancestors().len(), 0);

            for descendant_deme in [1_usize, 2] {
                let deme = graph.child_demes.get(descendant_deme).unwrap();
                assert!(deme.is_extant());
                assert_eq!(deme.ancestors().len(), 0);
            }
        }
        test_model_duration(&mut graph);
    }

    #[test]
    fn test_four_deme_model() {
        let demes_graph = graphs_for_testing::four_deme_model();
        let mut graph =
            ForwardGraph::new(demes_graph, 100, Some(demes::RoundTimeToInteger::F64)).unwrap();

        {
            graph.update_state(0).unwrap();
            for (i, deme) in graph.child_demes.iter().enumerate() {
                if i < 1 {
                    assert!(deme.is_extant());
                    assert!(graph
                        .ancestry_proportions(i)
                        .unwrap()
                        .iter()
                        .any(|p| p > &0.0));
                } else {
                    assert!(!graph
                        .ancestry_proportions(i)
                        .unwrap()
                        .iter()
                        .any(|p| p > &0.0));
                }
            }
        }

        {
            graph.update_state(100).unwrap();
            let deme = graph.parent_demes.get(0).unwrap();
            assert!(deme.is_extant());
            assert_eq!(deme.ancestors().len(), 0);

            for descendant_deme in [1_usize, 2] {
                let deme = graph.child_demes.get(descendant_deme).unwrap();
                assert!(deme.is_extant());
                assert_eq!(deme.ancestors().len(), 1);
                assert_eq!(deme.ancestors()[0], 0_usize);
                match graph.ancestry_proportions(descendant_deme) {
                    Some(ancestry) => {
                        for ancestor in deme.ancestors() {
                            assert_eq!(ancestry[*ancestor], 1.0);
                        }
                    }
                    None => panic!("expected Some(ancestry)"),
                }
            }
        }
        {
            graph.update_state(101).unwrap();
            let deme = graph.parent_demes.get(0).unwrap();
            assert_eq!(deme.ancestors().len(), 0);

            for descendant_deme in [1_usize, 2] {
                let deme = graph.child_demes.get(descendant_deme).unwrap();
                if descendant_deme == 2 {
                    assert_eq!(
                        graph.child_deme_sizes().unwrap().get(2),
                        Some(&demes::DemeSize::from(0.0))
                    );
                } else {
                    assert!(deme.is_extant());
                }
                assert_eq!(deme.ancestors().len(), 0);
            }
            let deme = graph.child_demes.get(3).unwrap();
            assert!(deme.is_extant());
            assert_eq!(deme.ancestors().len(), 2);
            assert_eq!(deme.ancestors(), &[2, 1]);
            assert_eq!(deme.proportions().len(), 2);
            assert_eq!(deme.proportions(), &[0.5, 0.5]);

            for ancestor in deme.ancestors() {
                assert!(graph.parent_demes.get(*ancestor).unwrap().is_extant());
            }
        }

        {
            graph.update_state(102).unwrap();
            for deme in graph.child_demes.iter() {
                assert!(deme.ancestors().is_empty());
                assert!(deme.proportions().is_empty());
            }
        }
        test_model_duration(&mut graph);
    }

    #[test]
    fn test_four_deme_model_duration() {
        let demes_graph = graphs_for_testing::four_deme_model();
        let mut graph =
            ForwardGraph::new(demes_graph, 73, Some(demes::RoundTimeToInteger::F64)).unwrap();
        test_model_duration(&mut graph);
    }
}

#[cfg(test)]
mod test_pulses {
    use super::*;

    fn model_with_pulses() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 50
 - name: B
   epochs:
    - start_size: 50
pulses:
 - sources: [A]
   dest: B
   time: 100
   proportions: [0.5]
";
        demes::loads(yaml).unwrap()
    }

    #[test]
    fn test_pulses() {
        let demes_g = model_with_pulses();
        let mut g = ForwardGraph::new(demes_g, 200., None).unwrap();

        for time in [199, 200] {
            g.update_state(time).unwrap();
            for deme in 0..g.num_demes_in_model() {
                let ap = test_functions::ancestry_proportions_from_graph(&g, deme).unwrap();
                let p = g.ancestry_proportions(deme).unwrap();
                for pi in p {
                    assert!(pi <= &1.0);
                }
                p.iter()
                    .zip(ap.iter())
                    .for_each(|(a, b)| assert!((a - b).abs() <= 1e-9, "{} {}", a, b));
            }
        }
    }
}

#[cfg(test)]
mod test_migrations {
    use super::*;
    fn model_with_migrations() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 50
 - name: B
   epochs:
    - start_size: 50
migrations:
 - source: A
   dest: B
   rate: 0.25
   start_time: 50
   end_time: 25
 - source: B
   dest: A
   rate: 0.1
   start_time: 40
   end_time: 20
 - demes: [A, B]
   rate: 0.05
   start_time: 15
";
        demes::loads(yaml).unwrap()
    }

    #[test]
    fn test_migrations() {
        let demes_g = model_with_migrations();
        let mut g = ForwardGraph::new(demes_g, 200., None).unwrap();

        // Making sure ;)
        // g.update_state(250).unwrap();
        // assert!(g.child_demes().is_none());
        // assert!(g.parental_demes().is_some());

        // At forward time 200, we are at the
        // start of the first migration epoch,
        // meaning that children born at 201 can be migrants
        g.update_state(200).unwrap();
        assert_eq!(g.migrations.len(), 1);

        g.update_state(209).unwrap();
        assert_eq!(g.migrations.len(), 1);

        g.update_state(210).unwrap();
        assert_eq!(g.migrations.len(), 2);

        g.update_state(230).unwrap();
        assert_eq!(g.migrations.len(), 0);

        // Symmetric mig, so 2 Asymmetric deals...
        g.update_state(235).unwrap();
        assert_eq!(g.migrations.len(), 2);
    }
}

#[cfg(test)]
mod test_fractional_times {
    use super::*;

    fn bad_epoch_end_time() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 100
      end_time: 10.2
";
        demes::loads(yaml).unwrap()
    }

    fn bad_deme_start_time() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 100
      end_time: 10
 - name: B
   start_time: 50.1
   ancestors: [A]
   epochs:
    - start_size: 50
";
        demes::loads(yaml).unwrap()
    }

    fn bad_pulse_time() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 100
      end_time: 10
 - name: B
   start_time: 50
   ancestors: [A]
   epochs:
    - start_size: 50
pulses:
 - sources: [A]
   dest: B
   proportions: [0.5]
   time: 30.2
";
        demes::loads(yaml).unwrap()
    }

    fn bad_migration_start_time() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 100
      end_time: 10
 - name: B
   start_time: 50
   ancestors: [A]
   epochs:
    - start_size: 50
migrations:
 - source: A
   dest: B
   rate: 0.5
   start_time: 30.2
   end_time: 10
";
        demes::loads(yaml).unwrap()
    }

    fn bad_migration_end_time() -> demes::Graph {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 100
      end_time: 10
 - name: B
   start_time: 50
   ancestors: [A]
   epochs:
    - start_size: 50
migrations:
 - source: A
   dest: B
   rate: 0.5
   start_time: 30
   end_time: 10.2
";
        demes::loads(yaml).unwrap()
    }

    fn run_invalid_model(f: fn() -> demes::Graph) {
        let demes_graph = f();
        assert!(ForwardGraph::new(demes_graph, 1, None).is_err());
    }

    #[test]
    fn test_invalid_models() {
        run_invalid_model(bad_epoch_end_time);
        run_invalid_model(bad_deme_start_time);
        run_invalid_model(bad_pulse_time);
        run_invalid_model(bad_migration_start_time);
        run_invalid_model(bad_migration_end_time);
    }
}

#[cfg(test)]
mod test_ancestry_proportions {
    use super::*;
    use test_functions::*;

    #[test]
    fn sequential_pulses_at_same_time_two_demes() {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 1000
 - name: B
   epochs:
    - start_size: 1000
 - name: C
   epochs:
    - start_size: 1000
pulses:
- sources: [A]
  dest: C
  proportions: [0.33]
  time: 10
- sources: [B]
  dest: C
  proportions: [0.25]
  time: 10
";
        let demes_graph = demes::loads(yaml).unwrap();
        let mut graph = ForwardGraph::new(demes_graph, 50, None).unwrap();
        let index_a: usize = 0;
        let index_b: usize = 1;
        let index_c: usize = 2;
        let mut ancestry_proportions = vec![0.0; 3];
        ancestry_proportions[index_c] = 1.0;
        update_ancestry_proportions(&[index_a], &[0.33], &mut ancestry_proportions);
        update_ancestry_proportions(&[index_b], &[0.25], &mut ancestry_proportions);
        graph.update_state(50.0).unwrap();
        assert_eq!(graph.ancestry_proportions(2).unwrap().len(), 3);
        graph
            .ancestry_proportions(2)
            .unwrap()
            .iter()
            .zip(ancestry_proportions.iter())
            .for_each(|(a, b)| assert!((a - b).abs() <= 1e-9));
    }

    #[test]
    fn sequential_pulses_at_same_time_two_demes_reverse_pulse_order() {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 1000
 - name: B
   epochs:
    - start_size: 1000
 - name: C
   epochs:
    - start_size: 1000
pulses:
- sources: [B]
  dest: C
  proportions: [0.25]
  time: 10
- sources: [A]
  dest: C
  proportions: [0.33]
  time: 10
";
        let demes_graph = demes::loads(yaml).unwrap();
        let mut graph = ForwardGraph::new(demes_graph, 50, None).unwrap();
        graph.update_state(50.0).unwrap();
        assert_eq!(graph.ancestry_proportions(2).unwrap().len(), 3);
        let ancestry_proportions = ancestry_proportions_from_graph(&graph, 2).unwrap();
        assert_eq!(ancestry_proportions.len(), 3);
        graph
            .ancestry_proportions(2)
            .unwrap()
            .iter()
            .zip(ancestry_proportions.iter())
            .for_each(|(a, b)| assert!((a - b).abs() <= 1e-9, "{} {}", a, b));

        for child_deme in [0, 1] {
            let independent = ancestry_proportions_from_graph(&graph, child_deme).unwrap();
            assert_eq!(
                graph.ancestry_proportions(child_deme),
                Some(independent.as_slice())
            );
        }
    }

    #[test]
    fn test_pulse_that_does_nothing() {
        let yaml = "
time_units: generations
description:
 The pulse does nothing because
 demes B and C are new after time 50.
 Demes with finite start_time must
 have 100% of their ancestry accounted
 for in the ancestors field.
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
pulses:
 - sources: [A]
   dest: B
   time: 50
   proportions: [0.5]
";
        let demes_graph = demes::loads(yaml).unwrap();
        let mut graph = ForwardGraph::new(demes_graph, 0, None).unwrap();
        graph.update_state(0.0).unwrap();
        for child_deme in [1, 2] {
            match graph.child_demes[child_deme].current_size() {
                Ok(value) => assert!(value.is_some()),
                Err(_) => panic!("unexpected Error"),
            }
            assert_eq!(graph.child_demes[child_deme].ancestors().len(), 1);
            assert_eq!(
                graph.ancestry_proportions(child_deme),
                Some([1., 0., 0.].as_slice())
            );
        }
    }
    #[test]
    fn sequential_pulses_at_same_time_two_demes_with_symmetric_migration() {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 1000
 - name: B
   epochs:
    - start_size: 1000
 - name: C
   epochs:
    - start_size: 1000
pulses:
- sources: [A]
  dest: C
  proportions: [0.33]
  time: 10
- sources: [B]
  dest: C
  proportions: [0.25]
  time: 10
migrations:
 - demes: [A, B, C]
   rate: 1e-3
   start_time: 11
   end_time: 0
";
        let demes_graph = demes::loads(yaml).unwrap();
        let mut graph = ForwardGraph::new(demes_graph, 50, None).unwrap();
        graph.update_state(51.0).unwrap();

        for child_deme in 0..3 {
            let a = graph.ancestry_proportions(child_deme).unwrap();
            let e = ancestry_proportions_from_graph(&graph, child_deme).unwrap();
            assert_eq!(a.len(), e.len());
            a.iter()
                .zip(e.iter())
                .for_each(|(a, b)| assert!((a - b).abs() <= 1e-9));
        }
    }
}

#[cfg(test)]
mod migration_matrix {
    use super::*;
    use test_functions::*;

    #[test]
    fn simple_two_way_migration() {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 1000
 - name: B
   epochs:
    - start_size: 1000
 - name: C
   epochs:
    - start_size: 1000
migrations:
- source: B
  dest: C
  rate: 0.25
- source: A
  dest: C
  rate: 0.25 
";
        let demes_graph = demes::loads(yaml).unwrap();
        let mut graph = ForwardGraph::new(demes_graph, 10, None).unwrap();
        for generation in 0..10 {
            graph.update_state(generation).unwrap();
            assert_eq!(
                graph.ancestry_proportions(2),
                Some([0.25, 0.25, 0.5].as_slice())
            );
            assert_eq!(
                graph.ancestry_proportions(2),
                Some([0.25, 0.25, 0.5].as_slice())
            );
        }
    }

    #[test]
    fn simple_two_way_symmetric_migration() {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - start_size: 1000
 - name: B
   epochs:
    - start_size: 1000
 - name: C
   epochs:
    - start_size: 1000
migrations:
- demes: [A, B, C]
  rate: 0.25
";
        let demes_graph = demes::loads(yaml).unwrap();
        let mut graph = ForwardGraph::new(demes_graph, 10, None).unwrap();
        for generation in 0..10 {
            graph.update_state(generation).unwrap();
            for deme in 0..3 {
                let expected = ancestry_proportions_from_graph(&graph, deme).unwrap();
                assert_eq!(
                    graph.ancestry_proportions(deme),
                    Some(expected.as_slice()),
                    "{} {:?}",
                    deme,
                    expected
                );
            }
        }
    }
}

#[cfg(test)]
mod test_cloning_selfing_rates {
    use super::*;

    #[test]
    fn test_rate_changes() {
        let yaml = "
time_units: generations
demes:
 - name: A
   epochs:
    - selfing_rate: 0.5
      end_time: 10
      start_size: 50
    - cloning_rate: 0.25
";
        let demes_graph = demes::loads(yaml).unwrap();
        let mut graph = ForwardGraph::new(demes_graph, 10, None).unwrap();
        graph.update_state(0.0).unwrap();
        match graph.selfing_rates() {
            Some(rates) => assert_eq!(rates[0], 0.5),
            None => panic!(),
        }
        match graph.cloning_rates() {
            Some(rates) => assert_eq!(rates[0], 0.0),
            None => panic!(),
        }
        graph.update_state(10.0).unwrap();
        match graph.selfing_rates() {
            Some(rates) => assert_eq!(rates[0], 0.0),
            None => panic!(),
        }
        match graph.cloning_rates() {
            Some(rates) => assert_eq!(rates[0], 0.25),
            None => panic!(),
        }
    }
}
