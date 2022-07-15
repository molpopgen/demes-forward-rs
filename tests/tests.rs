use demes_forward::demes;

struct ModelFirstLast {
    first: Option<demes_forward::ForwardTime>,
    last: Option<demes_forward::ForwardTime>,
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

fn iterate_all_generations(graph: &mut demes_forward::ForwardGraph) -> ModelFirstLast {
    let mut first_time_visited = None;
    let mut last_time_visited = None;
    for time in graph.time_iterator() {
        match first_time_visited {
            None => first_time_visited = Some(time),
            Some(_) => (),
        }
        if time == demes_forward::ForwardTime::from(0.0) {
            assert!(graph.last_time_updated().is_none());
        }
        last_time_visited = Some(time);
        graph.update_state(time).unwrap();
        assert_eq!(graph.last_time_updated(), Some(time));
        match graph.offspring_deme_sizes() {
            Some(child_deme_sizes) => {
                assert!(time < graph.end_time() - 1.0.into());
                assert!(graph.any_extant_parental_demes());
                assert!(graph.any_extant_offspring_demes());
                let parental_deme_sizes = graph.parental_deme_sizes().unwrap();
                let selfing_rates = graph.selfing_rates().unwrap();
                let cloning_rates = graph.cloning_rates().unwrap();
                assert_eq!(child_deme_sizes.len(), graph.num_demes_in_model());
                assert_eq!(parental_deme_sizes.len(), graph.num_demes_in_model());
                assert_eq!(selfing_rates.len(), graph.num_demes_in_model());
                assert_eq!(cloning_rates.len(), graph.num_demes_in_model());
                for i in 0..graph.num_demes_in_model() {
                    if selfing_rates[i] > 0.0 {
                        assert!(child_deme_sizes[i] > 0.0);
                    }
                    if cloning_rates[i] > 0.0 {
                        assert!(child_deme_sizes[i] > 0.0);
                    }
                    let ancestry_proportions = graph.ancestry_proportions(i).unwrap();
                    for j in 0..ancestry_proportions.len() {
                        if ancestry_proportions[j] > 0.0 {
                            assert!(parental_deme_sizes[j] > 0.0);
                            assert!(
                                child_deme_sizes[i] > 0.0,
                                "{:?}, {} => {:?}",
                                time,
                                i,
                                child_deme_sizes
                            );
                        }
                    }
                }
            }
            None => {
                assert!(!graph.any_extant_offspring_demes());
                assert!(graph.selfing_rates().is_none());
                assert!(graph.cloning_rates().is_none());
                assert!(time <= graph.end_time() - 1.0.into());
            }
        }
    }
    ModelFirstLast {
        first: first_time_visited,
        last: last_time_visited,
    }
}

#[test]
fn test_four_deme_model_pub_api_only() {
    let demes_graph = four_deme_model();
    let mut graph =
        demes_forward::ForwardGraph::new(demes_graph, 100, Some(demes::RoundTimeToInteger::F64))
            .unwrap();
    let last_time = iterate_all_generations(&mut graph);
    assert_eq!(
        last_time.last,
        Some(demes_forward::ForwardTime::from(150.0))
    );
    assert_eq!(last_time.first, Some(demes_forward::ForwardTime::from(0.0)));
}

#[test]
fn test_four_deme_model_pub_api_only_start_after_zero() {
    let demes_graph = four_deme_model();
    let mut graph =
        demes_forward::ForwardGraph::new(demes_graph, 100, Some(demes::RoundTimeToInteger::F64))
            .unwrap();
    graph.update_state(50.0).unwrap();
    let last_time = iterate_all_generations(&mut graph);
    assert_eq!(
        last_time.last,
        Some(demes_forward::ForwardTime::from(150.0))
    );
    assert_eq!(
        last_time.first,
        Some(demes_forward::ForwardTime::from(50.0))
    );
}

#[test]
fn gutenkunst2009() {
    let yaml = "
description: The Gutenkunst et al. (2009) OOA model.
doi:
- https://doi.org/10.1371/journal.pgen.1000695
time_units: years
generation_time: 25

demes:
- name: ancestral
  description: Equilibrium/root population
  epochs:
  - {end_time: 220e3, start_size: 7300}
- name: AMH
  description: Anatomically modern humans
  ancestors: [ancestral]
  epochs:
  - {end_time: 140e3, start_size: 12300}
- name: OOA
  description: Bottleneck out-of-Africa population
  ancestors: [AMH]
  epochs:
  - {end_time: 21.2e3, start_size: 2100}
- name: YRI
  description: Yoruba in Ibadan, Nigeria
  ancestors: [AMH]
  epochs:
  - start_size: 12300
- name: CEU
  description: Utah Residents (CEPH) with Northern and Western European Ancestry
  ancestors: [OOA]
  epochs:
  - {start_size: 1000, end_size: 29725}
- name: CHB
  description: Han Chinese in Beijing, China
  ancestors: [OOA]
  epochs:
  - {start_size: 510, end_size: 54090}

migrations:
- {demes: [YRI, OOA], rate: 25e-5}
- {demes: [YRI, CEU], rate: 3e-5}
- {demes: [YRI, CHB], rate: 1.9e-5}
- {demes: [CEU, CHB], rate: 9.6e-5}
";
    let demes_graph = demes::loads(yaml).unwrap();
    let mut graph =
        demes_forward::ForwardGraph::new(demes_graph, 0, Some(demes::RoundTimeToInteger::F64))
            .unwrap();
    // graph.update_state(0.0).unwrap();
    let _last_time = iterate_all_generations(&mut graph);
}
