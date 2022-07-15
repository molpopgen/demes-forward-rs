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

                // Stress-test that a deme > no. demes in model returns None
                assert!(graph
                    .ancestry_proportions(graph.num_demes_in_model())
                    .is_none());
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
