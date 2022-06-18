pub struct ForwardGraph {
    graph: demes::Graph,
}

impl ForwardGraph {
    pub fn new(
        graph: demes::Graph,
        rounding: Option<demes::RoundTimeToInteger>,
    ) -> Result<Self, crate::DemesForwardError> {
        let graph = match rounding {
            Some(r) => graph.to_integer_generations(r)?,
            None => graph.to_generations()?,
        };
        Ok(Self { graph })
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
    fn initialize_graph() {
        let demes_graph = two_epoch_model();
        ForwardGraph::new(demes_graph, None).unwrap();
    }

    #[test]
    fn invalid_conversion_error() {
        let demes_graph = two_epoch_model_invalid_conversion_to_generations();
        let result = ForwardGraph::new(demes_graph, Some(demes::RoundTimeToInteger::F64));
        assert!(matches!(
            result,
            Err(crate::DemesForwardError::DemesError(
                demes::DemesError::EpochError(_)
            ))
        ));
    }
}
