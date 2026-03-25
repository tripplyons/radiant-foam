use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq)]
pub enum ParameterError {
    GradientLengthMismatch { expected: usize, got: usize },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Parameter {
    pub values: Vec<f64>,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub step: u64,
}

impl Parameter {
    pub fn new(values: Vec<f64>, learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        let len = values.len();
        Self {
            values,
            m: vec![0.0; len],
            v: vec![0.0; len],
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-8,
            step: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn update_adam(&mut self, gradients: &[f64]) -> Result<(), ParameterError> {
        if gradients.len() != self.values.len() {
            return Err(ParameterError::GradientLengthMismatch {
                expected: self.values.len(),
                got: gradients.len(),
            });
        }

        self.step = self.step.saturating_add(1);
        let t = self.step as f64;
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);

        for (i, &g) in gradients.iter().enumerate() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;
            self.values[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Parameter, ParameterError};

    #[test]
    fn adam_update_changes_values_and_state() {
        let mut p = Parameter::new(vec![1.0, -2.0], 0.1, 0.9, 0.999);
        let grad = vec![0.5, -0.25];

        p.update_adam(&grad).expect("gradient length matches");

        assert_eq!(p.step, 1);
        assert!(p.values[0] < 1.0);
        assert!(p.values[1] > -2.0);
        assert_ne!(p.m[0], 0.0);
        assert_ne!(p.v[0], 0.0);
    }

    #[test]
    fn adam_update_rejects_gradient_length_mismatch() {
        let mut p = Parameter::new(vec![1.0, 2.0], 0.1, 0.9, 0.999);

        let result = p.update_adam(&[0.5]);

        assert!(matches!(
            result,
            Err(ParameterError::GradientLengthMismatch {
                expected: 2,
                got: 1
            })
        ));
    }
}
