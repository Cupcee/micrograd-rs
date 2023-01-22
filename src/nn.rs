use core::fmt;
use rand::Rng;
use std::{
    iter::zip,
    ops::{Range, RangeInclusive},
    time::Instant,
};

use crate::autograd::Parameter;

#[derive(Clone)]
pub struct Neuron {
    weights: Vec<Parameter>,
    bias: Parameter,
    nonlinear: bool,
    in_dim: usize,
}

fn uniform_sample(range: RangeInclusive<f32>) -> f32 {
    rand::thread_rng().gen_range(range) as f32
}

impl Neuron {
    pub fn new(in_dim: usize, nonlinear: bool) -> Neuron {
        let weights = (0..in_dim)
            .map(|_| Parameter::from_scalar(uniform_sample(-1.0..=1.0)))
            .collect();
        let bias = Parameter::from_scalar(0.0);
        Neuron {
            weights,
            bias,
            nonlinear,
            in_dim,
        }
    }
    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
    // Shape
    // weights: (2,) x: (2,)
    pub fn forward(&self, x: Vec<Parameter>) -> Parameter {
        let act =
            zip(self.weights.clone(), x).fold(self.bias.clone(), |sum, (wi, xi)| sum + (wi * xi));
        if self.nonlinear {
            act.relu()
        } else {
            act
        }
    }
}

impl fmt::Debug for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Neuron")
            .field("weights", &self.weights)
            .field("bias", &self.bias)
            .finish()
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(in_dim: usize, out_dim: usize, nonlinear: bool) -> Layer {
        Layer {
            neurons: (0..out_dim)
                .map(|_| Neuron::new(in_dim, nonlinear))
                .collect(),
        }
    }
    fn parameters(&self) -> Vec<Parameter> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }
    pub fn forward(&self, x: Vec<Parameter>) -> Vec<Parameter> {
        self.neurons
            .clone()
            .iter()
            .map(|neuron| neuron.forward(x.clone()))
            .collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(dims: Vec<usize>) -> MLP {
        let n_dims = dims.len() - 1;
        MLP {
            layers: (0..n_dims)
                .map(|i| Layer::new(dims[i], dims[i + 1], i != (n_dims - 1)))
                .collect(),
        }
    }
    /// Forward pass for MLP.
    /// `x` is an n-dimensional datapoint.
    pub fn forward(&self, mut x: Vec<Parameter>) -> Vec<Parameter> {
        for layer in self.layers.iter() {
            x = layer.forward(x);
        }
        x
    }
    /// Zero gradients for all neuron parameters.
    pub fn zero_grad(&self) -> () {
        for mut param in self.parameters() {
            param.zero_grad();
        }
    }
    pub fn lr_step(&self, new_lr: f32) -> () {
        for mut param in self.parameters() {
            param.lr_step(new_lr);
        }
    }
    pub fn parameters(&self) -> Vec<Parameter> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}

impl fmt::Display for MLP {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut msg = String::from("MLP:");
        for layer in self.layers.iter() {
            msg.push_str("\nLayer:");
            for neuron in layer.neurons.iter() {
                msg.push_str(
                    format!(
                        "\nNeuron: ({}, {})",
                        neuron.in_dim,
                        if neuron.nonlinear { "ReLU" } else { "Linear" },
                    )
                    .as_str(),
                );
            }
        }
        write!(f, "{}", msg)
    }
}

pub fn loss(model: &MLP, preds: Vec<Parameter>, y: &Vec<f32>) -> (Parameter, f32) {
    // svm max margin loss
    let losses: Vec<Parameter> = zip(y, preds.clone())
        .map(|(yi, pi)| (Parameter::from_scalar(1.0) + (-Parameter::from_scalar(*yi)) * pi).relu())
        .collect();
    let n = losses.len();
    let data_loss = losses
        .into_iter()
        .reduce(|acc, param| acc + param.clone())
        .unwrap();
    let data_loss = data_loss * (Parameter::from_scalar(1.0) / Parameter::from_scalar(n as f32));

    // l2 regularization
    let alpha = Parameter::from_scalar(1e-4);
    let reg_loss = alpha
        * model
            .parameters()
            .into_iter()
            .reduce(|acc, param| acc + param.clone() * param.clone())
            .unwrap();
    let total_loss = data_loss + reg_loss;
    let matches = zip(y, preds).map(|(yi, pi)| (*yi > 0.0) == (pi.data() > 0.0));
    let n_true = matches
        .into_iter()
        .filter(|b| *b)
        .collect::<Vec<bool>>()
        .len();
    let acc = (n_true as f32) / (n as f32);

    (total_loss, acc)
}
