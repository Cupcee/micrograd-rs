use crate::autograd::Parameter;

pub struct Neuron {
    parameters: Vec<Parameter>,
}

impl Neuron {
    pub fn zero_grad(&mut self) -> () {
        for param in self.parameters.iter_mut() {
            param.zero_grad();
        }
    }
}
