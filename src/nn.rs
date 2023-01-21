use crate::autograd::Parameter;

pub struct Neuron<'a> {
    parameters: Vec<Parameter<'a>>,
}

impl<'a> Neuron<'a> {
    pub fn zero_grad(&mut self) -> () {
        for param in self.parameters.iter_mut() {
            param.zero_grad();
        }
    }
}
