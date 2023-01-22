use core::fmt;
use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};
use uuid::Uuid;

#[derive(Debug)]
enum Operation {
    Init,
    Add,
    Sub,
    Mul,
    Neg,
    Div,
    Pow,
    ReLU,
}

/// A differentiable scalar value.
/// Wrapped into Parameter.
pub struct Value {
    hash: Uuid,
    data: f32,
    grad: f32,
    backward: Option<Box<dyn FnOnce() -> () + Send>>,
    previous: HashSet<Parameter>,
    op: Operation,
}

fn build_topo(param: Parameter, topo: &mut Vec<Parameter>, visited: &mut HashSet<Uuid>) {
    let hash = param.0.lock().unwrap().hash;
    if !visited.contains(&hash) {
        visited.insert(hash);
        param
            .0
            .lock()
            .unwrap()
            .previous
            .iter()
            .for_each(|child| build_topo(child.clone(), topo, visited));
        topo.push(param);
    }
}

/// Parameter is Value with reference counting and mutex support.
/// Backward passes form a recursive graph structure so Arc and Mutex
/// are needed for multithreading.
#[derive(Clone, Debug)]
pub struct Parameter(pub Arc<Mutex<Value>>);

impl Value {
    fn from_scalar(data: f32) -> Parameter {
        Parameter(Arc::new(Mutex::new(Value {
            hash: Uuid::new_v4(),
            data,
            grad: 0.0,
            backward: None,
            previous: HashSet::new(),
            op: Operation::Init,
        })))
    }
    fn new(data: f32, previous: HashSet<Parameter>, op: Operation) -> Value {
        Value {
            hash: Uuid::new_v4(),
            data,
            grad: 0.0,
            backward: None,
            previous,
            op,
        }
    }
}

impl Parameter {
    pub fn from_scalar(scalar: f32) -> Parameter {
        Value::from_scalar(scalar)
    }
    /// Passes Parameter through ReLU.
    pub fn relu(self) -> Parameter {
        let data = self.0.lock().unwrap().data;
        let out = Value::new(
            if data < 0.0 { 0.0 } else { data },
            HashSet::from([self.clone()]),
            Operation::ReLU,
        );
        let out = Arc::new(Mutex::new(out));
        let out_ref = Arc::clone(&out);

        out.lock().unwrap().backward = Some(Box::new(move || {
            let out_ref = out_ref.lock().unwrap();
            let out_data = out_ref.data;
            let out_grad = out_ref.grad;
            self.0.lock().unwrap().grad += if out_data > 0.0 { out_grad } else { 0.0 }
        }));
        Parameter(out)
    }
    /// Set Parameter gradient to zero.
    pub fn zero_grad(&mut self) -> () {
        self.0.lock().unwrap().grad = 0.0;
    }
    /// Increase reference count of this Parameter.
    pub fn clone(&self) -> Parameter {
        Parameter(Arc::clone(&self.0))
    }
    /// Performs a backward pass on the Parameter if it's defined.
    fn _backward(&self) -> () {
        let try_backward = self.0.lock().unwrap().backward.take();
        match try_backward {
            Some(back) => back(),
            None => (),
        }
    }
    /// Initiates a recursive backward pass from this Parameter through the
    /// computation graph in topological order.
    pub fn backward(&self) -> () {
        let mut topo_nodes: Vec<Parameter> = vec![];
        let mut visited_nodes: HashSet<Uuid> = HashSet::new();
        build_topo(self.clone(), &mut topo_nodes, &mut visited_nodes);
        self.0.lock().unwrap().grad = 1.0;
        topo_nodes.iter().rev().for_each(|value| value._backward());
    }
    /// Raises Parameter to power of `power`.
    pub fn pow(self, power: f32) -> Parameter {
        let data = self.0.lock().unwrap().data;
        let out = Value::new(
            data.powf(power),
            HashSet::from([self.clone()]),
            Operation::Pow,
        );
        let out = Arc::new(Mutex::new(out));
        let out_ref = Arc::clone(&out);

        out.lock().unwrap().backward = Some(Box::new(move || {
            let mut self_ref = self.0.lock().unwrap();
            let out_grad = out_ref.lock().unwrap().grad;
            self_ref.grad += (power * self_ref.data.powf(power - 1.0)) * out_grad;
        }));
        Parameter(out)
    }
    pub fn data(&self) -> f32 {
        self.0.lock().unwrap().data
    }
    pub fn lr_step(&mut self, new_lr: f32) -> () {
        let mut self_ref = self.0.lock().unwrap();
        self_ref.data -= new_lr * self_ref.grad;
    }
}

impl Hash for Parameter {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let borrow = self.0.lock().unwrap();
        borrow.hash.hash(state);
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Value {}

impl PartialEq for Parameter {
    fn eq(&self, other: &Self) -> bool {
        let borrow = self.0.lock().unwrap();
        let try_borrow_other = other.0.try_lock();
        match try_borrow_other {
            Ok(borrow_other) => borrow.hash == borrow_other.hash,
            Err(_) => true, // if referencing two same objects locking other would deadlock
        }
    }
}

impl std::ops::Add for Parameter {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let self_data = self.0.lock().unwrap().data;
        let other_data = other.0.lock().unwrap().data;

        let out = Value::new(
            self_data + other_data,
            HashSet::from([self.clone(), other.clone()]),
            Operation::Add,
        );

        let out = Arc::new(Mutex::new(out));
        let out_ref = Arc::clone(&out);

        out.lock().unwrap().backward = Some(Box::new(move || {
            let out_grad = out_ref.lock().unwrap().grad;
            self.0.lock().unwrap().grad += out_grad;
            other.0.lock().unwrap().grad += out_grad;
        }));
        Parameter(out)
    }
}

impl std::ops::Mul for Parameter {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let self_data = self.0.lock().unwrap().data;
        let other_data = other.0.lock().unwrap().data;

        let out = Value::new(
            self_data * other_data,
            HashSet::from([self.clone(), other.clone()]),
            Operation::Mul,
        );

        let out = Arc::new(Mutex::new(out));
        let out_ref = Arc::clone(&out);

        out.lock().unwrap().backward = Some(Box::new(move || {
            let out_grad = out_ref.lock().unwrap().grad;
            self.0.lock().unwrap().grad += other_data * out_grad;
            other.0.lock().unwrap().grad += self_data * out_grad;
        }));
        Parameter(out)
    }
}

impl std::ops::Neg for Parameter {
    type Output = Self;
    fn neg(self) -> Self {
        let out = self * Value::from_scalar(-1.0);
        out.0.lock().unwrap().op = Operation::Neg;
        out
    }
}

impl std::ops::Sub for Parameter {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let out = self + (-other);
        out.0.lock().unwrap().op = Operation::Sub;
        out
    }
}

impl std::ops::Div for Parameter {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let out = self * other.pow(-1.0);
        out.0.lock().unwrap().op = Operation::Div;
        out
    }
}

impl Eq for Parameter {}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "id: {}, data: {}, grad: {}, op: {:?}",
            self.hash.to_string(),
            self.data,
            self.grad,
            self.op
        )
    }
}
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Value")
            .field("hash", &self.hash)
            .field("data", &self.data)
            .field("grad", &self.grad)
            .finish()
    }
}

#[cfg(test)]
mod tests {

    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_scalar() {
        let x = Value::from_scalar(4.0);
        assert_eq!(x.0.lock().unwrap().data, 4.0);
    }

    #[test]
    fn test_add() {
        let x = Value::from_scalar(4.0);
        let y = Value::from_scalar(2.0);
        let z = x + y;
        assert_eq!(z.0.lock().unwrap().data, 6.0);
    }

    #[test]
    fn test_mul() {
        let x = Value::from_scalar(2.0);
        let y = Value::from_scalar(6.0);
        let z = x * y;
        assert_eq!(z.0.lock().unwrap().data, 12.0);
        let x = Value::from_scalar(-2.0);
        let y = Value::from_scalar(6.0);
        let z = x * y;
        assert_eq!(z.0.lock().unwrap().data, -12.0);
    }

    #[test]
    fn test_sanity_check() {
        let x = Value::from_scalar(-4.0);
        let z = Value::from_scalar(2.0) * x.clone() + Value::from_scalar(2.0) + x.clone();
        let q = z.clone().relu() + z.clone() * x.clone();
        let h = (z.clone() * z).relu();
        let y = h + q.clone() + q.clone() * x.clone();
        y.backward();

        // pytorch results for above
        // forward pass
        assert_eq!(y.0.lock().unwrap().data, -20.0);
        // backward pass
        assert_eq!(x.0.lock().unwrap().grad, 46.0);
    }

    #[test]
    fn test_more_ops() {
        let a = Value::from_scalar(-4.0);
        let b = Value::from_scalar(2.0);
        let c = a.clone() + b.clone();
        let d = a.clone() * b.clone() + b.clone().pow(3.0);
        let c = c.clone() + (c.clone() + Value::from_scalar(1.0));
        let c = c.clone() + (Value::from_scalar(1.0) + c.clone() + (-a.clone()));
        let d = d.clone() + (d.clone() * Value::from_scalar(2.0) + (b.clone() + a.clone()).relu());
        let d = d.clone() + (Value::from_scalar(3.0) * d + (b.clone() - a.clone()).relu());
        let e = c.clone() - d.clone();
        let f = e.pow(2.0);
        let g = f.clone() / Value::from_scalar(2.0);
        let g = g + (Value::from_scalar(10.0) / f.clone());
        g.backward();
        let (amg, bmg, gmg) = (a, b, g);

        // pytorch results for above
        // forward pass
        assert_approx_eq!(gmg.0.lock().unwrap().data, 24.7040816327, 1e-6);
        // backward pass
        assert_approx_eq!(amg.0.lock().unwrap().grad, 138.8338192420, 1e-6);
        assert_approx_eq!(bmg.0.lock().unwrap().grad, 645.5772594752, 1e-6);
    }
}
