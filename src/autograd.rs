use core::fmt;
use std::{
    cell::RefCell,
    collections::HashSet,
    hash::{Hash, Hasher},
    rc::Rc,
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
    // AddAssign,
    ReLU,
}

/// A differentiable scalar value.
/// Wrapped into Parameter.
struct Value {
    hash: Uuid,
    data: f64,
    grad: f64,
    backward: Option<Box<dyn FnMut() -> ()>>,
    previous: HashSet<Parameter>,
    op: Operation,
}

fn build_topo(param: Parameter, topo: &mut Vec<Parameter>, visited: &mut HashSet<Uuid>) {
    let hash = param.0.borrow().hash;
    if !visited.contains(&hash) {
        visited.insert(hash);
        param
            .0
            .borrow()
            .previous
            .iter()
            .for_each(|child| build_topo(child.clone(), topo, visited));
        topo.push(param);
    }
}

/// Parameter is Value with reference counting and mutable borrows.
/// These are needed because an autograd system is modeled as a graph.
pub struct Parameter(Rc<RefCell<Value>>);

impl Value {
    pub fn from_scalar(data: f64) -> Parameter {
        Parameter(Rc::new(RefCell::new(Value {
            hash: Uuid::new_v4(),
            data,
            grad: 0.0,
            backward: None,
            previous: HashSet::new(),
            op: Operation::Init,
        })))
    }
    fn new(data: f64, previous: HashSet<Parameter>, op: Operation) -> Value {
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
    /// Passes Parameter through a ReLU.
    pub fn relu(self) -> Parameter {
        let data = self.0.borrow().data;
        let out = Value::new(
            if data < 0.0 { 0.0 } else { data },
            HashSet::from([self.clone()]),
            Operation::ReLU,
        );
        let out = Rc::new(RefCell::new(out));
        let out_ref = Rc::clone(&out);

        out.borrow_mut().backward = Some(Box::new(move || {
            let out_ref = out_ref.borrow();
            self.0.borrow_mut().grad += if out_ref.data > 0.0 {
                out_ref.grad
            } else {
                0.0
            }
        }));
        Parameter(out)
    }
    /// Set Parameter gradient to zero.
    pub fn zero_grad(&mut self) -> () {
        self.0.borrow_mut().grad = 0.0;
    }
    /// Increase reference count of this Parameter.
    pub fn clone(&self) -> Parameter {
        Parameter(Rc::clone(&self.0))
    }
    /// Performs a backward pass on the Parameter if it's defined.
    fn _backward(&self) -> () {
        println!("{}", self.0.borrow());
        let try_backward = self.0.borrow_mut().backward.take();
        match try_backward {
            Some(mut back) => back(),
            None => (),
        }
    }
    /// Initiates a recursive backward pass from this Parameter to graph root.
    pub fn backward(&self) -> () {
        let mut topo_nodes: Vec<Parameter> = vec![];
        let mut visited_nodes: HashSet<Uuid> = HashSet::new();
        build_topo(self.clone(), &mut topo_nodes, &mut visited_nodes);
        println!("Topo");
        topo_nodes
            .iter()
            .for_each(|node| println!("{}", node.0.borrow()));
        println!("\n");
        self.0.borrow_mut().grad = 1.0;
        println!("_backward");
        topo_nodes.iter().rev().for_each(|value| value._backward());
        println!("\n");
    }
    /// Raises Parameter to power of `power`.
    pub fn pow(self, power: f64) -> Parameter {
        let data = self.0.borrow().data;
        let out = Value::new(
            data.powf(power),
            HashSet::from([self.clone()]),
            Operation::Pow,
        );
        let out = Rc::new(RefCell::new(out));
        let out_ref = Rc::clone(&out);

        out.borrow_mut().backward = Some(Box::new(move || {
            let mut self_ref = self.0.borrow_mut();
            self_ref.grad += (power * self_ref.data.powf(power - 1.0)) * out_ref.borrow().grad;
        }));
        Parameter(out)
    }
}

impl Hash for Parameter {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let borrow = self.0.borrow();
        let borrow = &*borrow;
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
        let borrow = self.0.borrow();
        let borrow = &*borrow;
        let borrow_other = other.0.borrow();
        let borrow_other = &*borrow_other;

        borrow.hash == borrow_other.hash
    }
}

impl std::ops::Add for Parameter {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let data = self.0.borrow().data + other.0.borrow().data;
        let out = Value::new(
            data,
            HashSet::from([self.clone(), other.clone()]),
            Operation::Add,
        );

        let out = Rc::new(RefCell::new(out));
        let out_ref = Rc::clone(&out);

        out.borrow_mut().backward = Some(Box::new(move || {
            match self.0.try_borrow_mut() {
                Ok(mut sref) => {
                    sref.grad += out_ref.borrow().grad;
                }
                Err(_) => println!("Add: already borrowed: {}", self.0.borrow().hash),
            }
            match other.0.try_borrow_mut() {
                Ok(mut oref) => {
                    oref.grad += out_ref.borrow().grad;
                }
                Err(_) => println!("Add: already borrowed: {}", other.0.borrow().hash),
            }
        }));
        Parameter(out)
    }
}

impl std::ops::Mul for Parameter {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let data = self.0.borrow().data * other.0.borrow().data;
        let out = Value::new(
            data,
            HashSet::from([self.clone(), other.clone()]),
            Operation::Mul,
        );

        let out = Rc::new(RefCell::new(out));
        let out_ref = Rc::clone(&out);
        let self_data = self.0.borrow().data;
        let other_data = other.0.borrow().data;

        out.borrow_mut().backward = Some(Box::new(move || {
            match self.0.try_borrow_mut() {
                Ok(mut sref) => {
                    sref.grad += other_data * out_ref.borrow().grad;
                }
                Err(_) => println!("Mul: already borrowed: {}", self.0.borrow().hash),
            }
            match other.0.try_borrow_mut() {
                Ok(mut oref) => {
                    oref.grad += self_data * out_ref.borrow().grad;
                }
                Err(_) => println!("Mul: already borrowed: {}", other.0.borrow().hash),
            }
        }));
        Parameter(out)
    }
}

impl std::ops::AddAssign for Parameter {
    fn add_assign(&mut self, _other: Self) -> () {
        // TODO: implement?
        unimplemented!("below does not work");
        // *self = self.clone() + other;
        // self.0.borrow_mut().op = Operation::AddAssign;
    }
}

impl std::ops::Neg for Parameter {
    type Output = Self;
    fn neg(self) -> Self {
        let out = self * Value::from_scalar(-1.0);
        out.0.borrow_mut().op = Operation::Neg;
        out
    }
}

impl std::ops::Sub for Parameter {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let out = self + (-other);
        out.0.borrow_mut().op = Operation::Sub;
        out
    }
}

impl std::ops::Div for Parameter {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let out = self * other.pow(-1.0);
        out.0.borrow_mut().op = Operation::Div;
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

#[cfg(test)]
mod tests {

    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_scalar() {
        let x = Value::from_scalar(4.0);
        assert_eq!(x.0.borrow().data, 4.0);
    }

    #[test]
    fn test_add() {
        let x = Value::from_scalar(4.0);
        let y = Value::from_scalar(2.0);
        let z = x + y;
        assert_eq!(z.0.borrow().data, 6.0);
    }

    #[test]
    fn test_mul() {
        let x = Value::from_scalar(2.0);
        let y = Value::from_scalar(6.0);
        let z = x * y;
        assert_eq!(z.0.borrow().data, 12.0);
        let x = Value::from_scalar(-2.0);
        let y = Value::from_scalar(6.0);
        let z = x * y;
        assert_eq!(z.0.borrow().data, -12.0);
    }

    #[test]
    fn test_sanity_check() {
        let x = Value::from_scalar(-4.0);
        let z = Value::from_scalar(2.0) * x.clone() + Value::from_scalar(2.0) + x.clone();
        let q = z.clone().relu() + z.clone() * x.clone();
        let h = (z.clone() * z.clone()).relu();
        let y = h + q.clone() + q.clone() * x.clone();
        y.backward();

        // pytorch results for above
        // forward pass
        assert_eq!(y.0.borrow().data, -20.0);
        // backward pass
        assert_eq!(x.0.borrow().grad, 46.0);
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
        assert_approx_eq!(gmg.0.borrow().data, 24.7040816327, 1e-6);
        // backward pass
        assert_approx_eq!(amg.0.borrow().grad, 138.8338192420, 1e-6);
        assert_approx_eq!(bmg.0.borrow().grad, 645.5772594752, 1e-6);
    }
}
