use core::fmt;
use std::{
    cell::RefCell,
    collections::HashSet,
    hash::{Hash, Hasher},
    rc::Rc,
};
use uuid::Uuid;

#[derive(Clone, Copy, Debug)]
pub enum Operation {
    Init, // given to value initialized from literal
    Add,
    Mul,
    ReLU,
}

/// Differentiable Value
pub struct Value<'a> {
    pub hash: Uuid,
    pub data: f64,
    pub grad: f64,
    pub backward: Option<Box<dyn FnOnce() -> () + 'a>>,
    pub previous: HashSet<MutValue<'a>>,
    pub op: Operation,
}

impl<'a> Value<'a> {
    pub fn from_scalar(data: f64) -> MutValue<'a> {
        let out = Value {
            hash: Uuid::new_v4(),
            data,
            grad: 0.0,
            backward: None,
            previous: HashSet::new(),
            op: Operation::Init,
        };
        MutValue(Rc::new(RefCell::new(out)))
    }
    fn new(
        data: f64,
        backward: Option<Box<dyn FnOnce() -> () + 'a>>,
        previous: HashSet<MutValue<'a>>,
        op: Operation,
    ) -> Value<'a> {
        Value {
            hash: Uuid::new_v4(),
            data,
            grad: 0.0,
            backward,
            previous,
            op,
        }
    }
    pub fn new_mut(
        data: f64,
        backward: Option<Box<dyn FnOnce() -> () + 'a>>,
        previous: HashSet<MutValue<'a>>,
        op: Operation,
    ) -> MutValue<'a> {
        MutValue(Rc::new(RefCell::new(Value::new(
            data, backward, previous, op,
        ))))
    }
}

pub struct MutValue<'a>(Rc<RefCell<Value<'a>>>);

impl<'a> MutValue<'a> {
    pub fn relu(self) -> MutValue<'a> {
        let data = self.0.borrow().data;
        let out = Value::new(
            if data < 0.0 { 0.0 } else { data },
            None,
            HashSet::from([MutValue(Rc::clone(&self.0))]),
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
        MutValue(out)
    }
    pub fn clone(&self) -> MutValue<'a> {
        MutValue(Rc::clone(&self.0))
    }
    pub fn backward(&self) -> () {
        let backward_pass = self.0.borrow_mut().backward.take().unwrap();
        backward_pass()
    }
}

impl Hash for MutValue<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let borrow = self.0.borrow();
        let borrow = &*borrow;
        borrow.hash.hash(state);
    }
}

impl PartialEq for Value<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Value<'_> {}

impl PartialEq for MutValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        let borrow = self.0.borrow();
        let borrow = &*borrow;
        let borrow_other = other.0.borrow();
        let borrow_other = &*borrow_other;

        borrow.hash == borrow_other.hash
    }
}

impl std::ops::Add for MutValue<'_> {
    type Output = Self;
    fn add<'a>(self, other: Self) -> Self {
        let data = self.0.borrow().data + other.0.borrow().data;
        let out = Value::new(
            data,
            None,
            HashSet::from([MutValue(Rc::clone(&self.0)), MutValue(Rc::clone(&other.0))]),
            Operation::Mul,
        );

        let out = Rc::new(RefCell::new(out));
        let out_ref = Rc::clone(&out);

        out.borrow_mut().backward = Some(Box::new(move || {
            self.0.borrow_mut().grad += out_ref.borrow().grad;
            other.0.borrow_mut().grad += out_ref.borrow().grad;
        }));
        MutValue(out)
    }
}

impl std::ops::Mul for MutValue<'_> {
    type Output = Self;
    fn mul<'a>(self, other: Self) -> Self {
        let data = self.0.borrow().data * other.0.borrow().data;
        let out = Value::new(
            data,
            None,
            HashSet::from([MutValue(Rc::clone(&self.0)), MutValue(Rc::clone(&other.0))]),
            Operation::Mul,
        );

        let out = Rc::new(RefCell::new(out));
        let out_ref = Rc::clone(&out);

        out.borrow_mut().backward = Some(Box::new(move || {
            self.0.borrow_mut().grad *= out_ref.borrow().grad;
            other.0.borrow_mut().grad *= out_ref.borrow().grad;
        }));
        MutValue(out)
    }
}

impl Eq for MutValue<'_> {}

pub fn add<'a>(lhs: MutValue<'a>, rhs: MutValue<'a>) -> MutValue<'a> {
    let data = lhs.0.borrow().data + rhs.0.borrow().data;
    // lhs and rhs value references are passed to addition result
    // and the backward pass
    // 1) Rc needed for multiple references to Value
    // 2) RefCell needed for mutable borrows in callback
    let out = Value::new(
        data,
        None,
        HashSet::from([MutValue(Rc::clone(&lhs.0)), MutValue(Rc::clone(&rhs.0))]),
        Operation::Add,
    );

    // initialize out with reference counting
    let out = Rc::new(RefCell::new(out));
    // new ref for closure
    let out_ref = Rc::clone(&out);

    out.borrow_mut().backward = Some(Box::new(move || {
        // lhs, rhs, out_ref ownership moved to closure
        // backward pass values are evaluated at some point in time
        lhs.0.borrow_mut().grad += out_ref.borrow().grad;
        rhs.0.borrow_mut().grad += out_ref.borrow().grad;
    }));
    MutValue(out)
}

pub fn mul<'a>(lhs: MutValue<'a>, rhs: MutValue<'a>) -> MutValue<'a> {
    let data = lhs.0.borrow().data * rhs.0.borrow().data;
    let out = Value::new(
        data,
        None,
        HashSet::from([MutValue(Rc::clone(&lhs.0)), MutValue(Rc::clone(&rhs.0))]),
        Operation::Mul,
    );

    let out = Rc::new(RefCell::new(out));
    let out_ref = Rc::clone(&out);

    out.borrow_mut().backward = Some(Box::new(move || {
        lhs.0.borrow_mut().grad *= out_ref.borrow().grad;
        rhs.0.borrow_mut().grad *= out_ref.borrow().grad;
    }));
    MutValue(out)
}

pub fn relu<'a>(value: MutValue<'a>) -> MutValue<'a> {
    let data = value.0.borrow().data;
    let out = Value::new(
        if data < 0.0 { 0.0 } else { data },
        None,
        HashSet::from([MutValue(Rc::clone(&value.0))]),
        Operation::ReLU,
    );
    let out = Rc::new(RefCell::new(out));
    let out_ref = Rc::clone(&out);

    out.borrow_mut().backward = Some(Box::new(move || {
        let out_ref = out_ref.borrow();
        value.0.borrow_mut().grad += if out_ref.data > 0.0 {
            out_ref.grad
        } else {
            0.0
        }
    }));
    MutValue(out)
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "data: {}, grad: {}, previous: Vec[...], op: {:?}",
            self.data, self.grad, self.op
        )
    }
}

#[cfg(test)]
mod tests {
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
    }

    #[test]
    fn test_sanity_check() {
        let x = Value::from_scalar(-4.0);
        let z = Value::from_scalar(2.0) * x.clone() + Value::from_scalar(2.0) + x.clone();
        let q = z.clone().relu() + z.clone() * x.clone();
        let h = (z.clone() * z.clone()).relu();
        let y = h + q.clone() + q.clone() * x.clone();
        y.backward();
        let (xmg, ymg) = (x, y);

        // pytorch results for above
        // xpt = tensor([-4.], dtype=torch.float64, requires_grad=True)
        // ypt = tensor([-20.], dtype=torch.float64, grad_fn=<AddBackward0>)
        assert_eq!(ymg.0.borrow().data, -20.0);
        assert_eq!(xmg.0.borrow().data, -4.0);
    }
}
