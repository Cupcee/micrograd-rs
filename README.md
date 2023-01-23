# micrograd-rs

## About

Rust rewrite of Andrej Karpathy's Python [micrograd](https://github.com/karpathy/micrograd)
repo for self-study.

## Dependencies

* Rust compiler
* Built and tested on `rustc 1.65.0`
* Cargo crates

## Contents

* `autograd.rs` contains a simple graph-like data structure of nodes
called `Parameter`s
* `nn.rs` contains definitions for `Neuron`, `Layer` and `MLP`, building
on top of `Parameter` definitions
* `math.rs` has util functions
* `main.rs` has example training code for `MLP` displaying that it works
