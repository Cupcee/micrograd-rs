use std::{f32::consts::PI, iter::zip};

use num_traits::Float;
use rand::Rng;
use rand::{seq::SliceRandom, thread_rng};
use rand_distr::{Distribution, Normal};

pub fn shuffle<T: Copy>(slices: &mut [&mut [T]]) {
    if slices.len() > 0 {
        let mut rng = thread_rng();

        let len = slices[0].len();
        assert!(slices.iter().all(|s| s.len() == len));

        for i in 0..len {
            let next = rng.gen_range(i..len);

            for slice in slices.iter_mut() {
                let tmp: T = slice[i];
                slice[i] = slice[next];
                slice[next] = tmp;
            }
        }
    }
}

pub fn linspace<T: Float + std::convert::From<u16>>(l: T, h: T, n: usize) -> Vec<T> {
    let size: T = (n as u16 - 1)
        .try_into()
        .expect("too many elements: max is 2^16");
    let dx = (h - l) / size;

    (1..=n)
        .scan(-dx, |a, _| {
            *a = *a + dx;
            Some(*a)
        })
        .collect()
}

// Credit: https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/datasets/_samples_generator.py#L724
pub fn make_moons(
    n_samples: usize,
    should_shuffle: bool,
    noise: f32,
) -> (Vec<(f32, f32)>, Vec<f32>) {
    let n_samples_in = n_samples;
    let n_samples_out = n_samples;

    let outer_circ_x: Vec<f32> = linspace(0.0, PI, n_samples_out)
        .iter()
        .map(|v| v.cos())
        .collect();
    let outer_circ_y: Vec<f32> = linspace(0.0, PI, n_samples_out)
        .iter()
        .map(|v| v.sin())
        .collect();
    let inner_circ_x: Vec<f32> = linspace(0.0, PI, n_samples_out)
        .iter()
        .map(|v| 1.0 - v.cos())
        .collect();
    let inner_circ_y: Vec<f32> = linspace(0.0, PI, n_samples_out)
        .iter()
        .map(|v| 0.5 - v.sin())
        .collect();

    let concat_x: Vec<f32> = outer_circ_x
        .into_iter()
        .chain(inner_circ_x.into_iter())
        .collect();
    let concat_y: Vec<f32> = outer_circ_y
        .into_iter()
        .chain(inner_circ_y.into_iter())
        .collect();
    let mut x: Vec<(f32, f32)> = zip(concat_x, concat_y).collect();
    let mut y: Vec<f32> = vec![0.0; n_samples_out]
        .into_iter()
        .chain(vec![1.0; n_samples_in])
        .collect();

    if should_shuffle {
        let mut x1: Vec<f32> = x.clone().into_iter().map(|(x1, _)| x1).collect();
        let mut x2: Vec<f32> = x.clone().into_iter().map(|(_, x2)| x2).collect();
        shuffle(&mut [&mut x1, &mut x2, &mut y]);
        x = zip(x1, x2).collect();
    }

    if noise > 0.0 {
        // add gaussian noise to x
        let normal = Normal::new(0.0, noise).unwrap();
        x = x
            .into_iter()
            .map(|(x, y)| {
                (
                    x + normal.sample(&mut thread_rng()),
                    y + normal.sample(&mut thread_rng()),
                )
            })
            .collect();
    }
    (x, y)
}
