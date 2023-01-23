use micrograd_rs::{
    autograd::Parameter,
    math::{make_moons, shuffle},
    nn::{loss, MLP},
    plotting::draw_chart,
};
use std::{iter::zip, sync::Arc, thread, time::Instant};

fn main() -> () {
    let (x, y01) = make_moons(100, true, 0.1);

    draw_chart(&x, &y01).ok();

    // make y between -1 or 1
    let mut y: Vec<f32> = y01.iter().map(|yi| yi * 2.0 - 1.0).collect();

    let model = Arc::new(MLP::new(vec![2, 16, 16, 1]));

    println!("{}", model);
    println!("Number of parameters: {}", model.parameters().len());

    // need to split for shuffle
    let mut x1: Vec<f32> = x.clone().into_iter().map(|(x1, _)| x1).collect();
    let mut x2: Vec<f32> = x.clone().into_iter().map(|(_, x2)| x2).collect();

    for epoch in 0..100 {
        let start = Instant::now();
        shuffle(&mut [&mut x1, &mut x2, &mut y]);
        let mut handles = Vec::<thread::JoinHandle<Vec<Parameter>>>::new();
        zip(&x1, &x2).for_each(|(x1, x2)| {
            let model_ref = Arc::clone(&model);
            let (x1, x2) = (*x1, *x2);
            // process each point's forward pass in a separate thread
            let jh = thread::spawn(move || {
                model_ref.forward(vec![Parameter::from_scalar(x1), Parameter::from_scalar(x2)])
            });
            handles.push(jh);
        });
        let preds: Vec<Parameter> = handles
            .into_iter()
            .flat_map(|jh| jh.join().unwrap())
            .collect();

        // compute loss
        let (total_loss, acc) = loss(&model, preds.clone(), &y);

        // backward pass
        model.zero_grad();
        total_loss.backward();

        // update learning rate
        let lr = 1.0 - 0.9 * (epoch as f32) / 100.0;
        model.lr_step(lr);

        if epoch % 1 == 0 {
            println!(
                "Epoch: {}, time: {}ms, loss: {:.6}, accuracy: {:.4}%",
                epoch,
                start.elapsed().as_millis(),
                total_loss.data(),
                acc * 100.0
            );
        }
    }
}
