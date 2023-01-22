use micrograd_rs::{
    autograd::Parameter,
    math::{make_moons, shuffle},
    nn::{loss, MLP},
};
use plotters::prelude::*;
use std::{iter::zip, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plots/test.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let gradient = colorous::VIRIDIS;
    let (x, y01) = make_moons(100, true, 0.1);
    // make y between -1 or 1
    let mut y: Vec<f32> = y01.iter().map(|yi| yi * 2.0 - 1.0).collect();

    let mut chart = ChartBuilder::on(&root)
        .caption("moons", ("sans-serif", 24).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-2f32..2f32, -2f32..2f32)?;

    chart.configure_mesh().draw()?;

    let xy = zip(x.clone(), y01);
    chart
        .draw_series(xy.into_iter().map(|(x, y)| {
            let color = gradient.eval_continuous(y.into());
            Circle::new(x, 3, RGBColor(color.r, color.g, color.b).filled())
        }))
        .unwrap();

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    // Ok(())

    let model = MLP::new(vec![2, 16, 16, 1]);

    println!("{}", model);
    println!("Number of parameters: {}", model.parameters().len());

    let mut x1: Vec<f32> = x.clone().into_iter().map(|(x1, _)| x1).collect();
    let mut x2: Vec<f32> = x.clone().into_iter().map(|(_, x2)| x2).collect();
    for epoch in 0..100 {
        // forward pass
        shuffle(&mut [&mut x1, &mut x2, &mut y]);
        let start = Instant::now();
        let preds: Vec<Parameter> = zip(&x1, &x2)
            .flat_map(|(x1, x2)| {
                model.forward(vec![
                    Parameter::from_scalar(*x1),
                    Parameter::from_scalar(*x2),
                ])
            })
            .collect();

        println!("after forward {}", start.elapsed().as_millis());

        // compute loss
        let (total_loss, acc) = loss(&model, preds.clone(), &y);

        println!("after loss {}", start.elapsed().as_millis());
        // backward pass
        model.zero_grad();
        println!("after zero_grad {}", start.elapsed().as_millis());
        total_loss.backward();

        println!("after backward {}", start.elapsed().as_millis());
        // update learning rate
        let lr = 1.0 - 0.9 * (epoch as f32) / 100.0;
        model.lr_step(lr);

        dbg!("after lr_step {}", start.elapsed().as_millis());
        if epoch % 1 == 0 {
            println!(
                "Epoch: {}, time: {}, loss: {}, accuracy: {}%",
                epoch,
                start.elapsed().as_millis(),
                total_loss.data(),
                acc * 100.0
            );
        }
    }

    Ok(())
}
